# ml_copilot_agent/workflow.py
import os
import asyncio
import json
import shlex # For safer parsing if needed
from typing import Optional, Union, List, Dict, Any

from llama_index.core.workflow import (
    Workflow, Context, Event, StartEvent, StopEvent, step,
    WorkflowDefinition, InputStep, JoinStep, ConditionalStep
)
from llama_index.core.tools import CodeInterpreterToolSpec # Use core tool spec
from llama_index.core.agent import AgentRunner # Use core agent runner
from llama_index.core.base.llms.types import ChatMessage, MessageRole # For constructing agent messages
from llama_index.core import Settings
# Import GeminiAgent if using directly, otherwise rely on Settings.llm and core AgentRunner
# from llama_index.agent.gemini import GeminiAgent
from llama_index.agent.openai import OpenAIAgent # Keep for potential OpenAI-specific features if needed

# --- Events ---
# Keep StartEvent, StopEvent

class GetCommandEvent(Event):
    """Event carrying the raw user input command."""
    user_input: str

class ParseCommandEvent(Event):
    """Event carrying the parsed command type and parameters."""
    command_type: str
    parameters: Dict[str, Any] = {}
    original_input: str # Keep original for context

class ExecuteTaskEvent(Event):
    """Event carrying the instructions for the LLM Agent to execute."""
    prompt: str
    step_name: str
    expected_outputs: Optional[List[str]] = None # Variable names agent should aim to create/update

class ReportResultEvent(Event):
    """Event carrying the result/output from the agent execution."""
    agent_response: Any # Can be text, structured data, etc.
    status: str # 'success', 'error'
    step_name: str

class AskNextActionEvent(Event):
    """Event to prompt the user for the next action."""
    previous_step_name: str
    previous_status: str

# --- Workflow Definition ---

class HNSCCAnalysisWorkflow(Workflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure Settings.llm is configured
        if not Settings.llm:
            raise ValueError("LLM not configured in Settings. Please set Settings.llm via initialize().")

        # Initialize the Code Interpreter tool
        # Ensure OPENAI_API_KEY is set in env if using OpenAI for the tool
        if isinstance(Settings.llm, OpenAI) and not os.getenv("OPENAI_API_KEY"):
             print("Warning: OPENAI_API_KEY not found in environment, CodeInterpreterTool might fail if using OpenAI LLM.")
        # Note: CodeInterpreterTool might have limitations or different setup needs when used with Gemini.
        # Assume for now it works via the generic LLM interface or adapt if necessary.
        code_tool_spec = CodeInterpreterToolSpec()
        self.code_tool = code_tool_spec.to_tool_list()[0] # Get the single tool

        # Create the AgentRunner (more generic than specific OpenAIAgent)
        self.agent = AgentRunner(tools=[self.code_tool], llm=Settings.llm, verbose=kwargs.get('verbose', True))

        self.llm = Settings.llm # Keep reference if needed

        # --- Workflow Definition using functional API ---
        # start -> get_command -> parse_command -> decide_action -> execute_task? -> report_result -> ask_next -> get_command ...
        #                                          -> list_files? -> report_result -> ask_next -> get_command ...
        #                                          -> stop?

        # Define Steps
        start_step = InputStep(input_type=StartEvent)
        get_command_step = step(self.get_command, name="GetCommand")(start_step) # Initial command
        parse_command_step = step(self.parse_command, name="ParseCommand")(get_command_step)
        decide_action_step = step(self.decide_action, name="DecideAction")(parse_command_step)

        # Conditional Branches
        is_execute_task = ConditionalStep(
            condition=lambda ctx, ev: ev.command_type == "custom_task",
            if_step_name="ExecuteTask",
            else_step_name="ListFiles" # Example: default to list files if not custom
        )
        is_list_files = ConditionalStep(
             condition=lambda ctx, ev: ev.command_type == "list_files",
             if_step_name="ListFiles",
             else_step_name="StopWorkflow" # Example: Stop if not list or custom
        )
        is_stop = ConditionalStep(
             condition=lambda ctx, ev: ev.command_type == "exit",
             if_step_name="StopWorkflow",
             else_step_name="DecideAction" # Re-decide if not stop
        )

        execute_task_step = step(self.execute_task, name="ExecuteTask")(decide_action_step)
        list_files_step = step(self.list_files, name="ListFiles")(decide_action_step) # Needs input from DecideAction
        stop_workflow_step = step(self.stop_workflow, name="StopWorkflow")(decide_action_step) # Needs input from DecideAction

        report_result_step = step(self.report_result, name="ReportResult") # Takes input from execute_task or list_files
        ask_next_step = step(self.ask_next_action, name="AskNextAction")(report_result_step)
        get_next_command_step = step(self.get_command, name="GetNextCommand")(ask_next_step) # Loop back

        # --- Define Workflow ---
        # This part needs the add_edge/set_entry_step/set_exit_step methods if using WorkflowDefinition explicitly
        # Or rely on the decorator-based linking if simpler structure is sufficient.
        # For this complex branching/looping, explicit definition is better.

        # Example structure (needs adjustment based on exact Workflow API):
        # wf_def = WorkflowDefinition(...)
        # wf_def.add_step(start_step)
        # wf_def.add_step(get_command_step)
        # ... add all steps
        # wf_def.add_edge(start_step, get_command_step)
        # wf_def.add_edge(get_command_step, parse_command_step)
        # ... add all edges, including conditional logic triggers

        # Simpler approach for now: Rely on direct returns between steps for linear flow + router
        # Keep the @step decorated methods below.

    # --- Helper Methods ---
    def _get_input(self, prompt: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        """Gets user input."""
        while True:
            default_str = f" (default: {default})" if default else ""
            required_str = " (required)" if required else ""
            full_prompt = f"{prompt}{required_str}{default_str}: "
            try:
                value = input(full_prompt).strip()
                if value: return value
                if default is not None: return default
                if required: print("This input is required.")
                else: return None
            except EOFError:
                print("
Input stream closed. Exiting.")
                return "exit" # Treat EOF as exit command


    # --- Step Implementations ---

    @step
    async def get_command(self, ctx: Context, ev: Union[StartEvent, AskNextActionEvent]) -> GetCommandEvent:
        """Gets the user's command."""
        if isinstance(ev, StartEvent):
            print("
--- ML Copilot for HNSCC Analysis ---")
            print("Enter your command (e.g., 'run step 1: setup', 'list files', 'help', 'exit').")
        # If ev is AskNextActionEvent, the message is already printed by ask_next_action
        user_input = self._get_input("> ", required=True)
        return GetCommandEvent(user_input=user_input)

    @step
    async def parse_command(self, ctx: Context, ev: GetCommandEvent) -> ParseCommandEvent:
        """Parses the user's command (basic version)."""
        user_input = ev.user_input.lower().strip()
        command_type = "unknown"
        parameters = {}

        # Enhanced parsing - more flexible
        if user_input == "exit" or user_input == "quit":
            command_type = "exit"
        elif user_input == "help":
            command_type = "help"
        elif user_input.startswith("list files") or user_input.startswith("show files"):
             command_type = "list_files"
             # Example: could parse path like "list files /data"
             parts = shlex.split(ev.user_input)
             if len(parts) > 2: parameters['path'] = parts[2]
             else: parameters['path'] = '.' # Default path
        # Prioritize "custom task" or specific steps for HNSCC
        elif user_input.startswith("run step") or user_input.startswith("custom task"):
            command_type = "custom_task"
            # Extract the actual instruction
            parameters['instruction'] = ev.user_input # Pass the full original command as instruction
        else:
             # Fallback to treating unrecognized input as a custom task
             print(f"Treating unrecognized input as custom task: '{ev.user_input}'")
             command_type = "custom_task"
             parameters['instruction'] = ev.user_input

        return ParseCommandEvent(
            command_type=command_type,
            parameters=parameters,
            original_input=ev.user_input
        )

    @step
    async def decide_action(self, ctx: Context, ev: ParseCommandEvent) -> Union[ExecuteTaskEvent, ListFilesEvent, StopEvent, GetCommandEvent]:
        """Decides the next event based on the parsed command."""
        command_type = ev.command_type
        parameters = ev.parameters

        if command_type == "exit":
            return StopEvent(result="Workflow terminated by user.")
        elif command_type == "help":
            print("
--- Help ---")
            print("This agent assists with HNSCC analysis by executing Python code based on your instructions.")
            print("Example Commands:")
            print(" - 'run step 1: setup and load TCGA data'")
            print(" - 'run step 2: perform initial clustering evaluation'")
            print(" - 'run step 3: prepare data for binary classification'")
            print(" - 'run step 4: train classifiers and select features'")
            print(" - 'run step 5: aggregate and visualize classification results'")
            print(" - 'run step 6: validate models on full cohorts and summarize'")
            print(" - 'run step 7: select overall best model using combined criteria'")
            print(" - 'run step 8: generate final KM plots for best model'")
            print(" - 'run step 9: re-cluster using selected features'")
            print(" - 'list files [path]' - List files in the specified path (default: current directory).")
            print(" - 'exit' or 'quit' - Terminate the workflow.")
            print("----------------
")
            # Ask for a new command after help
            new_input = self._get_input("> ", required=True)
            return GetCommandEvent(user_input=new_input) # Send back to get_command/parse
        elif command_type == "list_files":
             # Create ExecuteTaskEvent to run 'ls' command via agent
             ls_command = f"ls -lh {parameters.get('path', '.')}"
             prompt = f"Please execute the following terminal command and show the output:
```bash
{ls_command}
```"
             return ExecuteTaskEvent(prompt=prompt, step_name="List Files")
        elif command_type == "custom_task":
             instruction = parameters.get('instruction', 'No instruction provided.')
             print(f"Preparing custom task: {instruction[:100]}...") # Show snippet

             # Construct a more robust prompt for the HNSCC context
             prompt = f"""
Objective: Execute the following step in an HNSCC biomarker discovery analysis pipeline using Python.

User Instruction:
"{instruction}"

Context:
- This is part of a larger analysis involving TCGA and CPTAC HNSCC data.
- Assume standard libraries like pandas, numpy, scikit-learn, matplotlib, seaborn, lifelines, pickle, os are available.
- Assume previous steps might have created variables or files (e.g., DataFrames like df_expr_tcga, df_surv_tcga, df_expr_cptac, df_surv_cptac; results like clustering info, models, metrics).
- Generate and execute Python code to fulfill the instruction.
- Import necessary libraries within the code block.
- Handle potential errors gracefully.
- Print key outputs, shapes of dataframes, results of analyses (like p-values), and confirmation messages (e.g., "File saved to /path/to/output.csv").
- If saving files, use descriptive names and organize them into subdirectories like 'data', 'plots', 'models', 'results' within a main output directory (e.g., 'hnscc_robust_analysis_output'). Create directories if they don't exist.

Example Libraries/Functions Often Used:
- pandas: read_csv, loc, iloc, groupby, merge, value_counts, apply
- numpy: array, mean, std, min, max, isnan, where
- sklearn.model_selection: train_test_split
- sklearn.feature_selection: SelectKBest, f_classif
- sklearn.cluster: KMeans, AgglomerativeClustering, GaussianMixture
- sklearn.linear_model: LogisticRegression
- sklearn.ensemble: RandomForestClassifier, GradientBoostingClassifier
- sklearn.svm: SVC
- sklearn.neighbors: KNeighborsClassifier
- sklearn.metrics: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
- lifelines: KaplanMeierFitter, logrank_test
- matplotlib.pyplot: figure, subplot, plot, scatter, boxplot, savefig, show, title, xlabel, ylabel, legend, grid, tight_layout
- seaborn: boxplot, lineplot, heatmap
- pickle: dump, load
- os: makedirs, path.join

Execute the Python code required for the user instruction.
"""
             return ExecuteTaskEvent(prompt=prompt, step_name=f"Custom Task: {instruction[:50]}...")
        else:
             # Should not happen if parse_command is exhaustive
             print("Error: Unknown command type reached DecideAction.")
             return StopEvent(result="Internal error: Unknown command type.")

    @step
    async def execute_task(self, ctx: Context, ev: ExecuteTaskEvent) -> ReportResultEvent:
        """Executes the prompt using the LLM agent."""
        print(f"
--- Executing Step: {ev.step_name} ---")
        print("Sending request to LLM Agent...")
        # For debugging: print(f"Prompt:
```
{ev.prompt}
```")
        status = "error"
        response_content = None
        try:
            # Use agent.chat for conversational interaction which might be better for code execution tasks
            # Construct a chat message history if needed, or just send the prompt
            response = await self.agent.achat(ev.prompt)
            response_content = response.response # Extract text response
            print("
--- Agent Response ---")
            print(response_content) # This should include code output
            print("--------------------
")
            # Basic check for errors in agent's text output
            if "error" in str(response_content).lower() or "exception" in str(response_content).lower():
                 print(f"Warning: Potential error detected in agent response for '{ev.step_name}'.")
                 # More robust check would involve analyzing code execution results if available
                 status = "potential_error"
            else:
                 status = "success"

        except Exception as e:
            response_content = f"An exception occurred during agent execution: {e}"
            print(f"
--- Error during {ev.step_name} ---")
            print(response_content)
            print("------------------------------------
")
            status = "error"

        return ReportResultEvent(agent_response=response_content, status=status, step_name=ev.step_name)

    @step
    async def list_files(self, ctx: Context, ev: ListFilesEvent) -> ReportResultEvent:
         """Handles listing files (kept separate for clarity, but ExecuteTask can handle it too)."""
         path = ev.parameters.get('path', '.') # Get path from parameters if provided
         print(f"
--- Listing Files in: {path} ---")
         status = "success"
         output = ""
         try:
             # More robust listing with error handling
             if not os.path.exists(path):
                 output = f"Error: Path '{path}' does not exist."
                 status = "error"
             elif not os.path.isdir(path):
                 output = f"Error: Path '{path}' is not a directory."
                 status = "error"
             else:
                 files = os.listdir(path)
                 if not files:
                      output = f"Directory '{path}' is empty."
                 else:
                      output = f"Files in '{path}':
" + "
".join(f"- {f}" for f in files)
                 print(output) # Print the list
         except Exception as e:
             output = f"Error listing files in '{path}': {e}"
             print(output)
             status = "error"
         print("-----------------------------
")
         return ReportResultEvent(agent_response=output, status=status, step_name="List Files")


    @step
    async def report_result(self, ctx: Context, ev: ReportResultEvent) -> AskNextActionEvent:
        """Reports the result of the previous step."""
        # The result is already printed in execute_task or list_files
        print(f"Step '{ev.step_name}' finished with status: {ev.status}.")
        # Potentially log results here or store critical info in context
        # e.g., await ctx.set(f'{ev.step_name}_result', ev.agent_response)
        return AskNextActionEvent(previous_step_name=ev.step_name, previous_status=ev.status)

    @step
    async def ask_next_action(self, ctx: Context, ev: AskNextActionEvent) -> GetCommandEvent:
        """Asks the user what to do next."""
        print("
What would you like to do next? (Enter command, 'help', or 'exit')")
        # This step transitions back to get_command by returning its expected input event
        # The actual input reading happens in get_command
        pass # No return needed, connects implicitly to get_command

    @step
    async def stop_workflow(self, ctx: Context, ev: StopEvent) -> None:
        """Handles the stop event."""
        print(f"
--- Workflow Terminated ---")
        print(ev.result)
        print("---------------------------
")

# Main execution part (if run directly)
async def run_hnscc_workflow():
    # Initialization should happen outside, via __main__.py
    if not Settings.llm:
        print("ERROR: LLM is not configured in llama_index.core.Settings.")
        return

    workflow = HNSCCAnalysisWorkflow(timeout=1800, verbose=True) # Longer timeout
    await workflow.run() # Starts with the InputStep

if __name__ == "__main__":
    # This block is for testing the workflow directly.
    # In practice, initialization and run are called from __main__.py
    print("Running workflow directly for testing...")
    print("Please ensure LLM is configured (e.g., API keys set).")

    # Example direct initialization for testing
    from ml_copilot_agent import initialize
    try:
         # Replace with your actual key or ensure env var is set
         # initialize(llm_provider="openai", api_key="YOUR_OPENAI_KEY")
         # initialize(llm_provider="gemini", api_key="YOUR_GOOGLE_KEY")
         initialize() # Tries default (openai) using env var
    except ValueError as e:
         print(f"Initialization failed: {e}")
         exit()
    except ImportError:
         print("Ensure llama_index and necessary LLM extensions are installed.")
         exit()

    asyncio.run(run_hnscc_workflow())
