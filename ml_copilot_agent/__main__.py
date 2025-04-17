# ml_copilot_agent/__main__.py

import asyncio
import sys
import os
import argparse # Use argparse for better argument handling

from .workflow import MLWorkflow
from . import initialize

def main():
    parser = argparse.ArgumentParser(description="Run the ML Copilot Agent.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider to use ('openai' or 'gemini'). Default: openai"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the selected provider. Uses environment variables (OPENAI_API_KEY or GOOGLE_API_KEY) if not provided."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific LLM model name to use (e.g., 'gpt-4o', 'models/gemini-1.5-pro-latest'). Uses provider defaults if not set."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200, # Increased default timeout
        help="Workflow step timeout in seconds. Default: 1200"
    )
    parser.add_argument(
        "--verbose",
        action="store_true", # Make verbose a flag
        help="Enable verbose output from the workflow and agent."
    )

    args = parser.parse_args()

    try:
        initialize(
            llm_provider=args.provider,
            api_key=args.api_key,
            model=args.model
            # Temperature could be added as an arg if needed
        )
    except ValueError as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)

    async def run_workflow():
        # Pass verbose flag to workflow if it uses it
        workflow = MLWorkflow(timeout=args.timeout, verbose=args.verbose)
        await workflow.run()

    asyncio.run(run_workflow())

if __name__ == "__main__":
    main()
