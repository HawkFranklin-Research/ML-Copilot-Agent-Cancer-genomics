# ml_copilot_agent/__init__.py

import os
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import logging

def initialize(
    llm_provider: str = "openai", # 'openai' or 'gemini'
    api_key: str = None,
    model: str = None,
    temperature: float = 0.1,
):
    """Initialize the ML Copilot with LLM provider, API key, and settings."""

    if llm_provider == "openai":
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env var.")
        os.environ["OPENAI_API_KEY"] = api_key # Ensure env var is set for potential downstream use
        llm_model = model or "gpt-4o" # Default OpenAI model
        print(f"Initializing LLM with OpenAI provider. Model: {llm_model}")
        Settings.llm = OpenAI(model=llm_model, temperature=temperature)

    elif llm_provider == "gemini":
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key must be provided via argument or GOOGLE_API_KEY env var.")
        # Gemini doesn't require setting env var explicitly for the library
        llm_model = model or "models/gemini-1.5-flash" # Default Gemini model
        print(f"Initializing LLM with Google Gemini provider. Model: {llm_model}")
        # Note: Gemini might require safety_settings configuration depending on the content.
        # Add safety_settings=... if needed.
        Settings.llm = Gemini(model_name=llm_model, api_key=api_key, temperature=temperature)

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}. Choose 'openai' or 'gemini'.")

    print(f"LLM settings configured: Provider={llm_provider}, Model={Settings.llm.metadata.model_name}, Temperature={temperature}")
