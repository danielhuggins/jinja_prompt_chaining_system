"""Jinja Prompt Chaining System - A Jinja-based prompt chaining engine for LLM interactions."""

from jinja2 import Environment, FileSystemLoader
from .parser import LLMQueryExtension

__version__ = "0.1.0"

def create_environment(template_path=None) -> Environment:
    """Create a Jinja environment with the LLMQuery extension registered."""
    # Create environment with basic settings
    env = Environment(
        loader=FileSystemLoader(template_path) if template_path else None,
        enable_async=True,  # Enable async support for potential future use
        extensions=[LLMQueryExtension]
    )
    
    # Make the extension instance available in the global namespace
    env.globals['extension'] = env.extensions[LLMQueryExtension.identifier]
    
    return env 