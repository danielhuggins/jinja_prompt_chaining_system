"""Jinja Prompt Chaining System - A Jinja-based prompt chaining engine for LLM interactions."""

# Import and expose parallel versions by default
from .parallel_integration import create_environment_with_parallel as create_environment
from .parallel_integration import render_template_parallel

# Keep original API functions for backward compatibility
from .api import render_prompt, render_prompt_async

__version__ = "0.1.0" 