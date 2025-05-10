"""API functions for the Jinja Prompt Chaining System.

This module provides exportable functions that can be used by applications
importing this library, providing the same functionality as the CLI.
"""

import os
import yaml
import asyncio
from typing import Dict, Any, Optional, Union
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from .parser import LLMQueryExtension
from .logger import RunLogger

def create_environment(template_path=None) -> Environment:
    """Create a Jinja environment with the LLMQuery extension registered."""
    # Create environment with basic settings
    env = Environment(
        loader=FileSystemLoader(template_path) if template_path else None,
        enable_async=True,  # Enable async support for potential future use
        extensions=[LLMQueryExtension],
        autoescape=False  # Disable HTML escaping by default
    )
    
    # Make the extension instance available in the global namespace
    env.globals['extension'] = env.extensions[LLMQueryExtension.identifier]
    
    return env

async def render_template_async(template_obj, context):
    """Render a Jinja template asynchronously."""
    return await template_obj.render_async(**context)

def render_template_sync(template_obj, context):
    """Render a Jinja template synchronously, handling async calls if necessary."""
    # First try in sync mode
    try:
        return template_obj.render(**context)
    except RuntimeError as e:
        if "async" in str(e).lower():
            # Fall back to async rendering
            return asyncio.run(render_template_async(template_obj, context))
        else:
            # Re-raise other errors
            raise

def render_prompt(
    template_path: Union[str, Path],
    context_data: Union[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Render a Jinja template containing LLM queries.
    
    This function provides the same functionality as the CLI command but can be
    called directly from Python code. It processes a prompt template file with
    LLM query tags and returns the rendered output.
    
    Args:
        template_path: Path to the Jinja template file.
        context_data: Either a path to a YAML context file or a dictionary with context data.
        output_path: Optional path where the rendered output will be saved.
        log_dir: Optional directory for storing logs.
        
    Returns:
        The rendered prompt output as a string.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
        ValueError: If the context data is invalid.
        RuntimeError: If there's an error during template rendering.
    """
    # Convert paths to strings if they're Path objects
    template_path = str(template_path) if isinstance(template_path, Path) else template_path
    output_path = str(output_path) if isinstance(output_path, Path) and output_path else output_path
    log_dir = str(log_dir) if isinstance(log_dir, Path) and log_dir else log_dir
    
    # Check if template file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load context data
    if isinstance(context_data, str):
        # It's a file path to a YAML file
        context_path = context_data
        if not os.path.exists(context_path):
            raise FileNotFoundError(f"Context file not found: {context_path}")
        
        try:
            with open(context_path, 'r') as f:
                ctx = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in context file: {str(e)}")
    else:
        # It's a dictionary
        ctx = context_data
        context_path = None  # No file path since it's a dict
    
    # Setup Jinja environment
    template_dir = os.path.dirname(os.path.abspath(template_path))
    env = create_environment(template_dir)
    
    # Load template
    template_name = os.path.basename(template_path)
    template_obj = env.get_template(template_name)
    
    # Get the extension instance and set template name
    extension = env.globals['extension']
    extension.set_template_name(template_path)
    
    # Setup run-based logging if logdir is provided
    run_id = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        run_logger = RunLogger(log_dir)
        
        # Start a new run with template metadata and context
        run_metadata = {
            "template": template_path,
            "context_file": context_path
        }
        run_id = run_logger.start_run(metadata=run_metadata, context=ctx)
        
        # Get the LLM logger for this run
        llm_logger = run_logger.get_llm_logger(run_id)
        extension.logger = llm_logger
    
    try:
        # Render template - use manual sync rendering to avoid async issues
        result = render_template_sync(template_obj, ctx)
        
        # End the run if we started one
        if log_dir and run_id:
            run_logger.end_run()
        
        # Handle output
        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(result)
        
        return result
    except Exception as e:
        if log_dir and run_id:
            # Still try to end the run even if there was an error
            try:
                run_logger.end_run()
            except:
                pass
        raise RuntimeError(f"Error rendering template: {str(e)}")

async def render_prompt_async(
    template_path: Union[str, Path],
    context_data: Union[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Asynchronously render a Jinja template containing LLM queries.
    
    This is the async version of render_prompt which should be used in async contexts.
    It processes a prompt template file with LLM query tags and returns the rendered output.
    
    Args:
        template_path: Path to the Jinja template file.
        context_data: Either a path to a YAML context file or a dictionary with context data.
        output_path: Optional path where the rendered output will be saved.
        log_dir: Optional directory for storing logs.
        
    Returns:
        The rendered prompt output as a string.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
        ValueError: If the context data is invalid.
        RuntimeError: If there's an error during template rendering.
    """
    # Convert paths to strings if they're Path objects
    template_path = str(template_path) if isinstance(template_path, Path) else template_path
    output_path = str(output_path) if isinstance(output_path, Path) and output_path else output_path
    log_dir = str(log_dir) if isinstance(log_dir, Path) and log_dir else log_dir
    
    # Check if template file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load context data
    if isinstance(context_data, str):
        # It's a file path to a YAML file
        context_path = context_data
        if not os.path.exists(context_path):
            raise FileNotFoundError(f"Context file not found: {context_path}")
        
        try:
            with open(context_path, 'r') as f:
                ctx = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in context file: {str(e)}")
    else:
        # It's a dictionary
        ctx = context_data
        context_path = None  # No file path since it's a dict
    
    # Setup Jinja environment
    template_dir = os.path.dirname(os.path.abspath(template_path))
    env = create_environment(template_dir)
    
    # Load template
    template_name = os.path.basename(template_path)
    template_obj = env.get_template(template_name)
    
    # Get the extension instance and set template name
    extension = env.globals['extension']
    extension.set_template_name(template_path)
    
    # Setup run-based logging if logdir is provided
    run_id = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        run_logger = RunLogger(log_dir)
        
        # Start a new run with template metadata and context
        run_metadata = {
            "template": template_path,
            "context_file": context_path
        }
        run_id = run_logger.start_run(metadata=run_metadata, context=ctx)
        
        # Get the LLM logger for this run
        llm_logger = run_logger.get_llm_logger(run_id)
        extension.logger = llm_logger
    
    try:
        # Render template asynchronously
        result = await template_obj.render_async(**ctx)
        
        # End the run if we started one
        if log_dir and run_id:
            run_logger.end_run()
        
        # Handle output
        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(result)
        
        return result
    except Exception as e:
        if log_dir and run_id:
            # Still try to end the run even if there was an error
            try:
                run_logger.end_run()
            except:
                pass
        raise RuntimeError(f"Error rendering template: {str(e)}") 