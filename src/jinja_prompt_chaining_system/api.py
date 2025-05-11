"""API functions for the Jinja Prompt Chaining System.

This module provides exportable functions that can be used by applications
importing this library, providing the same functionality as the CLI.
"""

import os
import yaml
import asyncio
from typing import Dict, Any, Optional, Union
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from .parser import LLMQueryExtension
from .logger import RunLogger

def create_environment(template_path=None) -> Environment:
    """Create a Jinja environment with the LLMQuery extension registered."""
    # Create environment with basic settings
    env = Environment(
        loader=FileSystemLoader(template_path) if template_path else None,
        enable_async=True,  # Enable async support for potential future use
        extensions=[LLMQueryExtension],
        autoescape=False,  # Disable HTML escaping by default
        undefined=StrictUndefined  # Ensure undefined variables raise errors
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
    context: Union[str, Dict[str, Any]],
    out: Optional[Union[str, Path]] = None,
    logdir: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    enable_parallel: bool = True,
    max_concurrent: int = 100000
) -> str:
    """
    Render a Jinja template containing LLM queries.
    
    This function provides the same functionality as the CLI command but can be
    called directly from Python code. It processes a prompt template file with
    LLM query tags and returns the rendered output.
    
    Args:
        template_path: Path to the Jinja template file.
        context: Either a path to a YAML context file or a dictionary with context data.
        out: Optional path where the rendered output will be saved.
        logdir: Optional directory for storing logs.
        name: Optional name for the run, which will be appended to the run directory name.
        enable_parallel: Whether to enable parallel execution of LLM queries.
        max_concurrent: Maximum number of concurrent queries when parallel is enabled.
        
    Returns:
        The rendered prompt output as a string.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
        ValueError: If the context data is invalid.
        RuntimeError: If there's an error during template rendering.
    """
    # Convert paths to strings if they're Path objects
    template_path = str(template_path) if isinstance(template_path, Path) else template_path
    out = str(out) if isinstance(out, Path) and out else out
    logdir = str(logdir) if isinstance(logdir, Path) and logdir else logdir
    
    # Check if template file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load context data
    if isinstance(context, str):
        # It's a file path to a YAML file
        context_path = context
        if not os.path.exists(context_path):
            raise FileNotFoundError(f"Context file not found: {context_path}")
        
        try:
            with open(context_path, 'r') as f:
                ctx = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in context file: {str(e)}")
    else:
        # It's a dictionary
        ctx = context
        context_path = None  # No file path since it's a dict
    
    # Setup run-based logging if logdir is provided
    run_id = None
    run_logger = None
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        run_logger = RunLogger(logdir)
        
        # Start a new run with template metadata and context
        run_metadata = {
            "template": template_path,
            "context_file": context_path
        }
        run_id = run_logger.start_run(metadata=run_metadata, context=ctx, name=name)
        
        # Get the LLMLogger for this run to pass to the template engine
        llm_logger = run_logger.get_llm_logger(run_id)
    
    try:
        # Use parallel rendering by default
        from .parallel_integration import render_template_parallel
        result = render_template_parallel(
            template_path, 
            ctx,
            enable_parallel=enable_parallel,
            max_concurrent=max_concurrent
        )
        
        # End the run if we started one
        if logdir and run_id and run_logger:
            run_logger.end_run()
        
        # Handle output
        if out:
            output_dir = os.path.dirname(os.path.abspath(out))
            os.makedirs(output_dir, exist_ok=True)
            with open(out, 'w') as f:
                f.write(result)
        
        return result
    except Exception as e:
        if logdir and run_id and run_logger:
            # Still try to end the run even if there was an error
            try:
                run_logger.end_run()
            except:
                pass
        raise RuntimeError(f"Error rendering template: {str(e)}")

async def render_prompt_async(
    template_path: Union[str, Path],
    context: Union[str, Dict[str, Any]],
    out: Optional[Union[str, Path]] = None,
    logdir: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    enable_parallel: bool = True,
    max_concurrent: int = 4
) -> str:
    """
    Asynchronously render a Jinja template containing LLM queries.
    
    This is the async version of render_prompt which should be used in async contexts.
    It processes a prompt template file with LLM query tags and returns the rendered output.
    
    Args:
        template_path: Path to the Jinja template file.
        context: Either a path to a YAML context file or a dictionary with context data.
        out: Optional path where the rendered output will be saved.
        logdir: Optional directory for storing logs.
        name: Optional name for the run, which will be appended to the run directory name.
        enable_parallel: Whether to enable parallel execution of LLM queries.
        max_concurrent: Maximum number of concurrent queries when parallel is enabled.
        
    Returns:
        The rendered prompt output as a string.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
        ValueError: If the context data is invalid.
        RuntimeError: If there's an error during template rendering.
    """
    # Convert paths to strings if they're Path objects
    template_path = str(template_path) if isinstance(template_path, Path) else template_path
    out = str(out) if isinstance(out, Path) and out else out
    logdir = str(logdir) if isinstance(logdir, Path) and logdir else logdir
    
    # Check if template file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load context data
    if isinstance(context, str):
        # It's a file path to a YAML file
        context_path = context
        if not os.path.exists(context_path):
            raise FileNotFoundError(f"Context file not found: {context_path}")
        
        try:
            with open(context_path, 'r') as f:
                ctx = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in context file: {str(e)}")
    else:
        # It's a dictionary
        ctx = context
        context_path = None  # No file path since it's a dict
    
    # Setup run-based logging if logdir is provided
    run_id = None
    run_logger = None
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        run_logger = RunLogger(logdir)
        
        # Start a new run with template metadata and context
        run_metadata = {
            "template": template_path,
            "context_file": context_path
        }
        run_id = run_logger.start_run(metadata=run_metadata, context=ctx, name=name)
        
        # Get the LLMLogger for this run to pass to the template engine
        llm_logger = run_logger.get_llm_logger(run_id)
    
    try:
        # Import async-compatible rendering function
        from .parallel_integration import render_template_parallel_async
        
        # Use the async version for rendering
        result = await render_template_parallel_async(
            template_path, 
            ctx,
            enable_parallel=enable_parallel,
            max_concurrent=max_concurrent
        )
        
        # End the run if we started one
        if logdir and run_id and run_logger:
            run_logger.end_run()
        
        # Handle output
        if out:
            output_dir = os.path.dirname(os.path.abspath(out))
            os.makedirs(output_dir, exist_ok=True)
            with open(out, 'w') as f:
                f.write(result)
        
        return result
    except Exception as e:
        if logdir and run_id and run_logger:
            # Still try to end the run even if there was an error
            try:
                run_logger.end_run()
            except:
                pass
        raise RuntimeError(f"Error rendering template: {str(e)}") 