"""
Integration of parallel execution with the LLMQueryExtension.

This module extends the LLMQueryExtension to support parallel execution of LLM queries.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Set
import re

from jinja2 import Environment, Template, nodes, FileSystemLoader
from jinja2.ext import Extension

from .parser import LLMQueryExtension
from .parallel import ParallelExecutor, Query, ParallelQueryTracker, extract_dependencies
from .llm import LLMClient

class ParallelLLMQueryExtension(LLMQueryExtension):
    """Extended LLMQueryExtension with parallel execution support."""
    
    def __init__(self, environment, enable_parallel=True, max_concurrent=4):
        super().__init__(environment)
        
        # Parallel execution settings
        self.enable_parallel = enable_parallel
        self.max_concurrent = max_concurrent
        self.parallel_executor = ParallelExecutor(max_concurrent=max_concurrent)
        self.query_tracker = ParallelQueryTracker()
        
        # Track if we're in the collection phase
        self.collecting_queries = False
        
        # Override the global function with our version
        environment.globals['llmquery'] = self.parallel_global_llmquery
    
    def parallel_global_llmquery(self, prompt: str, **params):
        """
        Enhanced global llmquery function that supports parallel execution.
        
        Args:
            prompt: The prompt to send to the LLM
            **params: Additional parameters including parallel=True/False
            
        Returns:
            The LLM response or a placeholder if in collection phase
        """
        # Check if parallel execution is explicitly disabled for this query
        parallel_enabled = params.pop('parallel', self.enable_parallel)
        
        # If we're not in collection phase or parallel is disabled,
        # fall back to the original implementation
        if not self.collecting_queries or not parallel_enabled:
            return super().global_llmquery(prompt, **params)
        
        # Check if we're in async context
        try:
            asyncio.get_running_loop()
            # Async context is not fully supported for collection phase
            return super().global_llmquery(prompt, **params)
        except RuntimeError:
            pass
        
        # We're in collection phase and parallel is enabled
        # Extract dependencies from the prompt
        context = self.environment.globals
        dependencies = extract_dependencies(prompt, context)
        
        # Create a query object
        query = Query(
            prompt=prompt,
            params=params,
            dependencies=dependencies
        )
        
        # Add to the tracker and return the result variable
        result_var = self.query_tracker.add_query(query)
        
        # Return a placeholder that will be replaced during rendering
        return f"{{{{ {result_var} }}}}"
    
    def render_template_with_parallel(self, template, context):
        """
        Render a template with parallel execution of LLM queries.
        
        Args:
            template: The template to render
            context: The context to use for rendering
            
        Returns:
            The rendered template
        """
        if not self.enable_parallel:
            # Fall back to normal rendering if parallel is disabled
            return template.render(**context)
        
        # First pass: collect queries
        try:
            self.collecting_queries = True
            self.query_tracker.clear()
            
            # Render the template to collect queries
            # The actual output is discarded
            template.render(**context)
            
            # No queries collected, just render normally
            if not self.query_tracker.queries:
                self.collecting_queries = False
                return template.render(**context)
            
            # Execute the queries in parallel
            queries = self.query_tracker.queries
            
            # Use asyncio to run the executor
            loop = asyncio.new_event_loop()
            try:
                # Make a copy of the context for parallel execution
                parallel_context = dict(context)
                
                # Execute all queries
                updated_context = loop.run_until_complete(
                    self.parallel_executor.execute_all(queries, parallel_context)
                )
                
                # Update the original context with the results
                for var, value in updated_context.items():
                    if var not in context and var in [q.result_var for q in queries]:
                        context[var] = value
                
            finally:
                loop.close()
            
            # Second pass: actual rendering with results
            self.collecting_queries = False
            return template.render(**context)
            
        finally:
            # Make sure we reset the flag
            self.collecting_queries = False


def create_environment_with_parallel(template_path=None, enable_parallel=True, max_concurrent=4) -> Environment:
    """
    Create a Jinja environment with parallel LLM query support.
    
    Args:
        template_path: Optional path to templates
        enable_parallel: Whether to enable parallel execution
        max_concurrent: Maximum number of concurrent queries
        
    Returns:
        Configured Jinja environment
    """
    # Create environment with basic settings
    env = Environment(
        loader=FileSystemLoader(template_path) if template_path else None,
        enable_async=True,
        extensions=[ParallelLLMQueryExtension],
        autoescape=False  # Disable HTML escaping by default
    )
    
    # Configure the extension
    extension = env.extensions[ParallelLLMQueryExtension.identifier]
    extension.enable_parallel = enable_parallel
    extension.max_concurrent = max_concurrent
    
    # Make the extension instance available in the global namespace
    env.globals['extension'] = extension
    
    return env


# For testing, simple render functions
def render_template_parallel(template_path, context, enable_parallel=True, max_concurrent=4):
    """
    Render a template with parallel LLM query execution.
    
    Args:
        template_path: Path to the template
        context: Context for rendering
        enable_parallel: Whether to enable parallel execution
        max_concurrent: Maximum number of concurrent queries
        
    Returns:
        Rendered template
    """
    # Create environment
    env = create_environment_with_parallel(
        os.path.dirname(template_path),
        enable_parallel=enable_parallel,
        max_concurrent=max_concurrent
    )
    
    # Load template
    template = env.get_template(os.path.basename(template_path))
    
    # Render with parallel
    extension = env.globals['extension']
    return extension.render_template_with_parallel(template, context) 