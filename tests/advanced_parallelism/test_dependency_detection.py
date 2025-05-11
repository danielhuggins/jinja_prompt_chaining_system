import pytest
import os
import tempfile
import asyncio
import time
from unittest.mock import patch, AsyncMock, Mock

from src.jinja_prompt_chaining_system.parallel_integration import render_template_parallel


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_complex_expression_dependencies(mock_query_async, mock_query):
    """Test that complex expression dependencies are correctly identified and respected."""
    
    # Track execution order
    execution_order = []
    execution_times = {}
    
    # Set up synchronous mock with timing tracking
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        execution_times[prompt] = time.time()
        print(f"Sync query called with prompt: {prompt}")
        
        # Return appropriate responses for different queries
        if "first" in prompt.lower():
            time.sleep(0.1)  # Add a small delay
            return "FIRST_RESULT"
        elif "second" in prompt.lower():
            time.sleep(0.1)  # Add a small delay
            return "SECOND_RESULT"
        elif "combined" in prompt.lower():
            # This should be called after both first and second
            time.sleep(0.1)  # Add a small delay
            return "COMBINED_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        execution_times[prompt] = time.time()
        print(f"Async query called with prompt: {prompt}")
        
        # Add a small delay to simulate network latency
        await asyncio.sleep(0.1)
        
        # Return appropriate responses for different queries
        if "first" in prompt.lower():
            return "FIRST_RESULT"
        elif "second" in prompt.lower():
            return "SECOND_RESULT"
        elif "combined" in prompt.lower():
            # This should be called after both first and second
            return "COMBINED_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with complex expression dependencies
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Test complex expression dependencies #}
        {% set first = llmquery(prompt="Get first value") %}
        {% set second = llmquery(prompt="Get second value") %}
        
        First result: {{ first }}
        Second result: {{ second }}
        
        {# This query depends on both first and second through a complex expression #}
        {% set combined = llmquery(prompt="Combined query using " + (first if first else "default") + " and " + ("nothing" if not second else second)) %}
        
        Combined result: {{ combined }}
        
        {# This uses complex conditionals and filters as dependencies #}
        {% set processed = llmquery(prompt="Process data with " + (first|default("default")|upper if second else "fallback")) %}
        
        Processed result: {{ processed }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nExecution order:")
        for i, prompt in enumerate(execution_order):
            print(f"{i+1}. {prompt}")
        
        print("\nTemplate result:")
        print(result)
        
        # Verify the combined query was executed after both first and second
        first_idx = next((i for i, p in enumerate(execution_order) if "first" in p.lower()), -1)
        second_idx = next((i for i, p in enumerate(execution_order) if "second" in p.lower()), -1)
        combined_idx = next((i for i, p in enumerate(execution_order) if "combined" in p.lower()), -1)
        processed_idx = next((i for i, p in enumerate(execution_order) if "process" in p.lower()), -1)
        
        # Assert that indices exist
        assert first_idx >= 0, "First query not executed"
        assert second_idx >= 0, "Second query not executed"
        assert combined_idx >= 0, "Combined query not executed"
        assert processed_idx >= 0, "Processed query not executed"
        
        # Check execution order - combined should come after both first and second
        assert combined_idx > first_idx, "Combined query executed before first query"
        assert combined_idx > second_idx, "Combined query executed before second query"
        
        # Also check the processed query that depends on filtered values
        assert processed_idx > first_idx, "Processed query executed before first query"
        assert processed_idx > second_idx, "Processed query executed before second query"
        
        # Verify content
        assert "First result: FIRST_RESULT" in result
        assert "Second result: SECOND_RESULT" in result
        assert "Combined result: COMBINED_RESULT" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_implicit_dependencies_in_jinja_syntax(mock_query_async, mock_query):
    """Test that implicit dependencies in Jinja syntax are correctly identified."""
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "condition" in prompt.lower():
            return "true"  # Return a string that will be evaluated as truthy
        elif "true branch" in prompt.lower():
            return "TRUE_BRANCH_RESULT"
        elif "false branch" in prompt.lower():
            return "FALSE_BRANCH_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "condition" in prompt.lower():
            return "true"  # Return a string that will be evaluated as truthy
        elif "true branch" in prompt.lower():
            return "TRUE_BRANCH_RESULT"
        elif "false branch" in prompt.lower():
            return "FALSE_BRANCH_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with implicit dependencies through Jinja syntax
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# This query result will be used in an if statement #}
        {% set condition = llmquery(prompt="Get condition value") %}
        
        Condition: {{ condition }}
        
        {# Implicit dependency: This branch should only execute if condition is truthy #}
        {% if condition %}
            {% set true_result = llmquery(prompt="True branch query") %}
            True result: {{ true_result }}
        {% else %}
            {% set false_result = llmquery(prompt="False branch query") %}
            False result: {{ false_result }}
        {% endif %}
        
        {# Another implicit dependency where the condition is more complex #}
        {% if condition and condition|length > 0 %}
            {% set complex_result = llmquery(prompt="Complex condition query") %}
            Complex result: {{ complex_result }}
        {% endif %}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nExecution order:")
        for i, prompt in enumerate(execution_order):
            print(f"{i+1}. {prompt}")
        
        print("\nTemplate result:")
        print(result)
        
        # Check which branch was executed based on the condition query result
        condition_idx = next((i for i, p in enumerate(execution_order) if "condition" in p.lower()), -1)
        true_branch_idx = next((i for i, p in enumerate(execution_order) if "true branch" in p.lower()), -1)
        false_branch_idx = next((i for i, p in enumerate(execution_order) if "false branch" in p.lower()), -1)
        complex_idx = next((i for i, p in enumerate(execution_order) if "complex condition" in p.lower()), -1)
        
        # Assert condition was executed
        assert condition_idx >= 0, "Condition query not executed"
        
        # Since our mock returns "true" for the condition, the true branch should be executed
        # but not the false branch
        assert true_branch_idx >= 0, "True branch query not executed"
        assert true_branch_idx > condition_idx, "True branch executed before condition"
        
        # False branch should not be executed
        assert false_branch_idx == -1, "False branch query was executed but should not have been"
        
        # Complex condition should be executed
        assert complex_idx >= 0, "Complex condition query not executed"
        assert complex_idx > condition_idx, "Complex condition executed before condition"
        
        # Verify content
        assert "Condition: true" in result
        assert "True result: TRUE_BRANCH_RESULT" in result
        assert "False result:" not in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path) 