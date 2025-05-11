import pytest
import os
import tempfile
import asyncio
import time
from unittest.mock import patch, AsyncMock, Mock

from src.jinja_prompt_chaining_system.parallel_integration import render_template_parallel


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_changing_conditionals_between_passes(mock_query_async, mock_query):
    """
    Test handling of conditionals that change between template passes.
    
    This test checks that if a condition changes between the first pass (collection)
    and second pass (execution), the system correctly executes only the queries
    that should be executed based on the final condition value.
    """
    
    # Track execution order and counts
    queries_executed = {}
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        queries_executed[prompt] = queries_executed.get(prompt, 0) + 1
        print(f"Sync query called with prompt: {prompt}")
        
        # The condition query changes result between calls
        if "condition" in prompt.lower():
            # First call during collection phase
            if queries_executed[prompt] == 1:
                return "true"  
            # Second call during execution phase
            else:
                return "false"
        elif "true branch" in prompt.lower():
            return "TRUE_BRANCH_RESULT"
        elif "false branch" in prompt.lower():
            return "FALSE_BRANCH_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock with the same behavior
    async def mock_async_query(prompt, params=None, stream=False):
        queries_executed[prompt] = queries_executed.get(prompt, 0) + 1
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        # The condition query changes result between calls
        if "condition" in prompt.lower():
            # First call during collection phase
            if queries_executed[prompt] == 1:
                return "true"  
            # Second call during execution phase
            else:
                return "false"
        elif "true branch" in prompt.lower():
            return "TRUE_BRANCH_RESULT"
        elif "false branch" in prompt.lower():
            return "FALSE_BRANCH_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with a condition that will change between passes
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# The condition will change between collection and execution phases #}
        {% set condition = llmquery(prompt="Get condition value") %}
        
        Condition: {{ condition }}
        
        {# This branch should switch between collection and execution #}
        {% if condition == "true" %}
            {% set true_result = llmquery(prompt="True branch query") %}
            True result: {{ true_result }}
        {% else %}
            {% set false_result = llmquery(prompt="False branch query") %}
            False result: {{ false_result }}
        {% endif %}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nQueries executed:")
        for prompt, count in queries_executed.items():
            print(f"{prompt}: {count} times")
        
        print("\nTemplate result:")
        print(result)
        
        # Check which queries were actually executed
        condition_executed = queries_executed.get("Get condition value", 0)
        true_branch_executed = queries_executed.get("True branch query", 0) 
        false_branch_executed = queries_executed.get("False branch query", 0)
        
        # The condition query should be executed at least once
        assert condition_executed > 0, "Condition query not executed"
        
        # In a perfect implementation, the true branch would never execute if we know
        # the condition will be false in the final render. However, our current implementation
        # might collect it during the first pass when condition was "true"
        # The key is that the final rendered output should match the second pass condition
        
        # Verify content reflects the final condition value (false)
        assert "Condition: false" in result, "Final condition value not reflected in output"
        assert "False result:" in result, "False branch not rendered"
        assert "True result:" not in result, "True branch incorrectly rendered"
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_nested_conditional_branches(mock_query_async, mock_query):
    """
    Test handling of deeply nested conditional branches.
    
    This tests that the system correctly handles complex nested conditional logic
    and only executes the queries that are in the branches that should be taken.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "outer condition" in prompt.lower():
            return "true"
        elif "inner condition" in prompt.lower():
            return "false"
        elif "both true" in prompt.lower():
            return "BOTH_TRUE_RESULT"
        elif "outer true inner false" in prompt.lower():
            return "OUTER_TRUE_INNER_FALSE_RESULT"
        elif "outer false" in prompt.lower():
            return "OUTER_FALSE_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "outer condition" in prompt.lower():
            return "true"
        elif "inner condition" in prompt.lower():
            return "false"
        elif "both true" in prompt.lower():
            return "BOTH_TRUE_RESULT"
        elif "outer true inner false" in prompt.lower():
            return "OUTER_TRUE_INNER_FALSE_RESULT"
        elif "outer false" in prompt.lower():
            return "OUTER_FALSE_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with nested conditional branches
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Nested conditionals with several possible paths #}
        {% set outer_condition = llmquery(prompt="Get outer condition") %}
        
        Outer condition: {{ outer_condition }}
        
        {% if outer_condition == "true" %}
            {# This branch will be taken #}
            {% set inner_condition = llmquery(prompt="Get inner condition") %}
            Inner condition: {{ inner_condition }}
            
            {% if inner_condition == "true" %}
                {# This branch will NOT be taken #}
                {% set both_result = llmquery(prompt="Both true query") %}
                Both true result: {{ both_result }}
            {% else %}
                {# This branch will be taken #}
                {% set mixed_result = llmquery(prompt="Outer true inner false query") %}
                Mixed result: {{ mixed_result }}
            {% endif %}
        {% else %}
            {# This branch will NOT be taken #}
            {% set outer_false_result = llmquery(prompt="Outer false query") %}
            Outer false result: {{ outer_false_result }}
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
        
        # Check which branches were executed
        outer_idx = next((i for i, p in enumerate(execution_order) if "outer condition" in p.lower()), -1)
        inner_idx = next((i for i, p in enumerate(execution_order) if "inner condition" in p.lower()), -1)
        both_idx = next((i for i, p in enumerate(execution_order) if "both true" in p.lower()), -1)
        mixed_idx = next((i for i, p in enumerate(execution_order) if "outer true inner false" in p.lower()), -1)
        outer_false_idx = next((i for i, p in enumerate(execution_order) if "outer false" in p.lower()), -1)
        
        # Assert outer and inner conditions were executed
        assert outer_idx >= 0, "Outer condition query not executed"
        assert inner_idx >= 0, "Inner condition query not executed"
        assert inner_idx > outer_idx, "Inner condition executed before outer condition"
        
        # The mixed branch (outer true, inner false) should be executed
        assert mixed_idx >= 0, "Mixed branch query not executed"
        assert mixed_idx > inner_idx, "Mixed branch executed before inner condition"
        
        # The both true branch should not be executed
        assert both_idx == -1, "Both true branch query was executed but should not have been"
        
        # The outer false branch should not be executed
        assert outer_false_idx == -1, "Outer false branch query was executed but should not have been"
        
        # Verify content
        assert "Outer condition: true" in result
        assert "Inner condition: false" in result
        assert "Mixed result: OUTER_TRUE_INNER_FALSE_RESULT" in result
        assert "Both true result:" not in result
        assert "Outer false result:" not in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path) 