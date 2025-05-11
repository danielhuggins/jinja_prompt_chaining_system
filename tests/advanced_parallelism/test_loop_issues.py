import pytest
import os
import tempfile
import asyncio
import time
from unittest.mock import patch, AsyncMock, Mock

from src.jinja_prompt_chaining_system.parallel_integration import render_template_parallel


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_dynamic_loop_bounds(mock_query_async, mock_query):
    """
    Test handling of loops with bounds determined by an LLM query.
    
    This tests that the system correctly handles loop boundaries that are
    determined by the result of an LLM query.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "get loop count" in prompt.lower():
            return "3"  # Return 3 as the loop count
        elif "item 0" in prompt.lower():
            return "RESULT_0"
        elif "item 1" in prompt.lower():
            return "RESULT_1"
        elif "item 2" in prompt.lower():
            return "RESULT_2"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "get loop count" in prompt.lower():
            return "3"  # Return 3 as the loop count
        elif "item 0" in prompt.lower():
            return "RESULT_0"
        elif "item 1" in prompt.lower():
            return "RESULT_1"
        elif "item 2" in prompt.lower():
            return "RESULT_2"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with a loop having bounds determined by an LLM query
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Get the number of iterations from an LLM query #}
        {% set loop_count = llmquery(prompt="Get loop count") | int %}
        
        Loop count: {{ loop_count }}
        
        {# Loop the specified number of times #}
        {% set results = [] %}
        {% for i in range(loop_count) %}
            {% set result = llmquery(prompt="Process item " + i|string) %}
            {% set _ = results.append(result) %}
            Result {{ i }}: {{ result }}
        {% endfor %}
        
        {# Summarize the results #}
        All results: {{ results|join(', ') }}
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
        
        # Check that all expected queries were executed
        loop_count_idx = next((i for i, p in enumerate(execution_order) if "loop count" in p.lower()), -1)
        item0_idx = next((i for i, p in enumerate(execution_order) if "item 0" in p.lower()), -1)
        item1_idx = next((i for i, p in enumerate(execution_order) if "item 1" in p.lower()), -1)
        item2_idx = next((i for i, p in enumerate(execution_order) if "item 2" in p.lower()), -1)
        
        # Assert all queries were executed
        assert loop_count_idx >= 0, "Loop count query not executed"
        assert item0_idx >= 0, "Item 0 query not executed"
        assert item1_idx >= 0, "Item 1 query not executed"
        assert item2_idx >= 0, "Item 2 query not executed"
        
        # Verify the results
        assert "Loop count: 3" in result
        assert "Result 0: RESULT_0" in result
        assert "Result 1: RESULT_1" in result
        assert "Result 2: RESULT_2" in result
        assert "All results: RESULT_0, RESULT_1, RESULT_2" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_inter_iteration_dependencies(mock_query_async, mock_query):
    """
    Test dependencies between loop iterations.
    
    This tests that the system correctly handles dependencies between
    iterations of a loop, where each iteration depends on the previous one.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "iteration 0" in prompt.lower():
            return "ITER_0"
        elif "iteration 1" in prompt.lower() and "iter_0" in prompt.lower():
            return "ITER_1"
        elif "iteration 2" in prompt.lower() and "iter_1" in prompt.lower():
            return "ITER_2"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "iteration 0" in prompt.lower():
            return "ITER_0"
        elif "iteration 1" in prompt.lower() and "iter_0" in prompt.lower():
            return "ITER_1"
        elif "iteration 2" in prompt.lower() and "iter_1" in prompt.lower():
            return "ITER_2"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with inter-iteration dependencies
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Loop with dependencies between iterations #}
        {% set iterations = 3 %}
        {% set results = [] %}
        
        {# First iteration has no dependencies #}
        {% set result = llmquery(prompt="Process iteration 0") %}
        {% set _ = results.append(result) %}
        Result 0: {{ result }}
        
        {# Subsequent iterations depend on the previous one #}
        {% for i in range(1, iterations) %}
            {% set prev_result = results[i-1] %}
            {% set result = llmquery(prompt="Process iteration " + i|string + " using " + prev_result) %}
            {% set _ = results.append(result) %}
            Result {{ i }}: {{ result }}
        {% endfor %}
        
        {# Summarize the results #}
        Chain of results: {{ results|join(' -> ') }}
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
        
        # Check the execution order to verify dependencies were respected
        iter0_idx = next((i for i, p in enumerate(execution_order) if "iteration 0" in p.lower()), -1)
        iter1_idx = next((i for i, p in enumerate(execution_order) if "iteration 1" in p.lower()), -1)
        iter2_idx = next((i for i, p in enumerate(execution_order) if "iteration 2" in p.lower()), -1)
        
        # Assert all iterations were executed
        assert iter0_idx >= 0, "Iteration 0 query not executed"
        assert iter1_idx >= 0, "Iteration 1 query not executed"
        assert iter2_idx >= 0, "Iteration 2 query not executed"
        
        # Assert execution order
        assert iter0_idx < iter1_idx, "Iteration 1 executed before Iteration 0"
        assert iter1_idx < iter2_idx, "Iteration 2 executed before Iteration 1"
        
        # Verify the results
        assert "Result 0: ITER_0" in result
        assert "Result 1: ITER_1" in result
        assert "Result 2: ITER_2" in result
        assert "Chain of results: ITER_0 -> ITER_1 -> ITER_2" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_nested_loops_with_dependencies(mock_query_async, mock_query):
    """
    Test handling of nested loops with dependencies.
    
    This tests that the system correctly handles nested loops where the
    inner loop depends on the results of the outer loop.
    """
    
    # Track execution order
    execution_order = []
    execution_times = {}
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        execution_times[prompt] = time.time()
        print(f"Sync query called with prompt: {prompt}")
        
        if "outer" in prompt.lower() and "inner" not in prompt.lower():
            # Extract the outer index
            try:
                idx = int(prompt.split("outer")[1].split()[0])
                return f"OUTER_{idx}"
            except:
                return "OUTER_RESULT"
        elif "inner" in prompt.lower():
            # This should contain an outer result
            for i in range(3):
                if f"OUTER_{i}" in prompt:
                    # Extract the inner index
                    try:
                        idx = int(prompt.split("inner")[1].split()[0])
                        return f"INNER_{i}_{idx}"
                    except:
                        return f"INNER_RESULT_FOR_{i}"
            return "INNER_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        execution_times[prompt] = time.time()
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "outer" in prompt.lower() and "inner" not in prompt.lower():
            # Extract the outer index
            try:
                idx = int(prompt.split("outer")[1].split()[0])
                return f"OUTER_{idx}"
            except:
                return "OUTER_RESULT"
        elif "inner" in prompt.lower():
            # This should contain an outer result
            for i in range(3):
                if f"OUTER_{i}" in prompt:
                    # Extract the inner index
                    try:
                        idx = int(prompt.split("inner")[1].split()[0])
                        return f"INNER_{i}_{idx}"
                    except:
                        return f"INNER_RESULT_FOR_{i}"
            return "INNER_RESULT"
        
        return f"Response to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with nested loops and dependencies
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Nested loops with dependencies #}
        {% set outer_count = 3 %}
        {% set inner_count = 2 %}
        {% set results = [] %}
        
        {# Outer loop #}
        {% for i in range(outer_count) %}
            {% set outer_result = llmquery(prompt="Process outer " + i|string) %}
            Outer {{ i }}: {{ outer_result }}
            
            {# Inner loop depends on outer loop #}
            {% set inner_results = [] %}
            {% for j in range(inner_count) %}
                {% set inner_result = llmquery(prompt="Process inner " + j|string + " using " + outer_result) %}
                {% set _ = inner_results.append(inner_result) %}
                Inner {{ i }}.{{ j }}: {{ inner_result }}
            {% endfor %}
            
            {% set _ = results.append(inner_results) %}
        {% endfor %}
        
        {# Verify results #}
        All results: 
        {% for outer_idx, inner_results in results|enumerate %}
            Outer {{ outer_idx }}: {{ inner_results|join(', ') }}
        {% endfor %}
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
        
        # Get indices of the outer queries
        outer0_idx = next((i for i, p in enumerate(execution_order) if "outer 0" in p.lower()), -1)
        outer1_idx = next((i for i, p in enumerate(execution_order) if "outer 1" in p.lower()), -1)
        outer2_idx = next((i for i, p in enumerate(execution_order) if "outer 2" in p.lower()), -1)
        
        # Get indices of the inner queries
        inner00_idx = next((i for i, p in enumerate(execution_order) if "inner 0" in p.lower() and "OUTER_0" in p), -1)
        inner01_idx = next((i for i, p in enumerate(execution_order) if "inner 1" in p.lower() and "OUTER_0" in p), -1)
        inner10_idx = next((i for i, p in enumerate(execution_order) if "inner 0" in p.lower() and "OUTER_1" in p), -1)
        inner11_idx = next((i for i, p in enumerate(execution_order) if "inner 1" in p.lower() and "OUTER_1" in p), -1)
        inner20_idx = next((i for i, p in enumerate(execution_order) if "inner 0" in p.lower() and "OUTER_2" in p), -1)
        inner21_idx = next((i for i, p in enumerate(execution_order) if "inner 1" in p.lower() and "OUTER_2" in p), -1)
        
        # Assert all queries were executed
        assert outer0_idx >= 0, "Outer 0 query not executed"
        assert outer1_idx >= 0, "Outer 1 query not executed"
        assert outer2_idx >= 0, "Outer 2 query not executed"
        
        assert inner00_idx >= 0, "Inner 0.0 query not executed"
        assert inner01_idx >= 0, "Inner 0.1 query not executed"
        assert inner10_idx >= 0, "Inner 1.0 query not executed"
        assert inner11_idx >= 0, "Inner 1.1 query not executed"
        assert inner20_idx >= 0, "Inner 2.0 query not executed"
        assert inner21_idx >= 0, "Inner 2.1 query not executed"
        
        # Check dependencies
        assert inner00_idx > outer0_idx, "Inner 0.0 executed before Outer 0"
        assert inner01_idx > outer0_idx, "Inner 0.1 executed before Outer 0"
        assert inner10_idx > outer1_idx, "Inner 1.0 executed before Outer 1"
        assert inner11_idx > outer1_idx, "Inner 1.1 executed before Outer 1"
        assert inner20_idx > outer2_idx, "Inner 2.0 executed before Outer 2"
        assert inner21_idx > outer2_idx, "Inner 2.1 executed before Outer 2"
        
        # Verify content
        assert "Outer 0: OUTER_0" in result
        assert "Inner 0.0: INNER_0_0" in result
        assert "Inner 0.1: INNER_0_1" in result
        assert "Outer 1: OUTER_1" in result
        assert "Inner 1.0: INNER_1_0" in result
        assert "Inner 1.1: INNER_1_1" in result
        assert "Outer 2: OUTER_2" in result
        assert "Inner 2.0: INNER_2_0" in result
        assert "Inner 2.1: INNER_2_1" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path) 