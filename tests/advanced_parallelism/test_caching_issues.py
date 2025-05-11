import pytest
import os
import tempfile
import asyncio
import time
from unittest.mock import patch, AsyncMock, Mock

from src.jinja_prompt_chaining_system.parallel_integration import render_template_parallel


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_cache_key_collisions(mock_query_async, mock_query):
    """
    Test for potential cache key collisions.
    
    This test checks that similar queries with different parameters
    don't collide in the cache.
    """
    
    # Track query execution
    query_calls = {}
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        key = f"{prompt}::{params}"
        query_calls[key] = query_calls.get(key, 0) + 1
        print(f"Sync query called with prompt: {prompt}, params: {params}")
        
        # Return different results based on the model parameter
        if "model" in params and params["model"] == "gpt-4":
            return f"GPT-4 response: {prompt}"
        elif "model" in params and params["model"] == "gpt-3.5-turbo":
            return f"GPT-3.5 response: {prompt}"
        else:
            return f"Default response: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        key = f"{prompt}::{params}"
        query_calls[key] = query_calls.get(key, 0) + 1
        print(f"Async query called with prompt: {prompt}, params: {params}")
        await asyncio.sleep(0.01)
        
        # Return different results based on the model parameter
        if "model" in params and params["model"] == "gpt-4":
            return f"GPT-4 response: {prompt}"
        elif "model" in params and params["model"] == "gpt-3.5-turbo":
            return f"GPT-3.5 response: {prompt}"
        else:
            return f"Default response: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with similar queries but different parameters
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Same prompt with different models #}
        {% set query_text = "Tell me about AI" %}
        
        {# GPT-4 version #}
        {% set gpt4_response = llmquery(prompt=query_text, model="gpt-4") %}
        GPT-4 says: {{ gpt4_response }}
        
        {# GPT-3.5 version #}
        {% set gpt35_response = llmquery(prompt=query_text, model="gpt-3.5-turbo") %}
        GPT-3.5 says: {{ gpt35_response }}
        
        {# Default version #}
        {% set default_response = llmquery(prompt=query_text) %}
        Default model says: {{ default_response }}
        
        {# Similar prompts with slight differences #}
        {% set similar1 = llmquery(prompt=query_text + "?", model="gpt-4") %}
        Similar 1: {{ similar1 }}
        
        {% set similar2 = llmquery(prompt=query_text + "!", model="gpt-4") %}
        Similar 2: {{ similar2 }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nQuery calls:")
        for key, count in query_calls.items():
            print(f"{key}: {count} time(s)")
        
        print("\nTemplate result:")
        print(result)
        
        # Check that all distinct queries were executed
        gpt4_key = next((k for k in query_calls.keys() if "Tell me about AI" in k and "gpt-4" in k), None)
        gpt35_key = next((k for k in query_calls.keys() if "Tell me about AI" in k and "gpt-3.5-turbo" in k), None)
        default_key = next((k for k in query_calls.keys() if "Tell me about AI" in k and "gpt-4" not in k and "gpt-3.5-turbo" not in k), None)
        similar1_key = next((k for k in query_calls.keys() if "Tell me about AI?" in k), None)
        similar2_key = next((k for k in query_calls.keys() if "Tell me about AI!" in k), None)
        
        # Assert each query was executed
        assert gpt4_key is not None, "GPT-4 query was not executed"
        assert gpt35_key is not None, "GPT-3.5 query was not executed"
        assert default_key is not None, "Default query was not executed"
        assert similar1_key is not None, "Similar query 1 was not executed"
        assert similar2_key is not None, "Similar query 2 was not executed"
        
        # Each query should be executed only once (no duplicates)
        assert query_calls[gpt4_key] == 1, "GPT-4 query was executed multiple times"
        assert query_calls[gpt35_key] == 1, "GPT-3.5 query was executed multiple times"
        assert query_calls[default_key] == 1, "Default query was executed multiple times"
        assert query_calls[similar1_key] == 1, "Similar query 1 was executed multiple times"
        assert query_calls[similar2_key] == 1, "Similar query 2 was executed multiple times"
        
        # Verify the content
        assert "GPT-4 says: GPT-4 response: Tell me about AI" in result
        assert "GPT-3.5 says: GPT-3.5 response: Tell me about AI" in result
        assert "Default model says: Default response: Tell me about AI" in result
        assert "Similar 1: GPT-4 response: Tell me about AI?" in result
        assert "Similar 2: GPT-4 response: Tell me about AI!" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_cache_invalidation(mock_query_async, mock_query):
    """
    Test cache invalidation between template renderings.
    
    This test checks that query results are properly cached and invalidated
    when appropriate.
    """
    
    # Track query execution count
    query_counts = {}
    dynamic_queries = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        query_counts[prompt] = query_counts.get(prompt, 0) + 1
        print(f"Sync query called with prompt: {prompt}, count: {query_counts[prompt]}")
        
        # Track dynamic queries separately
        if "Dynamic query" in prompt:
            dynamic_queries.append(prompt)
            
        # Return string that includes execution count
        return f"Response {query_counts[prompt]} to: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        query_counts[prompt] = query_counts.get(prompt, 0) + 1
        print(f"Async query called with prompt: {prompt}, count: {query_counts[prompt]}")
        await asyncio.sleep(0.01)
        
        # Track dynamic queries separately
        if "Dynamic query" in prompt:
            dynamic_queries.append(prompt)
            
        # Return string that includes execution count
        return f"Response {query_counts[prompt]} to: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with multiple identical queries
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Two identical queries that should be cached #}
        {% set query1 = llmquery(prompt="Cached query") %}
        First result: {{ query1 }}
        
        {% set query2 = llmquery(prompt="Cached query") %}
        Second result (should be cached): {{ query2 }}
        
        {# Dynamic query that should not be cached between renderings #}
        {% set dynamic = llmquery(prompt="Dynamic query " + range(1000)|random|string) %}
        Dynamic result: {{ dynamic }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Reset test counters before each rendering
        query_counts.clear()
        dynamic_queries.clear()
        
        # First rendering
        print("\n=== First rendering ===")
        result1 = render_template_parallel(template_path, {}, enable_parallel=True)
        
        print("\nFirst render result:")
        print(result1)
        print(f"Query counts after first render: {query_counts}")
        print(f"Dynamic queries after first render: {dynamic_queries}")
        
        # Save query counts from first run
        first_run_counts = dict(query_counts)
        first_run_dynamics = list(dynamic_queries)
        
        # Verify first rendering cache behavior
        # The "Cached query" should only be executed once despite appearing twice in the template
        assert "Cached query" in first_run_counts, "Cached query was not executed"
        assert first_run_counts["Cached query"] == 1, f"Expected 'Cached query' to execute once, but got {first_run_counts['Cached query']} executions"
        
        # Verify that we executed at least one dynamic query
        assert len(first_run_dynamics) == 1, f"Expected 1 dynamic query, but got {len(first_run_dynamics)}"
        
        # Reset counters for second rendering
        query_counts.clear()
        dynamic_queries.clear()
        
        # Second rendering
        print("\n=== Second rendering ===")
        result2 = render_template_parallel(template_path, {"__second_run": True}, enable_parallel=True)
        
        print("\nSecond render result:")
        print(result2)
        print(f"Query counts after second render: {query_counts}")
        print(f"Dynamic queries after second render: {dynamic_queries}")
        
        # Verify second rendering cache behavior
        # The "Cached query" should only be executed once despite appearing twice in the template
        assert "Cached query" in query_counts, "Cached query was not executed in second rendering"
        assert query_counts["Cached query"] == 1, f"Expected 'Cached query' to execute once in second rendering, but got {query_counts['Cached query']} executions"
        
        # Dynamic queries should be different between renderings and executed exactly once in each
        assert len(dynamic_queries) == 1, f"Expected 1 dynamic query in second rendering, but got {len(dynamic_queries)}"
        assert first_run_dynamics[0] != dynamic_queries[0], "Dynamic queries should be different between renderings"
        
        # Check first render results content
        assert "First result:" in result1
        assert "Second result (should be cached):" in result1
        assert "Dynamic result:" in result1
        
        # Check second render results content
        assert "First result:" in result2
        assert "Second result (should be cached):" in result2
        assert "Dynamic result:" in result2
        
        # Verify both renderings have cached query responses that are consistent within each rendering
        first_response1 = None
        second_response1 = None
        dynamic_response1 = None
        
        for line in result1.splitlines():
            if "First result:" in line:
                first_response1 = line.strip()
            elif "Second result" in line:
                second_response1 = line.strip()
            elif "Dynamic result:" in line:
                dynamic_response1 = line.strip()
        
        first_response2 = None
        second_response2 = None
        dynamic_response2 = None
        
        for line in result2.splitlines():
            if "First result:" in line:
                first_response2 = line.strip()
            elif "Second result" in line:
                second_response2 = line.strip()
            elif "Dynamic result:" in line:
                dynamic_response2 = line.strip()
        
        # Check that results were found
        assert first_response1 is not None, "Couldn't find first response in first rendering"
        assert second_response1 is not None, "Couldn't find second response in first rendering"
        assert dynamic_response1 is not None, "Couldn't find dynamic response in first rendering"
        
        assert first_response2 is not None, "Couldn't find first response in second rendering"
        assert second_response2 is not None, "Couldn't find second response in second rendering"
        assert dynamic_response2 is not None, "Couldn't find dynamic response in second rendering"
        
        # First and second responses in the same rendering should match
        assert first_response1.split(": ", 1)[1] == second_response1.split(": ", 1)[1], "Cache not working in first rendering"
        assert first_response2.split(": ", 1)[1] == second_response2.split(": ", 1)[1], "Cache not working in second rendering"
        
        # Dynamic responses should be different between renderings
        assert dynamic_response1 != dynamic_response2, "Dynamic responses should differ between renderings"
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_cache_with_complex_parameters(mock_query_async, mock_query):
    """
    Test caching with complex parameter objects.
    
    This test checks that queries with complex parameter objects
    are cached correctly.
    """
    
    # Track query execution
    executed_queries = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        executed_queries.append((prompt, params))
        print(f"Sync query called with prompt: {prompt}, params: {params}")
        
        # Return a response that includes the prompt and parameters
        return f"Response to {prompt} with {str(params)}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        executed_queries.append((prompt, params))
        print(f"Async query called with prompt: {prompt}, params: {params}")
        await asyncio.sleep(0.01)
        
        # Return a response that includes the prompt and parameters
        return f"Response to {prompt} with {str(params)}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with complex parameter objects
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Define some complex parameters #}
        {% set params1 = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100} %}
        {% set params2 = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100} %}
        {% set params3 = {"model": "gpt-4", "temperature": 0.8, "max_tokens": 100} %}
        
        {# Queries with same parameters should be cached #}
        {% set query1 = llmquery(prompt="Test query", model=params1.model, temperature=params1.temperature, max_tokens=params1.max_tokens) %}
        First query: {{ query1 }}
        
        {% set query2 = llmquery(prompt="Test query", model=params2.model, temperature=params2.temperature, max_tokens=params2.max_tokens) %}
        Second query (should be cached): {{ query2 }}
        
        {# Query with different parameters should not be cached #}
        {% set query3 = llmquery(prompt="Test query", model=params3.model, temperature=params3.temperature, max_tokens=params3.max_tokens) %}
        Third query (different params): {{ query3 }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nExecuted queries:")
        for i, (prompt, params) in enumerate(executed_queries):
            print(f"{i+1}. {prompt} with {params}")
        
        print("\nTemplate result:")
        print(result)
        
        # Verify the expected content is in the result
        assert "First query:" in result, "First query result not found"
        assert "Second query (should be cached):" in result, "Second query result not found"
        assert "Third query (different params):" in result, "Third query result not found"
        
        # Extract the response values to check if caching worked properly
        first_response = None
        second_response = None
        third_response = None
        
        for line in result.splitlines():
            if "First query:" in line:
                first_response = line.split("First query:")[1].strip()
            elif "Second query" in line:
                second_response = line.split("Second query (should be cached):")[1].strip()
            elif "Third query" in line:
                third_response = line.split("Third query (different params):")[1].strip()
        
        # First and second should match, third should be different
        assert first_response is not None, "Couldn't find first response"
        assert second_response is not None, "Couldn't find second response"
        assert third_response is not None, "Couldn't find third response"
        
        assert first_response == second_response, "First and second responses should be the same (cached)"
        assert first_response != third_response, "First and third responses should be different"
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path) 