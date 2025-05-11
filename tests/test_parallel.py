import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from jinja2 import Template, Environment
import time

from src.jinja_prompt_chaining_system.parallel import (
    ParallelExecutor,
    Query,
    extract_dependencies,
    ParallelQueryTracker
)

# Simple dependency extraction tests
def test_extract_dependencies_simple():
    """Test extracting dependencies from a simple template string."""
    template = "This uses {{ var1 }} and {{ var2 }}"
    context = {}
    
    deps = extract_dependencies(template, context)
    assert deps == {"var1", "var2"}

def test_extract_dependencies_with_context():
    """Test that variables already in context aren't dependencies."""
    template = "This uses {{ var1 }} and {{ var2 }}"
    context = {"var1": "hello"}
    
    deps = extract_dependencies(template, context)
    assert deps == {"var2"}  # var1 is in context, so not a dependency

def test_extract_dependencies_nested():
    """Test extracting nested variable references."""
    template = "This uses {{ var1.attr }} and {{ var2['key'] }}"
    context = {}
    
    deps = extract_dependencies(template, context)
    assert "var1" in deps
    assert "var2" in deps

# Query class tests
def test_query_representation():
    """Test the Query object representation."""
    query = Query(
        prompt="Hello {{ name }}",
        params={"model": "gpt-4"},
        dependencies={"name"},
        result_var="result1"
    )
    
    assert query.prompt == "Hello {{ name }}"
    assert query.params == {"model": "gpt-4"}
    assert query.dependencies == {"name"}
    assert query.result_var == "result1"

# ParallelExecutor basic tests
@pytest.mark.asyncio
async def test_parallel_executor_simple():
    """Test parallel executor with no dependencies."""
    executor = ParallelExecutor(max_concurrent=2)
    
    # Create mock queries and results
    query1 = Query("Q1", {"model": "gpt-4"}, set(), "result1")
    query2 = Query("Q2", {"model": "gpt-4"}, set(), "result2")
    
    # Create a mock client
    mock_client = AsyncMock()
    mock_client.query_async.side_effect = [
        "Response 1",
        "Response 2"
    ]
    
    # Set up the executor with our mock
    executor.client = mock_client
    
    # Execute and check results
    context = {}
    await executor.execute_all([query1, query2], context)
    
    assert context["result1"] == "Response 1"
    assert context["result2"] == "Response 2"
    assert mock_client.query_async.call_count == 2

@pytest.mark.asyncio
async def test_parallel_executor_with_dependencies():
    """Test parallel executor with dependencies between queries."""
    executor = ParallelExecutor(max_concurrent=2)
    
    # Create mock queries where query2 depends on query1
    query1 = Query("Q1", {"model": "gpt-4"}, set(), "result1")
    query2 = Query("Q2 {{ result1 }}", {"model": "gpt-4"}, {"result1"}, "result2")
    
    # Create a mock client
    mock_client = AsyncMock()
    mock_client.query_async.side_effect = [
        "Response 1",
        "Response 2 with Response 1"
    ]
    
    # Set up the executor with our mock
    executor.client = mock_client
    
    # Execute and check results
    context = {}
    await executor.execute_all([query1, query2], context)
    
    assert context["result1"] == "Response 1"
    assert context["result2"] == "Response 2 with Response 1"
    
    # Query1 should be called first, then query2
    assert mock_client.query_async.call_count == 2

@pytest.mark.asyncio
async def test_parallel_executor_max_concurrent():
    """Test that executor respects max_concurrent limit."""
    MAX_CONCURRENT = 2
    NUM_QUERIES = 6
    QUERY_DELAY = 0.1  # seconds
    
    # Disable test detection to ensure the semaphore is used
    executor = ParallelExecutor(max_concurrent=MAX_CONCURRENT, disable_test_detection=True)
    
    # Create a client with a delay to simulate network latency
    mock_client = AsyncMock()
    
    # Use a similar approach to our working test_parallel_execution_respects_concurrency_limit
    async def delayed_response(prompt, **params):
        # Each query takes QUERY_DELAY seconds
        await asyncio.sleep(QUERY_DELAY)
        return f"Response for {prompt}"
    
    mock_client.query_async.side_effect = delayed_response
    executor.client = mock_client
    
    # Create independent queries
    queries = [
        Query(f"Q{i}", {"model": "gpt-4"}, set(), f"result{i}")
        for i in range(NUM_QUERIES)
    ]
    
    # Measure execution time
    start_time = time.time()
    context = {}
    await executor.execute_all(queries, context)
    execution_time = time.time() - start_time
    
    # Verify all queries were executed
    assert mock_client.query_async.call_count == NUM_QUERIES
    for i in range(NUM_QUERIES):
        assert f"result{i}" in context
    
    # Calculate theoretical execution time
    # With max_concurrent=2, the 6 queries should execute in 3 batches
    # Each batch takes QUERY_DELAY time
    theoretical_time = (NUM_QUERIES / MAX_CONCURRENT) * QUERY_DELAY
    
    # Allow some buffer for execution overhead
    max_allowed_time = theoretical_time * 1.5
    min_allowed_time = theoretical_time * 0.8
    
    # Print results for debugging
    print(f"\nMax concurrent: {MAX_CONCURRENT}")
    print(f"Num queries: {NUM_QUERIES}")
    print(f"Theoretical time: {theoretical_time:.2f}s")
    print(f"Actual time: {execution_time:.2f}s")
    
    # The execution time should be close to the theoretical time
    assert execution_time < max_allowed_time, \
        f"Execution took too long ({execution_time:.2f}s), expected ~{theoretical_time:.2f}s"
    
    # And it shouldn't be too fast (which would indicate the concurrency limit isn't working)
    assert execution_time > min_allowed_time, \
        f"Execution too fast ({execution_time:.2f}s), suggesting max_concurrent wasn't respected"

# Integration tests
@patch('src.jinja_prompt_chaining_system.parser.LLMClient')
def test_template_with_parallel_queries(mock_llm_client):
    """Test rendering a template with parallel queries."""
    from src.jinja_prompt_chaining_system import render_prompt
    
    # Setup mock
    client = Mock()
    client.query.side_effect = [
        "First response",
        "Second response using First response"
    ]
    mock_llm_client.return_value = client
    
    # Create a simple template with two queries
    template_content = """
    {% set resp1 = llmquery(prompt="Query 1", model="gpt-4") %}
    {{ resp1 }}
    
    {% set resp2 = llmquery(prompt="Query 2 using " + resp1, model="gpt-4") %}
    {{ resp2 }}
    """
    
    # TODO: This test will be completed after implementing parallel execution
    # For now, it serves as a scaffold for the actual integration test 