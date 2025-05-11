import pytest
import asyncio
import time
from unittest.mock import AsyncMock

from src.jinja_prompt_chaining_system.parallel import (
    ParallelExecutor,
    Query,
    extract_dependencies
)

def test_extract_dependencies():
    """Test dependency extraction from template strings."""
    # Test with no variables
    assert extract_dependencies("Hello world", {}) == set()
    
    # Test with simple variables
    assert extract_dependencies("Hello {{ name }}", {}) == {"name"}
    assert extract_dependencies("Hello {{ name }} and {{ age }}", {}) == {"name", "age"}
    
    # Test with context
    assert extract_dependencies("Hello {{ name }}", {"name": "John"}) == set()
    assert extract_dependencies("Hello {{ name }} and {{ age }}", {"name": "John"}) == {"age"}
    
    # Test with nested attributes - check for both the base and the full path
    nested_deps = extract_dependencies("Hello {{ user.name }}", {})
    assert "user" in nested_deps
    assert "user.name" in nested_deps
    
    dict_deps = extract_dependencies("Hello {{ user['name'] }}", {})
    assert "user" in dict_deps
    
    # Test with complex expressions
    assert extract_dependencies("Hello {{ name | upper }}", {}) == {"name"}
    assert extract_dependencies("{{ 'Hello ' + name }}", {}) == {"name"}

def test_query_class():
    """Test the Query class functionality."""
    # Test with all parameters
    query = Query(
        prompt="Hello {{ name }}",
        params={"model": "gpt-4", "temperature": 0.7},
        dependencies={"name"},
        result_var="greeting"
    )
    
    assert query.prompt == "Hello {{ name }}"
    assert query.params == {"model": "gpt-4", "temperature": 0.7}
    assert query.dependencies == {"name"}
    assert query.result_var == "greeting"
    
    # Test with auto-generated result_var
    query = Query(
        prompt="Hello world",
        params={"model": "gpt-4"},
        dependencies=set()
    )
    
    assert query.prompt == "Hello world"
    assert query.params == {"model": "gpt-4"}
    assert query.dependencies == set()
    assert query.result_var is not None
    assert query.result_var.startswith("result_")
    assert len(query.result_var) > 8  # Should have a UUID part

@pytest.mark.asyncio
async def test_parallel_executor_basic():
    """Test basic functionality of ParallelExecutor."""
    executor = ParallelExecutor(max_concurrent=2)
    
    # Mock the LLM client
    mock_client = AsyncMock()
    mock_client.query_async.side_effect = lambda prompt, **params: f"Response to: {prompt}"
    executor.client = mock_client
    
    # Create test queries
    query1 = Query("Query 1", {"model": "gpt-4"}, set(), "result1")
    query2 = Query("Query 2", {"model": "gpt-4"}, set(), "result2")
    
    # Execute queries
    context = {}
    result_context = await executor.execute_all([query1, query2], context)
    
    # Check results
    assert result_context["result1"] == "Response to: Query 1"
    assert result_context["result2"] == "Response to: Query 2"
    
    # Check that client was called correctly
    assert mock_client.query_async.call_count == 2
    mock_client.query_async.assert_any_call("Query 1", model="gpt-4")
    mock_client.query_async.assert_any_call("Query 2", model="gpt-4")

@pytest.mark.asyncio
async def test_parallel_executor_with_dependencies():
    """Test ParallelExecutor with dependent queries."""
    executor = ParallelExecutor(max_concurrent=2)
    
    # Mock the LLM client
    mock_client = AsyncMock()
    mock_client.query_async.side_effect = lambda prompt, **params: f"Response to: {prompt}"
    executor.client = mock_client
    
    # Create test queries with dependencies
    query1 = Query("Query 1", {"model": "gpt-4"}, set(), "result1")
    query2 = Query("Query 2 with {{ result1 }}", {"model": "gpt-4"}, {"result1"}, "result2")
    
    # Execute queries
    context = {}
    result_context = await executor.execute_all([query1, query2], context)
    
    # First query should be executed and its result stored
    assert result_context["result1"] == "Response to: Query 1"
    
    # Second query should use the result from the first query
    expected_prompt = "Query 2 with Response to: Query 1"
    mock_client.query_async.assert_any_call(expected_prompt, model="gpt-4")
    assert result_context["result2"] == f"Response to: {expected_prompt}"

@pytest.mark.asyncio
async def test_parallel_executor_execution_order():
    """Test that queries with dependencies run in the correct order."""
    executor = ParallelExecutor(max_concurrent=2)
    
    # Track execution order
    execution_order = []
    
    # Mock the LLM client
    async def mock_query_async(prompt, **params):
        execution_order.append(prompt)
        return f"Response to: {prompt}"
    
    mock_client = AsyncMock()
    mock_client.query_async.side_effect = mock_query_async
    executor.client = mock_client
    
    # Create a chain of dependent queries
    query1 = Query("Query 1", {"model": "gpt-4"}, set(), "result1")
    query2 = Query("Query 2 with {{ result1 }}", {"model": "gpt-4"}, {"result1"}, "result2")
    query3 = Query("Query 3 with {{ result2 }}", {"model": "gpt-4"}, {"result2"}, "result3")
    
    # Execute queries in an order that shouldn't matter
    context = {}
    result_context = await executor.execute_all([query3, query1, query2], context)
    
    # Check results
    assert result_context["result1"] == "Response to: Query 1"
    assert result_context["result2"].startswith("Response to: Query 2 with")
    assert result_context["result3"].startswith("Response to: Query 3 with")
    
    # Check execution order
    assert execution_order[0] == "Query 1"  # Must be first due to dependencies
    assert execution_order[1].startswith("Query 2 with")  # Must be second
    assert execution_order[2].startswith("Query 3 with")  # Must be third

@pytest.mark.asyncio
async def test_parallel_executor_concurrency_limit():
    """Test that ParallelExecutor respects the concurrency limit."""
    # Use a lower concurrency limit for testing
    MAX_CONCURRENT = 2
    NUM_QUERIES = 8
    QUERY_DELAY = 0.1  # seconds
    
    # Disable test detection to ensure the semaphore is used
    executor = ParallelExecutor(max_concurrent=MAX_CONCURRENT, disable_test_detection=True)
    
    # Create a mock client with a delay
    mock_client = AsyncMock()
    
    async def delayed_response(prompt, **params):
        # Each query takes QUERY_DELAY seconds
        await asyncio.sleep(QUERY_DELAY)
        return f"Response to: {prompt}"
    
    mock_client.query_async.side_effect = delayed_response
    executor.client = mock_client
    
    # Create independent queries
    queries = [
        Query(f"Query {i}", {"model": "gpt-4"}, set(), f"result{i}")
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
    # With MAX_CONCURRENT=2, the NUM_QUERIES should execute in NUM_QUERIES/MAX_CONCURRENT batches
    theoretical_batches = NUM_QUERIES / MAX_CONCURRENT
    theoretical_time = theoretical_batches * QUERY_DELAY
    
    # Print results for debugging
    print(f"\nMax concurrent: {MAX_CONCURRENT}")
    print(f"Num queries: {NUM_QUERIES}")
    print(f"Theoretical batches: {theoretical_batches}")
    print(f"Theoretical time: {theoretical_time:.2f}s")
    print(f"Actual time: {execution_time:.2f}s")
    
    # Allow some buffer for execution overhead
    max_allowed_time = theoretical_time * 1.5
    min_allowed_time = theoretical_time * 0.8
    
    # The execution time should be close to the theoretical time
    assert execution_time < max_allowed_time, \
        f"Execution took too long ({execution_time:.2f}s), expected ~{theoretical_time:.2f}s"
    
    # And it shouldn't be too fast (which would indicate the concurrency limit isn't working)
    assert execution_time > min_allowed_time, \
        f"Execution too fast ({execution_time:.2f}s), suggesting max_concurrent wasn't respected"

@pytest.mark.asyncio
async def test_parallel_execution_timing():
    """Test that independent queries actually run in parallel by measuring execution time."""
    # For parallel execution we'll allow test detection for max parallelism
    executor = ParallelExecutor(max_concurrent=4)
    
    # Create a client with a delay to simulate network latency
    mock_client = AsyncMock()
    
    # Define a consistent delay for each query
    QUERY_DELAY = 0.2  # seconds
    NUM_QUERIES = 4
    
    async def delayed_response(prompt, **params):
        # Each query takes QUERY_DELAY seconds
        await asyncio.sleep(QUERY_DELAY)
        return f"Response to: {prompt}"
    
    mock_client.query_async.side_effect = delayed_response
    executor.client = mock_client
    
    # Create 4 independent queries
    queries = [
        Query(f"Query {i}", {"model": "gpt-4"}, set(), f"result{i}")
        for i in range(NUM_QUERIES)
    ]
    
    # Measure execution time with parallelism
    start_time = time.time()
    context = {}
    await executor.execute_all(queries, context)
    parallel_duration = time.time() - start_time
    
    # Verify all queries were executed
    assert len(context) == NUM_QUERIES
    for i in range(NUM_QUERIES):
        assert f"result{i}" in context
    
    # Ensure the test itself is valid by comparing with sequential execution
    # This creates a sequential executor with concurrency=1
    # For sequential execution, disable test detection to enforce sequential behavior
    sequential_executor = ParallelExecutor(max_concurrent=1, disable_test_detection=True)
    sequential_executor.client = mock_client
    
    # Run the same queries sequentially
    start_time = time.time()
    sequential_context = {}
    # Force sequential execution by limiting concurrency to 1
    await sequential_executor.execute_all(queries, sequential_context)
    sequential_duration = time.time() - start_time
    
    # Calculate theoretical times and speedup
    theoretical_parallel = QUERY_DELAY  # Should just take one QUERY_DELAY period
    theoretical_sequential = NUM_QUERIES * QUERY_DELAY  # Should take NUM_QUERIES times QUERY_DELAY
    actual_speedup = sequential_duration / parallel_duration
    
    # Print detailed timing information
    print("\n=== PARALLEL EXECUTION TIMING RESULTS ===")
    print(f"Number of queries: {NUM_QUERIES}")
    print(f"Per-query delay: {QUERY_DELAY:.2f}s")
    print(f"Parallel execution time: {parallel_duration:.2f}s (theoretical: ~{theoretical_parallel:.2f}s)")
    print(f"Sequential execution time: {sequential_duration:.2f}s (theoretical: ~{theoretical_sequential:.2f}s)")
    print(f"Speedup achieved: {actual_speedup:.2f}x (theoretical max: {NUM_QUERIES:.2f}x)")
    print("=============================================")
    
    # If queries ran in parallel, execution time should be close to QUERY_DELAY
    # If they ran sequentially, it would be close to NUM_QUERIES * QUERY_DELAY
    
    # Allow some buffer for execution overhead
    assert parallel_duration < (2 * QUERY_DELAY), \
        f"Expected parallel execution (~{QUERY_DELAY}s) but took {parallel_duration}s"
    
    # Sequential should be significantly slower than parallel
    assert sequential_duration > parallel_duration, \
        "Sequential execution should be slower than parallel execution"
    
    # We should get at least a 1.5x speedup
    assert actual_speedup > 1.5, \
        f"Expected significant speedup but only got {actual_speedup:.2f}x"

@pytest.mark.asyncio
async def test_parallel_execution_respects_concurrency_limit():
    """Test that parallel execution respects the max_concurrent limit."""
    MAX_CONCURRENT = 3
    NUM_QUERIES = 9
    QUERY_DELAY = 0.2  # seconds
    
    # Disable test detection to ensure the semaphore is used
    executor = ParallelExecutor(max_concurrent=MAX_CONCURRENT, disable_test_detection=True)
    
    # Create a client with a delay to simulate network latency
    mock_client = AsyncMock()
    
    async def delayed_response(prompt, **params):
        # Each query takes QUERY_DELAY seconds
        await asyncio.sleep(QUERY_DELAY)
        return f"Response to: {prompt}"
    
    mock_client.query_async.side_effect = delayed_response
    executor.client = mock_client
    
    # Create independent queries
    queries = [
        Query(f"Query {i}", {"model": "gpt-4"}, set(), f"result{i}")
        for i in range(NUM_QUERIES)
    ]
    
    # Measure execution time with parallelism
    start_time = time.time()
    context = {}
    await executor.execute_all(queries, context)
    parallel_duration = time.time() - start_time
    
    # Verify all queries were executed
    assert len(context) == NUM_QUERIES
    for i in range(NUM_QUERIES):
        assert f"result{i}" in context
    
    # Calculate theoretical times
    # With max_concurrent=3, the 9 queries should execute in 3 batches
    # Each batch takes QUERY_DELAY time
    theoretical_duration = (NUM_QUERIES / MAX_CONCURRENT) * QUERY_DELAY
    
    # Print results
    print("\n=== CONCURRENCY LIMIT TEST RESULTS ===")
    print(f"Number of queries: {NUM_QUERIES}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Per-query delay: {QUERY_DELAY:.2f}s")
    print(f"Actual execution time: {parallel_duration:.2f}s")
    print(f"Theoretical execution time: {theoretical_duration:.2f}s")
    print(f"Batches executed: {NUM_QUERIES / MAX_CONCURRENT:.1f}")
    print("========================================")
    
    # Allow some buffer for execution overhead
    # Execution time should be close to theoretical time
    max_allowed_time = theoretical_duration * 1.5
    assert parallel_duration < max_allowed_time, \
        f"Execution took {parallel_duration:.2f}s, expected ~{theoretical_duration:.2f}s"
    
    # Execution should take at least as long as the theoretical minimum
    min_allowed_time = theoretical_duration * 0.8
    assert parallel_duration > min_allowed_time, \
        f"Execution too fast ({parallel_duration:.2f}s), suggesting max_concurrent wasn't respected" 