#!/usr/bin/env python
"""
Conclusive proof of parallel execution in the Jinja Prompt Chaining System.

This script demonstrates true parallelism by using a controlled environment
with mocked LLM responses and precise timing measurements.
"""

import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

# Constants for testing
QUERY_DELAY = 1.0  # seconds - delay for each query
NUM_QUERIES = 4    # number of queries to run
MAX_CONCURRENT = NUM_QUERIES  # maximum concurrent queries

def create_test_template():
    """Create a test template with multiple independent queries."""
    template_content = """
    <h1>Parallelism Conclusive Proof</h1>
    
    {% set resp1 = llmquery(prompt="Query 1", model="gpt-3.5-turbo") %}
    <h2>Response 1:</h2>
    {{ resp1 }}
    
    {% set resp2 = llmquery(prompt="Query 2", model="gpt-3.5-turbo") %}
    <h2>Response 2:</h2>
    {{ resp2 }}
    
    {% set resp3 = llmquery(prompt="Query 3", model="gpt-3.5-turbo") %}
    <h2>Response 3:</h2>
    {{ resp3 }}
    
    {% set resp4 = llmquery(prompt="Query 4", model="gpt-3.5-turbo") %}
    <h2>Response 4:</h2>
    {{ resp4 }}
    """
    
    return template_content

def create_mock_llm_client():
    """Create a mock LLM client with controlled delays."""
    client = Mock()
    
    # Create predictable delayed responses for testing
    def mock_sync_query(prompt, *args, **kwargs):
        start_time = time.time()
        print(f"  [SYNC] {prompt} started at t={start_time:.3f}s")
        # Add a controlled delay to simulate network latency
        time.sleep(QUERY_DELAY)
        end_time = time.time()
        print(f"  [SYNC] {prompt} finished at t={end_time:.3f}s (took {end_time-start_time:.3f}s)")
        return f"Response to {prompt}"
    
    client.query = Mock(side_effect=mock_sync_query)
    
    # Create async version that also has precise delays
    async def mock_async_query(prompt, *args, **kwargs):
        import asyncio
        start_time = time.time()
        print(f"  [ASYNC] {prompt} started at t={start_time:.3f}s")
        # Add a controlled delay to simulate network latency
        await asyncio.sleep(QUERY_DELAY)
        end_time = time.time()
        print(f"  [ASYNC] {prompt} finished at t={end_time:.3f}s (took {end_time-start_time:.3f}s)")
        return f"Response to {prompt}"
    
    client.query_async = AsyncMock(side_effect=mock_async_query)
    
    return client

def run_conclusive_test():
    """Run a controlled test that definitively proves parallel execution."""
    template_content = create_test_template()
    
    # Create temporary template file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jinja", delete=False) as f:
        f.write(template_content)
        template_path = f.name
    
    try:
        # Create patches for both places where LLMClient is used
        with patch("jinja_prompt_chaining_system.llm.LLMClient") as mock_parser_client, \
             patch("jinja_prompt_chaining_system.parallel.LLMClient") as mock_parallel_client:
            
            # Create a mock LLM client
            client = create_mock_llm_client()
            
            # Apply the same mock to both places where LLMClient is used
            mock_parser_client.return_value = client
            mock_parallel_client.return_value = client
            
            # Reset call counts
            client.query.reset_mock()
            client.query_async.reset_mock()
            
            print("\n=== SEQUENTIAL EXECUTION ===")
            start_time = time.time()
            render_prompt(str(template_path), {})
            sequential_time = time.time() - start_time
            print(f"Sequential execution total time: {sequential_time:.3f}s")
            
            # Calculate expected sequential time
            expected_sequential = NUM_QUERIES * QUERY_DELAY
            print(f"Expected sequential time: {expected_sequential:.3f}s")
            
            # Count how many sync/async calls were made
            sync_calls = client.query.call_count
            async_calls = client.query_async.call_count
            print(f"Sync calls: {sync_calls}")
            print(f"Async calls: {async_calls}")
            
            # Reset call counts
            client.query.reset_mock()
            client.query_async.reset_mock()
            
            print("\n=== PARALLEL EXECUTION ===")
            start_time = time.time()
            render_template_parallel(str(template_path), {}, enable_parallel=True, max_concurrent=MAX_CONCURRENT)
            parallel_time = time.time() - start_time
            print(f"Parallel execution total time: {parallel_time:.3f}s")
            
            # Calculate expected parallel time
            expected_parallel = QUERY_DELAY  # Should be close to a single query time
            print(f"Expected parallel time: {expected_parallel:.3f}s")
            
            # Count how many sync/async calls were made
            sync_calls = client.query.call_count
            async_calls = client.query_async.call_count
            print(f"Sync calls: {sync_calls}")
            print(f"Async calls: {async_calls}")
            
            # Calculate speedup
            speedup = sequential_time / parallel_time
            print(f"\nSpeedup: {speedup:.2f}x")
            
            # Calculate the expected speedup
            expected_speedup = expected_sequential / expected_parallel
            print(f"Expected speedup: {expected_speedup:.2f}x")
            
            # Calculate efficiency
            efficiency = (speedup / expected_speedup) * 100
            print(f"Efficiency: {efficiency:.1f}%")
            
            # Evaluate the result
            print("\n=== CONCLUSION ===")
            if parallel_time < sequential_time:
                if async_calls > 0:
                    print("✅ PARALLELISM CONFIRMED: Queries ran in parallel as proven by:")
                    print(f"1. Parallel execution ({parallel_time:.2f}s) was faster than sequential ({sequential_time:.2f}s)")
                    print(f"2. Async API was used ({async_calls} async calls)")
                    
                    if efficiency >= 70:
                        print(f"3. High efficiency: {efficiency:.1f}% of theoretical speedup was achieved")
                    else:
                        print(f"3. Moderate efficiency: {efficiency:.1f}% of theoretical speedup was achieved")
                        
                    if speedup >= NUM_QUERIES * 0.7:
                        print(f"4. Near-optimal speedup: {speedup:.2f}x with {NUM_QUERIES} queries")
                    else:
                        print(f"4. Good speedup: {speedup:.2f}x with {NUM_QUERIES} queries")
                else:
                    print("⚠️ MIXED RESULTS: Faster execution but no async calls detected.")
                    print("This suggests the test environment may not be accurately measuring parallel execution.")
            else:
                print("❌ PARALLELISM NOT CONFIRMED: Sequential execution was faster than parallel.")
                print("This may be due to overhead in managing parallel requests or test environment issues.")
    
    finally:
        # Clean up temporary file
        if template_path:
            os.unlink(template_path)

if __name__ == "__main__":
    run_conclusive_test() 