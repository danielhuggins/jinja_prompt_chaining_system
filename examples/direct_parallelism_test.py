#!/usr/bin/env python
"""
Direct test of parallel execution in Jinja Prompt Chaining System.

This script demonstrates true parallelism by directly accessing
the internal parallel execution mechanism with controlled delays.
"""

import os
import time
import asyncio
import tempfile
from pathlib import Path
import concurrent.futures

# Import the core parallel components directly
from jinja_prompt_chaining_system.parallel import ParallelExecutor, Query
from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

# Constants for testing
QUERY_DELAY = 1.0  # seconds - delay for each query
NUM_QUERIES = 4    # number of queries to run
MAX_CONCURRENT = NUM_QUERIES  # maximum concurrent queries

# Create a list to track execution times
execution_timestamps = []

async def delayed_query(prompt, delay=QUERY_DELAY):
    """Simulate an LLM query with a controlled delay."""
    # Record start time
    start_time = time.time()
    execution_timestamps.append((prompt, "start", start_time))
    print(f"  Query '{prompt}' started at t={start_time:.3f}s")
    
    # Simulate network delay
    await asyncio.sleep(delay)
    
    # Record end time
    end_time = time.time()
    execution_timestamps.append((prompt, "end", end_time))
    print(f"  Query '{prompt}' finished at t={end_time:.3f}s (took {end_time-start_time:.3f}s)")
    
    return f"Response to {prompt}"

def run_sequential_test():
    """Run queries sequentially."""
    # Clear tracking data
    execution_timestamps.clear()
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Sequential execution
        print("\n=== SEQUENTIAL EXECUTION ===")
        start_time = time.time()
        
        # Run each query in sequence
        results = {}
        for i in range(1, NUM_QUERIES + 1):
            prompt = f"Query {i}"
            result = loop.run_until_complete(delayed_query(prompt))
            results[f"result{i}"] = result
        
        sequential_time = time.time() - start_time
        print(f"Sequential total time: {sequential_time:.3f}s")
        
        return sequential_time, results
    finally:
        loop.close()

async def run_parallel_test_async():
    """Run queries in parallel using asyncio."""
    # Clear tracking data
    execution_timestamps.clear()
    
    print("\n=== PARALLEL EXECUTION ===")
    start_time = time.time()
    
    # Create tasks for all queries
    tasks = []
    for i in range(1, NUM_QUERIES + 1):
        prompt = f"Query {i}"
        tasks.append(delayed_query(prompt))
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Compile results
    result_dict = {}
    for i, result in enumerate(results, 1):
        result_dict[f"result{i}"] = result
    
    parallel_time = time.time() - start_time
    print(f"Parallel total time: {parallel_time:.3f}s")
    
    return parallel_time, result_dict

def run_parallel_test():
    """Run the parallel test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(run_parallel_test_async())
    finally:
        loop.close()

def analyze_execution_timestamps():
    """Analyze the execution timestamps to verify parallelism."""
    # Group by prompt
    execution_by_prompt = {}
    for prompt, stage, timestamp in execution_timestamps:
        if prompt not in execution_by_prompt:
            execution_by_prompt[prompt] = {}
        execution_by_prompt[prompt][stage] = timestamp
    
    # Check for overlaps
    overlaps = []
    prompts = list(execution_by_prompt.keys())
    
    for i in range(len(prompts)):
        prompt_i = prompts[i]
        start_i = execution_by_prompt[prompt_i]["start"]
        end_i = execution_by_prompt[prompt_i]["end"]
        
        for j in range(i+1, len(prompts)):
            prompt_j = prompts[j]
            start_j = execution_by_prompt[prompt_j]["start"]
            end_j = execution_by_prompt[prompt_j]["end"]
            
            # Check if there's an overlap in execution time
            if start_i < end_j and end_i > start_j:
                overlaps.append((prompt_i, prompt_j))
    
    return overlaps, execution_by_prompt

def direct_test_with_template():
    """Test parallelism using the template system."""
    # Create a test template with multiple queries
    template_content = """
    <h1>Parallelism Test with {0} Queries</h1>
    
    {{% set resp1 = llmquery(prompt="Direct Test Query 1", model="gpt-3.5-turbo") %}}
    <h2>Response 1:</h2>
    {{{{ resp1 }}}}
    
    {{% set resp2 = llmquery(prompt="Direct Test Query 2", model="gpt-3.5-turbo") %}}
    <h2>Response 2:</h2>
    {{{{ resp2 }}}}
    
    {{% set resp3 = llmquery(prompt="Direct Test Query 3", model="gpt-3.5-turbo") %}}
    <h2>Response 3:</h2>
    {{{{ resp3 }}}}
    
    {{% set resp4 = llmquery(prompt="Direct Test Query 4", model="gpt-3.5-turbo") %}}
    <h2>Response 4:</h2>
    {{{{ resp4 }}}}
    """.format(NUM_QUERIES)
    
    # Create a temporary directory for the template
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the template file
        template_path = Path(tmpdir) / "parallel_test.jinja"
        with open(template_path, "w") as f:
            f.write(template_content)
        
        # Direct test using ParallelExecutor
        # Build queries manually
        queries = []
        for i in range(1, NUM_QUERIES + 1):
            query = Query(
                prompt=f"Direct Test Query {i}",
                params={"model": "gpt-3.5-turbo"},
                dependencies=set(),
                result_var=f"result{i}"
            )
            queries.append(query)
        
        # Create an executor
        executor = ParallelExecutor(max_concurrent=MAX_CONCURRENT)
        
        # Replace the executor's execute method with our controlled version
        original_execute = executor.execute
        
        async def mock_execute(query, context):
            """Mock execution with controlled timing."""
            return await delayed_query(query.prompt, QUERY_DELAY)
        
        executor.execute = mock_execute
        
        # Sequential execution first
        print("\n=== DIRECT TEMPLATE SEQUENTIAL EXECUTION ===")
        start_time = time.time()
        context = {}
        
        # Execute queries sequentially
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            execution_timestamps.clear()
            for query in queries:
                result = loop.run_until_complete(mock_execute(query, context))
                context[query.result_var] = result
            
            sequential_time = time.time() - start_time
            print(f"Sequential total time: {sequential_time:.3f}s")
            
            # Analyze sequential execution
            sequential_overlaps, sequential_timestamps = analyze_execution_timestamps()
            print(f"Sequential overlaps: {len(sequential_overlaps)}")
            
            # Now test parallel execution
            print("\n=== DIRECT TEMPLATE PARALLEL EXECUTION ===")
            start_time = time.time()
            execution_timestamps.clear()
            
            # Execute all queries in parallel
            context = {}
            updated_context = loop.run_until_complete(executor.execute_all(queries, context))
            
            parallel_time = time.time() - start_time
            print(f"Parallel total time: {parallel_time:.3f}s")
            
            # Analyze parallel execution
            parallel_overlaps, parallel_timestamps = analyze_execution_timestamps()
            print(f"Parallel overlaps: {len(parallel_overlaps)}")
            if parallel_overlaps:
                print(f"Overlapping queries: {parallel_overlaps}")
            
            # Calculate speedup
            speedup = sequential_time / parallel_time
            print(f"\nSpeedup: {speedup:.2f}x")
            
            # Calculate theoretical speedup
            theoretical_speedup = min(NUM_QUERIES, MAX_CONCURRENT)
            print(f"Theoretical speedup: {theoretical_speedup:.1f}x")
            
            # Calculate efficiency
            efficiency = (speedup / theoretical_speedup) * 100
            print(f"Efficiency: {efficiency:.1f}%")
            
            # Evaluate results
            print("\n=== CONCLUSION ===")
            if parallel_time < sequential_time:
                if len(parallel_overlaps) > 0:
                    print("✅ PARALLELISM CONFIRMED: Queries ran in parallel as proven by:")
                    print(f"1. Parallel execution ({parallel_time:.2f}s) was faster than sequential ({sequential_time:.2f}s)")
                    print(f"2. {len(parallel_overlaps)} overlapping query executions detected")
                    
                    if efficiency >= 70:
                        print(f"3. High efficiency: {efficiency:.1f}% of theoretical speedup was achieved")
                    else:
                        print(f"3. Moderate efficiency: {efficiency:.1f}% of theoretical speedup was achieved")
                else:
                    print("⚠️ MIXED RESULTS: Faster execution but no overlaps detected.")
                    print("This suggests parallelism is happening but may not be optimal.")
            else:
                if len(parallel_overlaps) > 0:
                    print("⚠️ PARTIAL PARALLELISM: Overlapping execution detected but not faster overall.")
                    print("This suggests parallel execution has high overhead.")
                else:
                    print("❌ NO PARALLELISM DETECTED: Neither timing nor execution overlap indicates parallelism.")
        finally:
            loop.close()
            # Restore original execute method
            executor.execute = original_execute

def run_comprehensive_test():
    """Run all tests to comprehensively prove parallelism."""
    # Run basic sequential and parallel tests
    sequential_time, sequential_results = run_sequential_test()
    parallel_time, parallel_results = run_parallel_test()
    
    # Calculate speedup
    speedup = sequential_time / parallel_time
    
    # Calculate theoretical speedup
    theoretical_speedup = min(NUM_QUERIES, MAX_CONCURRENT)
    
    # Calculate efficiency
    efficiency = (speedup / theoretical_speedup) * 100
    
    # Analyze timestamps to detect parallelism
    overlaps, timestamps = analyze_execution_timestamps()
    
    # Print summary
    print("\n=== BASIC TEST SUMMARY ===")
    print(f"Sequential execution time: {sequential_time:.3f}s")
    print(f"Parallel execution time: {parallel_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Theoretical speedup: {theoretical_speedup:.1f}x")
    print(f"Efficiency: {efficiency:.1f}%")
    print(f"Concurrent executions detected: {len(overlaps)}")
    
    if overlaps:
        print(f"Overlapping queries: {overlaps}")
    
    # Now run the direct template test
    direct_test_with_template()

if __name__ == "__main__":
    run_comprehensive_test() 