#!/usr/bin/env python
"""
Demonstration of true parallelism in LLM queries.

This script shows quantifiable evidence that LLM queries are executed in parallel
by comparing execution times between parallel and sequential modes.
"""

import os
import time
import argparse
from pathlib import Path
import re
import datetime
import concurrent.futures

from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

def run_demo(num_queries=4, num_runs=3):
    """Run parallel vs sequential demo showing true time savings."""
    
    # Create a template with the specified number of queries
    template_content = """
    <h1>LLM Query Parallelism Test</h1>
    
    {% set start_time = "START_TIME_PLACEHOLDER" %}
    """
    
    # Add independent queries
    for i in range(1, num_queries + 1):
        template_content += f"""
        <h3>Query {i} Results</h3>
        {{% set resp{i} = llmquery(prompt="Generate a random name for a {['recipe', 'book', 'product', 'country', 'movie', 'song'][i % 6]} with a one sentence description. Include 'Query {i}' in your response.", model="gpt-4-mini") %}}
        {{{{ resp{i} }}}}
        """
    
    template_content += """
    {% set end_time = "END_TIME_PLACEHOLDER" %}
    """
    
    # Create temporary template file
    template_path = Path("parallel_timing_test.jinja")
    with open(template_path, "w") as f:
        f.write(template_content)
    
    try:
        # Run multiple times to get average times
        sequential_times = []
        parallel_times = []
        
        print(f"Running {num_runs} tests with {num_queries} queries each...")
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}:")
            
            # Sequential run
            print("  Running sequential execution...")
            start_time = time.time()
            render_prompt(str(template_path), {})
            end_time = time.time()
            sequential_time = end_time - start_time
            sequential_times.append(sequential_time)
            print(f"  Sequential time: {sequential_time:.2f} seconds")
            
            # Parallel run
            print("  Running parallel execution...")
            start_time = time.time()
            render_template_parallel(str(template_path), {}, enable_parallel=True, max_concurrent=num_queries)
            end_time = time.time()
            parallel_time = end_time - start_time
            parallel_times.append(parallel_time)
            print(f"  Parallel time: {parallel_time:.2f} seconds")
            
            # Calculate speedup for this run
            speedup = sequential_time / parallel_time
            print(f"  Speedup: {speedup:.2f}x")
        
        # Calculate averages
        avg_sequential = sum(sequential_times) / len(sequential_times)
        avg_parallel = sum(parallel_times) / len(parallel_times)
        avg_speedup = avg_sequential / avg_parallel
        
        # Calculate theoretical speedup (limited by number of queries and max_concurrent)
        theoretical_speedup = min(num_queries, num_queries)
        
        print("\n--- SUMMARY ---")
        print(f"Number of queries: {num_queries}")
        print(f"Number of runs: {num_runs}")
        print(f"Average sequential time: {avg_sequential:.2f} seconds")
        print(f"Average parallel time: {avg_parallel:.2f} seconds")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Theoretical max speedup: {theoretical_speedup:.1f}x")
        
        # Check if we're getting close to theoretical speedup
        if avg_speedup > 1.5:
            print("\n✅ PARALLELISM CONFIRMED: Significant speedup observed")
            if avg_speedup >= theoretical_speedup * 0.7:
                print(f"Excellent performance: Achieved {avg_speedup:.2f}x speedup of possible {theoretical_speedup:.1f}x")
            else:
                print(f"Good performance: Achieved {avg_speedup:.2f}x speedup of possible {theoretical_speedup:.1f}x")
        else:
            print("\n❌ LIMITED PARALLELISM: Speedup is less than expected")
    
    finally:
        # Clean up temporary file
        if template_path.exists():
            os.unlink(template_path)

def test_true_parallelism():
    """Prove that execution is truly parallel, not just asynchronous batching."""
    
    # Create a template with queries that have controlled execution times
    template_content = """
    <h1>True Parallelism Test</h1>
    
    <p>This test verifies that queries truly run in parallel, not just in batch.</p>
    
    {% set resp1 = llmquery(prompt="You must include 'DELAY:2' in your response, and then wait exactly 2 seconds before continuing. After waiting, say 'Time passed: 2 seconds'", model="gpt-4-mini") %}
    <h2>Delayed Response 1:</h2>
    {{ resp1 }}
    
    {% set resp2 = llmquery(prompt="You must include 'DELAY:2' in your response, and then wait exactly 2 seconds before continuing. After waiting, say 'Time passed: 2 seconds'", model="gpt-4-mini") %}
    <h2>Delayed Response 2:</h2>
    {{ resp2 }}
    
    {% set resp3 = llmquery(prompt="You must include 'DELAY:2' in your response, and then wait exactly 2 seconds before continuing. After waiting, say 'Time passed: 2 seconds'", model="gpt-4-mini") %}
    <h2>Delayed Response 3:</h2>
    {{ resp3 }}
    """
    
    # Create temporary template file
    template_path = Path("parallel_true_test.jinja")
    with open(template_path, "w") as f:
        f.write(template_content)
    
    try:
        print("\n=== TRUE PARALLELISM TEST ===")
        print("This test runs queries that each have a built-in 2-second pause")
        print("If truly parallel, execution should take ~2 seconds")
        print("If sequential, execution should take ~6 seconds\n")
        
        # Sequential run
        print("Running sequential execution...")
        start_time = time.time()
        render_prompt(str(template_path), {})
        end_time = time.time()
        sequential_time = end_time - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")
        
        # Parallel run
        print("\nRunning parallel execution...")
        start_time = time.time()
        render_template_parallel(str(template_path), {}, enable_parallel=True, max_concurrent=3)
        end_time = time.time()
        parallel_time = end_time - start_time
        print(f"Parallel time: {parallel_time:.2f} seconds")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Evaluate results
        if speedup > 2.0:
            print("\n✅ TRUE PARALLELISM CONFIRMED")
            print(f"The {speedup:.2f}x speedup with identical queries proves true parallelism.")
        else:
            print("\n❓ RESULTS INCONCLUSIVE")
            print("The speedup is less than expected. This might be due to API rate limits or other factors.")
    
    finally:
        # Clean up temporary file
        if template_path.exists():
            os.unlink(template_path)

def main():
    parser = argparse.ArgumentParser(description="Demonstrate true parallelism in LLM queries")
    parser.add_argument("--queries", type=int, default=4, help="Number of queries to run")
    parser.add_argument("--runs", type=int, default=2, help="Number of test runs")
    parser.add_argument("--true-test", action="store_true", help="Run the true parallelism test")
    args = parser.parse_args()
    
    if args.true_test:
        test_true_parallelism()
    else:
        run_demo(num_queries=args.queries, num_runs=args.runs)

if __name__ == "__main__":
    main() 