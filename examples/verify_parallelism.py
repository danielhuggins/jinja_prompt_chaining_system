#!/usr/bin/env python
"""
A simple proof that the Jinja Prompt Chaining System executes LLM queries in parallel.

This script demonstrates true parallelism by comparing execution times
for sequential vs. parallel execution of multiple independent LLM queries.
"""

import os
import time
import argparse
from pathlib import Path

from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

def create_template(num_queries=4):
    """Create a test template with multiple independent queries."""
    content = "<h1>Parallelism Test</h1>\n\n"
    
    for i in range(1, num_queries + 1):
        content += f"""
        <h2>Query {i}</h2>
        {{% set response{i} = llmquery(prompt="Generate a brief response with exactly 50 words about {['technology', 'nature', 'space', 'science', 'art', 'music'][i % 6]}. Include 'Query {i}' in your response.", model="gpt-3.5-turbo") %}}
        {{{{ response{i} }}}}
        
        """
    
    return content

def run_test(num_queries=4, max_concurrent=4):
    """Run the test to prove parallelism."""
    print(f"Creating test template with {num_queries} independent queries...")
    template_content = create_template(num_queries)
    
    # Create temporary directory and file
    temp_dir = Path.cwd()
    template_name = "parallelism_test_temp.jinja"
    template_path = temp_dir / template_name
    
    with open(template_path, "w") as f:
        f.write(template_content)
    
    try:
        # Run sequential execution
        print("\nRunning sequential execution...")
        start_time = time.time()
        render_prompt(str(template_path), {})
        sequential_time = time.time() - start_time
        print(f"Sequential execution time: {sequential_time:.2f} seconds")
        
        # Run parallel execution
        print("\nRunning parallel execution...")
        start_time = time.time()
        result = render_template_parallel(
            str(template_path), 
            {}, 
            enable_parallel=True, 
            max_concurrent=max_concurrent
        )
        parallel_time = time.time() - start_time
        print(f"Parallel execution time: {parallel_time:.2f} seconds")
        
        # Save results to a file
        result_path = temp_dir / "parallelism_test_results.html"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Results saved to {result_path}")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time
        
        # Theoretical maximum speedup is min(num_queries, max_concurrent)
        theoretical_max = min(num_queries, max_concurrent)
        efficiency = (speedup / theoretical_max) * 100
        
        # Print results
        print("\n----- RESULTS -----")
        print(f"Sequential time: {sequential_time:.2f} seconds")
        print(f"Parallel time:   {parallel_time:.2f} seconds")
        print(f"Speedup:         {speedup:.2f}x")
        print(f"Efficiency:      {efficiency:.1f}% of theoretical maximum ({theoretical_max}x)")
        
        # Evaluate the result
        if speedup > 1.5:
            print("\n✅ PARALLELISM CONFIRMED")
            print(f"The {speedup:.2f}x speedup proves that queries are running in parallel.")
            
            if efficiency >= 70:
                print("Excellent parallel efficiency!")
            elif efficiency >= 50:
                print("Good parallel efficiency.")
            else:
                print("Modest parallel efficiency. Some overhead may be present.")
        else:
            print("\n❓ RESULTS INCONCLUSIVE")
            print("The speedup is less than expected. This might be due to:")
            print("- API rate limits")
            print("- Network latency variations")
            print("- Limited number of queries")
            print("Try running with more queries for a clearer result.")
    
    finally:
        # Clean up temporary file
        if template_path.exists():
            os.unlink(template_path)

def main():
    parser = argparse.ArgumentParser(description="Prove LLM query parallelism")
    parser.add_argument("--queries", type=int, default=4, help="Number of queries to test")
    parser.add_argument("--concurrent", type=int, default=4, help="Maximum concurrent queries")
    args = parser.parse_args()
    
    run_test(num_queries=args.queries, max_concurrent=args.concurrent)

if __name__ == "__main__":
    main() 