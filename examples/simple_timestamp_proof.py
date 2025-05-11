#!/usr/bin/env python
"""
Simplified timestamp-based proof of parallel execution.

This script runs a template with multiple LLM queries that each report timestamps,
then analyzes those timestamps to prove that queries run in parallel.
"""

import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta

from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

def extract_timestamps(output):
    """Extract start and end timestamps from the output."""
    pattern = r"Query (\d+) started at TIMESTAMP: START-([^]]+) and finished at TIMESTAMP: END-([^]]+)"
    matches = re.findall(pattern, output, re.DOTALL)
    
    results = {}
    for match in matches:
        query_num = int(match[0])
        start_time = match[1].strip()
        end_time = match[2].strip()
        results[query_num] = {
            "start": start_time,
            "end": end_time
        }
    
    return results

def analyze_parallelism(timestamps):
    """Analyze the timestamps to check for parallel execution."""
    # Check for any overlaps between queries
    queries = list(timestamps.keys())
    overlaps = []
    
    for i in range(len(queries)):
        query_i = queries[i]
        for j in range(i+1, len(queries)):
            query_j = queries[j]
            
            # Get start and end times
            start_i = timestamps[query_i]["start"]
            end_i = timestamps[query_i]["end"]
            start_j = timestamps[query_j]["start"]
            end_j = timestamps[query_j]["end"]
            
            # Check for overlap (any overlap in execution time)
            if (start_i <= end_j and end_i >= start_j):
                overlaps.append((query_i, query_j))
    
    return overlaps

def format_timestamps(timestamps):
    """Format the timestamps for display."""
    result = []
    for query_num in sorted(timestamps.keys()):
        times = timestamps[query_num]
        result.append(f"Query {query_num}: START={times['start']} -> END={times['end']}")
    
    return "\n".join(result)

def run_timestamp_proof():
    """Run the proof with timestamps."""
    # Create the template file
    template_content = """
    <h1>Parallel Execution Timestamp Proof</h1>
    
    <p>This test demonstrates that LLM queries run in parallel by comparing timestamps.</p>
    
    {% set query1 = llmquery(prompt="Please respond with exactly: 'Query 1 started at TIMESTAMP: START-[current UTC time] and finished at TIMESTAMP: END-[current UTC time]'", model="gpt-3.5-turbo") %}
    <h2>Query 1 Timestamps:</h2>
    {{ query1 }}
    
    {% set query2 = llmquery(prompt="Please respond with exactly: 'Query 2 started at TIMESTAMP: START-[current UTC time] and finished at TIMESTAMP: END-[current UTC time]'", model="gpt-3.5-turbo") %}
    <h2>Query 2 Timestamps:</h2>
    {{ query2 }}
    
    {% set query3 = llmquery(prompt="Please respond with exactly: 'Query 3 started at TIMESTAMP: START-[current UTC time] and finished at TIMESTAMP: END-[current UTC time]'", model="gpt-3.5-turbo") %}
    <h2>Query 3 Timestamps:</h2>
    {{ query3 }}
    
    {% set query4 = llmquery(prompt="Please respond with exactly: 'Query 4 started at TIMESTAMP: START-[current UTC time] and finished at TIMESTAMP: END-[current UTC time]'", model="gpt-3.5-turbo") %}
    <h2>Query 4 Timestamps:</h2>
    {{ query4 }}
    """
    
    # Create temporary template file
    template_path = Path.cwd() / "examples" / "timestamp_proof_temp.jinja"
    with open(template_path, "w") as f:
        f.write(template_content)
    
    try:
        print("\n=== TIMESTAMP-BASED PARALLEL EXECUTION PROOF ===")
        print("This test uses timestamps from LLM responses to detect parallelism")
        
        # Run sequential execution first
        print("\nRunning sequential execution...")
        start_time = time.time()
        sequential_output = render_prompt(str(template_path), {})
        sequential_time = time.time() - start_time
        print(f"Sequential execution time: {sequential_time:.2f} seconds")
        
        # Extract and analyze sequential timestamps
        sequential_timestamps = extract_timestamps(sequential_output)
        sequential_overlaps = analyze_parallelism(sequential_timestamps)
        
        print("\nSequential Execution Timestamps:")
        print(format_timestamps(sequential_timestamps))
        print(f"Overlaps detected: {len(sequential_overlaps)}")
        if sequential_overlaps:
            print(f"Overlapping queries: {sequential_overlaps}")
        
        # Wait briefly between tests
        time.sleep(2)
        
        # Run parallel execution
        print("\nRunning parallel execution...")
        start_time = time.time()
        parallel_output = render_template_parallel(
            str(template_path), {}, enable_parallel=True, max_concurrent=4
        )
        parallel_time = time.time() - start_time
        print(f"Parallel execution time: {parallel_time:.2f} seconds")
        
        # Extract and analyze parallel timestamps
        parallel_timestamps = extract_timestamps(parallel_output)
        parallel_overlaps = analyze_parallelism(parallel_timestamps)
        
        print("\nParallel Execution Timestamps:")
        print(format_timestamps(parallel_timestamps))
        print(f"Overlaps detected: {len(parallel_overlaps)}")
        if parallel_overlaps:
            print(f"Overlapping queries: {parallel_overlaps}")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time
        print(f"\nOverall speedup: {speedup:.2f}x")
        
        # Evaluate results
        if len(parallel_overlaps) > 0 and len(sequential_overlaps) == 0:
            print("\n✅ PARALLELISM CONFIRMED")
            print(f"Found {len(parallel_overlaps)} overlapping query executions in parallel mode")
            print("No overlaps in sequential mode, as expected")
        elif len(parallel_overlaps) > len(sequential_overlaps):
            print("\n✅ PARALLELISM INDICATED")
            print(f"More overlaps in parallel mode ({len(parallel_overlaps)}) than sequential mode ({len(sequential_overlaps)})")
        elif speedup > 1.5:
            print("\n✅ PARALLELISM CONFIRMED BY TIMING")
            print(f"Achieved {speedup:.2f}x speedup, which proves parallel execution")
        else:
            print("\n❓ RESULTS INCONCLUSIVE")
            print("No conclusive evidence of parallel execution from timestamps")
            print("This might be due to how timestamps are reported by the LLM")
        
        # Save results
        with open("parallel_proof_results.html", "w") as f:
            f.write("<html><body>\n")
            f.write("<h1>Parallel Execution Proof Results</h1>\n")
            f.write("<h2>Sequential Execution</h2>\n")
            f.write(f"<pre>{sequential_output}</pre>\n")
            f.write("<h2>Parallel Execution</h2>\n")
            f.write(f"<pre>{parallel_output}</pre>\n")
            f.write("</body></html>")
        
        print("\nDetailed results saved to parallel_proof_results.html")
        
    finally:
        # Clean up temporary file
        if template_path.exists():
            os.unlink(template_path)

if __name__ == "__main__":
    run_timestamp_proof() 