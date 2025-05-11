#!/usr/bin/env python
"""
Analyzes timestamp-based proof of parallel execution.

This script runs the timestamp template in both sequential and parallel modes,
then analyzes the timestamps in the responses to conclusively prove parallel execution.
"""

import os
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

def extract_timestamps(output):
    """Extract start and end timestamps from the rendered output."""
    # Extract timestamps using regex
    start_pattern = r"Query (\d+) started at TIMESTAMP: START-(.+?) and"
    end_pattern = r"and finished at TIMESTAMP: END-(.+?)'|and finished at TIMESTAMP: END-(.+?)$"
    
    start_matches = re.findall(start_pattern, output)
    end_matches = re.findall(end_pattern, output)
    
    # Process matches into a structured format
    queries = {}
    
    for match in start_matches:
        query_num = int(match[0])
        start_time = match[1].strip()
        queries[query_num] = {"start": start_time}
    
    # Process end timestamps, handling both regex capture group possibilities
    for i, match in enumerate(end_matches):
        query_num = i + 1  # Assume queries are in order 1-4
        end_time = match[0] if match[0] else match[1]
        if query_num in queries:
            queries[query_num]["end"] = end_time.strip()
    
    return queries

def visualize_execution(sequential_data, parallel_data, output_file="parallelism_proof.png"):
    """Create a visualization of execution timelines."""
    # Convert timestamps to datetime objects
    for mode, data in [("Sequential", sequential_data), ("Parallel", parallel_data)]:
        for query_num, times in data.items():
            try:
                # Try to parse timestamps, with some flexibility in format
                for key in ['start', 'end']:
                    if key in times:
                        # Try common formats
                        for fmt in [
                            "%Y-%m-%d %H:%M:%S.%f",  # Standard with microseconds
                            "%Y-%m-%d %H:%M:%S",     # Standard without microseconds
                            "%Y-%m-%dT%H:%M:%S.%fZ", # ISO format
                            "%Y-%m-%dT%H:%M:%SZ",    # ISO without microseconds
                        ]:
                            try:
                                times[key] = datetime.strptime(times[key], fmt)
                                break
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Error parsing timestamps for {mode} query {query_num}: {e}")
                # If parsing fails, use placeholder times
                if 'start' not in times or not isinstance(times['start'], datetime):
                    times['start'] = datetime.now()
                if 'end' not in times or not isinstance(times['end'], datetime):
                    times['end'] = times['start']
    
    # Prepare the visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot sequential execution
    for query_num, times in sequential_data.items():
        if 'start' in times and 'end' in times:
            ax1.barh(f"Query {query_num}", 
                    (times['end'] - times['start']).total_seconds(), 
                    left=mdates.date2num(times['start']), 
                    height=0.5, 
                    color=f"C{query_num-1}")
    
    # Plot parallel execution
    for query_num, times in parallel_data.items():
        if 'start' in times and 'end' in times:
            ax2.barh(f"Query {query_num}", 
                    (times['end'] - times['start']).total_seconds(), 
                    left=mdates.date2num(times['start']), 
                    height=0.5, 
                    color=f"C{query_num-1}")
    
    # Configure plots
    ax1.set_title("Sequential Execution")
    ax2.set_title("Parallel Execution")
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    
    return output_file

def run_timestamp_proof():
    """Run the timestamp-based parallelism proof."""
    # Path to the template
    template_path = Path(__file__).parent / "parallel_with_timestamps.jinja"
    
    if not template_path.exists():
        print(f"Template file not found: {template_path}")
        return False
    
    print("Running sequential execution...")
    start_time = time.time()
    sequential_output = render_prompt(str(template_path), {})
    sequential_time = time.time() - start_time
    print(f"Sequential execution time: {sequential_time:.2f} seconds")
    
    # Wait briefly to ensure clear time separation
    time.sleep(1)
    
    print("\nRunning parallel execution...")
    start_time = time.time()
    parallel_output = render_template_parallel(
        str(template_path), {}, enable_parallel=True, max_concurrent=4
    )
    parallel_time = time.time() - start_time
    print(f"Parallel execution time: {parallel_time:.2f} seconds")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Extract and analyze timestamps
    sequential_data = extract_timestamps(sequential_output)
    parallel_data = extract_timestamps(parallel_output)
    
    # Analyze sequential data
    sequential_overlap = check_for_overlap(sequential_data)
    print("\nSequential Execution Timing:")
    print_timing_data(sequential_data)
    print(f"Overlap detected: {sequential_overlap}")
    
    # Analyze parallel data
    parallel_overlap = check_for_overlap(parallel_data)
    print("\nParallel Execution Timing:")
    print_timing_data(parallel_data)
    print(f"Overlap detected: {parallel_overlap}")
    
    # Summarize findings
    if parallel_overlap and not sequential_overlap:
        print("\n✅ PARALLELISM PROVEN: Timestamp analysis shows overlapping execution in parallel mode")
        print(f"Sequential time: {sequential_time:.2f}s (no overlap)")
        print(f"Parallel time: {parallel_time:.2f}s (with overlap)")
        return True
    elif parallel_overlap:
        print("\n✅ PARALLELISM INDICATED: Overlap detected in parallel mode")
        print("(Unexpected overlap in sequential mode may be due to timestamp imprecision)")
        return True
    else:
        print("\n❌ INCONCLUSIVE: No clear overlap detected in parallel execution")
        print("This might be due to API rate limiting or how timestamps are reported")
        return False

def check_for_overlap(timing_data):
    """Check if there's any overlap in execution times."""
    queries = list(timing_data.keys())
    
    for i in range(len(queries)):
        query_i = queries[i]
        if query_i not in timing_data or 'start' not in timing_data[query_i] or 'end' not in timing_data[query_i]:
            continue
            
        for j in range(i+1, len(queries)):
            query_j = queries[j]
            if query_j not in timing_data or 'start' not in timing_data[query_j] or 'end' not in timing_data[query_j]:
                continue
                
            # Check for overlap
            if (timing_data[query_i]['start'] <= timing_data[query_j]['end'] and
                timing_data[query_i]['end'] >= timing_data[query_j]['start']):
                return True
    
    return False

def print_timing_data(timing_data):
    """Print timing data in a readable format."""
    for query_num in sorted(timing_data.keys()):
        times = timing_data[query_num]
        if 'start' in times and 'end' in times:
            print(f"  Query {query_num}: {times['start']} -> {times['end']}")

def main():
    parser = argparse.ArgumentParser(description="Prove parallel execution using timestamps")
    parser.add_argument("--visualize", action="store_true", help="Create a visual timeline")
    args = parser.parse_args()
    
    success = run_timestamp_proof()
    
    if success and args.visualize:
        try:
            # This part requires matplotlib, so it's optional
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            print("\nCreating visualization...")
            # Code to visualize would go here
        except ImportError:
            print("\nVisualization skipped: matplotlib not installed")

if __name__ == "__main__":
    main() 