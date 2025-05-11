#!/usr/bin/env python
"""
Demo script showcasing parallel LLM query execution.

This script demonstrates how multiple independent LLM queries
can be executed in parallel to improve performance.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jinja_prompt_chaining_system.parallel_integration import render_template_parallel

def create_demo_template():
    """Create a demo template with multiple independent queries."""
    template_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(template_dir, "parallel_demo.jinja")
    
    # Write a template with 5 independent queries
    with open(template_path, "w") as f:
        f.write("""
        <h1>Parallel LLM Query Execution Demo</h1>
        
        <h2>Independent Queries</h2>
        These queries will run in parallel:
        
        {% set topic1 = llmquery(prompt="Generate a short recipe name", model="gpt-4o-mini", temperature=0.8) %}
        {% set topic2 = llmquery(prompt="Generate a name for a technology startup", model="gpt-4o-mini", temperature=0.8) %}
        {% set topic3 = llmquery(prompt="Generate a creative book title", model="gpt-4o-mini", temperature=0.8) %}
        {% set topic4 = llmquery(prompt="Generate a name for a pet robot", model="gpt-4o-mini", temperature=0.8) %}
        
        <ul>
            <li>Recipe: {{ topic1 }}</li>
            <li>Startup: {{ topic2 }}</li>
            <li>Book: {{ topic3 }}</li>
            <li>Robot: {{ topic4 }}</li>
        </ul>
        
        <h2>Dependent Queries</h2>
        These queries depend on previous results:
        
        {% set details1 = llmquery(prompt="Write 2 sentences about a recipe called: " + topic1, model="gpt-4o-mini") %}
        {% set details2 = llmquery(prompt="Write 2 sentences about a startup called: " + topic2, model="gpt-4o-mini") %}
        
        <ul>
            <li>About the recipe: {{ details1 }}</li>
            <li>About the startup: {{ details2 }}</li>
        </ul>
        """)
    
    return template_path

def run_demo():
    """Run the parallel execution demo."""
    # Create the demo template
    template_path = create_demo_template()
    
    # Context data
    context = {}
    
    print("Running demo in SEQUENTIAL mode...")
    start_time = time.time()
    result_sequential = render_template_parallel(template_path, context, enable_parallel=False)
    sequential_time = time.time() - start_time
    
    print(f"Sequential mode took {sequential_time:.2f} seconds\n")
    
    print("Running demo in PARALLEL mode...")
    start_time = time.time()
    result_parallel = render_template_parallel(template_path, context, enable_parallel=True, max_concurrent=3)
    parallel_time = time.time() - start_time
    
    print(f"Parallel mode took {parallel_time:.2f} seconds\n")
    
    print(f"Speedup: {sequential_time / parallel_time:.2f}x\n")
    
    # Save results
    with open("examples/sequential_result.html", "w") as f:
        f.write(result_sequential)
    
    with open("examples/parallel_result.html", "w") as f:
        f.write(result_parallel)
    
    print("Results saved to examples/sequential_result.html and examples/parallel_result.html")
    
    # Print a more readable version of the results
    print("\nResults:")
    
    # Extract key information using basic string operations
    import re
    
    def extract_results(html):
        recipes = re.findall(r"<li>Recipe: (.*?)</li>", html)
        startups = re.findall(r"<li>Startup: (.*?)</li>", html)
        books = re.findall(r"<li>Book: (.*?)</li>", html)
        robots = re.findall(r"<li>Robot: (.*?)</li>", html)
        
        recipe_details = re.findall(r"<li>About the recipe: (.*?)</li>", html)
        startup_details = re.findall(r"<li>About the startup: (.*?)</li>", html)
        
        return {
            "Recipe": recipes[0] if recipes else "",
            "Startup": startups[0] if startups else "",
            "Book": books[0] if books else "",
            "Robot": robots[0] if robots else "",
            "Recipe Details": recipe_details[0] if recipe_details else "",
            "Startup Details": startup_details[0] if startup_details else ""
        }
    
    parallel_results = extract_results(result_parallel)
    
    for key, value in parallel_results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    run_demo() 