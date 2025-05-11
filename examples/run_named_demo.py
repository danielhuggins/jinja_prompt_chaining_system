#!/usr/bin/env python
"""
Demo script showing how to use the run naming feature with the API.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jinja_prompt_chaining_system import render_prompt

def main():
    # Get the absolute path to the examples directory
    examples_dir = Path(__file__).parent.absolute()
    
    # Define paths
    template_path = examples_dir / "named_run_demo.jinja"
    context_path = examples_dir / "named_run_demo_context.yaml"
    log_dir = examples_dir / "logs"
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Running template with named run...")
    
    # Run the template with a named run
    result = render_prompt(
        template_path=str(template_path),
        context=str(context_path),
        logdir=str(log_dir),
        name="experiment-demo"
    )
    
    print("\nResult:")
    print("-" * 40)
    print(result)
    print("-" * 40)
    
    # Show the created log directory
    print("\nCreated log directories:")
    for run_dir in sorted(os.listdir(log_dir)):
        if run_dir.startswith("run_") and "experiment-demo" in run_dir:
            print(f"  - {run_dir}")
    
    print("\nDone! Check the logs directory for the named run.")

if __name__ == "__main__":
    main() 