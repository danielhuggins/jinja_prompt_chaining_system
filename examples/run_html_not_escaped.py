#!/usr/bin/env python
"""
Demo script for showing that HTML is not escaped in Jinja templates with our system.
"""

import os
import yaml
import argparse
from pathlib import Path

from jinja_prompt_chaining_system.parallel_integration import render_template_parallel
from jinja_prompt_chaining_system.api import render_prompt

def load_yaml_context(path):
    """Load a YAML context file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Demo HTML non-escaping in templates")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel LLM query execution")
    parser.add_argument("--output", type=str, default="html_not_escaped_output.html", help="Output file path")
    args = parser.parse_args()
    
    # Get the path to the example files
    examples_dir = Path(__file__).parent
    template_path = examples_dir / "html_not_escaped.jinja"
    context_path = examples_dir / "html_not_escaped_context.yaml"
    
    # Load the context
    context = load_yaml_context(context_path)
    
    # Run with or without parallel execution
    if args.parallel:
        print(f"Rendering template with parallel execution...")
        result = render_template_parallel(
            str(template_path), 
            context, 
            enable_parallel=True,
            max_concurrent=2
        )
    else:
        print(f"Rendering template with sequential execution...")
        result = render_prompt(str(template_path), context)
    
    # Write the output
    output_path = examples_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main() 