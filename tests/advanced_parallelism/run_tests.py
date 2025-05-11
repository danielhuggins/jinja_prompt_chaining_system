#!/usr/bin/env python
"""
Run all advanced parallelism tests to thoroughly test the parallel execution system.

This script will run the advanced parallelism tests with detailed output to help
diagnose any issues with the parallel execution system.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run all advanced parallelism tests."""
    # Get the directory containing the tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("Running Advanced Parallelism Tests")
    print("=" * 80)
    
    # Run with pytest flags for verbosity and showing output
    args = [
        "-xvs",       # x: stop on first failure, v: verbose, s: show output
        "--no-header",  # Omit header
        "--durations=0",  # Show durations for all tests
        test_dir,     # Path to test directory
    ]
    
    # Run the tests and get the return code
    result = pytest.main(args)
    
    # Print summary
    print("\n" + "=" * 80)
    if result == 0:
        print("All advanced parallelism tests passed!")
    else:
        print(f"Some tests failed with exit code {result}")
    print("=" * 80)
    
    return result

if __name__ == "__main__":
    sys.exit(main()) 