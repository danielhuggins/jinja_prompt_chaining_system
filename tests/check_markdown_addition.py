"""
Tests how the post-processor adds # markdown markers
"""

import yaml
import re
from pathlib import Path
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper, LLMLogger

def main():
    print("\n=== TESTING MARKDOWN MARKER ADDITION ===\n")
    
    # Create test file
    test_file = Path("markdown_test.yaml")
    
    # Create an LLMLogger for post-processing
    logger = LLMLogger()
    
    # Test 1: Simple content with pipe style
    print("\nTEST 1: Simple content with pipe style")
    simple_content = {
        "messages": [
            {"role": "user", "content": "Simple one-line content"}
        ]
    }
    
    with open(test_file, "w") as f:
        yaml.dump(simple_content, f, Dumper=ContentAwareYAMLDumper)
    
    # Print original
    with open(test_file, "r") as f:
        print("Original:")
        print(f.read())
    
    # Post-process
    logger._post_process_yaml_file(test_file)
    
    # Print processed
    with open(test_file, "r") as f:
        processed = f.read()
        print("After post-processing:")
        print(processed)
    
    # Count markdown markers
    markdown_count = processed.count("# markdown")
    print(f"Markdown markers added: {markdown_count}")
    
    # Test 2: Multiline content
    print("\nTEST 2: Multiline content with pipe style")
    multiline_content = {
        "messages": [
            {"role": "user", "content": "Line 1\nLine 2\nLine 3"}
        ]
    }
    
    with open(test_file, "w") as f:
        yaml.dump(multiline_content, f, Dumper=ContentAwareYAMLDumper)
    
    # Print original
    with open(test_file, "r") as f:
        print("Original:")
        print(f.read())
    
    # Post-process
    logger._post_process_yaml_file(test_file)
    
    # Print processed
    with open(test_file, "r") as f:
        processed = f.read()
        print("After post-processing:")
        print(processed)
    
    # Count markdown markers
    markdown_count = processed.count("# markdown")
    print(f"Markdown markers added: {markdown_count}")
    
    # Test 3: Very long content that will trigger line continuation
    print("\nTEST 3: Very long content with line continuation")
    long_line = "This is a very long line " + "that will get broken with continuation markers " * 5
    long_content = {
        "messages": [
            {"role": "user", "content": long_line}
        ]
    }
    
    with open(test_file, "w") as f:
        yaml.dump(long_content, f, Dumper=ContentAwareYAMLDumper)
    
    # Print original
    with open(test_file, "r") as f:
        original = f.read()
        print("Original:")
        print(original)
        
    # Check for backslashes in original
    has_backslash = "\\" in original
    print(f"Has backslash continuation: {has_backslash}")
    
    # Post-process
    logger._post_process_yaml_file(test_file)
    
    # Print processed
    with open(test_file, "r") as f:
        processed = f.read()
        print("After post-processing:")
        print(processed)
    
    # Count markdown markers
    markdown_count = processed.count("# markdown")
    print(f"Markdown markers added: {markdown_count}")
    
    # Clean up
    test_file.unlink()

if __name__ == "__main__":
    main() 