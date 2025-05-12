"""
Direct test of YAML dumping and post-processing with large content
"""

import yaml
import re
from pathlib import Path
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper, LLMLogger

def main():
    print("\n=== DIRECT YAML TEST ===\n")
    
    # Create test content with problematic patterns
    test_content = """<AboutComplexityProcessing>
    
# This is a large document with markdown headers
## That might cause confusion

It has multiple lines and some # markdown annotations within it.
It might also have some content: fields that could be mistaken for YAML keys.

The issue appears when lines are very long and the YAML dumper adds backslashes,
then the post-processor adds # markdown markers to each continued line segment.
</AboutComplexityProcessing>"""

    # Create a test data structure
    test_data = {
        "request": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": test_content
                }
            ]
        }
    }
    
    # Output file
    test_file = Path("direct_test.yaml")
    
    # Step 1: Initial dump using ContentAwareYAMLDumper
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(test_data, f, Dumper=ContentAwareYAMLDumper, 
                 default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Read the result
    with open(test_file, "r", encoding="utf-8") as f:
        initial_yaml = f.read()
    
    print("Original YAML:")
    print(initial_yaml)
    print("-" * 80)
    
    # Create an LLMLogger for post-processing
    logger = LLMLogger()
    
    # Post-process the file
    logger._post_process_yaml_file(test_file)
    
    # Read the processed file
    with open(test_file, "r", encoding="utf-8") as f:
        processed_yaml = f.read()
    
    print("Post-processed YAML:")
    print(processed_yaml)
    print("-" * 80)
    
    # Count markdown markers
    initial_markers = re.findall(r'# markdown', initial_yaml)
    processed_markers = re.findall(r'# markdown', processed_yaml)
    
    print(f"Initial markdown markers: {len(initial_markers)}")
    print(f"After post-processing: {len(processed_markers)}")
    
    # Clean up
    test_file.unlink()
    
    # Now try with a much larger content string to see if it triggers more issues
    print("\n=== TESTING WITH VERY LARGE CONTENT ===\n")
    
    # Generate a large content
    large_section = "# markdown    # markdown    # markdown    # markdown    # markdown\n" * 30
    large_content = f"""<AboutComplexityProcessing>
{large_section}
This is a document with many repeated markdown markers that might cause confusion in post-processing.
{large_section}
</AboutComplexityProcessing>"""
    
    large_data = {
        "request": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": large_content
                }
            ]
        }
    }
    
    # Dump the large data
    large_file = Path("large_test.yaml")
    try:
        with open(large_file, "w", encoding="utf-8") as f:
            yaml.dump(large_data, f, Dumper=ContentAwareYAMLDumper, 
                    default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Count line continuations with backslashes
        with open(large_file, "r", encoding="utf-8") as f:
            large_yaml = f.read()
        
        backslashes = re.findall(r'\\$', large_yaml, re.MULTILINE)
        print(f"Found {len(backslashes)} line continuations with backslashes")
        
        # Show a few examples if found
        if backslashes:
            lines_with_backslashes = re.findall(r'.*\\$', large_yaml, re.MULTILINE)
            print("\nExamples of lines with backslashes:")
            for i, line in enumerate(lines_with_backslashes[:3]):
                print(f"{i+1}: {line}")
        
        # Post-process
        logger._post_process_yaml_file(large_file)
        
        # Read again
        with open(large_file, "r", encoding="utf-8") as f:
            processed_large = f.read()
        
        # Count markdown markers
        large_initial = re.findall(r'# markdown', large_yaml)
        large_processed = re.findall(r'# markdown', processed_large)
        
        print(f"Large content initial markdown markers: {len(large_initial)}")
        print(f"Large content after post-processing: {len(large_processed)}")
        
        # Check for repeated post-processing behavior
        # Post-process multiple times to see if markers multiply
        for i in range(3):
            logger._post_process_yaml_file(large_file)
        
        with open(large_file, "r", encoding="utf-8") as f:
            multi_processed = f.read()
        
        multi_markers = re.findall(r'# markdown', multi_processed)
        print(f"After {i+1} additional passes: {len(multi_markers)} markdown markers")
        
    finally:
        if large_file.exists():
            large_file.unlink()

if __name__ == "__main__":
    main() 