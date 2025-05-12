"""
Debug script to examine YAML formatting issues
Run with: python -m tests.debug_yaml_formatting
"""

import os
import yaml
import re
from pathlib import Path
from datetime import datetime, timezone
from jinja_prompt_chaining_system.logger import LLMLogger, ContentAwareYAMLDumper

def debug_yaml_formatting():
    """Debug YAML formatting issues with the # markdown comments."""
    
    print("\n===== DEBUG YAML FORMATTING ISSUES =====\n")
    
    # Test with a simpler yet problematic string
    test_content = """This is a test string with
# markdown in the content itself which might cause confusion
when being processed by the YAML dumper and post-processor.

It also has multiple lines that might trigger line continuation backslashes \\
if the line is particularly long like this one that could potentially exceed some internal limit.
"""
    
    # Create a simple test data structure
    test_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_field": test_content
    }
    
    # Create a test file in the current directory for inspection
    test_file = Path("debug_yaml_test.yaml")
    
    try:
        # Step 1: Dump with ContentAwareYAMLDumper
        with open(test_file, "w", encoding="utf-8") as f:
            yaml.dump(test_data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Read the raw content and print for inspection
        with open(test_file, "r", encoding="utf-8") as f:
            initial_content = f.read()
        
        print("\n=== Initial YAML dump content ===")
        print(initial_content)
        
        # Step 2: Manually post-process the file
        dummy_logger = LLMLogger()
        dummy_logger._post_process_yaml_file(test_file)
        
        # Read again and print for comparison
        with open(test_file, "r", encoding="utf-8") as f:
            processed_content = f.read()
        
        print("\n=== After post-processing ===")
        print(processed_content)
        
        # Count the # markdown markers
        initial_markers = re.findall(r'# markdown', initial_content)
        processed_markers = re.findall(r'# markdown', processed_content)
        
        print(f"\nBefore post-processing: {len(initial_markers)} markdown markers")
        print(f"After post-processing: {len(processed_markers)} markdown markers")
        
        # Check if the YAML can still be loaded
        with open(test_file, "r", encoding="utf-8") as f:
            try:
                reloaded_data = yaml.safe_load(f)
                assert reloaded_data is not None
                assert "content_field" in reloaded_data
                assert test_content == reloaded_data["content_field"]
                print("\nYAML file can still be loaded correctly")
            except yaml.YAMLError as e:
                print(f"\nError loading YAML: {e}")
                
        # Test with a large complex content to see if it causes backslash line breaks
        print("\n\n===== TEST WITH LARGE COMPLEX CONTENT =====\n")
        
        # Generate a more complex content with repeated patterns
        repeated_section = "# markdown    # markdown    # markdown    # markdown    # markdown\n" * 5
        large_content = f"""<AboutComplexityProcessing>
{repeated_section}
This is a test of very large content.
</AboutComplexityProcessing>"""
        
        complex_data = {
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
        
        # Write the complex data
        with open(test_file, "w", encoding="utf-8") as f:
            yaml.dump(complex_data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        # Read the raw content
        with open(test_file, "r", encoding="utf-8") as f:
            complex_initial = f.read()
            
        print("\n=== Complex content initial dump ===")
        print(complex_initial)
        
        # Post-process
        dummy_logger._post_process_yaml_file(test_file)
        
        # Read again
        with open(test_file, "r", encoding="utf-8") as f:
            complex_processed = f.read()
            
        print("\n=== Complex content after post-processing ===")
        print(complex_processed)
        
        # Check for line breaks with backslashes
        backslash_breaks = re.findall(r'\\$\n', complex_processed)
        
        print(f"\nFound {len(backslash_breaks)} line breaks with trailing backslashes")
        if backslash_breaks:
            print("Example of line with backslash continuation:")
            pattern = r'.*\\$\n'
            matches = re.findall(pattern, complex_processed)
            for i, match in enumerate(matches[:3]):  # Show first 3 examples
                print(f"Example {i+1}: {match.strip()}")
                
        # Count markers
        complex_initial_markers = re.findall(r'# markdown', complex_initial)
        complex_processed_markers = re.findall(r'# markdown', complex_processed)
        
        print(f"\nBefore post-processing: {len(complex_initial_markers)} markdown markers")
        print(f"After post-processing: {len(complex_processed_markers)} markdown markers")
        
        # Try multiple post-processing passes to see if markers multiply
        for i in range(3):
            dummy_logger._post_process_yaml_file(test_file)
            
        with open(test_file, "r", encoding="utf-8") as f:
            multi_processed = f.read()
            
        multi_processed_markers = re.findall(r'# markdown', multi_processed)
        print(f"After {i+1} additional passes: {len(multi_processed_markers)} markdown markers")
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

    print("\n===== DEBUG COMPLETE =====\n")

if __name__ == "__main__":
    debug_yaml_formatting() 