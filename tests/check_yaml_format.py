"""
Check YAML formatting issues by examining raw bytes
"""

import yaml
from pathlib import Path
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper, LLMLogger

def main():
    print("\n=== CHECKING YAML FORMATTING ISSUES (RAW BYTES) ===\n")
    
    # Create simple test content WITHOUT any # markdown text
    clean_content = """This is clean content
with multiple lines
but NO markdown symbols at all
to avoid any confusion with formatting markers."""
    
    # Create test data
    test_data = {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": clean_content
                }
            ]
        }
    }
    
    # Output file
    test_file = Path("format_check.yaml")
    
    # Step 1: Initial dump
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(test_data, f, Dumper=ContentAwareYAMLDumper, 
                 default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Read the result as bytes to see exact content
    with open(test_file, "rb") as f:
        initial_bytes = f.read()
    
    print("Original YAML bytes:")
    print(initial_bytes)
    print("-" * 50)
    
    # Post-process
    logger = LLMLogger()
    logger._post_process_yaml_file(test_file)
    
    # Read the processed result as bytes
    with open(test_file, "rb") as f:
        processed_bytes = f.read()
    
    print("Post-processed YAML bytes:")
    print(processed_bytes)
    print("-" * 50)
    
    # Check if "# markdown" was added despite not being in original content
    original_contains_marker = b"# markdown" in initial_bytes
    processed_contains_marker = b"# markdown" in processed_bytes
    
    print(f"Original contains '# markdown': {original_contains_marker}")
    print(f"Processed contains '# markdown': {processed_contains_marker}")
    
    # Now try with a long line that would trigger line continuation
    long_line = "This is a very long line " + "that will likely cause the YAML dumper to add continuation markers " * 5 + "at the end of each segment."
    
    long_data = {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": long_line
                }
            ]
        }
    }
    
    # Dump the long data
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(long_data, f, Dumper=ContentAwareYAMLDumper, 
                 default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Read as bytes
    with open(test_file, "rb") as f:
        long_initial_bytes = f.read()
    
    print("\nLong content initial bytes:")
    print(long_initial_bytes)
    
    # Check for continuation markers
    has_continuation = b"\\" in long_initial_bytes
    print(f"Has continuation markers: {has_continuation}")
    
    # Post-process
    logger._post_process_yaml_file(test_file)
    
    # Read processed as bytes
    with open(test_file, "rb") as f:
        long_processed_bytes = f.read()
    
    print("\nLong content processed bytes:")
    print(long_processed_bytes)
    
    # Check markdown in long content
    long_processed_contains_marker = b"# markdown" in long_processed_bytes
    print(f"Long processed contains '# markdown': {long_processed_contains_marker}")
    
    # Check number of markdown markers
    marker_count = long_processed_bytes.count(b"# markdown")
    print(f"Number of '# markdown' markers: {marker_count}")
    
    # Clean up
    test_file.unlink()

if __name__ == "__main__":
    main() 