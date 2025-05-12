import yaml
import re
from pathlib import Path
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper, LLMLogger

# Test data with different types of content
test_data = {
    "messages": [
        {"role": "user", "content": "Single line content that should use pipe style"},
        {"role": "assistant", "content": "Very long content " + "that would trigger line breaks " * 5},
        {"role": "system", "content": "Multiline\ncontent\nwith\nnewlines"}
    ]
}

def test_yaml_fix():
    print("\n=== Testing ContentAwareYAMLDumper Fix ===\n")
    
    # Output file for the test
    test_file = Path("test_yaml_output.yaml")
    
    # Write the test data using the modified ContentAwareYAMLDumper
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(test_data, f, Dumper=ContentAwareYAMLDumper, 
                 default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Read the generated YAML
    with open(test_file, "r", encoding="utf-8") as f:
        yaml_content = f.read()
    
    # Print the output
    print("Generated YAML:")
    print("-" * 50)
    print(yaml_content)
    print("-" * 50)
    
    # Analyze the formatting
    pipe_count = yaml_content.count("content: |")
    quoted_count = (yaml_content.count("content: '") + 
                   yaml_content.count('content: "'))
    backslash_count = len(re.findall(r'\\$', yaml_content, re.MULTILINE))
    
    # Report results
    print(f"\nPipe-style fields: {pipe_count}")
    print(f"Quoted-style fields: {quoted_count}")
    print(f"Line breaks with backslashes: {backslash_count}")
    
    # Check if fix was successful
    if pipe_count == 3 and quoted_count == 0 and backslash_count == 0:
        print("\n✅ YAML FIX SUCCESSFUL: All content fields use pipe style")
    else:
        print("\n❌ YAML FIX FAILED: Some content fields are not using pipe style")
    
    # Test post-processing of the YAML file
    print("\n=== Testing Post-processing ===\n")
    
    # Create a logger for testing the post-processing
    logger = LLMLogger()
    logger._post_process_yaml_file(test_file)
    
    # Read the post-processed file
    with open(test_file, "r", encoding="utf-8") as f:
        processed_yaml = f.read()
    
    # Print the processed YAML
    print("After post-processing:")
    print("-" * 50)
    print(processed_yaml)
    print("-" * 50)
    
    # Count # markdown markers
    marker_count = processed_yaml.count("# markdown")
    print(f"\nFound {marker_count} markdown markers")
    
    # Check for expected number of markers (one per content field)
    expected_markers = 3
    if marker_count == expected_markers:
        print(f"✅ POST-PROCESSING SUCCESSFUL: Found expected {expected_markers} markdown markers")
    else:
        print(f"❌ POST-PROCESSING ISSUE: Found {marker_count} markers, expected {expected_markers}")
    
    # Clean up
    test_file.unlink()

if __name__ == "__main__":
    test_yaml_fix() 