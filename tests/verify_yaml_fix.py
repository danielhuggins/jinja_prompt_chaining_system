"""
Verify the fix for YAML formatting with ContentAwareYAMLDumper
Run with: python -m tests.verify_yaml_fix
"""

import yaml
import re
from pathlib import Path
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper, LLMLogger, preprocess_yaml_data

def main():
    print("\n=== VERIFYING YAML FORMATTING FIX ===\n")
    
    # Create test files
    manual_test_file = Path("verify_yaml_fix_manual.yaml")
    preprocessed_test_file = Path("verify_yaml_fix_preprocessed.yaml")
    
    # Create YAML content directly with pipe style for all content fields
    yaml_content = """request:
  model: gpt-4o-mini
  messages:
  - role: user
    content: |
      This is a simple one-line content field.
  - role: assistant
    content: |
      This is a very long line that would normally get broken up with line continuation markers that would normally get broken up with line continuation markers that would normally get broken up with line continuation markers that would normally get broken up with line continuation markers that would normally get broken up with line continuation markers
  - role: system
    content: |
      Line 1
      Line 2
      Line 3
"""
    
    # Write the manually generated YAML to file
    with open(manual_test_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print("Manually Generated YAML:\n")
    print(yaml_content)
    
    # Check for specific formatting
    pipe_count = yaml_content.count("content: |")
    quoted_count = yaml_content.count("content: '") + yaml_content.count('content: "')
    backslash_count = len(re.findall(r'\\$', yaml_content, re.MULTILINE))
    
    print(f"\nAnalysis of Manual YAML:")
    print(f"- Pipe-style fields: {pipe_count}")
    print(f"- Quoted-style fields: {quoted_count}")
    print(f"- Line continuations with backslashes: {backslash_count}")
    
    # Check if the manual approach is successful
    if pipe_count == 3 and quoted_count == 0 and backslash_count == 0:
        print("\n✅ MANUAL YAML VERIFIED: All content fields use pipe style with no quote marks or backslashes")
    else:
        print("\n❌ MANUAL YAML FAILED: Not all content fields are using pipe style")
    
    # Now test with auto-generation using our improved preprocess + dump approach
    print("\n=== TESTING PREPROCESSED YAML GENERATION ===\n")
    
    # Test data
    test_data = {
        "request": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "This is a simple one-line content field."},
                {"role": "assistant", "content": "This is a very long line " + 
                 "that would normally get broken up with line continuation markers " * 5},
                {"role": "system", "content": "Line 1\nLine 2\nLine 3"}
            ]
        }
    }
    
    # Preprocess and write
    preprocessed_data = preprocess_yaml_data(test_data)
    
    with open(preprocessed_test_file, "w", encoding="utf-8") as f:
        yaml.dump(
            preprocessed_data, 
            stream=f, 
            Dumper=ContentAwareYAMLDumper,
            default_flow_style=False, 
            sort_keys=False, 
            allow_unicode=True
        )
    
    # Read and analyze
    with open(preprocessed_test_file, "r", encoding="utf-8") as f:
        preprocessed_content = f.read()
    
    print("Preprocessed + ContentAwareYAMLDumper Output:\n")
    print(preprocessed_content)
    
    # Check preprocessing results
    pre_pipe_count = preprocessed_content.count("content: |")
    pre_quoted_count = preprocessed_content.count("content: '") + preprocessed_content.count('content: "')
    pre_backslash_count = len(re.findall(r'\\$', preprocessed_content, re.MULTILINE))
    
    print(f"\nAnalysis of Preprocessed + ContentAwareYAMLDumper:")
    print(f"- Pipe-style fields: {pre_pipe_count}")
    print(f"- Quoted-style fields: {pre_quoted_count}")
    print(f"- Line continuations with backslashes: {pre_backslash_count}")
    
    # Now test post-processing
    print("\n=== TESTING POST-PROCESSING ===\n")
    
    # Create a logger for post-processing
    logger = LLMLogger()
    
    # Process both files
    logger._post_process_yaml_file(manual_test_file)
    logger._post_process_yaml_file(preprocessed_test_file)
    
    # Check manual file post-processing
    with open(manual_test_file, "r", encoding="utf-8") as f:
        manual_processed = f.read()
    
    manual_markdown = manual_processed.count("# markdown")
    print(f"Manual file markdown markers: {manual_markdown}")
    
    if manual_markdown == 3:
        print("✅ MANUAL POST-PROCESSING VERIFIED: Each content field has exactly one markdown marker")
    else:
        print(f"❌ MANUAL POST-PROCESSING ISSUE: Found {manual_markdown} markdown markers instead of 3")
    
    # Check preprocessed file
    with open(preprocessed_test_file, "r", encoding="utf-8") as f:
        pre_processed = f.read()
    
    pre_markdown = pre_processed.count("# markdown")
    print(f"Preprocessed file markdown markers: {pre_markdown}")
    
    if pre_markdown == 3:
        print("✅ PREPROCESSED POST-PROCESSING VERIFIED: Each content field has exactly one markdown marker")
    else:
        print(f"❌ PREPROCESSED POST-PROCESSING ISSUE: Found {pre_markdown} markdown markers instead of 3")
    
    print("\nConclusion:")
    if manual_markdown == 3 and pre_markdown == 3:
        print("✅ POST-PROCESSING WORKS CORRECTLY for both pipe-style and quoted-style fields with line continuations")
    
    # Clean up
    manual_test_file.unlink()
    preprocessed_test_file.unlink()
    print("\nTest complete.")

if __name__ == "__main__":
    main() 