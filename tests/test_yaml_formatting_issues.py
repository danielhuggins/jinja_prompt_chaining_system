import os
import yaml
import pytest
import re
from pathlib import Path
from datetime import datetime, timezone
from jinja_prompt_chaining_system.logger import LLMLogger, ContentAwareYAMLDumper, preprocess_yaml_data

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"

@pytest.fixture
def logger(log_dir):
    return LLMLogger(str(log_dir))

def test_huge_formatted_content_logging(logger, log_dir):
    """Test YAML formatting with very large content field."""
    from jinja_prompt_chaining_system.logger import preprocess_yaml_data
    
    # Create a test template name
    template_name = "test_huge_content"
    
    # Create a large request
    request = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": """<AboutComplexityProcessing>
I need to create a short video about how complexity in texts is measured. Can you help?

Passage Complexity Analysis:
passageId: 1234567890
words: 234
sentences: 12
avgWordLength: 4.23
avgWordsPerSentence: 19.5
letterCount: 987
numericCount: 32
punctuationCount: 41
complexityLevel: 1.0
</AboutComplexityProcessing>"""
            }
        ]
    }
    
    # Start a streaming request
    log_path = logger.log_request(template_name, request)
    
    # Read the initial file content
    with open(log_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    # Count markdown markers
    markdown_markers = re.findall(r'# markdown', raw_content)
    print(f"Found {len(markdown_markers)} markdown markers in the output")
    
    # Check for continued content with backslashes
    backslash_breaks = re.findall(r'\\$\n', raw_content)
    print(f"Found {len(backslash_breaks)} line breaks with trailing backslashes")
    
    # Create a large response content
    response_content = """```json
{
  "passageId": "1234567890",
  "words": 234,
  "sentences": 12,
  "avgWordLength": 4.23,
  "avgWordsPerSentence": 19.5,
  "letterCount": 987,
  "numericCount": 32,
  "punctuationCount": 41,
  "complexityLevel": 1.0
}
```
"""

    # Update with some chunks before sending the completion
    logger.update_response(template_name, response_content)
    
    # Create a response with this content
    completion_data = {
        "id": "chatcmpl-test123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "placeholder"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 4000,
            "completion_tokens": 500,
            "total_tokens": 4500
        }
    }
    
    # Finish the response
    logger.complete_response(template_name, completion_data)
    
    # Re-read the file to check for markdown formatting issues
    with open(log_path, 'r', encoding='utf-8') as f:
        updated_raw_content = f.read()
    
    # Count markdown markers again
    updated_markdown_markers = re.findall(r'# markdown', updated_raw_content)
    print(f"After completion: Found {len(updated_markdown_markers)} markdown markers in the output")
    
    # Check for continued content with backslashes
    updated_backslash_breaks = re.findall(r'\\$\n', updated_raw_content)
    print(f"After completion: Found {len(updated_backslash_breaks)} line breaks with trailing backslashes")
    
    # Use our testing-specific loader that strips trailing newlines
    def load_yaml_for_testing(file_path):
        with open(file_path, encoding='utf-8') as f:
            log_data = yaml.safe_load(f)
        
        # Strip newlines from content fields for tests
        return preprocess_yaml_data(log_data, strip_newlines=True)
    
    # Load YAML again to verify it can still be parsed, using our special loader
    updated_yaml_content = load_yaml_for_testing(log_path)
    
    # Ensure both request and response content are preserved
    assert "<AboutComplexityProcessing>" in updated_yaml_content["request"]["messages"][0]["content"]
    assert "passageId" in updated_yaml_content["response"]["choices"][0]["message"]["content"]

def test_direct_yaml_dumper_with_large_content():
    """Test the YAML dumper directly to isolate the issue."""
    # Create test data with large content
    repeated_section = "# markdown    # markdown    # markdown    # markdown    # markdown\n" * 10
    large_content = f"<AboutComplexityProcessing>\n\n{repeated_section}\nThis is a test of very large content.\n</AboutComplexityProcessing>"
    
    test_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    
    # Create a temporary file for the test
    temp_file_path = Path("temp_yaml_test.yaml")
    
    try:
        # Write the data using ContentAwareYAMLDumper
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Read the raw content to check formatting
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Check for indicators of formatting issues
        markdown_markers = re.findall(r'# markdown', raw_content)
        print(f"Direct YAML dumper test: Found {len(markdown_markers)} markdown markers")
        
        backslash_breaks = re.findall(r'\\$\n', raw_content)
        print(f"Direct YAML dumper test: Found {len(backslash_breaks)} line breaks with backslashes")
        
        # Try to reload the YAML
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            reloaded_data = yaml.safe_load(f)
        
        # Verify content is preserved
        assert "<AboutComplexityProcessing>" in reloaded_data["request"]["messages"][0]["content"]
        
        # Now run the post-processing step manually to see its effect
        from jinja_prompt_chaining_system.logger import LLMLogger
        dummy_logger = LLMLogger()
        dummy_logger._post_process_yaml_file(temp_file_path)
        
        # Read again after post-processing
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            post_process_content = f.read()
        
        # Check for increased markdown markers after post-processing
        post_markdown_markers = re.findall(r'# markdown', post_process_content)
        print(f"After post-processing: Found {len(post_markdown_markers)} markdown markers")
        
        # If count increased significantly, the issue is in the post-processing
        assert len(post_markdown_markers) > len(markdown_markers), "Post-processing should add markdown markers"
        
    finally:
        # Clean up
        if temp_file_path.exists():
            temp_file_path.unlink()

def test_debug_yaml_formatting():
    """A focused test to debug YAML formatting issues with the # markdown comments."""
    # Test with a simpler yet problematic string
    test_content = """This is a test string with
# markdown in the content itself which might cause confusion
when being processed by the YAML dumper and post-processor.

It also has multiple lines that might trigger line continuation backslashes \
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
        
        print("\n--- Initial YAML dump content ---")
        print(initial_content)
        
        # Step 2: Manually post-process the file
        dummy_logger = LLMLogger()
        dummy_logger._post_process_yaml_file(test_file)
        
        # Read again and print for comparison
        with open(test_file, "r", encoding="utf-8") as f:
            processed_content = f.read()
        
        print("\n--- After post-processing ---")
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
                assert False, "YAML could not be loaded after post-processing"
                
        # Test with a line that definitely contains "content:" in it
        test_data_2 = {
            "request": {
                "messages": [
                    {"role": "user", "content": "This is a content field with # markdown in it"},
                    {"role": "system", "content": "The content: field here has a colon which might confuse the post-processor"}
                ]
            }
        }
        
        # Write this to the file
        with open(test_file, "w", encoding="utf-8") as f:
            yaml.dump(test_data_2, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        # Read the raw content
        with open(test_file, "r", encoding="utf-8") as f:
            nested_initial_content = f.read()
            
        print("\n--- Nested content fields initial dump ---")
        print(nested_initial_content)
        
        # Post-process
        dummy_logger._post_process_yaml_file(test_file)
        
        # Read again
        with open(test_file, "r", encoding="utf-8") as f:
            nested_processed_content = f.read()
            
        print("\n--- Nested content fields after post-processing ---")
        print(nested_processed_content)
        
        # Count markers again
        nested_initial_markers = re.findall(r'# markdown', nested_initial_content)
        nested_processed_markers = re.findall(r'# markdown', nested_processed_content)
        
        print(f"\nBefore post-processing: {len(nested_initial_markers)} markdown markers")
        print(f"After post-processing: {len(nested_processed_markers)} markdown markers")
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink() 