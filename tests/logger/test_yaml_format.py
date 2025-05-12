import os
import tempfile
import yaml
from pathlib import Path
import pytest

from jinja_prompt_chaining_system.logger import LLMLogger, preprocess_yaml_data
from tests.logger.test_logger import load_yaml_for_testing


def test_content_format_streaming():
    """Test that streaming content is formatted correctly as a YAML literal block."""
    with tempfile.TemporaryDirectory() as log_dir:
        logger = LLMLogger(log_dir)
        
        # Log a request
        template_name = "test_streaming"
        request = {
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        log_path = logger.log_request(template_name, request)
        assert log_path is not None
        
        # Update with a multi-line response
        logger.update_response(template_name, "Line 1\n")
        logger.update_response(template_name, "Line 2\n")
        logger.update_response(template_name, "Line 3")
        
        # Complete the response
        completion_data = {
            "id": "test-id",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Line 1\nLine 2\nLine 3"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 10,
                "total_tokens": 11
            }
        }
        logger.complete_response(template_name, completion_data)
        
        # Read the log file
        with open(log_path, 'r') as f:
            raw_content = f.read()  # Keep this for checking formatting
            
        # Use our test-specific loader to strip newlines
        log_data = load_yaml_for_testing(log_path)
        
        # Verify that the content is in the correct format
        assert "response" in log_data
        assert "choices" in log_data["response"]
        assert len(log_data["response"]["choices"]) > 0
        assert "message" in log_data["response"]["choices"][0]
        assert "content" in log_data["response"]["choices"][0]["message"]
        
        # Verify that content is only in the message, not at the root level
        assert "content" not in log_data["response"], "Content should not be at the root level of the response"
        
        # Check that the content is correct
        expected_content = "Line 1\nLine 2\nLine 3"
        assert log_data["response"]["choices"][0]["message"]["content"] == expected_content
        
        # Now, check that the raw file uses the literal block format (|)
        assert "content: |" in raw_content or "content: >\n" in raw_content, "Content should use literal block format (|)"
        
        # Content should not have escape sequences
        assert "\\n" not in raw_content, "Content should not contain escape sequences"


def test_content_format_non_streaming():
    """Test that non-streaming content is formatted correctly as a YAML literal block."""
    with tempfile.TemporaryDirectory() as log_dir:
        logger = LLMLogger(log_dir)
        
        # Log a request with response
        template_name = "test_non_streaming"
        request = {
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = {
            "id": "test-id",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Line 1\nLine 2\nLine 3"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 10,
                "total_tokens": 11
            }
        }
        
        log_path = logger.log_request(template_name, request, response)
        assert log_path is not None
        
        # Read the log file
        with open(log_path, 'r') as f:
            raw_content = f.read()  # Keep this for checking formatting
            
        # Use our test-specific loader to strip newlines
        log_data = load_yaml_for_testing(log_path)
        
        # Verify that the content is in the correct format
        assert "response" in log_data
        assert "choices" in log_data["response"]
        assert len(log_data["response"]["choices"]) > 0
        assert "message" in log_data["response"]["choices"][0]
        assert "content" in log_data["response"]["choices"][0]["message"]
        
        # Check that the content is correct
        expected_content = "Line 1\nLine 2\nLine 3"
        assert log_data["response"]["choices"][0]["message"]["content"] == expected_content
        
        # Now, check that the raw file uses the literal block format (|)
        assert "content: |" in raw_content or "content: >\n" in raw_content, "Content should use literal block format (|)"
        
        # Content should not have escape sequences
        assert "\\n" not in raw_content, "Content should not contain escape sequences" 