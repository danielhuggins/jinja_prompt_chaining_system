import os
import tempfile
import yaml
import pytest
import re
from pathlib import Path

from jinja_prompt_chaining_system.llm import LLMClient
from jinja_prompt_chaining_system.logger import LLMLogger

# Skip this test if no OpenAI API key is available
pytestmark = pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY environment variable not set"
)

def test_long_streaming_response():
    """Test that a long streaming response is correctly logged with literal format."""
    with tempfile.TemporaryDirectory() as log_dir:
        # Initialize logger
        logger = LLMLogger(log_dir)
        
        # Initialize LLM client with API key from environment
        llm_client = LLMClient()
        
        # Template name for logging
        template_name = "test_long_streaming"
        
        # Create a text to ask the model to repeat
        test_text = "This is a test of streaming logging. It includes multiple sentences."
        
        # Create request
        request = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,  # Use 0 to get more deterministic responses
            "max_tokens": 100,
            "stream": True,
            "messages": [
                {"role": "user", "content": f"Please repeat this text verbatim: {test_text}"}
            ]
        }
        
        # Log the request
        log_path = logger.log_request(template_name, request)
        assert log_path is not None
        
        # Get streaming response from OpenAI
        response_generator = llm_client.query(
            f"Please repeat this text verbatim: {test_text}",
            {
                "model": "gpt-4o-mini", 
                "temperature": 0.0,
                "max_tokens": 100
            },
            stream=True
        )
        
        # Collect response chunks
        accumulated_content = ""
        for chunk in response_generator:
            # Update the logger
            logger.update_response(template_name, chunk)
            accumulated_content += chunk
            
        # Complete the response
        completion_data = {
            "id": f"chatcmpl-{id(test_text)}",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": accumulated_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(test_text) // 4,
                "completion_tokens": len(accumulated_content) // 4,
                "total_tokens": (len(test_text) + len(accumulated_content)) // 4
            }
        }
        logger.complete_response(template_name, completion_data)
        
        # Print the log file for debugging
        print(f"\nLog file path: {log_path}")
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            print(f"Log file content:\n{log_content}")
            
        # Read the log file as YAML
        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = yaml.safe_load(f)
        
        # Basic structure checks
        assert "response" in log_data
        assert "choices" in log_data["response"]
        assert len(log_data["response"]["choices"]) > 0
        assert "message" in log_data["response"]["choices"][0]
        assert "content" in log_data["response"]["choices"][0]["message"]
        
        # Check that content is not at the root level
        assert "content" not in log_data["response"], "Content should not be at the root level"
        
        # Check if the content appears properly formatted without escape sequences
        with open(log_path, 'r', encoding='utf-8') as f:
            log_text = f.read()
            
            # Check for quoted strings with escape sequences which we don't want
            escaped_newlines = re.search(r'content: ".*\\n.*"', log_text, re.DOTALL)
            assert not escaped_newlines, "Content contains escaped newlines, should use literal block format"
            
            # Check for multiline content that is properly indented
            content_match = re.search(r'content: .*?\n(\s+\S.*?)(?:\n\S|$)', log_text, re.DOTALL)
            assert content_match, "No properly indented content found, should be using literal block format"
            
            # Check the indentation of the first content line
            indented_content = content_match.group(1)
            assert indented_content.startswith(" "), "Content should be indented" 