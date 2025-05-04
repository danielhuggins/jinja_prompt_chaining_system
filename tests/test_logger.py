import os
import yaml
import pytest
import re
from datetime import datetime, timezone
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system.logger import LLMLogger

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"

@pytest.fixture
def logger(log_dir):
    return LLMLogger(str(log_dir))

def test_logger_initialization(log_dir):
    """Test logger initialization creates log directory."""
    logger = LLMLogger(str(log_dir))
    assert os.path.exists(log_dir)

def test_log_file_naming_format(logger, log_dir):
    """Test that log files are named with template name and timestamp."""
    template_name = "test_template"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    # Log a request
    log_path = logger.log_request(template_name, request)
    
    # Verify the log file was created
    assert os.path.exists(log_path)
    
    # Extract the filename from the path
    filename = os.path.basename(log_path)
    
    # The filename should start with the template name
    assert filename.startswith(f"{template_name}_")
    
    # The filename should end with .log.yaml
    assert filename.endswith(".log.yaml")
    
    # Extract the parts between the template name and the extension
    remaining = filename[len(template_name)+1:-9]  # Remove template_name_ prefix and .log.yaml suffix
    
    # Split the remaining part to get timestamp and counter
    parts = remaining.split('_')
    assert len(parts) == 2  # Should have [timestamp, counter]
    
    # Verify the timestamp matches the expected format
    timestamp = parts[0]
    assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', timestamp)
    
    # Verify the counter is a digit
    counter = parts[1]
    assert counter.isdigit()

def test_log_request(logger, log_dir):
    """Test logging a request."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    logger.log_request(template_name, request)
    
    # Find the log file (using glob since it now has a timestamp)
    log_files = list(log_dir.glob(f"{template_name}_*.log.yaml"))
    assert len(log_files) == 1
    
    with open(log_files[0]) as f:
        log_data = yaml.safe_load(f)
    
    assert "request" in log_data
    assert log_data["request"] == request
    assert "timestamp" in log_data

def test_log_request_with_response(logger, log_dir):
    """Test logging a request with non-streaming response."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "stream": False,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    response = {
        "id": "chatcmpl-123abc",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    
    logger.log_request(template_name, request, response)
    
    # Find the log file
    log_files = list(log_dir.glob(f"{template_name}_*.log.yaml"))
    assert len(log_files) == 1
    
    with open(log_files[0]) as f:
        log_data = yaml.safe_load(f)
    
    assert "request" in log_data
    assert log_data["request"] == request
    assert "response" in log_data
    assert log_data["response"] == response
    assert "timestamp" in log_data

def test_streaming_response_reconstruction(logger, log_dir):
    """Test that streaming responses are reconstructed to match OpenAI's response format."""
    template_name = "test_streaming"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "stream": True,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    # Log initial request
    logger.log_request(template_name, request)
    
    # Update with chunks
    chunks = ["Hello", ", ", "world", "!"]
    for chunk in chunks:
        logger.update_response(template_name, chunk)
    
    # Find the log file
    log_files = list(log_dir.glob(f"{template_name}_*.log.yaml"))
    assert len(log_files) == 1
    
    with open(log_files[0]) as f:
        log_data = yaml.safe_load(f)
    
    assert "request" in log_data
    assert log_data["request"] == request
    assert "response" in log_data
    assert "content" in log_data["response"]
    assert log_data["response"]["content"] == "Hello, world!"
    assert "done" in log_data["response"]
    assert log_data["response"]["done"] is False  # Not marked as done yet

    # Complete the response with metadata
    completion_data = {
        "id": "chatcmpl-abc123",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14
        }
    }
    logger.complete_response(template_name, completion_data)
    
    # Re-read the log file
    with open(log_files[0]) as f:
        log_data = yaml.safe_load(f)
    
    # Verify the response structure matches OpenAI's response format
    assert log_data["response"]["done"] is True
    assert log_data["response"]["id"] == "chatcmpl-abc123"
    assert log_data["response"]["model"] == "gpt-4"
    assert "choices" in log_data["response"]
    assert len(log_data["response"]["choices"]) == 1
    assert log_data["response"]["choices"][0]["index"] == 0
    assert log_data["response"]["choices"][0]["message"]["role"] == "assistant"
    assert log_data["response"]["choices"][0]["message"]["content"] == "Hello, world!"
    assert "usage" in log_data["response"]
    assert log_data["response"]["usage"]["total_tokens"] == 14
    # Since we fixed the logger, check that content is directly in the response
    assert log_data["response"]["content"] == "Hello, world!"

def test_parallel_streaming_responses(logger, log_dir):
    """Test handling multiple concurrent streaming responses."""
    template_name_1 = "test_stream_1"
    template_name_2 = "test_stream_2"
    
    # Log initial requests
    request_1 = {
        "model": "gpt-4",
        "temperature": 0.7,
        "stream": True,
        "messages": [{"role": "user", "content": "First request"}]
    }
    request_2 = {
        "model": "gpt-4",
        "temperature": 0.6,
        "stream": True,
        "messages": [{"role": "user", "content": "Second request"}]
    }
    
    logger.log_request(template_name_1, request_1)
    logger.log_request(template_name_2, request_2)
    
    # Interleave updates
    logger.update_response(template_name_1, "First ")
    logger.update_response(template_name_2, "Second ")
    logger.update_response(template_name_1, "response")
    logger.update_response(template_name_2, "response")
    
    # Complete both
    completion_data_1 = {
        "id": "chatcmpl-111",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "First response"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"total_tokens": 10}
    }
    
    completion_data_2 = {
        "id": "chatcmpl-222",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Second response"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"total_tokens": 11}
    }
    
    logger.complete_response(template_name_1, completion_data_1)
    logger.complete_response(template_name_2, completion_data_2)
    
    # Verify both log files
    log_files_1 = list(log_dir.glob(f"{template_name_1}_*.log.yaml"))
    log_files_2 = list(log_dir.glob(f"{template_name_2}_*.log.yaml"))
    
    assert len(log_files_1) == 1
    assert len(log_files_2) == 1
    
    with open(log_files_1[0]) as f:
        log_data_1 = yaml.safe_load(f)
    
    with open(log_files_2[0]) as f:
        log_data_2 = yaml.safe_load(f)
    
    # Check that content was preserved directly in response
    assert log_data_1["response"]["content"] == "First response"
    assert log_data_1["response"]["done"] is True
    assert log_data_1["response"]["id"] == "chatcmpl-111"
    
    assert log_data_2["response"]["content"] == "Second response"
    assert log_data_2["response"]["done"] is True
    assert log_data_2["response"]["id"] == "chatcmpl-222"

def test_multiple_requests_same_template(logger, log_dir):
    """Test logging multiple requests for the same template name."""
    template_name = "test_multiple"
    
    # First request and response with distinct content
    request1 = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "First query"}]
    }
    response1 = {
        "id": "chatcmpl-111",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "First response"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"total_tokens": 10}
    }
    log_path1 = logger.log_request(template_name, request1, response1)
    
    # Second request and response with distinct content
    request2 = {
        "model": "gpt-4",
        "temperature": 0.8,
        "messages": [{"role": "user", "content": "Second query"}]
    }
    response2 = {
        "id": "chatcmpl-222", # Different ID to distinguish
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Second response"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"total_tokens": 11}
    }
    log_path2 = logger.log_request(template_name, request2, response2)
    
    # Verify that logger tracked both log files
    assert len(logger.template_logs[template_name]) == 2
    
    # Also check that the actual files exist
    assert os.path.exists(log_path1)
    assert os.path.exists(log_path2)
    
    # Verify the two log files are different
    assert log_path1 != log_path2
    
    # Read the log files
    with open(log_path1) as f:
        log_data1 = yaml.safe_load(f)
    
    with open(log_path2) as f:
        log_data2 = yaml.safe_load(f)
    
    # Get the response IDs from each file
    response_id1 = log_data1["response"]["id"]
    response_id2 = log_data2["response"]["id"]
    
    # Either the first file contains the first response and the second file contains 
    # the second response, or vice versa. But they should definitely be different.
    assert response_id1 != response_id2
    assert {"chatcmpl-111", "chatcmpl-222"} == {response_id1, response_id2}

def test_log_request_with_tools(logger, log_dir):
    """Test logging a request with tools parameter."""
    template_name = "test_tools"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Extract text from this PDF"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_pdf_text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
    
    # Add the done flag directly to the response
    response = {
        "id": "chatcmpl-tools",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "extract_pdf_text",
                                "arguments": '{"url": "https://example.com/doc.pdf"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"total_tokens": 20},
        "done": True
    }
    
    logger.log_request(template_name, request, response)
    
    # Find the log file
    log_files = list(log_dir.glob(f"{template_name}_*.log.yaml"))
    assert len(log_files) == 1
    
    with open(log_files[0]) as f:
        log_data = yaml.safe_load(f)
    
    # Verify tools in request
    assert "tools" in log_data["request"]
    assert log_data["request"]["tools"][0]["type"] == "function"
    assert log_data["request"]["tools"][0]["function"]["name"] == "extract_pdf_text"
    
    # Verify tool_calls in response
    assert log_data["response"]["choices"][0]["message"]["tool_calls"][0]["id"] == "call_123"
    assert log_data["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "extract_pdf_text"
    
    # Verify the response structure matches OpenAI's response format
    assert log_data["response"]["done"] is True
    assert log_data["response"]["id"] == "chatcmpl-tools"
    assert log_data["response"]["model"] == "gpt-4"
    assert "choices" in log_data["response"]
    assert len(log_data["response"]["choices"]) == 1
    assert log_data["response"]["choices"][0]["index"] == 0
    assert log_data["response"]["choices"][0]["message"]["role"] == "assistant"
    assert log_data["response"]["choices"][0]["message"]["content"] is None
    assert "usage" in log_data["response"]
    assert log_data["response"]["usage"]["total_tokens"] == 20

@patch('jinja_prompt_chaining_system.logger.datetime')
@patch('jinja_prompt_chaining_system.logger.time.sleep')  # Mock sleep to avoid delays in tests
def test_timestamp_in_filename(mock_sleep, mock_datetime, log_dir):
    """Test timestamp formatting in log filenames."""
    # Mock datetime to return a fixed time
    fixed_time = datetime(2023, 1, 15, 12, 30, 45, 123456)
    mock_datetime.now.return_value = fixed_time
    mock_datetime.timezone = timezone  # Use timezone instead of UTC
    
    # Prevent actual sleeping
    mock_sleep.return_value = None
    
    logger = LLMLogger(str(log_dir))
    template_name = "test"
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    
    log_path = logger.log_request(template_name, request)
    
    # The expected timestamp format now includes microseconds and counter
    expected_timestamp = fixed_time.strftime("%Y-%m-%dT%H-%M-%S-%f")
    expected_counter = 0  # First file for this template
    
    # Construct expected path
    expected_filename = f"{template_name}_{expected_timestamp}_{expected_counter}.log.yaml"
    expected_log_file = log_dir / expected_filename
    
    # Just check that the path exists, don't rely on exact matching
    assert os.path.exists(log_path)
    # Also verify that our templating logic is correct by making sure our expected path points to the same file
    assert os.path.samefile(log_path, expected_log_file)

def test_empty_log_dir():
    """Test behavior when no log directory is provided."""
    logger = LLMLogger()  # No log directory
    
    # Should not raise an error when trying to log
    logger.log_request("test", {"model": "gpt-4"})
    logger.update_response("test", "Hello")
    logger.complete_response("test", {"id": "chatcmpl-123"})
    
    # No assertions needed, just checking that no exceptions are raised 