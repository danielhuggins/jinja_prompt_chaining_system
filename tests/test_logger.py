import os
import yaml
import pytest
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

def test_log_request(logger, log_dir):
    """Test logging a request."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    logger.log_request(template_name, request)
    
    log_file = log_dir / f"{template_name}.log.yaml"
    assert log_file.exists()
    
    with open(log_file) as f:
        log_data = yaml.safe_load(f)
    
    assert len(log_data) == 1
    assert log_data[0]["request"] == request
    assert "timestamp" in log_data[0]

def test_log_request_with_response(logger, log_dir):
    """Test logging a request with response."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    response = {
        "content": "Hi there!",
        "done": True
    }
    
    logger.log_request(template_name, request, response)
    
    log_file = log_dir / f"{template_name}.log.yaml"
    assert log_file.exists()
    
    with open(log_file) as f:
        log_data = yaml.safe_load(f)
    
    assert len(log_data) == 1
    assert log_data[0]["request"] == request
    assert log_data[0]["response"] == response

def test_update_response(logger, log_dir):
    """Test updating response with chunks."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    # Log initial request
    logger.log_request(template_name, request)
    
    # Update with chunks
    chunks = ["Hello", ", ", "world", "!"]
    for chunk in chunks:
        logger.update_response(template_name, chunk)
    
    log_file = log_dir / f"{template_name}.log.yaml"
    assert log_file.exists()
    
    with open(log_file) as f:
        log_data = yaml.safe_load(f)
    
    assert len(log_data) == 1
    assert log_data[0]["response"]["content"] == "Hello, world!"
    assert not log_data[0]["response"]["done"]

def test_multiple_requests_same_template(logger, log_dir):
    """Test logging multiple requests for the same template."""
    template_name = "test"
    
    # First request and response
    request1 = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "First query"}]
    }
    response1 = {
        "content": "First response",
        "done": True
    }
    logger.log_request(template_name, request1, response1)
    
    # Second request and response
    request2 = {
        "model": "gpt-4",
        "temperature": 0.8,
        "messages": [{"role": "user", "content": "Second query"}]
    }
    response2 = {
        "content": "Second response",
        "done": True
    }
    logger.log_request(template_name, request2, response2)
    
    log_file = log_dir / f"{template_name}.log.yaml"
    assert log_file.exists()
    
    with open(log_file) as f:
        log_data = yaml.safe_load(f)
    
    assert len(log_data) == 2
    assert log_data[0]["request"] == request1
    assert log_data[0]["response"] == response1
    assert log_data[1]["request"] == request2
    assert log_data[1]["response"] == response2

def test_complete_streaming_response(logger, log_dir):
    """Test completing a streaming response."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    # Log initial request
    logger.log_request(template_name, request)
    
    # Update with chunks
    chunks = ["Hello", ", ", "world", "!"]
    for chunk in chunks:
        logger.update_response(template_name, chunk)
    
    # Mark response as done
    logger.complete_response(template_name, {
        "id": "chatcmpl-abc123",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14
        }
    })
    
    log_file = log_dir / f"{template_name}.log.yaml"
    assert log_file.exists()
    
    with open(log_file) as f:
        log_data = yaml.safe_load(f)
    
    assert len(log_data) == 1
    assert log_data[0]["response"]["content"] == "Hello, world!"
    assert log_data[0]["response"]["done"]
    assert "id" in log_data[0]["response"]
    assert "usage" in log_data[0]["response"]
    assert log_data[0]["response"]["usage"]["total_tokens"] == 14

def test_log_request_with_tools(logger, log_dir):
    """Test logging a request with tools parameter."""
    template_name = "test"
    request = {
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}],
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
    
    logger.log_request(template_name, request)
    
    log_file = log_dir / f"{template_name}.log.yaml"
    assert log_file.exists()
    
    with open(log_file) as f:
        log_data = yaml.safe_load(f)
    
    assert len(log_data) == 1
    assert log_data[0]["request"] == request
    assert "tools" in log_data[0]["request"]
    assert log_data[0]["request"]["tools"][0]["type"] == "function"
    assert log_data[0]["request"]["tools"][0]["function"]["name"] == "extract_pdf_text" 