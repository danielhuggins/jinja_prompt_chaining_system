import os
import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from pathlib import Path

from jinja_prompt_chaining_system import render_prompt, render_prompt_async

@pytest.fixture
def template_file(tmp_path):
    template = tmp_path / "test.jinja"
    template.write_text("""
    {% llmquery model="gpt-4" temperature=0.7 %}
    Hello, {{ name }}!
    {% endllmquery %}
    """)
    return str(template)  # Return string path instead of Path

@pytest.fixture
def context_file(tmp_path):
    context = tmp_path / "context.yaml"
    context.write_text("""
    name: World
    """)
    return str(context)  # Return string path instead of Path

@pytest.fixture
def context_dict():
    return {"name": "World"}

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_render_prompt_basic(mock_logger, mock_llm_client, template_file, context_file):
    """Test basic API function with file paths."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Call the function
    result = render_prompt(template_file, context_file)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_render_prompt_with_dict(mock_logger, mock_llm_client, template_file, context_dict):
    """Test API function with context as dictionary."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Call the function
    result = render_prompt(template_file, context_dict)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_render_prompt_with_output(mock_logger, mock_llm_client, template_file, context_file, tmp_path):
    """Test API function with output file."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Set output path
    output_path = str(tmp_path / "output" / "output.txt")
    
    # Call the function
    result = render_prompt(template_file, context_file, out=output_path)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result
    
    # Check that the output file was created with the correct content
    assert os.path.exists(output_path)
    with open(output_path, 'r') as f:
        assert "Hello, World!" in f.read()

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
@patch('jinja_prompt_chaining_system.api.RunLogger')
def test_render_prompt_with_logdir(mock_run_logger, mock_llm_logger, mock_llm_client, template_file, context_file, tmp_path):
    """Test API function with log directory."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Mock RunLogger
    run_logger_instance = Mock()
    run_logger_instance.start_run.return_value = "test_run_id"
    run_logger_instance.get_llm_logger.return_value = Mock()
    mock_run_logger.return_value = run_logger_instance
    
    # Set log directory
    log_dir = str(tmp_path / "logs")
    
    # Call the function
    result = render_prompt(template_file, context_file, logdir=log_dir)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result
    
    # Check that RunLogger was called correctly
    mock_run_logger.assert_called_once_with(log_dir)
    run_logger_instance.start_run.assert_called_once()
    run_logger_instance.get_llm_logger.assert_called_once_with("test_run_id")
    run_logger_instance.end_run.assert_called_once()

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_render_prompt_file_not_found(mock_logger, mock_llm_client, tmp_path):
    """Test API function with nonexistent template file."""
    nonexistent_template = str(tmp_path / "nonexistent.jinja")
    context_file = str(tmp_path / "context.yaml")
    with open(context_file, 'w') as f:
        f.write("name: World")
    
    # Check that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        render_prompt(nonexistent_template, context_file)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_render_prompt_invalid_context_file(mock_logger, mock_llm_client, template_file, tmp_path):
    """Test API function with nonexistent context file."""
    nonexistent_context = str(tmp_path / "nonexistent.yaml")
    
    # Check that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        render_prompt(template_file, nonexistent_context)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_render_prompt_invalid_yaml(mock_logger, mock_llm_client, template_file, tmp_path):
    """Test API function with invalid YAML context."""
    invalid_yaml = str(tmp_path / "invalid.yaml")
    with open(invalid_yaml, 'w') as f:
        f.write("invalid: yaml: content")
    
    # Check that ValueError is raised
    with pytest.raises(ValueError):
        render_prompt(template_file, invalid_yaml)

@pytest.fixture
def async_template_file(tmp_path):
    template = tmp_path / "async_test.jinja"
    template.write_text("""
    {% llmquery model="gpt-4" temperature=0.7 %}
    Hello, {{ name }}!
    {% endllmquery %}
    """)
    return str(template)  # Return string path instead of Path

@pytest.mark.asyncio
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
async def test_render_prompt_async(mock_logger, mock_llm_client, async_template_file, context_dict):
    """Test async API function."""
    # Setup mocks
    client = Mock()
    client.query_async = AsyncMock()
    client.query_async.return_value = "Hello, World!"
    client.query = Mock(return_value="Hello, World!")  # For synchronous fallback
    mock_llm_client.return_value = client
    
    # Call the async function
    result = await render_prompt_async(async_template_file, context_dict)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result

@pytest.mark.asyncio
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
@patch('jinja_prompt_chaining_system.api.RunLogger')
async def test_render_prompt_async_with_logdir(mock_run_logger, mock_llm_logger, mock_llm_client, async_template_file, context_dict, tmp_path):
    """Test async API function with log directory."""
    # Setup mocks
    client = Mock()
    client.query_async = AsyncMock()
    client.query_async.return_value = "Hello, World!"
    client.query = Mock(return_value="Hello, World!")  # For synchronous fallback
    mock_llm_client.return_value = client
    
    # Mock RunLogger
    run_logger_instance = Mock()
    run_logger_instance.start_run.return_value = "test_run_id"
    run_logger_instance.get_llm_logger.return_value = Mock()
    mock_run_logger.return_value = run_logger_instance
    
    # Set log directory
    log_dir = str(tmp_path / "logs")
    
    # Call the async function
    result = await render_prompt_async(async_template_file, context_dict, logdir=log_dir)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result
    
    # Check that RunLogger was called correctly
    mock_run_logger.assert_called_once_with(log_dir)
    run_logger_instance.start_run.assert_called_once()
    run_logger_instance.get_llm_logger.assert_called_once_with("test_run_id")
    run_logger_instance.end_run.assert_called_once()

@pytest.mark.asyncio
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
async def test_render_prompt_async_with_output(mock_logger, mock_llm_client, async_template_file, context_dict, tmp_path):
    """Test async API function with output file."""
    # Setup mocks
    client = Mock()
    client.query_async = AsyncMock()
    client.query_async.return_value = "Hello, World!"
    client.query = Mock(return_value="Hello, World!")  # For synchronous fallback
    mock_llm_client.return_value = client
    
    # Set output path
    output_path = str(tmp_path / "output" / "async_output.txt")
    
    # Call the async function
    result = await render_prompt_async(async_template_file, context_dict, out=output_path)
    
    # Check that the result contains our mocked response
    assert "Hello, World!" in result
    
    # Check that the output file was created with the correct content
    assert os.path.exists(output_path)
    with open(output_path, 'r') as f:
        assert "Hello, World!" in f.read() 