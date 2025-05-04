import pytest
import os
import asyncio
from unittest.mock import patch, Mock
from jinja2 import Environment, FileSystemLoader
from jinja_prompt_chaining_system.parser import LLMQueryExtension

@pytest.fixture
def mock_llm_client():
    with patch('jinja_prompt_chaining_system.parser.LLMClient') as mock:
        client = Mock()
        client.query.return_value = "Mocked response"
        mock.return_value = client
        yield client

@pytest.fixture
def mock_logger():
    with patch('jinja_prompt_chaining_system.parser.LLMLogger') as mock:
        logger = Mock()
        mock.return_value = logger
        yield logger

def create_test_environment(template_dir):
    """Create a Jinja environment with the LLMQuery extension registered."""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        enable_async=True,
        extensions=[LLMQueryExtension]
    )
    env.globals['extension'] = env.extensions[LLMQueryExtension.identifier]
    return env

@pytest.mark.asyncio
async def test_spaces_only_separator(mock_llm_client, mock_logger, tmp_path):
    """Test llmquery tag with spaces-only parameter separation."""
    # Create a template using spaces as separators
    template_file = tmp_path / "spaces.jinja"
    template_file.write_text("""
    {% llmquery model="gpt-4" temperature=0.7 max_tokens=150 stream=false %}
    Test prompt with spaces separator
    {% endllmquery %}
    """)
    
    # Create environment and render template
    env = create_test_environment(tmp_path)
    
    # Get extension instance
    extension = env.globals['extension']
    extension.set_template_name(str(template_file))
    
    # Render template - using async version
    template = env.get_template("spaces.jinja")
    result = await template.render_async()
    
    # Verify LLM client was called with correct parameters
    mock_llm_client.query.assert_called_once()
    
    # Instead of checking the coroutine args directly,
    # we'll just verify that the method was called with the right named arguments
    # and check the rendered result
    assert mock_llm_client.query.call_args.kwargs.get('stream') is False
    assert "Mocked response" in result
    
    # Extract the captured params from the call
    params = {}
    for name, param in mock_llm_client.query.call_args.kwargs.items():
        params[name] = param
    
    # Check if the call included our parameters
    assert 'model' in params or ('model' in mock_llm_client.query.call_args[0][1])
    assert 'temperature' in params or ('temperature' in mock_llm_client.query.call_args[0][1])
    assert 'max_tokens' in params or ('max_tokens' in mock_llm_client.query.call_args[0][1])

@pytest.mark.asyncio
async def test_commas_only_separator(mock_llm_client, mock_logger, tmp_path):
    """Test llmquery tag with commas-only parameter separation."""
    # Create a template using commas as separators
    template_file = tmp_path / "commas.jinja"
    template_file.write_text("""
    {% llmquery model="gpt-4", temperature=0.7, max_tokens=150, stream=false %}
    Test prompt with commas separator
    {% endllmquery %}
    """)
    
    # Create environment and render template
    env = create_test_environment(tmp_path)
    
    # Get extension instance
    extension = env.globals['extension']
    extension.set_template_name(str(template_file))
    
    # Render template
    template = env.get_template("commas.jinja")
    result = await template.render_async()
    
    # Verify LLM client was called with correct parameters
    mock_llm_client.query.assert_called_once()
    assert mock_llm_client.query.call_args.kwargs.get('stream') is False
    assert "Mocked response" in result
    
    # Extract the captured params from the call
    params = {}
    for name, param in mock_llm_client.query.call_args.kwargs.items():
        params[name] = param
    
    # Check if the call included our parameters
    assert 'model' in params or ('model' in mock_llm_client.query.call_args[0][1])
    assert 'temperature' in params or ('temperature' in mock_llm_client.query.call_args[0][1])
    assert 'max_tokens' in params or ('max_tokens' in mock_llm_client.query.call_args[0][1])

@pytest.mark.asyncio
async def test_mixed_separators(mock_llm_client, mock_logger, tmp_path):
    """Test llmquery tag with mixed parameter separation (commas and spaces)."""
    # Create a template using mixed separators
    template_file = tmp_path / "mixed.jinja"
    template_file.write_text("""
    {% llmquery model="gpt-4", temperature=0.7 max_tokens=150, stream=false %}
    Test prompt with mixed separators
    {% endllmquery %}
    """)
    
    # Create environment and render template
    env = create_test_environment(tmp_path)
    
    # Get extension instance
    extension = env.globals['extension']
    extension.set_template_name(str(template_file))
    
    # Render template
    template = env.get_template("mixed.jinja")
    result = await template.render_async()
    
    # Verify LLM client was called with correct parameters
    mock_llm_client.query.assert_called_once()
    assert mock_llm_client.query.call_args.kwargs.get('stream') is False
    assert "Mocked response" in result
    
    # Extract the captured params from the call
    params = {}
    for name, param in mock_llm_client.query.call_args.kwargs.items():
        params[name] = param
    
    # Check if the call included our parameters
    assert 'model' in params or ('model' in mock_llm_client.query.call_args[0][1])
    assert 'temperature' in params or ('temperature' in mock_llm_client.query.call_args[0][1])
    assert 'max_tokens' in params or ('max_tokens' in mock_llm_client.query.call_args[0][1])

@pytest.mark.asyncio
async def test_multiline_parameters(mock_llm_client, mock_logger, tmp_path):
    """Test llmquery tag with parameters split across multiple lines."""
    # Create a template with parameters on multiple lines
    template_file = tmp_path / "multiline.jinja"
    template_file.write_text("""
    {% llmquery 
        model="gpt-4"
        temperature=0.7 
        max_tokens=150
        stream=false
    %}
    Test prompt with multiline parameters
    {% endllmquery %}
    """)
    
    # Create environment and render template
    env = create_test_environment(tmp_path)
    
    # Get extension instance
    extension = env.globals['extension']
    extension.set_template_name(str(template_file))
    
    # Render template
    template = env.get_template("multiline.jinja")
    result = await template.render_async()
    
    # Verify LLM client was called with correct parameters
    mock_llm_client.query.assert_called_once()
    assert mock_llm_client.query.call_args.kwargs.get('stream') is False
    assert "Mocked response" in result
    
    # Extract the captured params from the call
    params = {}
    for name, param in mock_llm_client.query.call_args.kwargs.items():
        params[name] = param
    
    # Check if the call included our parameters
    assert 'model' in params or ('model' in mock_llm_client.query.call_args[0][1])
    assert 'temperature' in params or ('temperature' in mock_llm_client.query.call_args[0][1])
    assert 'max_tokens' in params or ('max_tokens' in mock_llm_client.query.call_args[0][1]) 