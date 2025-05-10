import pytest
from jinja2 import Environment, Template
from jinja_prompt_chaining_system import create_environment
from jinja_prompt_chaining_system.parser import LLMQueryExtension
from unittest.mock import patch, Mock, AsyncMock

def test_html_escaping_disabled_by_default():
    """Test that HTML escaping is disabled by default in the environment."""
    env = create_environment()
    
    # Check the environment configuration
    assert env.autoescape is False
    
    # Create and render a template with HTML content
    template_str = 'HTML content: <div>{{ variable }}</div>'
    template = env.from_string(template_str)
    result = template.render(variable='<strong>test</strong>')
    
    # Verify HTML characters are not escaped
    assert 'HTML content: <div><strong>test</strong></div>' == result
    assert '&lt;' not in result
    assert '&gt;' not in result

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_llmquery_tag_preserves_html(mock_llm_client):
    """Test that HTML in llmquery tags is preserved."""
    # Setup mock
    client = Mock()
    client.query.return_value = "Response with <tags> preserved"
    mock_llm_client.return_value = client
    
    # Create environment
    env = create_environment()
    
    # Create template with HTML in llmquery tag, using trim blocks to handle whitespace
    template_str = '''
    {%- llmquery model="gpt-4" -%}
    Prompt with <html> & special characters
    {%- endllmquery -%}
    '''
    template = env.from_string(template_str)
    
    # Render template
    result = template.render()
    
    # Verify HTML was preserved in the prompt
    call_args = client.query.call_args
    prompt = call_args[0][0]
    assert "<html>" in prompt
    assert "&lt;html&gt;" not in prompt
    
    # Verify response HTML is also preserved
    assert "Response with <tags> preserved" == result
    assert "&lt;tags&gt;" not in result

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_global_llmquery_preserves_html(mock_llm_client):
    """Test that HTML in global llmquery function is preserved."""
    # Setup mock
    client = Mock()
    client.query.return_value = "Response with <b>bold</b> text"
    # Also provide async mock method
    client.query_async = AsyncMock(return_value="Response with <b>bold</b> text")
    mock_llm_client.return_value = client
    
    # Create environment
    env = create_environment()
    
    # Create template with HTML in global llmquery
    template_str = '{{ llmquery(prompt="Query with <p>paragraph</p> & ampersand", model="gpt-4") }}'
    template = env.from_string(template_str)
    
    # Render template
    result = template.render()
    
    # Verify HTML was preserved in the prompt
    # Check if either sync or async method was called
    called_method = client.query if client.query.called else client.query_async
    call_args = called_method.call_args
    
    # Extract prompt from args or kwargs
    prompt = None
    # Check args first
    for arg in call_args[0]:
        if isinstance(arg, str) and "<p>" in arg:
            prompt = arg
            break
    
    # If prompt not found in args, check kwargs
    if prompt is None:
        for arg_name, arg_value in call_args[1].items():
            if arg_name == "prompt" or (isinstance(arg_value, str) and "<p>" in arg_value):
                prompt = arg_value
                break
    
    assert prompt is not None, "Prompt not found in function call"
    assert "<p>paragraph</p>" in prompt
    assert "&lt;p&gt;" not in prompt
    
    # Verify response HTML is also preserved
    assert "Response with <b>bold</b> text" == result
    assert "&lt;b&gt;" not in result 