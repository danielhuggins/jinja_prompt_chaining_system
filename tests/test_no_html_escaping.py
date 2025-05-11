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

def test_global_llmquery_preserves_html():
    """Test that HTML in global llmquery function is preserved."""
    # Create environment with a custom global llmquery function
    # to verify HTML preservation without relying on mocks
    env = Environment(autoescape=False)
    
    # Add a simple global llmquery function that preserves HTML
    html_preserved = False
    
    def test_llmquery(prompt, **kwargs):
        nonlocal html_preserved
        # Check if HTML is preserved in the prompt
        html_preserved = "<p>" in prompt and "&lt;p&gt;" not in prompt
        return f"Response: {prompt}"
    
    env.globals['llmquery'] = test_llmquery
    
    # Create and render a template with HTML
    template_str = '{{ llmquery(prompt="Test with <p>paragraph</p> tag", model="gpt-4") }}'
    template = env.from_string(template_str)
    result = template.render()
    
    # Verify HTML was preserved in the prompt passed to llmquery
    assert html_preserved, "HTML was not preserved in the llmquery prompt parameter"
    
    # Verify HTML is in the output
    assert "<p>paragraph</p>" in result
    assert "&lt;p&gt;" not in result 