import os
import pytest
from unittest.mock import patch, Mock
import tempfile
from jinja2 import Environment, FileSystemLoader
from jinja_prompt_chaining_system import create_environment
from jinja_prompt_chaining_system.parser import LLMQueryExtension

@pytest.fixture
def temp_template_dir():
    """Create a temporary directory with template files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create main template
        with open(os.path.join(tmpdir, "main.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Main template content
            {% include 'include1.jinja' %}
            {% endllmquery %}
            """)
        
        # Create included template
        with open(os.path.join(tmpdir, "include1.jinja"), "w") as f:
            f.write("""
            Included content from include1.jinja
            """)
        
        # Create nested include template
        with open(os.path.join(tmpdir, "nested.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Nested template content
            {% include 'include2.jinja' %}
            {% endllmquery %}
            """)
        
        # Create second included template
        with open(os.path.join(tmpdir, "include2.jinja"), "w") as f:
            f.write("""
            Included content from include2.jinja
            """)
        
        # Create template with llmquery inside include
        with open(os.path.join(tmpdir, "with_llmquery.jinja"), "w") as f:
            f.write("""
            Content before include
            {% include 'llmquery_include.jinja' %}
            Content after include
            """)
        
        # Create included template containing llmquery
        with open(os.path.join(tmpdir, "llmquery_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            This is a query from an included template
            {% endllmquery %}
            """)
        
        # Create template with variable in include path
        with open(os.path.join(tmpdir, "variable_include.jinja"), "w") as f:
            f.write("""
            {% set include_file = 'include1.jinja' %}
            {% llmquery model="gpt-4" %}
            Content with variable include:
            {% include include_file %}
            {% endllmquery %}
            """)
        
        # Create template with conditional include
        with open(os.path.join(tmpdir, "conditional_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            {% if condition %}
                {% include 'include1.jinja' %}
            {% else %}
                {% include 'include2.jinja' %}
            {% endif %}
            {% endllmquery %}
            """)
        
        # Create template with include with context
        with open(os.path.join(tmpdir, "include_with_context.jinja"), "w") as f:
            f.write("""
            {% set local_var = "local value" %}
            {% llmquery model="gpt-4" %}
            Content before including with context
            {% include 'context_template.jinja' with context %}
            Content after including with context
            {% endllmquery %}
            """)
        
        # Create template for context testing
        with open(os.path.join(tmpdir, "context_template.jinja"), "w") as f:
            f.write("""
            Accessing context variable: {{ local_var }}
            """)
        
        # Create template with circular includes
        with open(os.path.join(tmpdir, "circular1.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Circular template 1
            {% include 'circular2.jinja' %}
            {% endllmquery %}
            """)
        
        with open(os.path.join(tmpdir, "circular2.jinja"), "w") as f:
            f.write("""
            Circular template 2
            {% include 'circular1.jinja' %}
            """)
        
        yield tmpdir

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_include_in_llmquery(mock_llm_client, temp_template_dir):
    """Test using {% include %} within {% llmquery %} tags."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("main.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"
    
    # Check that the LLM was called with the correct prompt including the included content
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Main template content" in prompt
    assert "Included content from include1.jinja" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nested_include_in_llmquery(mock_llm_client, temp_template_dir):
    """Test nested templates that both have {% include %} within {% llmquery %}."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock nested LLM response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("nested.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock nested LLM response"
    
    # Check that the LLM was called with the correct prompt including all nested content
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Nested template content" in prompt
    assert "Included content from include2.jinja" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_llmquery_in_included_template(mock_llm_client, temp_template_dir):
    """Test using a template with {% include %} that contains {% llmquery %} tags."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock included LLM response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("with_llmquery.jinja")
    result = template.render()
    
    # Verify the result contains content before/after and LLM response - ignore exact whitespace
    assert "Content before include" in result
    assert "Mock included LLM response" in result
    assert "Content after include" in result
    
    # Check that the LLM was called with the correct prompt from included template
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "This is a query from an included template" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_variable_include_path(mock_llm_client, temp_template_dir):
    """Test using a variable for the include path within {% llmquery %}."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock variable include response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("variable_include.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock variable include response"
    
    # Check that the LLM was called with content from the variable-path include
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Content with variable include:" in prompt
    assert "Included content from include1.jinja" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_conditional_include(mock_llm_client, temp_template_dir):
    """Test conditional include within {% llmquery %}."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock conditional include response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Test with condition=True
    template = env.get_template("conditional_include.jinja")
    result = template.render(condition=True)
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock conditional include response"
    
    # Check that the LLM was called with include1.jinja content
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Included content from include1.jinja" in prompt
    
    # Reset mock and test with condition=False
    client_instance.reset_mock()
    client_instance.query.return_value = "Mock alternate include response"
    
    result = template.render(condition=False)
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock alternate include response"
    
    # Check that the LLM was called with include2.jinja content
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Included content from include2.jinja" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_include_with_context(mock_llm_client, temp_template_dir):
    """Test include with context inside {% llmquery %}."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock context include response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("include_with_context.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock context include response"
    
    # Check that the LLM was called with the correct prompt including context variables
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Content before including with context" in prompt
    assert "Accessing context variable: local value" in prompt
    assert "Content after including with context" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_circular_include_in_llmquery(mock_llm_client, temp_template_dir):
    """Test behavior with circular includes within {% llmquery %}."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock circular include response"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get the template - this should raise a template error due to circular includes
    template = env.get_template("circular1.jinja")
    
    # Rendering should raise an exception due to circular includes
    with pytest.raises(Exception) as exc_info:
        template.render()
    
    # Verify the exception contains information about circular includes
    assert "circular" in str(exc_info.value).lower() or "recursion" in str(exc_info.value).lower()

# Now enabling this test since we've specified undefined variables should throw errors
@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_include_with_undefined_variables(mock_llm_client, temp_template_dir):
    """Test include with undefined variables inside {% llmquery %}."""
    # Create a new template for this test
    with open(os.path.join(temp_template_dir, "undefined_var.jinja"), "w") as f:
        f.write("""
        {% llmquery model="gpt-4" %}
        {{ undefined_variable }}
        {% include 'include1.jinja' %}
        {% endllmquery %}
        """)
    
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock response with undefined vars"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get the template
    template = env.get_template("undefined_var.jinja")
    
    # Rendering should raise an exception due to undefined variable
    with pytest.raises(Exception) as exc_info:
        template.render()
    
    # Verify the exception contains information about undefined variable
    assert "undefined" in str(exc_info.value).lower()

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_include_nonexistent_template(mock_llm_client, temp_template_dir):
    """Test behavior when including a non-existent template in {% llmquery %}."""
    # Create a new template for this test
    with open(os.path.join(temp_template_dir, "nonexistent_include.jinja"), "w") as f:
        f.write("""
        {% llmquery model="gpt-4" %}
        Content before including non-existent template
        {% include 'this_template_does_not_exist.jinja' %}
        {% endllmquery %}
        """)
    
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock response with nonexistent include"
    mock_llm_client.return_value = client_instance
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get the template
    template = env.get_template("nonexistent_include.jinja")
    
    # Rendering should raise an exception due to non-existent template
    with pytest.raises(Exception) as exc_info:
        template.render()
    
    # Verify the exception contains information about template not found
    assert "template" in str(exc_info.value).lower() and "not found" in str(exc_info.value).lower() 