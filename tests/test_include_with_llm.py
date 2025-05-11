import os
import pytest
from unittest.mock import patch, Mock, AsyncMock, MagicMock
import tempfile
from jinja2 import Environment, FileSystemLoader
from jinja_prompt_chaining_system import create_environment
from jinja_prompt_chaining_system.parallel_integration import ParallelLLMQueryExtension

# Add a function to configure environment to bypass parallel query evaluation
def setup_test_environment(template_dir, mock_response="Mock response"):
    """Create an environment with a controlled parallel query extension."""
    # Create environment 
    env = Environment(
        loader=FileSystemLoader(template_dir),
        enable_async=True,
        extensions=[ParallelLLMQueryExtension],
        autoescape=False
    )
    
    # Get the extension
    extension = env.extensions[ParallelLLMQueryExtension.identifier]
    
    # Override the query method to always return our mock response
    def mock_global_llmquery(prompt, **params):
        return mock_response
    
    # Replace the global function
    extension.parallel_global_llmquery = mock_global_llmquery
    env.globals['llmquery'] = mock_global_llmquery
    
    # Make extension accessible in globals
    env.globals['extension'] = extension
    
    return env

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

def test_include_in_llmquery(temp_template_dir, mock_parallel_llm_extension):
    """Test using {% include %} within {% llmquery %} tags."""
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("main.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"

def test_nested_include_in_llmquery(temp_template_dir, mock_parallel_llm_extension):
    """Test nested templates that both have {% include %} within {% llmquery %}."""
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("nested.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"

def test_llmquery_in_included_template(temp_template_dir, mock_parallel_llm_extension):
    """Test using a template with {% include %} that contains {% llmquery %} tags."""
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("with_llmquery.jinja")
    result = template.render()
    
    # Verify the result contains content before/after and LLM response - ignore exact whitespace
    assert "Content before include" in result
    assert "Mock LLM response" in result
    assert "Content after include" in result

def test_variable_include_path(temp_template_dir, mock_parallel_llm_extension):
    """Test using a variable for the include path within {% llmquery %}."""
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("variable_include.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"

def test_conditional_include(temp_template_dir, mock_parallel_llm_extension):
    """Test conditional include within {% llmquery %}."""
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Test with condition=True
    template = env.get_template("conditional_include.jinja")
    result = template.render(condition=True)
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"
    
    # Test with condition=False
    result = template.render(condition=False)
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"

def test_include_with_context(temp_template_dir, mock_parallel_llm_extension):
    """Test include with context inside {% llmquery %}."""
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get and render the template
    template = env.get_template("include_with_context.jinja")
    result = template.render()
    
    # Verify the result - ignore whitespace
    assert result.strip() == "Mock LLM response"

def test_circular_include_in_llmquery(temp_template_dir, mock_parallel_llm_extension):
    """Test behavior with circular includes within {% llmquery %}."""
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
def test_include_with_undefined_variables(temp_template_dir, mock_parallel_llm_extension):
    """Test include with undefined variables inside {% llmquery %}."""
    # Create a new template for this test
    with open(os.path.join(temp_template_dir, "undefined_var.jinja"), "w") as f:
        f.write("""
        {% llmquery model="gpt-4" %}
        {{ undefined_variable }}
        {% include 'include1.jinja' %}
        {% endllmquery %}
        """)
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get the template
    template = env.get_template("undefined_var.jinja")
    
    # Rendering should raise an exception due to undefined variable
    with pytest.raises(Exception) as exc_info:
        template.render()
    
    # Verify the exception contains information about undefined variable
    assert "undefined" in str(exc_info.value).lower()

def test_include_nonexistent_template(temp_template_dir, mock_parallel_llm_extension):
    """Test behavior when including a non-existent template in {% llmquery %}."""
    # Create a new template for this test
    with open(os.path.join(temp_template_dir, "nonexistent_include.jinja"), "w") as f:
        f.write("""
        {% llmquery model="gpt-4" %}
        Content before including non-existent template
        {% include 'this_template_does_not_exist.jinja' %}
        {% endllmquery %}
        """)
    
    # Create environment with temp dir as template path
    env = create_environment(temp_template_dir)
    
    # Get the template
    template = env.get_template("nonexistent_include.jinja")
    
    # Rendering should raise an exception due to non-existent template
    with pytest.raises(Exception) as exc_info:
        template.render()
    
    # Verify the exception contains information about template not found
    assert "template" in str(exc_info.value).lower() and "not found" in str(exc_info.value).lower() 