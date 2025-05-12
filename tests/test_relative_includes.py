import os
import pytest
import tempfile
from unittest.mock import patch, Mock
from click.testing import CliRunner
from jinja_prompt_chaining_system.cli import main
from jinja_prompt_chaining_system import create_environment
from jinja2 import Environment, FileSystemLoader

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def relative_template_dir():
    """Create a directory structure for testing relative includes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a more complex directory structure
        main_dir = tmpdir
        partials_dir = os.path.join(main_dir, "partials")
        sections_dir = os.path.join(partials_dir, "sections")
        shared_dir = os.path.join(main_dir, "shared")
        
        # Create all directories
        os.makedirs(partials_dir, exist_ok=True)
        os.makedirs(sections_dir, exist_ok=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Create main template with relative includes
        with open(os.path.join(main_dir, "main.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Main template for {{ topic }}
            
            {% include './partials/header.jinja' %}
            
            Main content here.
            
            {% include './partials/footer.jinja' %}
            {% endllmquery %}
            """)
        
        # Create header template
        with open(os.path.join(partials_dir, "header.jinja"), "w") as f:
            f.write("""
            # {{ topic | upper }} DOCUMENT
            Date: {{ current_date }}
            
            {% include './sections/intro.jinja' %}
            """)
        
        # Create footer template with parent directory reference
        with open(os.path.join(partials_dir, "footer.jinja"), "w") as f:
            f.write("""
            ---
            {% include '../shared/common.jinja' %}
            """)
        
        # Create intro template with parent directory reference
        with open(os.path.join(sections_dir, "intro.jinja"), "w") as f:
            f.write("""
            Introduction to {{ topic }}.
            
            {% include '../../shared/common.jinja' %}
            """)
        
        # Create common template
        with open(os.path.join(shared_dir, "common.jinja"), "w") as f:
            f.write("""
            Common footer text for all templates.
            """)
        
        # Create a template that combines both relative and absolute includes
        with open(os.path.join(main_dir, "mixed_includes.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Template with mixed includes
            
            Standard include: {% include 'partials/header.jinja' %}
            Relative include: {% include './partials/footer.jinja' %}
            {% endllmquery %}
            """)
        
        # Create context file
        with open(os.path.join(main_dir, "context.yaml"), "w") as f:
            f.write("""
            topic: "Relative Includes"
            current_date: "2023-08-15"
            """)
        
        yield tmpdir

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_relative_includes_basic(mock_llm_client, runner, relative_template_dir):
    """Test basic relative include functionality."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response with relative includes"
    mock_llm_client.return_value = client_instance
    
    # Run CLI command
    template_path = os.path.join(relative_template_dir, "main.jinja")
    context_path = os.path.join(relative_template_dir, "context.yaml")
    
    result = runner.invoke(main, [
        template_path,
        "--context", context_path
    ], catch_exceptions=False)
    
    # Check result
    assert result.exit_code == 0
    assert "Mock LLM response with relative includes" in result.output
    
    # Check that the LLM was called with content from all included templates
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Main template for Relative Includes" in prompt
    assert "# RELATIVE INCLUDES DOCUMENT" in prompt
    assert "Introduction to Relative Includes" in prompt
    assert "Common footer text for all templates" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_mixed_includes(mock_llm_client, runner, relative_template_dir):
    """Test a mixture of standard and relative includes."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response with mixed includes"
    mock_llm_client.return_value = client_instance
    
    # Run CLI command
    template_path = os.path.join(relative_template_dir, "mixed_includes.jinja")
    context_path = os.path.join(relative_template_dir, "context.yaml")
    
    result = runner.invoke(main, [
        template_path,
        "--context", context_path
    ], catch_exceptions=False)
    
    # Check result
    assert result.exit_code == 0
    assert "Mock LLM response with mixed includes" in result.output
    
    # Check that both standard and relative includes work
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Template with mixed includes" in prompt
    assert "# RELATIVE INCLUDES DOCUMENT" in prompt
    assert "Common footer text for all templates" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nested_relative_includes(mock_llm_client, relative_template_dir):
    """Test nested relative includes using the API directly."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response with nested includes"
    mock_llm_client.return_value = client_instance
    
    # Create environment with relative includes
    env = create_environment(relative_template_dir)
    
    # Load and render template
    template = env.get_template("main.jinja")
    
    # Prepare context
    context = {
        "topic": "Relative Includes API Test",
        "current_date": "2023-08-16"
    }
    
    # Render template
    result = template.render(**context)
    
    # Check result
    assert "Mock LLM response with nested includes" in result
    
    # Check that nested includes were processed
    client_instance.query.assert_called_once()
    prompt = client_instance.query.call_args[0][0]
    assert "Main template for Relative Includes API Test" in prompt
    assert "# RELATIVE INCLUDES API TEST DOCUMENT" in prompt
    assert "Introduction to Relative Includes API Test" in prompt
    assert "Common footer text for all templates" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nonexistent_relative_include(mock_llm_client, relative_template_dir):
    """Test error handling for non-existent relative include paths."""
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "This shouldn't be returned"
    mock_llm_client.return_value = client_instance
    
    # Create a template with a non-existent relative include
    nonexistent_path = os.path.join(relative_template_dir, "nonexistent.jinja")
    with open(nonexistent_path, "w") as f:
        f.write("""
        {% llmquery model="gpt-4" %}
        This template includes a non-existent file:
        {% include './does_not_exist.jinja' %}
        {% endllmquery %}
        """)
    
    # Create environment with relative includes
    env = create_environment(relative_template_dir)
    
    # Load template
    template = env.get_template("nonexistent.jinja")
    
    # Rendering should raise an exception
    with pytest.raises(Exception) as exc_info:
        template.render(topic="Error Test")
    
    # Verify the error message mentions the relative path
    assert "does_not_exist.jinja" in str(exc_info.value)
    
    # Verify that LLM client was not called
    client_instance.query.assert_not_called() 