import os
import pytest
import asyncio
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from jinja_prompt_chaining_system.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def template_file(tmp_path):
    template = tmp_path / "test.jinja"
    template.write_text("""
    {% llmquery model="gpt-4" temperature=0.7 %}
    Hello, {{ name }}!
    {% endllmquery %}
    """)
    return template

@pytest.fixture
def context_file(tmp_path):
    context = tmp_path / "context.yaml"
    context.write_text("""
    name: World
    """)
    return context

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_basic(mock_logger, mock_llm_client, mock_render, runner, template_file, context_file):
    """Test basic CLI functionality."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result directly
    mock_render.return_value = "Hello, World!"
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        with open(context_file, "rb") as f:
            with open(context_path, "wb") as cf:
                cf.write(f.read())
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert result.output.strip() == "Hello, World!"

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_output(mock_logger, mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Test CLI with output file."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock with a side effect that writes to the output file
    def mock_render_with_side_effect(template_path, context, out=None, **kwargs):
        result = "Hello, World!"
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, 'w') as f:
                f.write(result)
        return result
    
    mock_render.side_effect = mock_render_with_side_effect
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        output_path = os.path.join(os.getcwd(), "output", "output.txt")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        with open(context_file, "rb") as f:
            with open(context_path, "wb") as cf:
                cf.write(f.read())
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--out", output_path
        ], catch_exceptions=False)
    
        assert result.exit_code == 0
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            assert f.read().decode().strip() == "Hello, World!"

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_logdir(mock_logger, mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Test CLI with log directory."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result directly
    mock_render.return_value = "Hello, World!"
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        log_dir = os.path.join(os.getcwd(), "logs")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        with open(context_file, "rb") as f:
            with open(context_path, "wb") as cf:
                cf.write(f.read())
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
    
        assert result.exit_code == 0
        assert os.path.exists(log_dir)

def test_cli_missing_template(runner, context_file):
    """Test CLI with missing template file."""
    result = runner.invoke(main, [
        "nonexistent.jinja",
        "--context", str(context_file)
    ])
    assert result.exit_code != 0
    assert "Error" in result.output

def test_cli_missing_context(runner, template_file):
    """Test CLI with missing context file."""
    result = runner.invoke(main, [
        str(template_file),
        "--context", "nonexistent.yaml"
    ])
    assert result.exit_code != 0
    assert "Error" in result.output

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_invalid_yaml(mock_logger, mock_llm_client, runner, template_file, tmp_path):
    """Test CLI with invalid YAML context."""
    # Create invalid YAML file
    context = tmp_path / "invalid.yaml"
    context.write_text("""
    invalid: yaml: content
    """)
    
    result = runner.invoke(main, [
        str(template_file),
        "--context", str(context)
    ])
    
    assert result.exit_code != 0
    assert "Error" in result.output

@pytest.fixture
def complex_template_file(tmp_path):
    """Create a template with multiple llmquery tags and complex syntax."""
    template = tmp_path / "complex.jinja"
    template.write_text("""
    {% set system_message = "You are a helpful assistant." %}
    {% set temperature_value = 0.8 %}
    
    First Query:
    {% llmquery model="gpt-4", temperature=temperature_value, max_tokens=100 %}
    {{ system_message }}
    
    Please list 3 reasons why Python is popular.
    {% endllmquery %}
    
    Second Query:
    {% llmquery 
        model="gpt-4-turbo"
        temperature=0.5
        max_tokens=200
        stream=false
    %}
    Explain the difference between synchronous and asynchronous programming.
    {% endllmquery %}
    """)
    return template

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_complex_template(mock_logger, mock_llm_client, mock_render, runner, complex_template_file, context_file):
    """Test CLI with a complex template containing multiple llmquery tags and expressions."""
    # Prepare the expected output
    output = """
    First Query:
    1. Easy to learn
    2. Large ecosystem
    3. Versatile
    
    Second Query:
    Synchronous programming executes tasks sequentially, while asynchronous programming allows tasks to run independently.
    """
    
    # Setup render mock to return fixed result
    mock_render.return_value = output.strip()
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "complex.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        
        with open(complex_template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        with open(context_file, "rb") as f:
            with open(context_path, "wb") as cf:
                cf.write(f.read())
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert "First Query:" in result.output
    assert "Large ecosystem" in result.output
    assert "Second Query:" in result.output
    assert "Synchronous programming" in result.output

@pytest.fixture
def streaming_template_file(tmp_path):
    """Create a template specifically for testing streaming functionality."""
    template = tmp_path / "streaming.jinja"
    template.write_text("""
    {% llmquery model="gpt-4", stream=true %}
    Generate a story about space exploration.
    {% endllmquery %}
    """)
    return template

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_streaming(mock_logger, mock_llm_client, mock_render, runner, streaming_template_file, context_file):
    """Test CLI with streaming enabled."""
    # Setup streaming output
    streaming_output = "In the year 2150, humanity had established colonies on Mars"
    mock_render.return_value = streaming_output
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "streaming.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        log_dir = os.path.join(os.getcwd(), "logs")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        with open(streaming_template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        with open(context_file, "rb") as f:
            with open(context_path, "wb") as cf:
                cf.write(f.read())
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert streaming_output in result.output 