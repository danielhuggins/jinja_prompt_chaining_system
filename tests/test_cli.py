import os
import pytest
import asyncio
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from jinja_prompt_chaining_system.cli import main, render_template_sync

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

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
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

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_output(mock_logger, mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Test CLI with output file."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result directly
    mock_render.return_value = "Hello, World!"
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        output_path = os.path.join(os.getcwd(), "output", "output.txt")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        with open(context_file, "rb") as f:
            with open(context_path, "wb") as cf:
                cf.write(f.read())
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--out", output_path
        ], catch_exceptions=False)
    
        assert result.exit_code == 0
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            assert f.read().decode().strip() == "Hello, World!"

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
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

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
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

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
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

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_key_value_pairs(mock_logger, mock_llm_client, mock_render, runner, template_file, monkeypatch):
    """Test CLI with key-value pairs instead of context file."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, Alice!"
    mock_llm_client.return_value = client
    
    # Setup render mock to capture context
    context_capture = {}
    def mock_render_fn(template, context):
        nonlocal context_capture
        context_capture = context
        return f"Hello, {context.get('name', 'Unknown')}!"
    
    mock_render.side_effect = mock_render_fn
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        result = runner.invoke(main, [
            template_path,
            "name=Alice",
            "age=30"
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert "Hello, Alice!" in result.output
    assert context_capture == {"name": "Alice", "age": 30}

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_mixed_context_sources(mock_logger, mock_llm_client, mock_render, runner, template_file, tmp_path):
    """Test CLI with both key-value pairs and context file."""
    # Create a context file with some values
    context_file = tmp_path / "mixed_context.yaml"
    context_file.write_text("""
    name: Bob
    location: London
    preferences:
      color: blue
    """)
    
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, Alice from Paris!"
    mock_llm_client.return_value = client
    
    # Setup render mock to capture context
    context_capture = {}
    def mock_render_fn(template, context):
        nonlocal context_capture
        context_capture = context
        return f"Hello, {context.get('name', 'Unknown')} from {context.get('location', 'Nowhere')}!"
    
    mock_render.side_effect = mock_render_fn
    
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
        
        # Key-value pairs should override context file values
        result = runner.invoke(main, [
            template_path,
            "name=Alice",  # Override name from context file
            "location=Paris",  # Override location from context file
            "--context", context_path
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert "Hello, Alice from Paris!" in result.output
    
    # Verify that inline values overrode file values but other file values were preserved
    assert context_capture["name"] == "Alice"  # Overridden by inline
    assert context_capture["location"] == "Paris"  # Overridden by inline
    assert "preferences" in context_capture  # Preserved from file
    assert context_capture["preferences"]["color"] == "blue"  # Preserved from file

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_complex_key_values(mock_logger, mock_llm_client, mock_render, runner, template_file):
    """Test CLI with complex YAML values in key-value pairs."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Result with complex values"
    mock_llm_client.return_value = client
    
    # Setup render mock to capture context
    context_capture = {}
    def mock_render_fn(template, context):
        nonlocal context_capture
        context_capture = context
        return "Result with complex values"
    
    mock_render.side_effect = mock_render_fn
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        # Test with various YAML types
        result = runner.invoke(main, [
            template_path,
            "string_value=hello",
            "number_value=42",
            "boolean_value=true",
            "null_value=null",
            "list_value=[1, 2, 3]",
            "dict_value={'key': 'value', 'nested': {'data': 123}}"
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    
    # Verify the complex values were parsed correctly
    assert context_capture["string_value"] == "hello"
    assert context_capture["number_value"] == 42
    assert context_capture["boolean_value"] is True
    assert context_capture["null_value"] is None
    assert context_capture["list_value"] == [1, 2, 3]
    assert context_capture["dict_value"] == {"key": "value", "nested": {"data": 123}}

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_no_context(mock_logger, mock_llm_client, mock_render, runner, template_file):
    """Test CLI with no context provided at all."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, Unknown!"
    mock_llm_client.return_value = client
    
    # Setup render mock to verify empty context
    context_capture = None
    def mock_render_fn(template, context):
        nonlocal context_capture
        context_capture = context
        return "Hello, Unknown!"
    
    mock_render.side_effect = mock_render_fn
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        result = runner.invoke(main, [
            template_path
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert "Hello, Unknown!" in result.output
    assert context_capture == {}  # Empty dictionary for context

def test_cli_with_invalid_key_value(runner, template_file):
    """Test CLI with invalid key-value pair format."""
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        
        # Copy files to isolated filesystem
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        with open(template_file, "rb") as f:
            with open(template_path, "wb") as tf:
                tf.write(f.read())
        
        # Test with invalid key-value format (missing equals sign)
        result = runner.invoke(main, [
            template_path,
            "invalid_format"
        ])
    
    assert result.exit_code != 0
    assert "Invalid key-value pair" in result.output 