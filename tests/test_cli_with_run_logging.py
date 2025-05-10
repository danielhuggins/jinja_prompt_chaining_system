import os
import pytest
import yaml
from unittest.mock import patch, Mock
from click.testing import CliRunner
from jinja_prompt_chaining_system.cli import main
from jinja_prompt_chaining_system.logger import RunLogger, LLMLogger

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
@patch('jinja_prompt_chaining_system.cli.RunLogger')
def test_cli_with_run_logging(mock_run_logger, mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Test CLI integration with run-based logging."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result
    mock_render.return_value = "Hello, World!"
    
    # Setup RunLogger mock
    run_logger_instance = Mock()
    run_id = "run_2023-01-01T12-00-00-123456"
    run_logger_instance.start_run.return_value = run_id
    
    llm_logger_instance = Mock()
    run_logger_instance.get_llm_logger.return_value = llm_logger_instance
    mock_run_logger.return_value = run_logger_instance
    
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
    assert result.output.strip() == "Hello, World!"
    
    # Verify RunLogger was initialized
    mock_run_logger.assert_called_once_with(log_dir)
    
    # Verify run was started with template metadata and context
    run_logger_instance.start_run.assert_called_once()
    call_args = run_logger_instance.start_run.call_args[1]
    
    # Check that metadata dict contains the expected keys/values
    expected_metadata = {"template": template_path, "context_file": context_path}
    assert all(item in call_args["metadata"].items() for item in expected_metadata.items())
    
    # Verify that context was loaded and passed to start_run
    assert "context" in call_args
    assert call_args["context"] == {"name": "World"}
    
    # Verify LLM logger for the run was obtained
    run_logger_instance.get_llm_logger.assert_called_once_with(run_id)
    
    # Verify the run was ended
    run_logger_instance.end_run.assert_called_once()

@patch('jinja_prompt_chaining_system.cli.render_template_sync')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_cli_with_real_run_logging(mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Integration test with the actual RunLogger implementation."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result
    mock_render.return_value = "Hello, World!"
    
    log_dir = tmp_path / "logs"
    
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
            "--context", context_path,
            "--logdir", str(log_dir)
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    
    # Verify the run directory structure was created
    run_dirs = list(log_dir.glob("run_*"))
    assert len(run_dirs) == 1
    
    run_dir = run_dirs[0]
    assert (run_dir / "llmcalls").exists()
    assert (run_dir / "metadata.yaml").exists()
    assert (run_dir / "context.yaml").exists()
    
    # Verify metadata contains template info
    with open(run_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    
    assert "timestamp" in metadata
    assert "template" in metadata
    assert "context_file" in metadata
    
    # Verify context contains the loaded context data
    with open(run_dir / "context.yaml") as f:
        context = yaml.safe_load(f)
    
    assert context == {"name": "World"} 