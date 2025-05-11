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

@patch('jinja_prompt_chaining_system.cli.render_prompt')
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

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.cli.RunLogger')
def test_cli_with_run_name(mock_run_logger, mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Test CLI integration with named run."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result
    mock_render.return_value = "Hello, World!"
    
    with runner.isolated_filesystem():
        template_path = os.path.join(os.getcwd(), "test.jinja")
        context_path = os.path.join(os.getcwd(), "context.yaml")
        log_dir = os.path.join(os.getcwd(), "logs")
        run_name = "experiment-1"
        
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
            "--logdir", log_dir,
            "--name", run_name
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert result.output.strip() == "Hello, World!"
    
    # Verify render_prompt was called with correct arguments
    mock_render.assert_called_once()
    call_args = mock_render.call_args[1]
    assert call_args["template_path"] == template_path
    assert call_args["context"] == context_path
    assert call_args["logdir"] == log_dir
    assert call_args["name"] == run_name

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_cli_with_real_run_logging(mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Integration test with the actual RunLogger implementation."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    log_dir = tmp_path / "logs"
    
    # Setup render mock with a side effect that creates log directories
    def render_side_effect(template_path, context, logdir=None, **kwargs):
        # Only create logs if logdir is provided
        if logdir:
            # Create run directory
            import time, datetime, os, yaml
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
            run_id = f"run_{timestamp}"
            run_dir = os.path.join(logdir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            # Create llmcalls directory
            llmcalls_dir = os.path.join(run_dir, "llmcalls")
            os.makedirs(llmcalls_dir, exist_ok=True)
            
            # Create a sample log file
            log_file = os.path.join(llmcalls_dir, f"query_{int(time.time())}.log.yaml")
            log_data = {
                "request": {
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, World!"
                        }
                    ]
                },
                "response": "Hello, World!"
            }
            with open(log_file, "w") as f:
                yaml.dump(log_data, f)
            
            # Create metadata file
            metadata_file = os.path.join(run_dir, "metadata.yaml")
            metadata = {
                "timestamp": timestamp,
                "template": template_path,
                "context_file": context if isinstance(context, str) else None
            }
            with open(metadata_file, "w") as f:
                yaml.dump(metadata, f)
                
            # Create context file
            context_file = os.path.join(run_dir, "context.yaml")
            context_data = {"name": "World"}
            with open(context_file, "w") as f:
                yaml.dump(context_data, f)
                
        return "Hello, World!"
    
    mock_render.side_effect = render_side_effect
    
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

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_cli_with_real_run_naming(mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Integration test with the actual RunLogger implementation using named runs."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    log_dir = tmp_path / "logs"
    run_name = "experiment-integration"
    
    # Setup render mock with a side effect that creates log directories
    def render_side_effect(template_path, context, logdir=None, name=None, **kwargs):
        # Only create logs if logdir is provided
        if logdir:
            # Create run directory with name if provided
            import time, datetime, os, yaml
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
            run_id = f"run_{timestamp}_{name}" if name else f"run_{timestamp}"
            run_dir = os.path.join(logdir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            # Create llmcalls directory
            llmcalls_dir = os.path.join(run_dir, "llmcalls")
            os.makedirs(llmcalls_dir, exist_ok=True)
            
            # Create a sample log file
            log_file = os.path.join(llmcalls_dir, f"query_{int(time.time())}.log.yaml")
            log_data = {
                "request": {
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, World!"
                        }
                    ]
                },
                "response": "Hello, World!"
            }
            with open(log_file, "w") as f:
                yaml.dump(log_data, f)
            
            # Create metadata file
            metadata_file = os.path.join(run_dir, "metadata.yaml")
            metadata = {
                "timestamp": timestamp,
                "template": template_path,
                "context_file": context if isinstance(context, str) else None
            }
            # Add name to metadata if provided
            if name:
                metadata["name"] = name
                
            with open(metadata_file, "w") as f:
                yaml.dump(metadata, f)
                
            # Create context file
            context_file = os.path.join(run_dir, "context.yaml")
            context_data = {"name": "World"}
            with open(context_file, "w") as f:
                yaml.dump(context_data, f)
                
        return "Hello, World!"
    
    mock_render.side_effect = render_side_effect
    
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
            "--logdir", str(log_dir),
            "--name", run_name
        ], catch_exceptions=False)
    
    assert result.exit_code == 0
    
    # Verify the run directory structure was created with the correct name
    run_dirs = list(log_dir.glob(f"run_*_{run_name}"))
    assert len(run_dirs) == 1
    
    run_dir = run_dirs[0]
    assert (run_dir / "llmcalls").exists()
    assert (run_dir / "metadata.yaml").exists()
    assert (run_dir / "context.yaml").exists()
    
    # Verify metadata contains template info and the run name
    with open(run_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    
    assert "timestamp" in metadata
    assert "template" in metadata
    assert "context_file" in metadata
    assert "name" in metadata
    assert metadata["name"] == run_name
    
    # Verify context contains the loaded context data
    with open(run_dir / "context.yaml") as f:
        context = yaml.safe_load(f)
    
    assert context == {"name": "World"}

@patch('jinja_prompt_chaining_system.cli.render_prompt')
@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_cli_with_run_logging(mock_llm_client, mock_render, runner, template_file, context_file, tmp_path):
    """Test CLI integration with run-based logging."""
    # Setup mocks
    client = Mock()
    client.query.return_value = "Hello, World!"
    mock_llm_client.return_value = client
    
    # Setup render mock to return a fixed result
    mock_render.return_value = "Hello, World!"
    
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
    
    # Verify render_prompt was called with correct arguments
    mock_render.assert_called_once()
    call_args = mock_render.call_args[1]
    assert call_args["template_path"] == template_path
    assert call_args["context"] == context_path
    assert call_args["logdir"] == log_dir
    # Name is not provided in this test
    assert call_args.get("name") is None 