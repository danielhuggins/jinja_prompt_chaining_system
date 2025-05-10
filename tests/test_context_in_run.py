import os
import yaml
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system.logger import RunLogger
from jinja_prompt_chaining_system.cli import main
from click.testing import CliRunner

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_saves_context_in_run_dir(runner, tmp_path):
    """Test that the CLI saves the context in the run directory."""
    # Create a test template and context files
    template_path = tmp_path / "test.jinja"
    template_path.write_text("""
    {% llmquery model="gpt-4" %}
    Hello, {{ name }}!
    {% endllmquery %}
    """)
    
    context_path = tmp_path / "context.yaml"
    context_data = {
        "name": "World",
        "complex_data": {
            "nested": {
                "value": 42
            },
            "list": [1, 2, 3]
        }
    }
    
    with open(context_path, 'w') as f:
        yaml.dump(context_data, f)
    
    log_dir = tmp_path / "logs"
    
    # Mock response so we don't actually call the API
    with patch('jinja_prompt_chaining_system.parser.LLMClient') as mock_client:
        mock_instance = Mock()
        mock_instance.query.return_value = "Hello, World!"
        mock_client.return_value = mock_instance
        
        # Run the CLI
        runner.invoke(main, [
            str(template_path),
            "--context", str(context_path),
            "--logdir", str(log_dir)
        ], catch_exceptions=False)
    
    # Find the run directory
    run_dirs = list(log_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    
    # Verify context.yaml exists
    context_file = run_dir / "context.yaml"
    assert context_file.exists()
    
    # Verify content matches the original context
    with open(context_file) as f:
        saved_context = yaml.safe_load(f)
    
    assert saved_context == context_data
    
    # Verify metadata.yaml exists and refers to the context file
    metadata_file = run_dir / "metadata.yaml"
    assert metadata_file.exists()
    
    with open(metadata_file) as f:
        metadata = yaml.safe_load(f)
    
    assert "context_file" in metadata
    assert metadata["context_file"] == str(context_path)
    
    # Verify llmcalls directory exists
    llmcalls_dir = run_dir / "llmcalls"
    assert llmcalls_dir.exists()
    
    # Verify at least one log file exists in llmcalls
    log_files = list(llmcalls_dir.glob("*.log.yaml"))
    assert len(log_files) > 0
    
    # Check that the log file contains the rendered template text
    with open(log_files[0]) as f:
        log_data = yaml.safe_load(f)
    
    assert "request" in log_data
    assert "messages" in log_data["request"]
    assert len(log_data["request"]["messages"]) > 0
    assert "Hello, World!" in log_data["request"]["messages"][0]["content"] 