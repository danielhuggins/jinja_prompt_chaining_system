import os
import yaml
import pytest
import re
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system.logger import LLMLogger, RunLogger

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"

@pytest.fixture
def run_logger(log_dir):
    return RunLogger(str(log_dir))

def test_run_logger_initialization(log_dir):
    """Test run logger initialization creates log directory."""
    logger = RunLogger(str(log_dir))
    assert os.path.exists(log_dir)

def test_run_creation(run_logger, log_dir):
    """Test that a run creates a timestamped directory."""
    # Start a new run
    run_id = run_logger.start_run()
    
    # Verify the run directory was created with timestamp format
    run_dirs = list(log_dir.glob("run_*"))
    assert len(run_dirs) == 1
    
    # Check the run ID format (should be timestamp-based)
    assert re.match(r'run_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', run_id)
    
    # Verify that an llmcalls directory exists in the run directory
    llmcalls_dir = log_dir / run_id / "llmcalls"
    assert os.path.exists(llmcalls_dir)

def test_get_llm_logger_for_run(run_logger, log_dir):
    """Test that we can get an LLM logger for a specific run."""
    run_id = run_logger.start_run()
    llm_logger = run_logger.get_llm_logger(run_id)
    
    # Verify the logger is an LLMLogger instance
    assert isinstance(llm_logger, LLMLogger)
    
    # Verify the logger uses the correct path
    expected_log_dir = os.path.join(str(log_dir), run_id, "llmcalls")
    assert llm_logger.log_dir == expected_log_dir

def test_run_metadata(run_logger, log_dir):
    """Test that run metadata is correctly stored."""
    # Start a new run with metadata
    metadata = {
        "user": "test_user",
        "template": "test_template.jinja",
        "context_file": "test_context.yaml"
    }
    run_id = run_logger.start_run(metadata=metadata)
    
    # Verify the metadata file exists
    metadata_file = log_dir / run_id / "metadata.yaml"
    assert os.path.exists(metadata_file)
    
    # Check the metadata content
    with open(metadata_file) as f:
        stored_metadata = yaml.safe_load(f)
    
    assert "timestamp" in stored_metadata
    assert stored_metadata["user"] == "test_user"
    assert stored_metadata["template"] == "test_template.jinja"
    assert stored_metadata["context_file"] == "test_context.yaml"

def test_current_run(run_logger):
    """Test that the current run is tracked correctly."""
    # Initially there should be no current run
    assert run_logger.current_run_id is None
    
    # Start a run
    run_id = run_logger.start_run()
    assert run_logger.current_run_id == run_id
    
    # End the run
    run_logger.end_run()
    assert run_logger.current_run_id is None

def test_use_current_run_logger(run_logger, log_dir):
    """Test that we can use the logger for the current run."""
    # Start a run
    run_id = run_logger.start_run()
    
    # Get the logger for the current run
    llm_logger = run_logger.get_llm_logger()
    
    # Log a request
    template_name = "test_template"
    request = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    log_path = llm_logger.log_request(template_name, request)
    
    # Verify the log file was created in the correct location
    expected_location = os.path.join(str(log_dir), run_id, "llmcalls")
    assert log_path.startswith(expected_location)
    assert os.path.exists(log_path)

def test_multiple_runs(run_logger, log_dir):
    """Test handling multiple runs in sequence."""
    # Start first run
    run_id1 = run_logger.start_run({"name": "run1"})
    llm_logger1 = run_logger.get_llm_logger()
    
    # Log something in first run
    template_name = "test_template"
    request = {"model": "model1", "messages": [{"role": "user", "content": "Request 1"}]}
    log_path1 = llm_logger1.log_request(template_name, request)
    
    # End first run
    run_logger.end_run()
    
    # Start second run
    run_id2 = run_logger.start_run({"name": "run2"})
    llm_logger2 = run_logger.get_llm_logger()
    
    # Log something in second run
    request = {"model": "model2", "messages": [{"role": "user", "content": "Request 2"}]}
    log_path2 = llm_logger2.log_request(template_name, request)
    
    # Both log files should exist and be in their respective run directories
    assert os.path.exists(log_path1)
    assert os.path.exists(log_path2)
    assert run_id1 in log_path1
    assert run_id2 in log_path2
    assert run_id1 != run_id2

def test_get_specific_run_logger(run_logger, log_dir):
    """Test getting a logger for a specific run ID."""
    # Create two different run IDs manually
    run_id1 = "run_2023-01-01T12-00-00-000001"
    run_id2 = "run_2023-01-01T12-00-01-000002"
    
    # Create the directory structure
    os.makedirs(os.path.join(str(log_dir), run_id1, "llmcalls"), exist_ok=True)
    os.makedirs(os.path.join(str(log_dir), run_id2, "llmcalls"), exist_ok=True)
    
    # Store the run IDs in the logger's instance variables
    run_logger.run_loggers[run_id1] = LLMLogger(os.path.join(str(log_dir), run_id1, "llmcalls"))
    run_logger.current_run_id = run_id2
    
    # Get logger for the first run (not the current one)
    llm_logger1 = run_logger.get_llm_logger(run_id1)
    
    # Log something using this logger
    template_name = "test_template"
    request = {"model": "model1", "messages": [{"role": "user", "content": "Request 1"}]}
    log_path1 = llm_logger1.log_request(template_name, request)
    
    # Verify it went to the correct run directory
    assert run_id1 in log_path1
    assert run_id2 not in log_path1

@patch('jinja_prompt_chaining_system.logger.datetime')
def test_run_id_format(mock_datetime, log_dir):
    """Test that run IDs have the correct format."""
    # Mock datetime to return a fixed value
    mock_date = datetime(2023, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    mock_datetime.now.return_value = mock_date
    
    # Create a run logger and start a run
    run_logger = RunLogger(str(log_dir))
    run_id = run_logger.start_run()
    
    # Check the run ID format
    expected_run_id = "run_2023-01-01T12-00-00-123456"
    assert run_id == expected_run_id

def test_save_context_in_run(run_logger, log_dir):
    """Test that the context is saved in the run directory."""
    # Create a test context
    context = {
        "name": "Test User",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 100
        },
        "prompts": ["Hello, world!", "How are you?"]
    }
    
    # Start a run with context
    run_id = run_logger.start_run(context=context)
    
    # Verify the context file exists
    context_file = log_dir / run_id / "context.yaml"
    assert os.path.exists(context_file)
    
    # Check the context content
    with open(context_file) as f:
        stored_context = yaml.safe_load(f)
    
    # Verify the stored context matches the original
    assert stored_context == context

def test_save_context_not_overwritten(run_logger, log_dir):
    """Test that the context isn't overwritten by metadata."""
    # Create a test context with a 'timestamp' field
    context = {
        "timestamp": "should-not-be-overwritten",
        "data": "test data"
    }
    
    # Create metadata with the same field
    metadata = {
        "template": "test.jinja"
    }
    
    # Start a run with both context and metadata
    run_id = run_logger.start_run(context=context, metadata=metadata)
    
    # Verify both files exist
    context_file = log_dir / run_id / "context.yaml"
    metadata_file = log_dir / run_id / "metadata.yaml"
    assert os.path.exists(context_file)
    assert os.path.exists(metadata_file)
    
    # Check the context content
    with open(context_file) as f:
        stored_context = yaml.safe_load(f)
    
    # Check the metadata content
    with open(metadata_file) as f:
        stored_metadata = yaml.safe_load(f)
    
    # Verify context timestamp is preserved
    assert stored_context["timestamp"] == "should-not-be-overwritten"
    # Verify metadata has a timestamp field but it's a real timestamp (not "metadata-timestamp")
    assert "timestamp" in stored_metadata
    assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+00:00', stored_metadata["timestamp"])

def test_context_None_still_creates_empty_context_file(run_logger, log_dir):
    """Test that even when context is None, an empty context file is created."""
    # Start a run with no context
    run_id = run_logger.start_run()
    
    # Verify the context file exists but is empty (or contains empty data structure)
    context_file = log_dir / run_id / "context.yaml"
    assert os.path.exists(context_file)
    
    # Check the context content
    with open(context_file) as f:
        stored_context = yaml.safe_load(f)
    
    # Should be an empty dict, not None
    assert stored_context == {} 