import os
import pytest
import yaml
from unittest.mock import Mock, MagicMock, patch
import asyncio
import socket
import gc
import warnings

from src.jinja_prompt_chaining_system.llm import LLMClient
from src.jinja_prompt_chaining_system.logger import LLMLogger
from jinja_prompt_chaining_system.parallel_integration import ParallelLLMQueryExtension

# Mock classes
class MockLLM:
    """Mock LLM client for testing."""
    
    def __init__(self, response="Test response", is_streaming=False):
        self.response = response
        self.is_streaming = is_streaming
        
    def query(self, prompt, params=None, stream=False):
        """Mock query method."""
        return self.response
        
    async def query_async(self, prompt, params=None, stream=False):
        """Mock async query method."""
        return self.response

@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MockLLM()

@pytest.fixture
def mock_streaming_llm():
    """Create a mock LLM client with streaming support."""
    return MockLLM(is_streaming=True)

# Log directory utilities
@pytest.fixture
def log_dir(tmp_path):
    """Create a temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir

# Fixture to filter out ResourceWarnings from socket
@pytest.fixture(autouse=True)
def ignore_socket_warnings():
    """Filter out ResourceWarnings related to unclosed sockets."""
    # Store original filters
    original_filters = warnings.filters.copy()
    
    # Add filter to ignore ResourceWarning for sockets
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")
    
    # Run the test
    yield
    
    # After test, restore original filters and run garbage collection
    warnings.filters = original_filters
    gc.collect()

# Fixture to close any asyncio event loops after each test
@pytest.fixture(autouse=True)
def close_event_loops():
    """Fixture to clean up event loops after each test, preventing ResourceWarnings."""
    # Add a filter to ignore ResourceWarning for unclosed event loops
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed event loop.*")
    
    # Run the test
    yield
    
    # After the test completes, try to close any event loops
    try:
        # Use the safer get_running_loop() approach first
        try:
            # See if there's a running loop we can access
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                loop.close()
        except RuntimeError:
            # No running loop, try to create/get a new one and close it
            try:
                # Create a new event loop (since Python 3.10+ get_event_loop() produces warnings)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.close()
                # Reset the event loop policy for the next test
                asyncio.set_event_loop(None)
            except Exception:
                # If all else fails, just ignore
                pass
    except Exception as e:
        # If there's an error, just log it but don't fail the test
        print(f"Warning: Error cleaning up event loop: {e}")

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    with patch('src.jinja_prompt_chaining_system.llm.LLMClient') as mock:
        client = Mock()
        client.query.return_value = "Mock LLM response"
        client.query_async.return_value = "Mock LLM response"
        mock.return_value = client
        yield client

@pytest.fixture
def mock_parallel_llm_extension():
    """Mock the ParallelLLMQueryExtension for testing."""
    # Save the original class
    original_init = ParallelLLMQueryExtension.__init__
    original_parse = ParallelLLMQueryExtension.parse
    original_call = ParallelLLMQueryExtension.__call__
    original_global_llmquery = ParallelLLMQueryExtension.parallel_global_llmquery
    
    # Mock the initialization
    def mock_init(self, environment):
        original_init(self, environment)
        # Create a mock client for testing
        client = Mock()
        client.query.return_value = "Mock LLM response"
        client.query_async.return_value = "Mock LLM response"
        self.llm_client = client
    
    # Mock the parse method to simply wrap content
    def mock_parse(self, parser):
        return original_parse(self, parser)
    
    # Mock the call method to return fixed content
    def mock_call(self, *args, **kwargs):
        return "Mock LLM response"
    
    # Mock the global llmquery function
    def mock_global_llmquery(self, prompt, **kwargs):
        return "Mock LLM response"
    
    # Apply the mocks
    ParallelLLMQueryExtension.__init__ = mock_init
    ParallelLLMQueryExtension.__call__ = mock_call
    ParallelLLMQueryExtension.parallel_global_llmquery = mock_global_llmquery
    
    yield
    
    # Restore the original methods
    ParallelLLMQueryExtension.__init__ = original_init
    ParallelLLMQueryExtension.parse = original_parse
    ParallelLLMQueryExtension.__call__ = original_call
    ParallelLLMQueryExtension.parallel_global_llmquery = original_global_llmquery 