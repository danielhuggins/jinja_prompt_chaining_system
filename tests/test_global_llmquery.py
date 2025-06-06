import os
import pytest
import yaml
from unittest.mock import patch, Mock, AsyncMock
from jinja2 import Environment, FileSystemLoader
from jinja_prompt_chaining_system.parser import LLMQueryExtension
from jinja_prompt_chaining_system.logger import RunLogger, LLMLogger

class MockLLM:
    """Mock LLM client for testing."""
    
    def __init__(self, response="Test response"):
        self.response = response
        
    def query(self, prompt, params, stream=False):
        """Mock query method."""
        return self.response
        
    async def query_async(self, prompt, params, stream=False):
        """Mock async query method."""
        return self.response

@pytest.fixture
def mock_env():
    """Create a Jinja environment with a mock LLM."""
    # Create a new environment
    env = Environment(
        loader=FileSystemLoader("."),
        extensions=[LLMQueryExtension],
        enable_async=True,  # Enable async mode
        autoescape=False  # Disable HTML escaping by default
    )
    
    # Get the extension
    extension = env.extensions['jinja_prompt_chaining_system.parser.LLMQueryExtension']
    
    # Replace the LLM client with our mock
    extension.llm_client = MockLLM()
    
    return env, extension

def test_global_llmquery_function_basic(mock_env):
    """Test basic functionality of the global llmquery function."""
    env, extension = mock_env
    extension.llm_client = MockLLM("Test response")
    
    # Create a test template
    template_str = '{{ llmquery(prompt="Test prompt", model="gpt-4") }}'
    template = env.from_string(template_str)
    
    # Render the template
    result = template.render()
    
    # Verify result
    assert result == "Test response"

def test_global_llmquery_with_variables(mock_env):
    """Test using the global llmquery function with variables in the prompt."""
    env, extension = mock_env
    extension.llm_client = MockLLM("Hello, World!")
    
    # Create a template with variables
    template_str = '''
    {% set name = "World" %}
    {{ llmquery(prompt="Hello, " + name + "!", model="gpt-4") }}
    '''
    template = env.from_string(template_str)
    
    # Render the template
    result = template.render().strip()
    
    # Verify result
    assert result == "Hello, World!"

def test_global_llmquery_with_context(mock_env):
    """Test using the global llmquery function with context variables."""
    env, extension = mock_env
    extension.llm_client = MockLLM("Hello, Test User!")
    
    # Create a template using context
    template_str = '{{ llmquery(prompt="Hello, " + user + "!", model="gpt-4") }}'
    template = env.from_string(template_str)
    
    # Render with context
    result = template.render(user="Test User")
    
    # Verify result
    assert result == "Hello, Test User!"

def test_global_llmquery_with_multiline_prompt(mock_env):
    """Test using the global llmquery function with a multiline prompt."""
    env, extension = mock_env
    extension.llm_client = MockLLM("Multiline response")
    
    # Create a template with a multiline prompt
    template_str = '''
    {{ llmquery(
        prompt="""This is a
        multiline prompt
        with multiple lines.""",
        model="gpt-4"
    ) }}
    '''
    template = env.from_string(template_str)
    
    # Render the template
    result = template.render().strip()
    
    # Verify result
    assert result == "Multiline response"

def test_global_llmquery_with_logging(mock_env, tmp_path):
    """Test that the global llmquery function logs correctly."""
    env, extension = mock_env
    extension.llm_client = MockLLM("Logged response")
    
    # Setup logging directory
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup RunLogger
    run_logger = RunLogger(log_dir)
    run_id = run_logger.start_run(metadata={"test": True}, context={"test": True})
    llm_logger = run_logger.get_llm_logger(run_id)
    
    # Set the logger on the extension
    extension.logger = llm_logger
    extension.set_template_name("test_template.jinja")
    
    # Create a test template
    template_str = '{{ llmquery(prompt="Test prompt for logging", model="gpt-4") }}'
    template = env.from_string(template_str)
    
    # Render the template
    result = template.render()
    
    # Verify result
    assert result == "Logged response"
    
    # Verify log file was created in the llmcalls directory
    llmcalls_dir = os.path.join(log_dir, run_id, "llmcalls")
    assert os.path.exists(llmcalls_dir)
    
    # Should have at least one log file
    log_files = [f for f in os.listdir(llmcalls_dir) if f.endswith(".log.yaml")]
    assert len(log_files) > 0
    
    # Verify log content
    with open(os.path.join(llmcalls_dir, log_files[0]), 'r') as f:
        log_data = yaml.safe_load(f)
    
    assert log_data["request"]["model"] == "gpt-4"
    assert "Test prompt for logging" in log_data["request"]["messages"][0]["content"]
    assert "response" in log_data

@pytest.mark.asyncio
async def test_global_llmquery_async(mock_env):
    """Test the global llmquery function in async context."""
    env, extension = mock_env
    extension.llm_client = MockLLM("Async response")
    
    # Create a test template
    template_str = '{{ llmquery(prompt="Async test prompt", model="gpt-4") }}'
    template = env.from_string(template_str)
    
    # Render the template asynchronously
    result = await template.render_async()
    
    # Verify result
    assert result == "Async response" 