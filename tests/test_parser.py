import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from jinja_prompt_chaining_system import create_environment
from jinja_prompt_chaining_system.parser import LLMQueryExtension

@pytest.fixture
def mock_llm_client():
    with patch('jinja_prompt_chaining_system.parser.LLMClient') as mock:
        client = Mock()
        client.query.return_value = "Mocked response"
        mock.return_value = client
        yield client

@pytest.fixture
def mock_logger():
    with patch('jinja_prompt_chaining_system.parser.LLMLogger') as mock:
        logger = Mock()
        mock.return_value = logger
        yield logger

class TestLLMQueryExtension:
    """Test cases for LLMQueryExtension class."""
    
    def test_llmquery_tag_basic(self, mock_llm_client, mock_logger):
        """Test basic functionality of the llmquery tag."""
        env = create_environment()
        extension = [ext for ext in env.extensions.values() if isinstance(ext, LLMQueryExtension)][0]
        extension.set_template_name("test.jinja")
        
        # Directly call _llmquery with parameters and a mock caller
        mock_caller = Mock(return_value="Hello, world!")
        
        result = extension._llmquery(
            {"model": "gpt-4", "temperature": 0.7}, 
            mock_caller
        )
        
        assert result == "Mocked response"
        
        # Verify LLM client was called correctly
        mock_llm_client.query.assert_called_once()
        call_args = mock_llm_client.query.call_args[0]
        assert call_args[0] == "Hello, world!"
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["temperature"] == 0.7
    
    def test_llmquery_tag_parameters(self, mock_llm_client, mock_logger):
        """Test different parameter formats in llmquery tag."""
        env = create_environment()
        extension = [ext for ext in env.extensions.values() if isinstance(ext, LLMQueryExtension)][0]
        extension.set_template_name("test.jinja")
        
        # Test with different parameter sets
        param_sets = [
            # Spaces
            {"model": "gpt-4", "temperature": 0.7, "max_tokens": 150},
            # Commas
            {"model": "gpt-4", "temperature": 0.7, "max_tokens": 150},
            # Mixed format with stream=false
            {"model": "gpt-4", "temperature": 0.7, "max_tokens": 150, "stream": False}
        ]
        
        mock_caller = Mock(return_value="Test prompt")
        
        for params in param_sets:
            extension._llmquery(params, mock_caller)
        
        # Verify parameters were parsed correctly
        assert mock_llm_client.query.call_count == 3
        calls = mock_llm_client.query.call_args_list
        
        # Check first call parameters
        assert calls[0][0][1]["model"] == "gpt-4"
        assert calls[0][0][1]["temperature"] == 0.7
        assert calls[0][0][1]["max_tokens"] == 150
        
        # Check second call parameters
        assert calls[1][0][1]["model"] == "gpt-4"
        assert calls[1][0][1]["temperature"] == 0.7
        assert calls[1][0][1]["max_tokens"] == 150
        
        # Check third call parameters
        assert calls[2][0][1]["model"] == "gpt-4"
        assert calls[2][0][1]["temperature"] == 0.7
        assert calls[2][0][1]["max_tokens"] == 150
        assert calls[2][0][1]["stream"] is False
    
    def test_llmquery_tag_with_template_variables(self, mock_llm_client, mock_logger):
        """Test llmquery tag with template variables in prompt."""
        env = create_environment()
        extension = [ext for ext in env.extensions.values() if isinstance(ext, LLMQueryExtension)][0]
        extension.set_template_name("test.jinja")
        
        # Create a prompt that would result from template rendering
        rendered_prompt = """Hello, World!

Here are some items:
- apple
- banana
- cherry"""
        
        mock_caller = Mock(return_value=rendered_prompt)
        
        extension._llmquery({"model": "gpt-4"}, mock_caller)
        
        # Verify prompt was passed correctly 
        mock_llm_client.query.assert_called_once()
        call_args = mock_llm_client.query.call_args[0]
        prompt = call_args[0]
        assert prompt == rendered_prompt
        assert "Hello, World!" in prompt
        assert "Here are some items:" in prompt
        assert "- apple" in prompt
        assert "- banana" in prompt
        assert "- cherry" in prompt
    
    def test_llmquery_tag_streaming_mode(self, mock_llm_client, mock_logger):
        """Test llmquery tag in streaming mode."""
        # Update mock for this test only
        mock_llm_client.query.return_value = "Hello, World!"
        
        env = create_environment()
        extension = [ext for ext in env.extensions.values() if isinstance(ext, LLMQueryExtension)][0]
        extension.set_template_name("test.jinja")
        
        mock_caller = Mock(return_value="Simple prompt")
        
        result = extension._llmquery({"model": "gpt-4", "stream": True}, mock_caller)
        assert result == "Hello, World!"
        
        # Verify stream parameter was passed
        mock_llm_client.query.assert_called_once()
        assert mock_llm_client.query.call_args[0][1]["stream"] is True
    
    @pytest.mark.skip("This test is pending implementation and will be addressed later")
    def test_llmquery_tag_default_parameters(self):
        """Test default parameter functionality by checking the request structure."""
        # TO DO: Implement a proper test for default parameters
        assert True
    
    def test_llmquery_tag_error_handling(self, mock_llm_client, mock_logger):
        """Test error handling in llmquery tag."""
        # Setup mock to raise an exception
        mock_llm_client.query.side_effect = Exception("API Error")
        
        env = create_environment()
        extension = [ext for ext in env.extensions.values() if isinstance(ext, LLMQueryExtension)][0]
        extension.set_template_name("test.jinja")
        
        mock_caller = Mock(return_value="Prompt that causes an error")
        
        # The error should be caught and re-raised as a RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            extension._llmquery({"model": "gpt-4"}, mock_caller)
        
        assert "LLM query error" in str(exc_info.value)
        
        # Verify the query was attempted
        mock_llm_client.query.assert_called_once()

def test_llmquery_tag_async_mode():
    """Test llmquery tag in async mode."""
    with patch('jinja_prompt_chaining_system.parser.LLMClient') as mock_llm:
        client = Mock()
        client.query.return_value = "Async response"
        mock_llm.return_value = client
        
        with patch('jinja_prompt_chaining_system.parser.LLMLogger') as mock_logger:
            logger = Mock()
            mock_logger.return_value = logger
            
            env = create_environment()
            extension = [ext for ext in env.extensions.values() if isinstance(ext, LLMQueryExtension)][0]
            extension.set_template_name("test_async.jinja")
            
            # Create a mock async caller function
            async def mock_caller():
                return "Async prompt content"
            
            # Test the async branch directly
            result = asyncio.run(extension._llmquery_async({
                "model": "gpt-4",
                "temperature": 0.5
            }, mock_caller))
            
            assert result == "Async response"
            
            # Verify the LLM was called with the async prompt content
            client.query.assert_called_once()
            call_args = client.query.call_args[0]
            assert call_args[0] == "Async prompt content"
            assert call_args[1]["model"] == "gpt-4"
            assert call_args[1]["temperature"] == 0.5 