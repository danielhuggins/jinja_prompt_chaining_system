import pytest
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system.llm import LLMClient

@pytest.fixture
def mock_openai():
    with patch('jinja_prompt_chaining_system.llm.openai.OpenAI') as mock_openai_class:
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        yield mock_openai_class

def test_llm_client_initialization(mock_openai):
    """Test LLM client initialization."""
    client = LLMClient()
    assert client.client is not None
    mock_openai.assert_called_once_with(api_key=None)

def test_llm_client_initialization_with_api_key(mock_openai):
    """Test LLM client initialization with API key."""
    api_key = "test-api-key"
    client = LLMClient(api_key)
    assert client.client is not None
    mock_openai.assert_called_once_with(api_key=api_key)

def test_llm_client_query_streaming(mock_openai):
    """Test LLM client query with streaming."""
    # Setup mock chunks
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=", "))]),
        Mock(choices=[Mock(delta=Mock(content="world"))]),
        Mock(choices=[Mock(delta=Mock(content="!"))])
    ]
    mock_openai.return_value.chat.completions.create.return_value = mock_chunks
    
    client = LLMClient()
    prompt = "Say hello"
    params = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    # Test streaming response
    response = list(client.query(prompt, params, stream=True))
    assert response == ["Hello", ", ", "world", "!"]
    
    # Verify API call
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.7,
        max_tokens=150,
        stream=True
    )

def test_llm_client_query_non_streaming(mock_openai):
    """Test LLM client query without streaming."""
    # Setup mock response
    mock_response = Mock(
        choices=[Mock(message=Mock(content="Hello, world!"))]
    )
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    client = LLMClient()
    prompt = "Say hello"
    params = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    # Test non-streaming response
    response = client.query(prompt, params, stream=False)
    assert response == "Hello, world!"
    
    # Verify API call
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.7,
        max_tokens=150,
        stream=False
    )

def test_llm_client_query_default_params(mock_openai):
    """Test LLM client query with default parameters."""
    # Setup mock response
    mock_response = Mock(
        choices=[Mock(message=Mock(content="Hello, world!"))]
    )
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    client = LLMClient()
    prompt = "Say hello"
    
    # Test with minimal params
    response = client.query(prompt, {}, stream=False)
    assert response == "Hello, world!"
    
    # Verify API call with defaults
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.7,
        max_tokens=150,
        stream=False
    )

def test_llm_client_query_error(mock_openai):
    """Test LLM client query error handling."""
    # Setup mock to raise an error
    mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
    
    client = LLMClient()
    prompt = "Say hello"
    params = {"model": "gpt-4o-mini"}
    
    # Test error handling
    with pytest.raises(RuntimeError) as exc_info:
        list(client.query(prompt, params))  # Force generator to execute
    
    assert "LLM API error" in str(exc_info.value)

def test_llm_client_query_with_tools(mock_openai):
    """Test LLM client query with tools parameter."""
    # Setup mock response
    mock_response = Mock(
        choices=[Mock(message=Mock(content="Function result"))]
    )
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    client = LLMClient()
    prompt = "Extract text from PDF"
    params = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_pdf_text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
    
    response = client.query(prompt, params, stream=False)
    assert response == "Function result"
    
    # Verify API call with tools
    mock_openai.return_value.chat.completions.create.assert_called_once()
    call_args = mock_openai.return_value.chat.completions.create.call_args[1]
    assert "tools" in call_args
    assert call_args["tools"][0]["type"] == "function"
    assert call_args["tools"][0]["function"]["name"] == "extract_pdf_text"

def test_llm_client_streaming_empty_content(mock_openai):
    """Test handling of empty content chunks in streaming response."""
    # Setup mock response with empty content
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),  # Empty content
        Mock(choices=[Mock(delta=Mock(content=""))]),     # Empty string
        Mock(choices=[Mock(delta=Mock(content="world"))])
    ]
    mock_openai.return_value.chat.completions.create.return_value = mock_chunks
    
    client = LLMClient()
    prompt = "Say hello"
    params = {"model": "gpt-4o-mini"}
    
    # Test streaming response with empty chunks
    response = list(client.query(prompt, params, stream=True))
    assert response == ["Hello", "world"]  # Empty chunks should be filtered out
    
    # Verify API call
    mock_openai.return_value.chat.completions.create.assert_called_once()

def test_llm_client_with_all_openai_params(mock_openai):
    """Test LLM client with all OpenAI parameters."""
    # Setup mock response
    mock_response = Mock(
        choices=[Mock(message=Mock(content="Complete response"))]
    )
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    client = LLMClient()
    prompt = "Test prompt"
    params = {
        "model": "gpt-4-turbo",
        "temperature": 0.9,
        "max_tokens": 500,
        "top_p": 0.95,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2,
        "stop": ["END"],
        "n": 1,
        "logit_bias": {"50256": -100}
    }
    
    response = client.query(prompt, params, stream=False)
    assert response == "Complete response"
    
    # Verify all parameters were passed correctly
    mock_openai.return_value.chat.completions.create.assert_called_once()
    call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4-turbo"
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["max_tokens"] == 500
    assert call_kwargs["top_p"] == 0.95
    assert call_kwargs["frequency_penalty"] == 0.5
    assert call_kwargs["presence_penalty"] == 0.2
    assert call_kwargs["stop"] == ["END"]
    assert call_kwargs["n"] == 1
    assert call_kwargs["logit_bias"] == {"50256": -100}

def test_llm_client_streaming_with_tool_calls(mock_openai):
    """Test streaming with tool calls in response."""
    # Setup mock response with tool calls
    tool_call = {"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "New York"}'}}
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="", tool_calls=[tool_call]))]),
        Mock(choices=[Mock(delta=Mock(content="Weather result"))]),
    ]
    mock_openai.return_value.chat.completions.create.return_value = mock_chunks
    
    client = LLMClient()
    prompt = "What's the weather in New York?"
    params = {
        "model": "gpt-4o-mini",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
    
    # Test streaming response with tool calls
    response = list(client.query(prompt, params, stream=True))
    assert "Weather result" in response  # Content should be in response
    
    # Verify API call
    mock_openai.return_value.chat.completions.create.assert_called_once()
    assert "tools" in mock_openai.return_value.chat.completions.create.call_args[1] 