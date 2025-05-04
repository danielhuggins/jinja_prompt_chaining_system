from typing import Dict, Any, Optional, Generator, Union
import openai
from openai.types.chat import ChatCompletionChunk

class LLMClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM client."""
        self.client = openai.OpenAI(api_key=api_key)
    
    def query(
        self,
        prompt: str,
        params: Dict[str, Any],
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """Send a query to the LLM and return the response."""
        messages = [{"role": "user", "content": prompt}]
        
        # Extract basic parameters
        model = params.get("model", "gpt-3.5-turbo")
        temperature = float(params.get("temperature", 0.7))
        max_tokens = int(params.get("max_tokens", 150))
        
        # Build API parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if they exist in params
        optional_params = [
            "top_p", "frequency_penalty", "presence_penalty", 
            "stop", "n", "logit_bias"
        ]
        for param in optional_params:
            if param in params:
                api_params[param] = params[param]
                
        # Add tools if specified
        if "tools" in params:
            api_params["tools"] = params["tools"]
        
        try:
            if not stream:
                response = self.client.chat.completions.create(**api_params)
                return str(response.choices[0].message.content)
            
            def generate_chunks():
                try:
                    response = self.client.chat.completions.create(**api_params)
                    for chunk in response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                except Exception as e:
                    raise RuntimeError(f"LLM API error: {str(e)}")
            
            return generate_chunks()
        except Exception as e:
            raise RuntimeError(f"LLM API error: {str(e)}") 