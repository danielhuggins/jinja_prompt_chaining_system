from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.lexer import Token, TokenStream
from jinja2.parser import Parser
from typing import Dict, Any, Optional
import os
import asyncio

from .llm import LLMClient
from .logger import LLMLogger

class LLMQueryExtension(Extension):
    """Jinja2 extension that adds the llmquery tag for LLM interactions."""
    
    tags = {'llmquery'}
    identifier = 'llmquery'
    
    def __init__(self, environment):
        super().__init__(environment)
        self.llm_client = LLMClient()
        self.logger = LLMLogger()
        self.template_name = None
        
    def parse(self, parser: Parser) -> nodes.Node:
        """Parse the llmquery tag and its parameters."""
        lineno = next(parser.stream).lineno
        
        # Parse parameters
        params = {}
        while parser.stream.current.type != 'block_end':
            # Skip whitespace and commas
            while parser.stream.current.type in ('whitespace', 'comma'):
                next(parser.stream)
                
            if parser.stream.current.type == 'name':
                name = parser.stream.current.value
                next(parser.stream)
                
                if parser.stream.current.type != 'assign':
                    parser.fail('Expected "=" after parameter name')
                next(parser.stream)
                
                value = parser.parse_expression()
                params[name] = value
                
                # Skip whitespace and commas
                while parser.stream.current.type in ('whitespace', 'comma'):
                    next(parser.stream)
        
        # Parse the body
        body = parser.parse_statements(['name:endllmquery'], drop_needle=True)
        
        # Create a call to the _llmquery function with a caller
        args = [nodes.Dict([
            nodes.Pair(nodes.Const(name), value) for name, value in params.items()
        ]).set_lineno(lineno)]
        
        caller = nodes.CallBlock(
            nodes.Call(
                nodes.Getattr(nodes.Name('extension', 'load'), '_llmquery', lineno),
                args, [], None, None
            ).set_lineno(lineno),
            [], [], body
        ).set_lineno(lineno)
        
        return caller
    
    def query(self, prompt: str, **params) -> str:
        """Query the LLM with the given prompt and parameters."""
        try:
            response = self.llm_client.query(prompt, params, stream=params.get("stream", False))
            if self.template_name and self.logger:
                self.logger.log_request(
                    self.template_name,
                    {
                        "model": params.get("model", "gpt-3.5-turbo"),
                        "temperature": float(params.get("temperature", 0.7)),
                        "max_tokens": int(params.get("max_tokens", 150)),
                        "stream": params.get("stream", True),
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    {"content": response, "done": True}
                )
            return response
        except Exception as e:
            raise RuntimeError(f"LLM query error: {str(e)}")

    def set_template_name(self, name: str):
        """Set the current template name for logging."""
        self.template_name = os.path.splitext(os.path.basename(name))[0]

    async def _llmquery_async(self, params: Dict[str, Any], caller) -> str:
        """Process the llmquery tag asynchronously and return the result."""
        # Get the prompt from the template body
        prompt = await caller()
        
        # Prepare request for logging
        request = {
            "model": params.get("model", "gpt-3.5-turbo"),
            "temperature": float(params.get("temperature", 0.7)),
            "max_tokens": int(params.get("max_tokens", 150)),
            "stream": params.get("stream", True),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Copy any additional parameters from params to request
        for key, value in params.items():
            if key not in request and key != "stream":
                request[key] = value
        
        # Get response from LLM
        stream = params.get("stream", True)
        try:
            if stream:
                # Log the initial request before streaming
                if self.template_name:
                    log_path = self.logger.log_request(self.template_name, request)
                
                # Get streaming response
                result = []
                for chunk in self.llm_client.query(prompt, params, stream=True):
                    result.append(chunk)
                    if self.template_name:
                        self.logger.update_response(self.template_name, chunk)
                
                # Join the chunks and return
                response_text = "".join(result)
                
                # Complete the response with final metadata
                if self.template_name:
                    # Create a response object that mirrors OpenAI's format
                    completion_data = {
                        "id": f"chatcmpl-{id(prompt)}",  # Generate a unique ID
                        "model": request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response_text
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt) // 4,  # Rough estimation
                            "completion_tokens": len(response_text) // 4,
                            "total_tokens": (len(prompt) + len(response_text)) // 4
                        }
                    }
                    self.logger.complete_response(self.template_name, completion_data)
                
                return response_text
            else:
                # Non-streaming: Get the complete response at once
                response = self.llm_client.query(prompt, params, stream=False)
                
                if self.template_name:
                    # Create a response object that mirrors OpenAI's format
                    completion_data = {
                        "id": f"chatcmpl-{id(prompt)}",
                        "model": request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt) // 4,
                            "completion_tokens": len(response) // 4,
                            "total_tokens": (len(prompt) + len(response)) // 4
                        }
                    }
                    self.logger.log_request(self.template_name, request, completion_data)
                
                return response
        except Exception as e:
            raise RuntimeError(f"LLM query error: {str(e)}")
            
    def _llmquery(self, params: Dict[str, Any], caller) -> str:
        """Process the llmquery tag and return the result."""
        try:
            # Check if we're in async mode
            if asyncio.iscoroutinefunction(caller):
                # We need to await this - Jinja will handle this correctly in async mode
                return self._llmquery_async(params, caller)
            
            # Synchronous mode
            prompt = caller()
            
            # Prepare request for logging
            request = {
                "model": params.get("model", "gpt-3.5-turbo"),
                "temperature": float(params.get("temperature", 0.7)),
                "max_tokens": int(params.get("max_tokens", 150)),
                "stream": params.get("stream", True),
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Copy any additional parameters from params to request
            for key, value in params.items():
                if key not in request and key != "stream":
                    request[key] = value
            
            # Get response from LLM
            stream = params.get("stream", True)
            try:
                if stream:
                    # Log the initial request before streaming
                    if self.template_name:
                        log_path = self.logger.log_request(self.template_name, request)
                    
                    # Get streaming response
                    result = []
                    for chunk in self.llm_client.query(prompt, params, stream=True):
                        result.append(chunk)
                        if self.template_name:
                            self.logger.update_response(self.template_name, chunk)
                    
                    # Join the chunks and return
                    response_text = "".join(result)
                    
                    # Complete the response with final metadata
                    if self.template_name:
                        # Create a response object that mirrors OpenAI's format
                        completion_data = {
                            "id": f"chatcmpl-{id(prompt)}",  # Generate a unique ID
                            "model": request["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": response_text
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(prompt) // 4,  # Rough estimation
                                "completion_tokens": len(response_text) // 4,
                                "total_tokens": (len(prompt) + len(response_text)) // 4
                            }
                        }
                        self.logger.complete_response(self.template_name, completion_data)
                    
                    return response_text
                else:
                    # Non-streaming: Get the complete response at once
                    response = self.llm_client.query(prompt, params, stream=False)
                    
                    if self.template_name:
                        # Create a response object that mirrors OpenAI's format
                        completion_data = {
                            "id": f"chatcmpl-{id(prompt)}",
                            "model": request["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": response
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(prompt) // 4,
                                "completion_tokens": len(response) // 4,
                                "total_tokens": (len(prompt) + len(response)) // 4
                            }
                        }
                        self.logger.log_request(self.template_name, request, completion_data)
                    
                    return response
            except Exception as e:
                raise RuntimeError(f"LLM query error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"LLM query error: {str(e)}") 