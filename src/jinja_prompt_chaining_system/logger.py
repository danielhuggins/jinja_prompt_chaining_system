import os
import yaml
from typing import Dict, Any, Optional
from datetime import datetime, timezone

class LLMLogger:
    """Logger for LLM interactions that saves to YAML files."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the logger with an optional log directory."""
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Keep track of active streaming requests and mapping from template name to log files
        self.active_requests = {}
        # For each template, track the current log files to support multiple requests
        self.template_logs = {}
    
    def _generate_log_path(self, template_name: str) -> Optional[str]:
        """Generate a log file path with timestamp."""
        if not self.log_dir:
            return None
        
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{template_name}_{timestamp}.log.yaml"
        return os.path.join(self.log_dir, filename)
    
    def log_request(
        self,
        template_name: str,
        request: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an LLM request and optionally its response.
        
        Returns the log file path if logging was successful, otherwise None.
        """
        log_path = self._generate_log_path(template_name)
        if not log_path:
            return None
        
        # Create the log data structure
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request": request
        }
        
        # Add response if provided (non-streaming case)
        if response:
            log_data["response"] = response
        # Initialize response structure for streaming
        elif request.get("stream", True):
            log_data["response"] = {
                "content": "",
                "done": False
            }
            # Keep track of this log for streaming updates
            self.active_requests[template_name] = log_path
        
        # Write to file
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Track logs for this template
        if template_name not in self.template_logs:
            self.template_logs[template_name] = []
        self.template_logs[template_name].append(log_path)
        
        return log_path
    
    def update_response(
        self,
        template_name: str,
        response_chunk: str
    ) -> None:
        """Update the streaming response with a new chunk."""
        # Skip if we don't have an active request for this template
        if template_name not in self.active_requests:
            return
        
        log_path = self.active_requests[template_name]
        if not log_path or not os.path.exists(log_path):
            return
        
        # Read the current log
        with open(log_path, 'r') as f:
            try:
                log_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                return
        
        # Make sure we have a response structure
        if "response" not in log_data:
            log_data["response"] = {
                "content": "",
                "done": False
            }
        
        # Update the content with the new chunk
        log_data["response"]["content"] += response_chunk
        
        # Write back to file
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
    def complete_response(
        self,
        template_name: str,
        completion_data: Dict[str, Any]
    ) -> None:
        """
        Mark the streaming response as complete and add additional metadata.
        
        The completion_data should match the OpenAI API response format.
        """
        # Skip if we don't have an active request for this template
        if template_name not in self.active_requests:
            return
        
        log_path = self.active_requests[template_name]
        if not log_path or not os.path.exists(log_path):
            return
        
        # Read the current log
        with open(log_path, 'r') as f:
            try:
                log_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                return
        
        # Make sure we have a response structure
        if "response" not in log_data:
            log_data["response"] = {
                "content": "",
                "done": False
            }
        
        # Get the accumulated content
        content = log_data["response"].get("content", "")
        
        # Update with the completion data (maintaining OpenAI API response format)
        response = completion_data.copy()
        
        # Preserve the content field directly in response
        response["content"] = content
        
        # Update the content in choices[0].message.content if it exists
        if "choices" in response and len(response["choices"]) > 0:
            if "message" in response["choices"][0]:
                if response["choices"][0]["message"].get("content") is None:
                    # Don't overwrite content if it's explicitly None (e.g., for tool calls)
                    pass
                else:
                    response["choices"][0]["message"]["content"] = content
        
        # Add the done flag
        response["done"] = True
        
        # Replace the response in the log data
        log_data["response"] = response
        
        # Write back to file
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Clean up the active request
        del self.active_requests[template_name] 