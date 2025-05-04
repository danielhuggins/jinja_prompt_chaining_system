import os
import time
import yaml
from typing import Dict, Any, Optional
from datetime import datetime, timezone

class SafeLineBreakDumper(yaml.SafeDumper):
    """A custom YAML dumper that uses the pipe (|) style for multiline strings."""
    def represent_scalar(self, tag, value, style=None):
        if isinstance(value, str) and '\n' in value:
            style = '|'
        return super().represent_scalar(tag, value, style)

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
        # Counter for unique filenames
        self.log_counters = {}
    
    def _generate_log_path(self, template_name: str) -> Optional[str]:
        """Generate a log file path with timestamp and counter to ensure uniqueness."""
        if not self.log_dir:
            return None
        
        # Get timestamp with microsecond precision
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
        
        # Add a counter to ensure uniqueness even for extremely close calls
        if template_name not in self.log_counters:
            self.log_counters[template_name] = 0
        
        counter = self.log_counters[template_name]
        self.log_counters[template_name] += 1
        
        # Sleep a tiny bit to ensure different timestamp when test runs are extremely fast
        time.sleep(0.001)
        
        # Include counter in filename to ensure uniqueness
        filename = f"{template_name}_{timestamp}_{counter}.log.yaml"
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
                "done": False
            }
            # Keep track of this log for streaming updates
            self.active_requests[template_name] = log_path
        
        # Write to file using our custom dumper
        with open(log_path, 'w', encoding='utf-8') as f:
            yaml.dump(log_data, f, Dumper=SafeLineBreakDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
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
        with open(log_path, 'r', encoding='utf-8') as f:
            try:
                log_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                return
        
        # Make sure we have a response structure
        if "response" not in log_data:
            log_data["response"] = {
                "done": False
            }
        
        # Initialize a temporary content buffer if it doesn't exist
        if "_content_buffer" not in log_data["response"]:
            log_data["response"]["_content_buffer"] = ""
        
        # Update the buffer with the new chunk
        log_data["response"]["_content_buffer"] += response_chunk
        
        # Write back to file using our custom dumper
        with open(log_path, 'w', encoding='utf-8') as f:
            yaml.dump(log_data, f, Dumper=SafeLineBreakDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
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
        with open(log_path, 'r', encoding='utf-8') as f:
            try:
                log_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                return
        
        # Make sure we have a response structure
        if "response" not in log_data:
            log_data["response"] = {
                "_content_buffer": "",
                "done": False
            }
        
        # Get the accumulated content from the buffer
        content = log_data["response"].get("_content_buffer", "")
        
        # Update with the completion data (maintaining OpenAI API response format)
        response = completion_data.copy()
        
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
        
        # Remove any existing content field at the root level
        if "content" in response:
            del response["content"]
        
        # Replace the response in the log data
        log_data["response"] = response
        
        # Remove the temporary buffer
        if "_content_buffer" in log_data["response"]:
            del log_data["response"]["_content_buffer"]
        
        # Write back to file using our custom dumper
        with open(log_path, 'w', encoding='utf-8') as f:
            yaml.dump(log_data, f, Dumper=SafeLineBreakDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        # Clean up the active request
        del self.active_requests[template_name] 