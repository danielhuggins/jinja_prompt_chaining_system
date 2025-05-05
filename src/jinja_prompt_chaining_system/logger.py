import os
import time
import yaml
import re
from typing import Dict, Any, Optional
from datetime import datetime, timezone

class ContentAwareYAMLDumper(yaml.SafeDumper):
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
    
    def _post_process_yaml_file(self, file_path: str) -> None:
        """
        Post-process the YAML file to change content field formatting without disturbing
        the actual YAML structure or content values.
        """
        if not os.path.exists(file_path):
            return
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all content field declarations with various formats and add the markdown comment
        # Case 1: content: | (multiline block)
        content = re.sub(r'(\s+content:\s*\|)(\s*\n)', r'\1   # markdown\2', content, flags=re.MULTILINE)
        
        # Case 2: content: |- (multiline compact block)
        content = re.sub(r'(\s+content:\s*\|-)', r'\1   # markdown', content, flags=re.MULTILINE)
        
        # Case 3: content: |+ (multiline keep trailing whitespace)
        content = re.sub(r'(\s+content:\s*\|\+)', r'\1   # markdown', content, flags=re.MULTILINE)
        
        # Case 4: content: |2 or any other number (explicit indentation)
        content = re.sub(r'(\s+content:\s*\|\d+)', r'\1   # markdown', content, flags=re.MULTILINE)
        
        # Case 5: content: "..." (quoted string)
        # Handle differently - add comment after the quotes
        quoted_pattern = r'(\s+content:\s*")([^"]*)(")(\s*\n)'
        replacement = lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}   # markdown{m.group(4)}"
        content = re.sub(quoted_pattern, replacement, content, flags=re.MULTILINE)
        
        # Case 6: content: '...' (single-quoted string)
        single_quoted_pattern = r"(\s+content:\s*')([^']*)(')"
        replacement_sq = lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}   # markdown"
        content = re.sub(single_quoted_pattern, replacement_sq, content, flags=re.MULTILINE)
        
        # Case 7: Plain scalar values (no quotes, no pipe)
        # Match content: followed by any text until end of line, but not if it already has markdown or pipe
        scalar_pattern = r'(\s+content:\s*)([^|"\'\n#][^\n]*)$'
        replacement_scalar = lambda m: f"{m.group(1)}{m.group(2)}   # markdown"
        content = re.sub(scalar_pattern, replacement_scalar, content, flags=re.MULTILINE)
        
        # Write back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
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
        
        # Write the YAML using the dumper
        with open(log_path, 'w', encoding='utf-8') as f:
            yaml.dump(log_data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Post-process the file for content field formatting
        self._post_process_yaml_file(log_path)
        
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
        
        # Note: Do not add the content field at root level
        # Keep only _content_buffer for internal tracking
        
        # Write to file
        with open(log_path, 'w', encoding='utf-8') as f:
            yaml.dump(log_data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Post-process for content field formatting
        self._post_process_yaml_file(log_path)
            
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
        
        # Remove any root-level content field for compliance with tests
        if "content" in response:
            del response["content"]
        
        # Replace the response in the log data
        log_data["response"] = response
        
        # Remove the temporary buffer
        if "_content_buffer" in log_data["response"]:
            del log_data["response"]["_content_buffer"]
        
        # Write to file
        with open(log_path, 'w', encoding='utf-8') as f:
            yaml.dump(log_data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Post-process for content field formatting
        self._post_process_yaml_file(log_path)
            
        # Clean up the active request
        del self.active_requests[template_name] 