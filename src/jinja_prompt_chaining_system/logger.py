import os
import yaml
import re
from typing import Dict, Any, Optional
from datetime import datetime, UTC

class LLMLogger:
    """Logger for LLM interactions that saves to YAML files."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the logger with an optional log directory."""
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def _get_log_path(self, template_name: str) -> str:
        """Get the path for the log file."""
        if not self.log_dir:
            return None
        return os.path.join(self.log_dir, f"{template_name}.log.yaml")
    
    def log_request(
        self,
        template_name: str,
        request: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an LLM request and optionally its response."""
        log_path = self._get_log_path(template_name)
        if not log_path:
            return
        
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request": request
        }
        
        if response:
            log_entry["response"] = response
        
        # Read existing log if it exists
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                try:
                    log_data = yaml.safe_load(f) or []
                except yaml.YAMLError:
                    log_data = []
        else:
            log_data = []
        
        # Append new entry
        log_data.append(log_entry)
        
        # Write back to file
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def update_response(
        self,
        template_name: str,
        response_chunk: str
    ) -> None:
        """Update the latest response with a new chunk."""
        log_path = self._get_log_path(template_name)
        if not log_path or not os.path.exists(log_path):
            return
        
        with open(log_path, 'r') as f:
            try:
                log_data = yaml.safe_load(f) or []
            except yaml.YAMLError:
                return
        
        if not log_data:
            return
        
        latest_entry = log_data[-1]
        if "response" not in latest_entry:
            latest_entry["response"] = {
                "content": "",
                "done": False
            }
        
        # Handle YAML-breaking sequences
        chunk = response_chunk
        
        # Add a space before any YAML document markers at start of lines
        chunk = re.sub(r'(\n|^)(---|\.\.\.)(\n|$)', r'\1 \2\3', chunk)
        
        latest_entry["response"]["content"] += chunk
        
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
    def complete_response(
        self,
        template_name: str,
        completion_data: Dict[str, Any]
    ) -> None:
        """Mark the streaming response as complete and add additional metadata."""
        log_path = self._get_log_path(template_name)
        if not log_path or not os.path.exists(log_path):
            return
        
        with open(log_path, 'r') as f:
            try:
                log_data = yaml.safe_load(f) or []
            except yaml.YAMLError:
                return
        
        if not log_data:
            return
        
        latest_entry = log_data[-1]
        if "response" not in latest_entry:
            latest_entry["response"] = {
                "content": "",
                "done": False
            }
        
        # Mark response as done and add any additional metadata
        latest_entry["response"]["done"] = True
        
        # Add completion data fields 
        for key, value in completion_data.items():
            if key != "content":  # Don't overwrite content
                latest_entry["response"][key] = value
        
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True) 