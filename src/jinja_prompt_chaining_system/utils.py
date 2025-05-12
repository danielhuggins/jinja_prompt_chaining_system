"""Utility functions for the Jinja Prompt Chaining System."""

import os
import re
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
from jinja2 import FileSystemLoader, TemplateNotFound, Template

class RelativePathFileSystemLoader(FileSystemLoader):
    """
    A custom Jinja FileSystemLoader that supports relative includes.
    
    This loader extends Jinja's default FileSystemLoader to support include statements
    with paths relative to the including template rather than the template root.
    For example, {% include './relative/path.jinja' %} or {% include '../parent/path.jinja' %}.
    
    Non-relative includes (those not starting with './' or '../') are handled normally.
    """
    
    def __init__(
        self, 
        searchpath: Union[str, os.PathLike, List[Union[str, os.PathLike]]], 
        encoding: str = 'utf-8', 
        followlinks: bool = False
    ):
        """
        Initialize the loader with the given search path(s).
        
        Args:
            searchpath: A path or list of paths to the template directory
            encoding: The encoding of the templates
            followlinks: Whether to follow symbolic links in the path
        """
        super().__init__(searchpath, encoding, followlinks)
        # Dictionary to track template directories by template path
        self._template_dirs: Dict[str, str] = {}
        # The last template loaded - used to track the current include context
        self._last_loaded_template: Optional[str] = None
    
    def get_source(self, environment, template):
        """
        Get the template source, filename, and uptodate function.
        
        This overrides the parent method to handle relative paths starting with './' or '../'.
        If the template name starts with these prefixes, the path is resolved relative to
        the directory of the including template rather than the template root.
        
        Args:
            environment: The Jinja environment
            template: The template name to load
            
        Returns:
            A tuple of (source, filename, uptodate_func)
            
        Raises:
            TemplateNotFound: If the template cannot be found
        """
        # Check if this is a relative include (starts with ./ or ../)
        is_relative = template.startswith('./') or template.startswith('../')
        
        if is_relative and self._last_loaded_template and self._last_loaded_template in self._template_dirs:
            # Get the directory of the including template
            parent_dir = self._template_dirs.get(self._last_loaded_template)
            
            # Resolve path relative to the current template directory
            resolved_path = os.path.normpath(os.path.join(parent_dir, template))
            
            # Try to load the template from this resolved path
            try:
                with open(resolved_path, 'r', encoding=self.encoding) as f:
                    contents = f.read()
                
                # Store the directory of this template for nested includes
                template_dir = os.path.dirname(resolved_path)
                self._template_dirs[resolved_path] = template_dir
                
                # Set this as the last loaded template
                previous_template = self._last_loaded_template
                self._last_loaded_template = resolved_path
                
                # Prepare the uptodate function
                mtime = os.path.getmtime(resolved_path)
                def uptodate():
                    try:
                        return os.path.getmtime(resolved_path) == mtime
                    except OSError:
                        return False
                
                return contents, resolved_path, uptodate
            except (IOError, OSError):
                # If we can't load the relative path, continue to try standard loading
                pass
        
        # If not a relative include or if relative include failed, 
        # try standard FileSystemLoader behavior
        source, filename, uptodate = super().get_source(environment, template)
        
        # Store the directory of this template for future relative includes
        template_dir = os.path.dirname(filename)
        self._template_dirs[filename] = template_dir
        
        # Set this as the last loaded template
        self._last_loaded_template = filename
        
        return source, filename, uptodate
    
    def load(self, environment, name, globals=None):
        """
        Load a template by name with proper template directory tracking.
        
        Args:
            environment: The Jinja environment
            name: The template name
            globals: Optional template globals
            
        Returns:
            The loaded template
        """
        # Save the current last loaded template
        previous_template = self._last_loaded_template
        
        try:
            # Load the template
            template = super().load(environment, name, globals)
            return template
        finally:
            # No need to restore previous template here as get_source will set it correctly
            pass 