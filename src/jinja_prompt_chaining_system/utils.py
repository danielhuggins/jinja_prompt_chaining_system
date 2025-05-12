"""Utility functions for the Jinja Prompt Chaining System."""

import os
import re
import posixpath
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
from jinja2 import FileSystemLoader, TemplateNotFound, Template

def split_template_path(template):
    """
    Split a template path into segments.
    
    This function replicates Jinja2's internal path splitting logic for templates,
    with improved handling for Windows paths and special characters.
    
    Args:
        template: The template path to split
        
    Returns:
        A list of path segments
    """
    # Special case handling for relative paths
    if template.startswith('./'):
        normalized_path = template[2:]  # Remove the leading ./
        segments = [part for part in normalized_path.split('/') if part]
        segments.insert(0, '.')
        return segments
    
    elif template.startswith('../'):
        # Count how many ../s we have at the beginning
        count = 0
        i = 0
        while template[i:i+3] == '../':
            count += 1
            i += 3
        
        # The rest of the path without the leading ../s
        normalized_path = template[count*3:]
        segments = [part for part in normalized_path.split('/') if part]
        
        # Add the ..s at the beginning
        for _ in range(count):
            segments.insert(0, '..')
        
        return segments
    
    # Normalize path separators to POSIX style (forward slashes)
    normalized_path = template.replace('\\', '/')
    
    # Split by forward slashes and filter out empty segments
    segments = [part for part in normalized_path.split('/') if part]
    
    return segments

class EnhancedTemplateNotFound(TemplateNotFound):
    """
    Enhanced version of TemplateNotFound that includes additional context about attempted paths.
    
    This exception provides more detailed information about where the system looked for templates,
    including absolute paths and relative paths from different bases.
    """
    
    def __init__(self, name, message=None, attempted_paths=None):
        """
        Initialize the exception with more context information.
        
        Args:
            name: The template name that was not found
            message: Optional message override
            attempted_paths: List of absolute paths that were checked
        """
        self.attempted_paths = attempted_paths or []
        
        # Build a detailed error message with all attempted paths
        if not message:
            message = f"Template {name!r} not found."
            
        if self.attempted_paths:
            paths_str = "\n - " + "\n - ".join(self.attempted_paths)
            message += f"\nAttempted paths:{paths_str}"
                
        super().__init__(name, message)
        
        # Ensure the message is also available in the Exception superclass
        self.args = (name, message)


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
            EnhancedTemplateNotFound: If the template cannot be found, with detailed path information
        """
        # Track all attempted paths for better error messages
        attempted_paths = []
        
        # Check if this is a relative include (starts with ./ or ../)
        is_relative = template.startswith('./') or template.startswith('../')
        
        if is_relative and self._last_loaded_template and self._last_loaded_template in self._template_dirs:
            # Get the directory of the including template
            parent_dir = self._template_dirs.get(self._last_loaded_template)
            including_template = self._last_loaded_template
            
            # Resolve path relative to the current template directory
            resolved_path = os.path.abspath(os.path.normpath(os.path.join(parent_dir, template)))
            attempted_paths.append(f"{resolved_path} (relative to {including_template})")
            
            # Try to load the template from this resolved path
            try:
                with open(resolved_path, 'r', encoding=self.encoding) as f:
                    contents = f.read()
                
                # Store the directory of this template for nested includes
                template_dir = os.path.dirname(resolved_path)
                self._template_dirs[resolved_path] = template_dir
                
                # Set this as the last loaded template
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
        
        # Try the standard approach, but catch and enhance any errors
        try:
            # Use the parent's get_source method to find the template
            source, filename, uptodate = super().get_source(environment, template)
            
            # Store the directory of this template for future relative includes
            template_dir = os.path.dirname(filename)
            self._template_dirs[filename] = template_dir
            
            # Set this as the last loaded template
            self._last_loaded_template = filename
            
            return source, filename, uptodate
        except TemplateNotFound as e:
            # If we get here, the template was not found
            # Collect all the paths we tried for better error messages
            for searchpath in self.searchpath:
                if is_relative:
                    # For relative includes, we also show the direct path
                    direct_path = os.path.join(searchpath, template)
                    attempted_paths.append(f"{os.path.abspath(direct_path)} (from searchpath, treating relative as absolute)")
                else:
                    # For standard includes, collect the full resolved path
                    pieces = split_template_path(template)
                    resolved_path = os.path.join(searchpath, *pieces)
                    attempted_paths.append(f"{os.path.abspath(resolved_path)} (from searchpath)")
            
            # Create a more detailed error message
            plural = "path" if len(self.searchpath) == 1 else "paths"
            paths_str = ", ".join(repr(p) for p in self.searchpath)
            
            # Raise with enhanced error information
            raise EnhancedTemplateNotFound(
                template,
                f"{template!r} not found in search {plural}: {paths_str}",
                attempted_paths
            ) from e
    
    def load(self, environment, name, globals=None):
        """
        Load a template by name with proper template directory tracking.
        
        Args:
            environment: The Jinja environment
            name: The template name
            globals: Optional template globals
            
        Returns:
            The loaded template
            
        Raises:
            EnhancedTemplateNotFound: If the template cannot be found
        """
        # Save the current last loaded template
        previous_template = self._last_loaded_template
        
        try:
            # Use the original FileSystemLoader's load method to avoid compatibility issues
            return super().load(environment, name, globals)
        except TemplateNotFound as e:
            # Convert standard TemplateNotFound to our enhanced version
            if not isinstance(e, EnhancedTemplateNotFound):
                # Get the attempted paths if available
                attempted_paths = getattr(e, 'attempted_paths', [])
                # Add absolute path information
                if not attempted_paths:
                    for searchpath in self.searchpath:
                        path = os.path.join(searchpath, name)
                        attempted_paths.append(f"{os.path.abspath(path)} (from searchpath)")
                
                # Create enhanced error with absolute path information
                raise EnhancedTemplateNotFound(
                    name, 
                    f"Template {name!r} not found with absolute paths",
                    attempted_paths
                ) from e
            raise 