import unittest
import os
import sys
from jinja_prompt_chaining_system.utils import split_template_path

class TestPathSplitting(unittest.TestCase):
    """Test the custom split_template_path function that handles template path resolution."""
    
    def test_posix_path_splitting(self):
        """Test that POSIX-style paths (with forward slashes) are properly split."""
        path = "templates/includes/header.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, ["templates", "includes", "header.jinja"])
    
    def test_windows_path_handling(self):
        """Test that Windows-style paths are normalized to forward slashes."""
        # Windows backslashes should be normalized to forward slashes
        path = "templates\\includes\\header.jinja"
        segments = split_template_path(path)
        # Our implementation should normalize backslashes to forward slashes
        self.assertEqual(segments, ["templates", "includes", "header.jinja"])
        
        # Test with mixed slash styles
        path = "templates/includes\\header.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, ["templates", "includes", "header.jinja"])
    
    def test_relative_path_splitting(self):
        """Test that relative paths are split correctly."""
        # Path with ./ prefix
        path = "./templates/header.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, [".", "templates", "header.jinja"])
        
        # Path with ../ prefix
        path = "../templates/header.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, ["..", "templates", "header.jinja"])
        
        # Path with multiple relative components
        path = "../../templates/header.jinja"
        segments = split_template_path(path)
        # Our implementation handles this separately
        self.assertTrue(all(s == ".." for s in segments[:2]))
        self.assertEqual(segments[2:], ["templates", "header.jinja"])
    
    def test_empty_segments_are_removed(self):
        """Test that empty segments from consecutive slashes are removed."""
        path = "templates//includes///header.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, ["templates", "includes", "header.jinja"])
        
        # Test with trailing slash
        path = "templates/includes/header.jinja/"
        segments = split_template_path(path)
        self.assertEqual(segments, ["templates", "includes", "header.jinja"])
        
        # Test with leading slash
        path = "/templates/includes/header.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, ["templates", "includes", "header.jinja"])
    
    def test_absolute_path_handling(self):
        """Test that absolute paths are handled properly."""
        if sys.platform.startswith('win'):
            # Windows absolute path (drive letter)
            path = "C:/templates/header.jinja"
            segments = split_template_path(path)
            self.assertEqual(segments, ["C:", "templates", "header.jinja"])
            
            # UNC path 
            path = "\\\\server\\share\\templates\\header.jinja"
            segments = split_template_path(path)
            # UNC paths should be normalized
            self.assertEqual(segments, ["server", "share", "templates", "header.jinja"])
        else:
            # Unix absolute path
            path = "/usr/local/templates/header.jinja"
            segments = split_template_path(path)
            self.assertEqual(segments, ["usr", "local", "templates", "header.jinja"])
    
    def test_path_with_special_characters(self):
        """Test that paths with special characters are handled correctly."""
        path = "templates/includes/special_$&@!.jinja"
        segments = split_template_path(path)
        self.assertEqual(segments, ["templates", "includes", "special_$&@!.jinja"]) 