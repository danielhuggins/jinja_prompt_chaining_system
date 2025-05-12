import os
import pytest
import tempfile
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system import create_environment
from jinja_prompt_chaining_system.utils import EnhancedTemplateNotFound
from jinja2 import TemplateNotFound

@pytest.fixture
def error_test_dirs():
    """Create a directory structure for testing error messages on includes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a nested directory structure
        main_dir = tmpdir
        template_dir = os.path.join(main_dir, "templates")
        nested_dir = os.path.join(template_dir, "nested")
        
        # Create directories
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        
        # Create a template with a relative include that doesn't exist
        with open(os.path.join(nested_dir, "relative_error.jinja"), "w") as f:
            f.write("""
            {% include '../non_existent.jinja' %}
            """)
        
        # Create a template with an absolute include that doesn't exist
        with open(os.path.join(nested_dir, "absolute_error.jinja"), "w") as f:
            f.write("""
            {% include 'non_existent.jinja' %}
            """)
            
        # Create a valid template for reference
        with open(os.path.join(template_dir, "valid.jinja"), "w") as f:
            f.write("Valid template content")
            
        yield {
            "main_dir": main_dir,
            "template_dir": template_dir,
            "nested_dir": nested_dir
        }

def test_relative_include_error_message(error_test_dirs):
    """Test that relative include errors show absolute paths."""
    # Create environment with nested directory as template path
    env = create_environment(error_test_dirs["nested_dir"])
    
    # Get the template with the relative include error
    template = env.get_template("relative_error.jinja")
    
    # Attempt to render, which should fail with our enhanced error
    with pytest.raises(EnhancedTemplateNotFound) as excinfo:
        template.render()
    
    # Check that the error message contains both relative and absolute paths
    error_msg = str(excinfo.value)
    
    # The error should reference the exact file we're looking for
    assert "non_existent.jinja" in error_msg
    
    # The error should contain absolute path information
    abs_path = os.path.abspath(os.path.join(error_test_dirs["template_dir"], "non_existent.jinja"))
    assert abs_path in error_msg
    
    # The error should mention it was a relative include
    assert "relative to" in error_msg
    
    # The error should include the searchpath information
    assert "Attempted paths:" in error_msg
    
    # The error should be an instance of our enhanced exception
    assert isinstance(excinfo.value, EnhancedTemplateNotFound)
    assert isinstance(excinfo.value, TemplateNotFound)  # Should still be compatible
    
    # Check attempted_paths list
    assert len(excinfo.value.attempted_paths) > 0
    assert any("relative to" in path for path in excinfo.value.attempted_paths)

def test_absolute_include_error_message(error_test_dirs):
    """Test that standard include errors show absolute paths."""
    # Create environment with template directory as template path
    env = create_environment(error_test_dirs["template_dir"])
    
    # Get the template with the absolute include error
    template = env.get_template("nested/absolute_error.jinja")
    
    # Attempt to render, which should fail with our enhanced error
    with pytest.raises(EnhancedTemplateNotFound) as excinfo:
        template.render()
    
    # Check that the error message contains the attempted paths
    error_msg = str(excinfo.value)
    
    # The error should reference the exact file we're looking for
    assert "non_existent.jinja" in error_msg
    
    # The error should contain absolute path information
    abs_path = os.path.abspath(os.path.join(error_test_dirs["template_dir"], "non_existent.jinja"))
    assert abs_path in error_msg or error_test_dirs["template_dir"] in error_msg
    
    # The error should be an instance of our enhanced exception
    assert isinstance(excinfo.value, EnhancedTemplateNotFound)
    assert isinstance(excinfo.value, TemplateNotFound)  # Should still be compatible
    
    # The error should include the searchpath information
    assert "Attempted paths:" in error_msg
    
    # Check attempted_paths list
    assert len(excinfo.value.attempted_paths) > 0
    assert any("from searchpath" in path for path in excinfo.value.attempted_paths)

def test_deeply_nested_include_error_messages(error_test_dirs):
    """Test that errors with multiple levels of includes show full chain."""
    # Create a multi-level include situation
    level1_path = os.path.join(error_test_dirs["template_dir"], "level1.jinja")
    level2_path = os.path.join(error_test_dirs["nested_dir"], "level2.jinja")
    
    # Level 1 includes level 2
    with open(level1_path, "w") as f:
        f.write("""
        Level 1 template
        {% include 'nested/level2.jinja' %}
        """)
    
    # Level 2 includes non-existent with relative path
    with open(level2_path, "w") as f:
        f.write("""
        Level 2 template
        {% include '../non_existent_nested.jinja' %}
        """)
    
    # Create environment with template directory as template path
    env = create_environment(error_test_dirs["template_dir"])
    
    # Get the top level template
    template = env.get_template("level1.jinja")
    
    # Attempt to render, which should fail with our enhanced error
    with pytest.raises(EnhancedTemplateNotFound) as excinfo:
        template.render()
    
    # Check that the error message contains the attempted paths
    error_msg = str(excinfo.value)
    
    # The error should reference the exact file we're looking for
    assert "non_existent_nested.jinja" in error_msg
    
    # The error should be an instance of our enhanced exception
    assert isinstance(excinfo.value, EnhancedTemplateNotFound)
    
    # For deeply nested includes, we should see the full chain of attempted lookups
    assert "Attempted paths:" in error_msg
    
    # Check attempted_paths list
    assert len(excinfo.value.attempted_paths) > 0 