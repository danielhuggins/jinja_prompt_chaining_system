import os
import tempfile
import pytest
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja_prompt_chaining_system import create_environment

@pytest.fixture(scope="module")
def test_dirs():
    """Create a directory structure for testing CWD-relative paths."""
    # Create a base temporary directory that will be manually cleaned up
    temp_dir = tempfile.mkdtemp(prefix="jinja_test_")
    
    try:
        # Create a directory structure for testing
        template_dir = os.path.join(temp_dir, "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # Create a subdirectory to change to during tests
        cwd_dir = os.path.join(temp_dir, "current_dir")
        os.makedirs(cwd_dir, exist_ok=True)
        
        # Create a template file in the template directory
        with open(os.path.join(template_dir, "main.jinja"), "w") as f:
            f.write("Main template - will include another template\n{% include 'included.jinja' %}")
        
        # Create the included template in the CWD directory
        with open(os.path.join(cwd_dir, "included.jinja"), "w") as f:
            f.write("This is included from the CWD directory")
        
        yield {
            "temp_dir": temp_dir,
            "template_dir": template_dir,
            "cwd_dir": cwd_dir
        }
    finally:
        # Clean up after all tests in this module are done
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_cwd_relative_includes(test_dirs):
    """Test that includes without ./ or ../ are resolved relative to the CWD."""
    original_cwd = os.getcwd()
    
    try:
        # Change to the test directory for the test
        os.chdir(test_dirs["cwd_dir"])
        
        # Create the environment with the template directory
        env = create_environment(test_dirs["template_dir"])
        
        # The main template includes 'included.jinja' without ./ or ../
        # This should resolve to included.jinja in the CWD, not in the template dir
        template = env.get_template("main.jinja")
        
        # The rendered output should contain the content from the included file in CWD
        result = template.render()
        assert "This is included from the CWD directory" in result
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def test_cwd_fallback_for_includes(test_dirs):
    """
    Test that if a non-relative include isn't found in CWD, 
    it falls back to the template directory.
    """
    original_cwd = os.getcwd()
    
    try:
        # Change to a directory where the included file doesn't exist
        os.chdir(test_dirs["temp_dir"])  # Not the cwd_dir where included.jinja exists
        
        # Create the same file in the template directory
        with open(os.path.join(test_dirs["template_dir"], "included.jinja"), "w") as f:
            f.write("This is included from the TEMPLATE directory")
        
        # Create the environment with the template directory
        env = create_environment(test_dirs["template_dir"])
        
        # The main template includes 'included.jinja' without ./ or ../
        # Since it's not in CWD, it should fall back to the template directory
        template = env.get_template("main.jinja")
        
        # The rendered output should contain the content from the template directory
        result = template.render()
        assert "This is included from the TEMPLATE directory" in result
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def test_explicit_relative_paths_not_affected(test_dirs):
    """Test that explicit relative paths (./file.jinja) are still relative to template."""
    original_cwd = os.getcwd()
    
    try:
        # Setup: Create a template that uses explicit relative include
        subfolder = os.path.join(test_dirs["template_dir"], "subfolder")
        os.makedirs(subfolder, exist_ok=True)
        
        with open(os.path.join(test_dirs["template_dir"], "explicit.jinja"), "w") as f:
            f.write("{% include './subfolder/explicit_include.jinja' %}")
        
        # Create the target file
        with open(os.path.join(subfolder, "explicit_include.jinja"), "w") as f:
            f.write("This is included with explicit relative path")
        
        # Change to the test directory for the test
        os.chdir(test_dirs["cwd_dir"])
        
        # Create a file with the same name in CWD to verify it's NOT used
        os.makedirs(os.path.join(test_dirs["cwd_dir"], "subfolder"), exist_ok=True)
        with open(os.path.join(test_dirs["cwd_dir"], "subfolder", "explicit_include.jinja"), "w") as f:
            f.write("This should NOT be included")
        
        # Create the environment with the template directory
        env = create_environment(test_dirs["template_dir"])
        
        # The template uses an explicit relative path ./subfolder/...
        template = env.get_template("explicit.jinja")
        
        # The rendered output should contain the content from the template directory's subfolder
        result = template.render()
        assert "This is included with explicit relative path" in result
        assert "This should NOT be included" not in result
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def test_cwd_relative_error_message(test_dirs):
    """Test that error messages properly show CWD paths for non-relative includes."""
    original_cwd = os.getcwd()
    
    try:
        # Change to the test directory for the test
        os.chdir(test_dirs["cwd_dir"])
        
        # Create a template that includes a non-existent file
        with open(os.path.join(test_dirs["template_dir"], "error.jinja"), "w") as f:
            f.write("{% include 'nonexistent.jinja' %}")
        
        # Create the environment with the template directory
        env = create_environment(test_dirs["template_dir"])
        
        # Try to render the template, which should fail
        template = env.get_template("error.jinja")
        
        with pytest.raises(Exception) as excinfo:
            template.render()
        
        # Verify the error message includes the CWD path
        error_msg = str(excinfo.value)
        assert "Attempted paths:" in error_msg
        
        # It should attempt to load from CWD
        cwd_path = os.path.abspath(os.path.join(test_dirs["cwd_dir"], "nonexistent.jinja"))
        assert cwd_path in error_msg or cwd_path.replace("\\", "/") in error_msg
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd) 