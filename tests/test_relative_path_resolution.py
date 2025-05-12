import os
import pytest
import tempfile
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system import create_environment

@pytest.fixture
def resolution_test_dirs():
    """Create a directory structure to test resolution of relative paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a unique identifier for this test
        test_id = "test_content_123456"
        
        # Create a nested directory structure
        main_dir = tmpdir
        template_dir = os.path.join(main_dir, "templates")
        nested_dir = os.path.join(template_dir, "nested")
        
        # Create all directories
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        
        # Create a same-named file in both the root and template directory with different content
        # This will let us determine which file is actually being included
        
        # File in root directory - this should NOT be included if resolution is correct
        with open(os.path.join(main_dir, "common.jinja"), "w") as f:
            f.write("WRONG CONTENT - from root directory")
        
        # File in template directory - this SHOULD be included if resolution is correct
        with open(os.path.join(template_dir, "common.jinja"), "w") as f:
            f.write(f"CORRECT CONTENT - from template directory - {test_id}")
        
        # Create main template in nested directory that includes a file using relative path
        with open(os.path.join(nested_dir, "main.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Template in nested directory.
            Including file using relative path:
            {% include '../common.jinja' %}
            {% endllmquery %}
            """)
            
        # Create a second test with absolute path for comparison
        with open(os.path.join(nested_dir, "absolute.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Template in nested directory.
            Including file using absolute path:
            {% include 'common.jinja' %}
            {% endllmquery %}
            """)
        
        yield {
            "main_dir": main_dir,
            "template_dir": template_dir,
            "nested_dir": nested_dir,
            "test_id": test_id
        }

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_relative_path_correctly_resolved(mock_llm_client, resolution_test_dirs):
    """
    Test that relative paths in includes are resolved relative to the template's directory,
    not the current working directory.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory to simulate running from there
        os.chdir(resolution_test_dirs["main_dir"])
        
        # Create environment with nested_dir as search path
        env = create_environment(resolution_test_dirs["nested_dir"])
        
        # Load and render the template
        template = env.get_template("main.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content from the template directory
        # not from the CWD (main_dir)
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # This should contain the content from template_dir/common.jinja, not main_dir/common.jinja
        assert "CORRECT CONTENT - from template directory" in prompt
        assert resolution_test_dirs["test_id"] in prompt
        assert "WRONG CONTENT - from root directory" not in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_absolute_path_resolution(mock_llm_client, resolution_test_dirs):
    """
    Test the resolution of absolute paths with same-named files for comparison.
    This helps us understand standard Jinja behavior vs. our relative path enhancement.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path (standard behavior)
        # This is important - absolute paths are resolved against the search path
        env = create_environment(resolution_test_dirs["template_dir"])
        
        # Load and render the nested/absolute.jinja template
        template = env.get_template("nested/absolute.jinja")
        result = template.render()
        
        # Check that the LLM was called with content that would be found in the search path
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # This should contain the content from template_dir/common.jinja
        # since that's in the search path at the root level
        assert "CORRECT CONTENT - from template directory" in prompt
        assert resolution_test_dirs["test_id"] in prompt
        assert "WRONG CONTENT - from root directory" not in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd) 