import os
import pytest
import tempfile
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system import create_environment
from jinja_prompt_chaining_system.utils import EnhancedTemplateNotFound

@pytest.fixture
def path_resolution_test_dirs():
    """Create a complex directory structure to test resolution edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a unique identifier for this test
        test_id = "edge_case_test_id"
        
        # Create a nested directory structure
        main_dir = tmpdir
        template_dir = os.path.join(main_dir, "templates")
        nested_dir = os.path.join(template_dir, "nested")
        deep_dir = os.path.join(nested_dir, "deep")
        sibling_dir = os.path.join(template_dir, "sibling")
        
        # Create all directories
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        os.makedirs(deep_dir, exist_ok=True)
        os.makedirs(sibling_dir, exist_ok=True)
        
        # Create same-named files in different locations with distinct content
        # This allows us to verify which specific file was loaded
        
        # Create duplicate.jinja in multiple locations
        with open(os.path.join(main_dir, "duplicate.jinja"), "w") as f:
            f.write("CONTENT FROM ROOT - " + test_id)
            
        with open(os.path.join(template_dir, "duplicate.jinja"), "w") as f:
            f.write("CONTENT FROM TEMPLATE DIR - " + test_id)
            
        with open(os.path.join(nested_dir, "duplicate.jinja"), "w") as f:
            f.write("CONTENT FROM NESTED DIR - " + test_id)
            
        with open(os.path.join(sibling_dir, "duplicate.jinja"), "w") as f:
            f.write("CONTENT FROM SIBLING DIR - " + test_id)

        # Create unique.jinja files in specific locations only
        with open(os.path.join(deep_dir, "deep_unique.jinja"), "w") as f:
            f.write("DEEP UNIQUE CONTENT - " + test_id)
            
        with open(os.path.join(template_dir, "template_unique.jinja"), "w") as f:
            f.write("TEMPLATE UNIQUE CONTENT - " + test_id)
        
        # Create test templates in various locations to test different include paths
        
        # Test absolute path include from nested directory
        with open(os.path.join(nested_dir, "test_absolute.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing absolute path include:
            {% include 'duplicate.jinja' %}
            {% endllmquery %}
            """)
        
        # Test relative path with ./ prefix
        with open(os.path.join(nested_dir, "test_current_dir.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing current dir include:
            {% include './duplicate.jinja' %}
            {% endllmquery %}
            """)
        
        # Test relative path with ../ prefix
        with open(os.path.join(nested_dir, "test_parent_dir.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing parent dir include:
            {% include '../duplicate.jinja' %}
            {% endllmquery %}
            """)
        
        # Test relative path to sibling directory
        with open(os.path.join(nested_dir, "test_sibling_dir.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing sibling dir include:
            {% include '../sibling/duplicate.jinja' %}
            {% endllmquery %}
            """)
        
        # Test multiple levels up relative path from deep
        with open(os.path.join(deep_dir, "test_multiple_parent.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing multiple parent levels:
            {% include '../../duplicate.jinja' %}
            {% endllmquery %}
            """)
        
        # Test non-existent paths with various prefixes
        with open(os.path.join(nested_dir, "test_missing_no_prefix.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing missing file with no prefix:
            {% include 'non_existent.jinja' %}
            {% endllmquery %}
            """)
            
        with open(os.path.join(nested_dir, "test_missing_dot_prefix.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing missing file with ./ prefix:
            {% include './non_existent.jinja' %}
            {% endllmquery %}
            """)
            
        with open(os.path.join(nested_dir, "test_missing_dotdot_prefix.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing missing file with ../ prefix:
            {% include '../non_existent.jinja' %}
            {% endllmquery %}
            """)
            
        yield {
            "main_dir": main_dir,
            "template_dir": template_dir,
            "nested_dir": nested_dir,
            "deep_dir": deep_dir,
            "sibling_dir": sibling_dir,
            "test_id": test_id
        }

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_absolute_path_loads_from_searchpath(mock_llm_client, path_resolution_test_dirs):
    """
    Test that absolute paths are resolved against the search path, not CWD,
    even when multiple files with the same name exist.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/test_absolute.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content from template_dir
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from template_dir/duplicate.jinja, not from root or nested
        assert "CONTENT FROM TEMPLATE DIR" in prompt
        assert "CONTENT FROM ROOT" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert "CONTENT FROM SIBLING DIR" not in prompt
        assert path_resolution_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_current_dir_relative_path(mock_llm_client, path_resolution_test_dirs):
    """
    Test that ./ paths correctly load from the current template directory,
    not from any other location.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/test_current_dir.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content from the nested directory
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from nested_dir/duplicate.jinja
        assert "CONTENT FROM NESTED DIR" in prompt
        assert "CONTENT FROM ROOT" not in prompt
        assert "CONTENT FROM TEMPLATE DIR" not in prompt
        assert "CONTENT FROM SIBLING DIR" not in prompt
        assert path_resolution_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_parent_dir_relative_path(mock_llm_client, path_resolution_test_dirs):
    """
    Test that ../ paths correctly load from the parent template directory,
    not from any other location.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/test_parent_dir.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content from the parent template directory
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from template_dir/duplicate.jinja
        assert "CONTENT FROM TEMPLATE DIR" in prompt
        assert "CONTENT FROM ROOT" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert "CONTENT FROM SIBLING DIR" not in prompt
        assert path_resolution_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_sibling_dir_relative_path(mock_llm_client, path_resolution_test_dirs):
    """
    Test that paths with ../ correctly resolve to sibling directories.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/test_sibling_dir.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content from the sibling directory
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from sibling_dir/duplicate.jinja
        assert "CONTENT FROM SIBLING DIR" in prompt
        assert "CONTENT FROM ROOT" not in prompt
        assert "CONTENT FROM TEMPLATE DIR" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert path_resolution_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_multiple_levels_up_relative_path(mock_llm_client, path_resolution_test_dirs):
    """
    Test that paths with multiple ../ segments correctly resolve.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/deep/test_multiple_parent.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content from two directory levels up
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from template_dir/duplicate.jinja (two levels up from deep)
        assert "CONTENT FROM TEMPLATE DIR" in prompt
        assert "CONTENT FROM ROOT" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert "CONTENT FROM SIBLING DIR" not in prompt
        assert path_resolution_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

def test_missing_file_with_no_prefix(path_resolution_test_dirs):
    """
    Test that a proper error is thrown when attempting to include a non-existent file
    with no relative path prefix.
    """
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load template
        template = env.get_template("nested/test_missing_no_prefix.jinja")
        
        # Attempt to render - should raise EnhancedTemplateNotFound
        with pytest.raises(EnhancedTemplateNotFound) as excinfo:
            result = template.render()
        
        # Verify error details
        error_msg = str(excinfo.value)
        assert "non_existent.jinja" in error_msg
        assert "not found" in error_msg
        assert "Attempted paths:" in error_msg
        
        # Should mention the searchpath
        assert "from searchpath" in error_msg
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

def test_missing_file_with_dot_prefix(path_resolution_test_dirs):
    """
    Test that a proper error is thrown when attempting to include a non-existent file
    with ./ prefix.
    """
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load template
        template = env.get_template("nested/test_missing_dot_prefix.jinja")
        
        # Attempt to render - should raise EnhancedTemplateNotFound
        with pytest.raises(EnhancedTemplateNotFound) as excinfo:
            result = template.render()
        
        # Verify error details
        error_msg = str(excinfo.value)
        assert "./non_existent.jinja" in error_msg
        assert "not found" in error_msg
        assert "Attempted paths:" in error_msg
        
        # Should mention attempt to resolve relative to template
        assert "relative to" in error_msg
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

def test_missing_file_with_dotdot_prefix(path_resolution_test_dirs):
    """
    Test that a proper error is thrown when attempting to include a non-existent file
    with ../ prefix.
    """
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(path_resolution_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(path_resolution_test_dirs["template_dir"])
        
        # Load template
        template = env.get_template("nested/test_missing_dotdot_prefix.jinja")
        
        # Attempt to render - should raise EnhancedTemplateNotFound
        with pytest.raises(EnhancedTemplateNotFound) as excinfo:
            result = template.render()
        
        # Verify error details
        error_msg = str(excinfo.value)
        assert "../non_existent.jinja" in error_msg
        assert "not found" in error_msg
        assert "Attempted paths:" in error_msg
        
        # Should mention attempt to resolve relative to template
        assert "relative to" in error_msg
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd) 