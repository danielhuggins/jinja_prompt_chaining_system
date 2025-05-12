import os
import pytest
import tempfile
from unittest.mock import patch, Mock
from jinja_prompt_chaining_system import create_environment

@pytest.fixture
def conflict_test_dirs():
    """Create a directory structure with same-named files in different locations to test conflict resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a unique identifier for this test
        test_id = "conflict_test_id"
        
        # Create a nested directory structure
        main_dir = tmpdir
        template_dir = os.path.join(main_dir, "templates")
        nested_dir = os.path.join(template_dir, "nested")
        
        # Create all directories
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        
        # Create same-named files in different locations with distinct content
        # We'll create the same filename in:
        # 1. CWD (main_dir)
        # 2. Search path (template_dir)
        # 3. Nested directory
        
        # Create files with the same name but different content
        with open(os.path.join(main_dir, "conflict.jinja"), "w") as f:
            f.write("CONTENT FROM CWD - " + test_id)
            
        with open(os.path.join(template_dir, "conflict.jinja"), "w") as f:
            f.write("CONTENT FROM SEARCH PATH - " + test_id)
            
        with open(os.path.join(nested_dir, "conflict.jinja"), "w") as f:
            f.write("CONTENT FROM NESTED DIR - " + test_id)
        
        # Create test templates
        
        # Test direct absolute include from a main template
        with open(os.path.join(template_dir, "direct_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing direct absolute include:
            {% include 'conflict.jinja' %}
            {% endllmquery %}
            """)
        
        # Test absolute include from a nested template
        with open(os.path.join(nested_dir, "nested_absolute_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing nested absolute include:
            {% include 'conflict.jinja' %}
            {% endllmquery %}
            """)
        
        # Test current dir relative include
        with open(os.path.join(nested_dir, "nested_current_dir_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing nested current dir include:
            {% include './conflict.jinja' %}
            {% endllmquery %}
            """)
        
        # Test parent dir relative include
        with open(os.path.join(nested_dir, "nested_parent_dir_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing nested parent dir include:
            {% include '../conflict.jinja' %}
            {% endllmquery %}
            """)
            
        # Nested template that includes an absolute path from another template
        with open(os.path.join(nested_dir, "nested_include_in_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Testing nested include in include:
            {% include './include_absolute.jinja' %}
            {% endllmquery %}
            """)
            
        # Template that includes an absolute path (will be included by another template)
        with open(os.path.join(nested_dir, "include_absolute.jinja"), "w") as f:
            f.write("""
            This template includes an absolute path:
            {% include 'conflict.jinja' %}
            """)
        
        yield {
            "main_dir": main_dir,
            "template_dir": template_dir,
            "nested_dir": nested_dir,
            "test_id": test_id
        }

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_direct_absolute_include_prefers_searchpath(mock_llm_client, conflict_test_dirs):
    """
    Test that a direct absolute include (from a template loaded with get_template) 
    prefers the search path over CWD, even when an identically named file exists in CWD.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(conflict_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(conflict_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("direct_include.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from search path, not CWD
        assert "CONTENT FROM SEARCH PATH" in prompt
        assert "CONTENT FROM CWD" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert conflict_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nested_absolute_include_prefers_searchpath(mock_llm_client, conflict_test_dirs):
    """
    Test that an absolute include from a nested template 
    prefers the search path over CWD, even when an identically named file exists in CWD.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(conflict_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(conflict_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/nested_absolute_include.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from search path, not CWD or nested dir
        assert "CONTENT FROM SEARCH PATH" in prompt
        assert "CONTENT FROM CWD" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert conflict_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nested_current_dir_include(mock_llm_client, conflict_test_dirs):
    """
    Test that a ./ include from a nested template correctly resolves to the nested directory,
    not the search path or CWD.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(conflict_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(conflict_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/nested_current_dir_include.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from nested dir, not search path or CWD
        assert "CONTENT FROM NESTED DIR" in prompt
        assert "CONTENT FROM SEARCH PATH" not in prompt
        assert "CONTENT FROM CWD" not in prompt
        assert conflict_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nested_parent_dir_include(mock_llm_client, conflict_test_dirs):
    """
    Test that a ../ include from a nested template correctly resolves to the parent directory (search path),
    not the CWD.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(conflict_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(conflict_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/nested_parent_dir_include.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # Should contain content from search path (parent dir), not CWD or nested dir
        assert "CONTENT FROM SEARCH PATH" in prompt
        assert "CONTENT FROM CWD" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert conflict_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

@patch('jinja_prompt_chaining_system.parser.LLMClient')
def test_nested_include_in_include(mock_llm_client, conflict_test_dirs):
    """
    Test the complex case of a nested include that itself includes an absolute path.
    This tests that our context tracking correctly carries through multiple levels of includes.
    """
    # Setup mock
    client_instance = Mock()
    client_instance.query.return_value = "Mock LLM response"
    mock_llm_client.return_value = client_instance
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the main test directory
        os.chdir(conflict_test_dirs["main_dir"])
        
        # Create environment with template_dir as search path
        env = create_environment(conflict_test_dirs["template_dir"])
        
        # Load and render the template
        template = env.get_template("nested/nested_include_in_include.jinja")
        result = template.render()
        
        # Check that the LLM was called with the correct content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        
        # The include_absolute.jinja includes conflict.jinja as an absolute path,
        # so it should use the search path version even in a nested include
        assert "CONTENT FROM SEARCH PATH" in prompt
        assert "CONTENT FROM CWD" not in prompt
        assert "CONTENT FROM NESTED DIR" not in prompt
        assert "This template includes an absolute path:" in prompt
        assert conflict_test_dirs["test_id"] in prompt
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd) 