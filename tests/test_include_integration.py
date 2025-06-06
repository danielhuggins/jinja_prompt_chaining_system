import os
import pytest
import yaml
import tempfile
import time
from unittest.mock import patch, Mock
from click.testing import CliRunner
from jinja_prompt_chaining_system.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def template_dir():
    """Create a directory structure with templates for integration testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create template directories
        partials_dir = os.path.join(tmpdir, "partials")
        os.makedirs(partials_dir, exist_ok=True)
        
        # Create main template with include
        with open(os.path.join(tmpdir, "main.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" temperature=0.7 %}
            I need information about {{ topic }}.
            
            {% include 'partials/additional_info.jinja' %}
            {% endllmquery %}
            """)
        
        # Create included template
        with open(os.path.join(partials_dir, "additional_info.jinja"), "w") as f:
            f.write("""
            Please provide details about its:
            - Origin
            - Key features
            - Common use cases
            """)
        
        # Create nested includes template
        with open(os.path.join(tmpdir, "nested_includes.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" temperature=0.7 %}
            {% include 'partials/header.jinja' %}
            
            Main content about {{ topic }}.
            
            {% include 'partials/footer.jinja' %}
            {% endllmquery %}
            """)
        
        # Create header and footer templates
        with open(os.path.join(partials_dir, "header.jinja"), "w") as f:
            f.write("""
            # {{ topic | upper }} ANALYSIS
            Date: {{ current_date }}
            """)
        
        with open(os.path.join(partials_dir, "footer.jinja"), "w") as f:
            f.write("""
            ---
            Thank you for using our {{ service_name }} service.
            """)
        
        # Create a template with both llmquery inside include AND include inside llmquery
        with open(os.path.join(tmpdir, "complex_includes.jinja"), "w") as f:
            f.write("""
            Initial content
            
            {% include 'partials/llm_section.jinja' %}
            
            Final content with another query:
            {% llmquery model="gpt-4" temperature=0.7 %}
            Summarize the differences between {{ topic_a }} and {{ topic_b }}.
            {% include 'partials/comparison_template.jinja' %}
            {% endllmquery %}
            """)
        
        # Create a template with llmquery inside
        with open(os.path.join(partials_dir, "llm_section.jinja"), "w") as f:
            f.write("""
            Section heading for {{ topic_a }}:
            
            {% llmquery model="gpt-3.5-turbo" temperature=0.5 %}
            Give me a brief overview of {{ topic_a }} in 3 sentences.
            {% endllmquery %}
            """)
        
        # Create comparison template
        with open(os.path.join(partials_dir, "comparison_template.jinja"), "w") as f:
            f.write("""
            Consider these aspects:
            - Performance
            - Cost
            - Ease of use
            - Community support
            """)
        
        # Create context file
        with open(os.path.join(tmpdir, "context.yaml"), "w") as f:
            f.write("""
            topic: "Python programming language"
            current_date: "2023-06-15"
            service_name: "AI Template System"
            topic_a: "Python"
            topic_b: "JavaScript"
            """)
        
        # Create a template with include and context variables
        with open(os.path.join(tmpdir, "dynamic_include.jinja"), "w") as f:
            f.write("""
            {% set template_name = topic | lower | replace(" ", "_") %}
            
            {% llmquery model="gpt-4" %}
            Let me tell you about {{ topic }}:
            
            {% include 'partials/' ~ template_name ~ '.jinja' ignore missing %}
            
            Additional information would go here.
            {% endllmquery %}
            """)
        
        # Create template files for dynamic inclusion
        with open(os.path.join(partials_dir, "python_programming_language.jinja"), "w") as f:
            f.write("""
            Python is a high-level, interpreted programming language created by Guido van Rossum.
            """)
        
        # Create a template with multiple includes and multiple llmqueries
        with open(os.path.join(tmpdir, "multi_query_include.jinja"), "w") as f:
            f.write("""
            {% llmquery model="gpt-4" %}
            Tell me about {{ topic }}:
            {% include 'partials/definition.jinja' %}
            {% endllmquery %}
            
            Now for a more detailed analysis:
            
            {% llmquery model="gpt-4" %}
            Provide an advanced analysis of {{ topic }}:
            {% include 'partials/analysis.jinja' %}
            {% endllmquery %}
            """)
        
        # Create definition and analysis templates
        with open(os.path.join(partials_dir, "definition.jinja"), "w") as f:
            f.write("""
            Basic definition and origin.
            """)
        
        with open(os.path.join(partials_dir, "analysis.jinja"), "w") as f:
            f.write("""
            Advanced analysis including:
            - Technical deep-dive
            - Comparison with alternatives
            - Future prospects
            """)
        
        yield tmpdir

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_cli_with_include(mock_logger, mock_llm_client, runner, template_dir):
    """Test the CLI with a template that includes another template within the llmquery."""
    # Setup mocks
    client_instance = Mock()
    client_instance.query.return_value = "Information about Python programming language including origin, features, and use cases."
    mock_llm_client.return_value = client_instance
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    # Create temporary log directory
    with tempfile.TemporaryDirectory() as log_dir:
        template_path = os.path.join(template_dir, "main.jinja")
        context_path = os.path.join(template_dir, "context.yaml")
        
        # Run CLI command
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
        
        # Verify CLI output
        assert result.exit_code == 0
        assert "Information about Python programming language" in result.output
        
        # Verify LLM was called with the included content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        assert "I need information about Python programming language" in prompt
        assert "Please provide details about its:" in prompt
        assert "Origin" in prompt
        assert "Key features" in prompt
        assert "Common use cases" in prompt
        
        # Verify log file was created - skipping YAML validation for now
        log_files = os.listdir(log_dir)
        assert len(log_files) > 0

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_nested_includes_in_llmquery(mock_logger, mock_llm_client, runner, template_dir):
    """Test template with nested includes within llmquery."""
    # Setup mocks
    client_instance = Mock()
    client_instance.query.return_value = "Analysis of Python programming language with header and footer."
    mock_llm_client.return_value = client_instance
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    # Run CLI command
    with tempfile.TemporaryDirectory() as log_dir:
        template_path = os.path.join(template_dir, "nested_includes.jinja")
        context_path = os.path.join(template_dir, "context.yaml")
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
        
        # Verify CLI output
        assert result.exit_code == 0
        assert "Analysis of Python programming language with header and footer." in result.output
        
        # Verify LLM was called with all the nested includes
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        assert "# PYTHON PROGRAMMING LANGUAGE ANALYSIS" in prompt
        assert "Date: 2023-06-15" in prompt
        assert "Main content about Python programming language" in prompt
        assert "Thank you for using our AI Template System service" in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_complex_includes_with_multiple_llmqueries(mock_logger, mock_llm_client, runner, template_dir):
    """Test template with both llmquery inside include and include inside llmquery."""
    # Setup mocks
    client_instance = Mock()
    
    # Set up different responses for different calls
    responses = [
        "Brief overview of Python in 3 sentences.",
        "Comparison between Python and JavaScript considering performance, cost, ease of use, and community support."
    ]
    client_instance.query.side_effect = responses
    
    mock_llm_client.return_value = client_instance
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    # Run CLI command
    with tempfile.TemporaryDirectory() as log_dir:
        template_path = os.path.join(template_dir, "complex_includes.jinja")
        context_path = os.path.join(template_dir, "context.yaml")
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
        
        # Verify CLI output
        assert result.exit_code == 0
        assert "Initial content" in result.output
        assert "Brief overview of Python in 3 sentences." in result.output
        assert "Final content with another query:" in result.output
        assert "Comparison between Python and JavaScript" in result.output
        
        # Verify LLM was called twice with correct prompts
        assert client_instance.query.call_count == 2
        
        # First call should be for the included template
        first_call_prompt = client_instance.query.call_args_list[0][0][0]
        assert "Give me a brief overview of Python in 3 sentences" in first_call_prompt
        
        # Second call should include the comparison template
        second_call_prompt = client_instance.query.call_args_list[1][0][0]
        assert "Summarize the differences between Python and JavaScript" in second_call_prompt
        assert "Consider these aspects:" in second_call_prompt
        assert "Performance" in second_call_prompt
        assert "Cost" in second_call_prompt
        assert "Ease of use" in second_call_prompt
        assert "Community support" in second_call_prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_dynamic_include_path(mock_logger, mock_llm_client, runner, template_dir):
    """Test template with dynamically constructed include path."""
    # Setup mocks
    client_instance = Mock()
    client_instance.query.return_value = "Information about Python with dynamically included content."
    mock_llm_client.return_value = client_instance
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    # Run CLI command
    with tempfile.TemporaryDirectory() as log_dir:
        template_path = os.path.join(template_dir, "dynamic_include.jinja")
        context_path = os.path.join(template_dir, "context.yaml")
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
        
        # Verify CLI output
        assert result.exit_code == 0
        assert "Information about Python with dynamically included content." in result.output
        
        # Verify LLM was called with dynamically included content
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        assert "Let me tell you about Python programming language:" in prompt
        assert "Python is a high-level, interpreted programming language created by Guido van Rossum." in prompt
        assert "Additional information would go here." in prompt

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_multi_query_with_includes(mock_logger, mock_llm_client, runner, template_dir):
    """Test template with multiple llmquery tags each having includes."""
    # Setup mocks
    client_instance = Mock()
    
    # Set up different responses for different calls
    responses = [
        "Basic information about Python programming language.",
        "Advanced analysis of Python programming language covering technical aspects, comparisons, and future prospects."
    ]
    client_instance.query.side_effect = responses
    
    mock_llm_client.return_value = client_instance
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    # Run CLI command
    with tempfile.TemporaryDirectory() as log_dir:
        template_path = os.path.join(template_dir, "multi_query_include.jinja")
        context_path = os.path.join(template_dir, "context.yaml")
        
        result = runner.invoke(main, [
            template_path,
            "--context", context_path,
            "--logdir", log_dir
        ], catch_exceptions=False)
        
        # Verify CLI output
        assert result.exit_code == 0
        assert "Basic information about Python programming language." in result.output
        assert "Now for a more detailed analysis:" in result.output
        assert "Advanced analysis of Python programming language" in result.output
        
        # Verify LLM was called twice with correct prompts and includes
        assert client_instance.query.call_count == 2
        
        # First call should include the definition
        first_call_prompt = client_instance.query.call_args_list[0][0][0]
        assert "Tell me about Python programming language:" in first_call_prompt
        assert "Basic definition and origin." in first_call_prompt
        
        # Second call should include the analysis
        second_call_prompt = client_instance.query.call_args_list[1][0][0]
        assert "Provide an advanced analysis of Python programming language:" in second_call_prompt
        assert "Advanced analysis including:" in second_call_prompt
        assert "Technical deep-dive" in second_call_prompt
        assert "Comparison with alternatives" in second_call_prompt
        assert "Future prospects" in second_call_prompt
        
        # Verify log files were created for both queries
        # With the new run-based logging, logs are now in run_*/llmcalls/
        run_dirs = [d for d in os.listdir(log_dir) if d.startswith("run_")]
        assert len(run_dirs) == 1
        
        llmcalls_dir = os.path.join(log_dir, run_dirs[0], "llmcalls")
        assert os.path.exists(llmcalls_dir)
        
        log_files = os.listdir(llmcalls_dir)
        assert len(log_files) == 2

@patch('jinja_prompt_chaining_system.parser.LLMClient')
@patch('jinja_prompt_chaining_system.parser.LLMLogger')
def test_missing_dynamic_include(mock_logger, mock_llm_client, runner, template_dir):
    """Test behavior when a dynamically included template is missing but with ignore missing flag."""
    # Setup mocks
    client_instance = Mock()
    client_instance.query.return_value = "Information with missing include gracefully handled."
    mock_llm_client.return_value = client_instance
    
    logger_instance = Mock()
    mock_logger.return_value = logger_instance
    
    # Modify context to use a non-existent template
    modified_context = os.path.join(template_dir, "modified_context.yaml")
    with open(modified_context, "w") as f:
        f.write("""
        topic: "Rust programming language"
        current_date: "2023-06-15"
        service_name: "AI Template System"
        topic_a: "Rust"
        topic_b: "Go"
        """)
    
    # Run CLI command
    with tempfile.TemporaryDirectory() as log_dir:
        template_path = os.path.join(template_dir, "dynamic_include.jinja")
        
        result = runner.invoke(main, [
            template_path,
            "--context", modified_context,
            "--logdir", log_dir
        ], catch_exceptions=False)
        
        # Verify CLI executed successfully (ignore missing should prevent failure)
        assert result.exit_code == 0
        
        # Verify LLM was called without the included content (since it was missing)
        client_instance.query.assert_called_once()
        prompt = client_instance.query.call_args[0][0]
        assert "Let me tell you about Rust programming language:" in prompt
        assert "Additional information would go here." in prompt
        # The missing include content should not be present
        assert "Rust is a high-level" not in prompt 