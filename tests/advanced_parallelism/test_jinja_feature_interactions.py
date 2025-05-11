import pytest
import os
import tempfile
import asyncio
import time
from unittest.mock import patch, AsyncMock, Mock

from src.jinja_prompt_chaining_system.parallel_integration import render_template_parallel


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_macro_with_queries(mock_query_async, mock_query):
    """
    Test macros containing LLM queries.
    
    This tests that LLM queries inside macros are properly collected and executed.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "inside macro" in prompt.lower():
            return f"MACRO_RESULT: {prompt}"
        else:
            return f"NORMAL_RESULT: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "inside macro" in prompt.lower():
            return f"MACRO_RESULT: {prompt}"
        else:
            return f"NORMAL_RESULT: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with macros containing LLM queries
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Define a macro with an LLM query inside #}
        {% macro query_llm(topic) %}
            {% set result = llmquery(prompt="Query inside macro about " + topic) %}
            Macro result for {{ topic }}: {{ result }}
        {% endmacro %}
        
        {# Normal query outside macro #}
        {% set normal_result = llmquery(prompt="Normal query outside macro") %}
        Normal result: {{ normal_result }}
        
        {# Call the macro multiple times #}
        {{ query_llm("topic1") }}
        {{ query_llm("topic2") }}
        
        {# Store macro result in a variable #}
        {% set macro_output %}{{ query_llm("topic3") }}{% endset %}
        Captured macro output: {{ macro_output }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nExecution order:")
        for i, prompt in enumerate(execution_order):
            print(f"{i+1}. {prompt}")
        
        print("\nTemplate result:")
        print(result)
        
        # Check that all queries were executed
        normal_idx = next((i for i, p in enumerate(execution_order) if "normal query" in p.lower()), -1)
        topic1_idx = next((i for i, p in enumerate(execution_order) if "topic1" in p.lower()), -1)
        topic2_idx = next((i for i, p in enumerate(execution_order) if "topic2" in p.lower()), -1)
        topic3_idx = next((i for i, p in enumerate(execution_order) if "topic3" in p.lower()), -1)
        
        # Assert all queries were executed
        assert normal_idx >= 0, "Normal query not executed"
        assert topic1_idx >= 0, "Topic1 query inside macro not executed"
        assert topic2_idx >= 0, "Topic2 query inside macro not executed"
        assert topic3_idx >= 0, "Topic3 query inside macro not executed"
        
        # Verify content
        assert "Normal result: NORMAL_RESULT: Normal query outside macro" in result
        assert "Macro result for topic1: MACRO_RESULT: Query inside macro about topic1" in result
        assert "Macro result for topic2: MACRO_RESULT: Query inside macro about topic2" in result
        assert "Macro result for topic3: MACRO_RESULT: Query inside macro about topic3" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_template_inheritance(mock_query_async, mock_query):
    """
    Test template inheritance with LLM queries in both base and child templates.
    
    This tests that LLM queries in parent and child templates are properly collected
    and executed when using template inheritance.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "base" in prompt.lower():
            return f"BASE_RESULT: {prompt}"
        elif "child" in prompt.lower():
            return f"CHILD_RESULT: {prompt}"
        else:
            return f"RESULT: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "base" in prompt.lower():
            return f"BASE_RESULT: {prompt}"
        elif "child" in prompt.lower():
            return f"CHILD_RESULT: {prompt}"
        else:
            return f"RESULT: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create temporary directory for templates
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create base template
        base_template_path = os.path.join(temp_dir, "base.jinja")
        with open(base_template_path, "w") as f:
            f.write("""
            {% set base_result = llmquery(prompt="Query in base template") %}
            
            Base template header: {{ base_result }}
            
            {% block content %}
                Default content from base template
            {% endblock %}
            
            Base template footer
            """)
        
        # Create child template that extends the base
        child_template_path = os.path.join(temp_dir, "child.jinja")
        with open(child_template_path, "w") as f:
            f.write("""
            {% extends "base.jinja" %}
            
            {% block content %}
                {% set child_result = llmquery(prompt="Query in child template") %}
                
                Child template content: {{ child_result }}
                
                {% set combined_result = llmquery(prompt="Combined query using base=" + base_result + " and child=" + child_result) %}
                Combined result: {{ combined_result }}
            {% endblock %}
            """)
        
        try:
            # Execute with parallel enabled
            result = render_template_parallel(child_template_path, {}, enable_parallel=True)
            
            # Print debug information
            print("\nExecution order:")
            for i, prompt in enumerate(execution_order):
                print(f"{i+1}. {prompt}")
            
            print("\nTemplate result:")
            print(result)
            
            # Check that all queries were executed
            base_idx = next((i for i, p in enumerate(execution_order) if "base template" in p.lower()), -1)
            child_idx = next((i for i, p in enumerate(execution_order) if "child template" in p.lower()), -1)
            combined_idx = next((i for i, p in enumerate(execution_order) if "combined query" in p.lower()), -1)
            
            # Assert all queries were executed
            assert base_idx >= 0, "Base template query not executed"
            assert child_idx >= 0, "Child template query not executed"
            assert combined_idx >= 0, "Combined query not executed"
            
            # The combined query should be executed after both base and child queries
            assert combined_idx > base_idx, "Combined query executed before base query"
            assert combined_idx > child_idx, "Combined query executed before child query"
            
            # Verify content
            assert "Base template header: BASE_RESULT: Query in base template" in result
            assert "Child template content: CHILD_RESULT: Query in child template" in result
            
            # The combined result should contain both base and child results
            combined_line = None
            for line in result.splitlines():
                if "Combined result:" in line:
                    combined_line = line.strip()
                    break
            
            assert combined_line is not None, "Combined result line not found in output"
            assert "BASE_RESULT" in combined_line, "Combined result missing base result"
            assert "CHILD_RESULT" in combined_line, "Combined result missing child result"
            
        finally:
            # No need to clean up as the temporary directory will be removed automatically
            pass


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_filters_applied_to_query_results(mock_query_async, mock_query):
    """
    Test Jinja filters applied to LLM query results.
    
    This tests that Jinja filters can be applied to LLM query results
    and that dependencies are properly tracked.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "raw text" in prompt.lower():
            return "raw TEXT for FILTERING"
        elif "filtered" in prompt.lower():
            return f"RESULT using filtered input: {prompt}"
        
        return f"RESULT: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "raw text" in prompt.lower():
            return "raw TEXT for FILTERING"
        elif "filtered" in prompt.lower():
            return f"RESULT using filtered input: {prompt}"
        
        return f"RESULT: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with filters applied to LLM query results
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Get raw text from LLM #}
        {% set raw_text = llmquery(prompt="Get raw text") %}
        Raw text: {{ raw_text }}
        
        {# Apply filters to the raw text #}
        Uppercase: {{ raw_text|upper }}
        Lowercase: {{ raw_text|lower }}
        Title case: {{ raw_text|title }}
        Word count: {{ raw_text|split|length }}
        
        {# Use filtered text in another query #}
        {% set upper_text = raw_text|upper %}
        {% set filtered_result = llmquery(prompt="Process filtered text: " + upper_text) %}
        Result with uppercase input: {{ filtered_result }}
        
        {# Chain filters #}
        {% set chained_result = llmquery(prompt="Process with chained filters: " + raw_text|lower|replace("text", "content")|title) %}
        Result with chained filters: {{ chained_result }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nExecution order:")
        for i, prompt in enumerate(execution_order):
            print(f"{i+1}. {prompt}")
        
        print("\nTemplate result:")
        print(result)
        
        # Check that all queries were executed
        raw_idx = next((i for i, p in enumerate(execution_order) if "raw text" in p.lower()), -1)
        filtered_idx = next((i for i, p in enumerate(execution_order) if "filtered text" in p.lower()), -1)
        chained_idx = next((i for i, p in enumerate(execution_order) if "chained filters" in p.lower()), -1)
        
        # Assert all queries were executed
        assert raw_idx >= 0, "Raw text query not executed"
        assert filtered_idx >= 0, "Filtered text query not executed"
        assert chained_idx >= 0, "Chained filters query not executed"
        
        # Verify the order: filtered queries should come after the raw query
        assert filtered_idx > raw_idx, "Filtered query executed before raw query"
        assert chained_idx > raw_idx, "Chained filters query executed before raw query"
        
        # Verify content - the original raw text
        assert "Raw text: raw TEXT for FILTERING" in result
        
        # Verify filters were applied
        assert "Uppercase: RAW TEXT FOR FILTERING" in result
        assert "Lowercase: raw text for filtering" in result
        assert "Title case: Raw Text For Filtering" in result
        assert "Word count: 4" in result
        
        # Verify the filtered query contains uppercase
        filtered_line = None
        for line in result.splitlines():
            if "Result with uppercase input:" in line:
                filtered_line = line.strip()
                break
        
        assert filtered_line is not None, "Filtered result line not found"
        assert "RAW TEXT FOR FILTERING" in filtered_line, "Uppercase filter not applied correctly"
        
        # Verify the chained filters query
        chained_line = None
        for line in result.splitlines():
            if "Result with chained filters:" in line:
                chained_line = line.strip()
                break
        
        assert chained_line is not None, "Chained result line not found"
        assert "Raw Content For Filtering" in chained_line, "Chained filters not applied correctly"
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path)


@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query')
@patch('src.jinja_prompt_chaining_system.llm.LLMClient.query_async')
def test_set_expressions_with_queries(mock_query_async, mock_query):
    """
    Test set expressions with LLM queries.
    
    This tests that LLM queries can be used in set expressions and
    that dependencies are properly tracked.
    """
    
    # Track execution order
    execution_order = []
    
    # Set up synchronous mock
    def mock_sync_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Sync query called with prompt: {prompt}")
        
        if "first" in prompt.lower():
            return "FIRST PART"
        elif "second" in prompt.lower():
            return "SECOND PART"
        elif "combined" in prompt.lower():
            return f"COMBINED: {prompt}"
        
        return f"RESULT: {prompt}"
    
    mock_query.side_effect = mock_sync_query
    
    # Set up asynchronous mock
    async def mock_async_query(prompt, params=None, stream=False):
        execution_order.append(prompt)
        print(f"Async query called with prompt: {prompt}")
        await asyncio.sleep(0.01)
        
        if "first" in prompt.lower():
            return "FIRST PART"
        elif "second" in prompt.lower():
            return "SECOND PART"
        elif "combined" in prompt.lower():
            return f"COMBINED: {prompt}"
        
        return f"RESULT: {prompt}"
    
    mock_query_async.side_effect = mock_async_query
    
    # Create a template with set expressions using LLM queries
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jinja', delete=False) as f:
        template_content = """
        {# Basic set expressions #}
        {% set first_part = llmquery(prompt="Generate first part") %}
        {% set second_part = llmquery(prompt="Generate second part") %}
        
        First part: {{ first_part }}
        Second part: {{ second_part }}
        
        {# Set expression with captured content #}
        {% set combined %}
            {{ first_part }} + {{ second_part }}
        {% endset %}
        
        Combined (from set expression): {{ combined }}
        
        {# Query using the set expression result #}
        {% set query_with_combined = llmquery(prompt="Combined query using: " + combined|trim) %}
        Result using combined: {{ query_with_combined }}
        
        {# Nested set expressions with queries #}
        {% set outer %}
            {% set inner = llmquery(prompt="Nested query inside set expression") %}
            Outer wrapper with {{ inner }}
        {% endset %}
        
        Nested result: {{ outer }}
        """
        f.write(template_content)
        template_path = f.name
    
    try:
        # Execute with parallel enabled
        result = render_template_parallel(template_path, {}, enable_parallel=True)
        
        # Print debug information
        print("\nExecution order:")
        for i, prompt in enumerate(execution_order):
            print(f"{i+1}. {prompt}")
        
        print("\nTemplate result:")
        print(result)
        
        # Check that all queries were executed
        first_idx = next((i for i, p in enumerate(execution_order) if "first part" in p.lower()), -1)
        second_idx = next((i for i, p in enumerate(execution_order) if "second part" in p.lower()), -1)
        combined_idx = next((i for i, p in enumerate(execution_order) if "combined query" in p.lower()), -1)
        nested_idx = next((i for i, p in enumerate(execution_order) if "nested query" in p.lower()), -1)
        
        # Assert all queries were executed
        assert first_idx >= 0, "First part query not executed"
        assert second_idx >= 0, "Second part query not executed"
        assert combined_idx >= 0, "Combined query not executed"
        assert nested_idx >= 0, "Nested query not executed"
        
        # The combined query should come after both first and second
        assert combined_idx > first_idx, "Combined query executed before first part query"
        assert combined_idx > second_idx, "Combined query executed before second part query"
        
        # Verify content
        assert "First part: FIRST PART" in result
        assert "Second part: SECOND PART" in result
        assert "Combined (from set expression): FIRST PART + SECOND PART" in result
        assert "COMBINED: Combined query using: FIRST PART + SECOND PART" in result
        assert "Nested result: Outer wrapper with RESULT: Nested query inside set expression" in result
        
    finally:
        # Clean up the temporary file
        os.unlink(template_path) 