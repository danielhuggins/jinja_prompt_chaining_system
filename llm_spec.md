# Jinja Prompt Chaining System Specification

## 1. Objectives

1. Provide a simple Jinja‑based prompt‑chaining engine with an `{% llmquery … %}` / `{% endllmquery %}` tag.
2. Expose a single CLI (`jinja-run`) that takes:
   * **template** path (`.jinja`)
   * **context** YAML (`--context ctx.yaml`)
   * optional **output** path (`--out out.txt`)
   * optional **log directory** (`--logdir logs/`)
   * optional **run name** (`--name "experiment-1"`)
3. Let tag parameters be arbitrary Jinja expressions (no `with` keyword), where separators between `name=expr` pairs can be **spaces**, **commas**, or **both**.
4. Default to streaming; allow `stream=false`.
5. Implement a run-based logging system that organizes all related logs in a structured directory:
   * Create a unique timestamp-named directory (`run_YYYY-MM-DDThh-mm-ss-ffffff` or `run_YYYY-MM-DDThh-mm-ss-ffffff_name`) for each template execution 
   * Store a complete, unmodified copy of the context data in `context.yaml`
   * Preserve template metadata in `metadata.yaml` (template path, context file path, timestamp)
   * Log all LLM API interactions in the `llmcalls/` subdirectory
   * For non-streaming requests: Dump the exact OpenAI request and response
   * For streaming requests: Dump the exact request and reconstruct the full response from streamed chunks
   * Include a `done: true` flag to indicate when streaming is complete
6. Support both synchronous and asynchronous template rendering:
   * Enable `{% llmquery %}` tag usage in both sync and async environments
   * Allow seamless integration with async web frameworks and applications
   * Provide efficient non-blocking I/O during LLM API calls in async mode
7. Disable HTML escaping by default:
   * Prevent automatic HTML escaping of special characters like `<`, `>`, `&`
   * Ensure LLM queries and responses preserve exact formatting and characters
   * This behavior is appropriate for prompt engineering where HTML escaping is rarely desired
8. Provide exportable API functions for library users:
   * Enable programmatic rendering of prompt templates with the same semantics as the CLI
   * Support both synchronous and asynchronous rendering
   * Accept both file paths and Python dictionaries for context data
   * Maintain the same logging and output capabilities as the CLI
9. Support relative includes:
   * Treat includes beginning with './' or '../' as relative to the including template's directory
   * Maintain backward compatibility with standard Jinja includes
   * Enable more flexible template organization with nested directory structures
   * Allow template reuse with proper path resolution regardless of the invocation directory

## 2. Project Structure

```
jinja_prompt_chaining_system/        ← Git repo & distribution name
├─ src/
│  └─ jinja_prompt_chaining_system/  ← importable Python package
│     ├─ __init__.py
│     ├─ chains/
│     ├─ templates/
│     ├─ parser.py
│     ├─ cli.py
│     ├─ logger.py                   ← Contains LLMLogger and RunLogger classes
│     ├─ api.py                      ← Contains exportable API functions
│     └─ utils.py
├─ tests/
├─ examples/
├─ setup.cfg                         ← name = jinja_prompt_chaining_system
├─ pyproject.toml
└─ README.md
```

## 3. CLI Usage

```bash
jinja-run path/to/template.jinja \
  [key1=value1 key2=value2 ...] \
  [--context ctx.yaml] \
  [--out out.txt] \
  [--logdir logs/] \
  [--name "experiment-1"]
```

The CLI supports two methods for providing context data:

1. **Inline context** as key-value pairs (must come before any options):
   * Specify key-value pairs directly on the command line
   * Example: `name=Alice age=30 location=London`
   * Values are parsed as YAML (strings, numbers, booleans, etc.)
   * Complex values should be quoted: `preferences='{"color": "blue"}'`
   * File references: Use `@file.txt` to load file contents: `message=@input.txt`

2. **File-based context** with `--context` or `-c`:
   * Specify a YAML file path containing context data
   * Example: `--context data.yaml`

If both methods are used, inline key-value pairs will override values from the context file.

### 3.1 File References in Key-Value Pairs

When a value in a key-value pair starts with "@", it's treated as a file reference. The system reads the content of the specified file and uses it as the value.

For example:
```bash
jinja-run template.jinja message=@input.txt
```

This loads the content of "input.txt" and assigns it to the "message" variable in the template context.

This is particularly useful for:
* Long text content that would be unwieldy on the command line
* Pre-written content you want to reuse
* Multiline text that would be difficult to quote properly on the command line

The file path is relative to the current working directory. Both absolute and relative paths are supported:
```bash
# Relative path
jinja-run template.jinja message=@inputs/message.txt

# Absolute path
jinja-run template.jinja message=@/path/to/inputs/message.txt
```

## 4. API Usage

The library provides exportable functions for programmatically rendering prompt templates:

### 4.1. Synchronous API

```python
from jinja_prompt_chaining_system import render_prompt

# Basic usage with file paths
result = render_prompt(
    template_path="path/to/template.jinja",
    context="path/to/context.yaml"
)

# Using a Python dictionary for context
context = {"name": "World", "settings": {"model": "gpt-4o-mini"}}
result = render_prompt(
    template_path="path/to/template.jinja",
    context=context
)

# With output file and logging
result = render_prompt(
    template_path="path/to/template.jinja",
    context=context,
    out="output.txt",
    logdir="logs/"
)

# With named run 
result = render_prompt(
    template_path="path/to/template.jinja",
    context=context,
    logdir="logs/",
    name="experiment-1"
)
```

### 4.2. Asynchronous API

```python
import asyncio
from jinja_prompt_chaining_system import render_prompt_async

async def main():
    # Basic async usage with file paths
    result = await render_prompt_async(
        template_path="path/to/template.jinja",
        context="path/to/context.yaml"
    )
    
    # With dictionary context and logging
    context = {"name": "World", "settings": {"model": "gpt-4o-mini"}}
    result = await render_prompt_async(
        template_path="path/to/template.jinja",
        context=context,
        out="output.txt",
        logdir="logs/"
    )
    
    # With named run
    result = await render_prompt_async(
        template_path="path/to/template.jinja",
        context=context,
        logdir="logs/",
        name="experiment-a"
    )

# Run the async function
asyncio.run(main())
```

## 5. Run Naming

When using the `--name` parameter (CLI) or `name` parameter (API), the system will append the provided name to the timestamp in the run directory name:

```
logs/
└─ run_2023-07-25T14-32-18-567891_experiment-1/   # Named run
└─ run_2023-07-25T14-32-20-123456/                # Default unnamed run
```

Run names:
- Must not contain characters that are invalid in a directory name (/, \, etc.)
- Are sanitized by replacing invalid characters with underscores
- Allow you to more easily identify and organize related runs
- Are stored in the metadata.yaml file under the "name" field

## 6. `{% llmquery %}` Tag Semantics

### 6.1. Syntax

```jinja
{% llmquery
   param1=expr1 [ , param2=expr2 ] …
%}
  …template body…
{% endllmquery %}
```

* **Parameters**: one or more `name=expr` pairs. Separators between pairs can be **spaces**, **commas**, or **both**.
* **Values**: any valid Jinja expression (literals, variables, filters, etc.).
* **Body**: the single "user" message sent to the LLM.

### 6.2. Examples

Using **spaces** only:

```jinja
{% llmquery model="gpt-4o-mini" temperature=0.7 max_tokens=150 stream=false %}
Summarise the plot of Hamlet.
{% endllmquery %}
```

Using **commas** only:

```jinja
{% llmquery model="gpt-4o-mini", temperature=0.7, max_tokens=150, stream=false %}
Summarise the plot of Hamlet.
{% endllmquery %}
```

Mixing **spaces** and **commas**:

```jinja
{% llmquery
   model="gpt-4o-mini", temperature=0.7 max_tokens=150, stream=false
%}
Summarise the plot of Hamlet.
{% endllmquery %}
```

*Line breaks* are allowed between parameters without special escapes.

### 6.3. Async Support

The `{% llmquery %}` tag works seamlessly in both synchronous and asynchronous contexts:

```python
# Synchronous usage
template = env.get_template('template.jinja')
result = template.render()

# Asynchronous usage
template = env.get_template('template.jinja')
result = await template.render_async()
```

When using asynchronous rendering:
- LLM API calls are non-blocking, allowing for better performance in web applications
- Streaming responses can be processed more efficiently
- The system automatically detects and handles async vs. sync contexts

## 7. Global `llmquery()` Function

In addition to the `{% llmquery %}` tag, the system provides a global `llmquery()` function that can be called directly in Jinja expressions. This function has identical semantics to the tag version but uses a function call syntax.

### 7.1. Syntax

```jinja
{{ llmquery(prompt="Your prompt text", param1=expr1, param2=expr2, ...) }}
```

* **Prompt**: Required parameter containing the text to send to the LLM.
* **Parameters**: Additional parameters identical to those accepted by the tag version (model, temperature, max_tokens, etc.).
* **Return Value**: The function returns the LLM's response as a string.

### 7.2. Examples

Basic usage:

```jinja
{{ llmquery(prompt="Summarise the plot of Hamlet.", model="gpt-4o-mini", temperature=0.7) }}
```

Using a variable for the prompt:

```jinja
{% set my_prompt = "Summarise the plot of " + title %}
{{ llmquery(prompt=my_prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=150) }}
```

Multi-line prompts:

```jinja
{{ llmquery(
    prompt="Generate a list of 5 creative names for a pet " + animal_type + " that lives in " + location + " with the personality: " + personality,
    model="gpt-4o-mini",
    temperature=0.8
) }}
```

Alternatively, you can use the tag syntax for multi-line prompts:

```jinja
{% llmquery model="gpt-4o-mini" temperature=0.8 %}
Generate a list of 5 creative names for:
- A pet {{ animal_type }}
- That lives in {{ location }}
- With the personality: {{ personality }}
{% endllmquery %}
```

### 7.3. Async Support

Like the tag version, the `llmquery()` function works seamlessly in both synchronous and asynchronous contexts:

```python
# Works in both render() and render_async() contexts
template = env.get_template('template.jinja')

# Synchronous
result = template.render()

# Asynchronous
result = await template.render_async()
```

## 8. Logging Format

### 8.1. Run-Based Directory Structure

Each template execution creates a unique run with this precise directory structure:

```
logs/                                                   # Base log directory specified with --logdir
└─ run_2023-07-25T14-32-18-567891/                     # Run directory with UTC timestamp
   ├─ metadata.yaml                                    # Execution metadata
   ├─ context.yaml                                     # Exact copy of rendered context
   └─ llmcalls/                                        # All LLM API interactions
      ├─ template_2023-07-25T14-32-18-567923_0.log.yaml  # First LLM call
      └─ template_2023-07-25T14-32-19-123456_1.log.yaml  # Subsequent LLM calls if any
```

The naming convention follows:
- **Run directory**: `run_YYYY-MM-DDThh-mm-ss-ffffff` (UTC timestamp with microsecond precision)
- **LLM log files**: `<template_name>_YYYY-MM-DDThh-mm-ss-ffffff_<counter>.log.yaml`

The `RunLogger` class manages this directory structure:
1. Creates the timestamped run directory upon instantiation
2. Stores the rendered context as a complete YAML dump
3. Records template metadata (path, timestamp, context file path)
4. Provides an `LLMLogger` instance specifically for the `llmcalls/` subdirectory

### 8.2. Metadata and Context Files

The `metadata.yaml` file contains:

```yaml
# Example: logs/run_2023-07-25T14-32-18-567891/metadata.yaml
timestamp: '2023-07-25T14:32:18.567891+00:00'  # ISO 8601 UTC timestamp with timezone
template: /absolute/path/to/template.jinja     # Absolute path to the template file
context_file: /absolute/path/to/context.yaml   # Absolute path to the context file
```

The `context.yaml` file is an exact YAML serialization of the context object used in rendering:

```yaml
# Example: logs/run_2023-07-25T14-32-18-567891/context.yaml
name: World
settings:
  model: gpt-4o-mini
  temperature: 0.7
user_data:
  location: New York
  preferences:
    - topic: Science
    - topic: History
```

Both files are generated at the beginning of the run before template rendering starts.

### 8.3. LLM Call Log Format

Each LLM API interaction generates a log file in the `llmcalls/` directory that exactly mirrors the OpenAI API request and response structure. The implementation preserves:

- All request parameters (model, temperature, max_tokens, etc.)
- Complete message content with exact whitespace
- Full response structure including metadata (tokens, ID, etc.)
- Streaming vs. non-streaming behavior

#### 8.3.1. Non-Streaming Example

```yaml
# Example: logs/run_2023-07-25T14-32-18-567891/llmcalls/template_2023-07-25T14-32-18-567923_0.log.yaml
timestamp: '2023-07-25T14:32:18.567923+00:00'
request:
  model: gpt-4o-mini
  temperature: 0.7
  stream: false
  max_tokens: 400
  messages:
    - role: user
      content: |   # markdown
        Summarise the plot of Hamlet.

response:
  id: chatcmpl-abc123
  model: gpt-4o-mini
  choices:
    - index: 0
      message:
        role: assistant
        content: |   # markdown
          Prince Hamlet of Denmark encounters the ghost of his father, who reveals he was murdered by Hamlet's uncle Claudius. Claudius has since married Hamlet's mother and become king. Hamlet seeks to avenge his father through an elaborate plan, feigning madness while confirming Claudius's guilt. His erratic behavior leads to the accidental killing of Polonius, driving Polonius's daughter Ophelia to suicide and his son Laertes to seek revenge. In a climactic duel between Hamlet and Laertes, Hamlet is poisoned, but before dying, he kills Claudius and finally avenges his father.

  finish_reason: stop
  usage:
    prompt_tokens: 47
    completion_tokens: 123
    total_tokens: 170
```

#### 8.3.2. Streaming Example

```yaml
# Example: logs/run_2023-07-25T14-32-18-567891/llmcalls/template_2023-07-25T14-32-18-567923_0.log.yaml
timestamp: '2023-07-25T14:32:18.567923+00:00'
request:
  model: gpt-4o-mini
  temperature: 0.7
  stream: true
  max_tokens: 400
  messages:
    - role: user
      content: |   # markdown
        Summarise the plot of Hamlet.

response:
  id: chatcmpl-abc123
  model: gpt-4o-mini
  choices:
    - index: 0
      message:
        role: assistant
        content: |   # markdown
          Prince Hamlet of Denmark encounters the ghost of his father, who reveals he was murdered by Hamlet's uncle Claudius. Claudius has since married Hamlet's mother and become king. Hamlet seeks to avenge his father through an elaborate plan, feigning madness while confirming Claudius's guilt. His erratic behavior leads to the accidental killing of Polonius, driving Polonius's daughter Ophelia to suicide and his son Laertes to seek revenge. In a climactic duel between Hamlet and Laertes, Hamlet is poisoned, but before dying, he kills Claudius and finally avenges his father.

  finish_reason: stop
  usage:
    prompt_tokens: 47
    completion_tokens: 123
    total_tokens: 170
  done: true  # Indicates streaming is complete
```

### 8.4. Technical Implementation Details

The logging system consists of two main classes:

1. **RunLogger**: Manages the run-level organization
   - Creates the timestamped run directory
   - Saves context and metadata files
   - Provides LLMLogger instances for the run
   
2. **LLMLogger**: Handles individual LLM API call logging
   - Maintains the existing logging format for compatibility
   - Creates log files with properly formatted YAML
   - Handles both streaming and non-streaming responses

The `RunLogger` is initialized in the CLI with:

```python
run_logger = RunLogger(log_dir)
run_id = run_logger.start_run(
    metadata={"template": template_path, "context_file": context_path},
    context=context_data
)
llm_logger = run_logger.get_llm_logger(run_id)
```

All whitespace in log files is preserved exactly as in the original content:
- Leading whitespace is preserved without using indentation indicators
- Trailing whitespace/newlines are preserved without using chomp indicators (+/-)
- The 3-space buffer after the pipe character allows for format modifications without changing overall structure

## 9. Testing

* **Unit tests** with `pytest`.
* **CLI end-to-end tests** that invoke:

```bash
python -m jinja_prompt_chaining_system.cli <template.jinja> --context ctx.yaml --out out.txt --logdir logs/
```

and verify both output and the run directory structure with all its contents.

Run all tests with:

```bash
pytest -n auto
```

## 10. Release

* **Publish** `jinja_prompt_chaining_system` wheel to PyPI.
* **Docs** (optional) via MkDocs.

## 11. Relative Includes

The system supports relative includes in Jinja templates for more flexible and maintainable template organization.

### 11.1. Syntax

```jinja
{% include './relative/path/to/template.jinja' %}
{% include '../parent/directory/template.jinja' %}
```

* **Relative path**: Starts with `.` or `..` followed by a directory separator (`/`)
* **Resolution**: Paths are resolved relative to the directory of the template that includes them, not the directory of the main template or the current working directory

### 11.2. Examples

Given this directory structure:

```
templates/
├─ main.jinja
├─ partials/
│  ├─ header.jinja
│  ├─ footer.jinja
│  └─ sections/
│     └─ content.jinja
└─ shared/
   └─ common.jinja
```

**Example 1**: From `main.jinja` including a template in a subdirectory:

```jinja
{% include 'partials/header.jinja' %}           {# Standard include #}
{% include './partials/header.jinja' %}         {# Equivalent relative include #}

{% llmquery model="gpt-4" %}
Content with included section:
{% include './partials/sections/content.jinja' %}
{% endllmquery %}
```

**Example 2**: From `partials/sections/content.jinja` including a template in a parent directory:

```jinja
Here is the main content.

{% include '../footer.jinja' %}                 {# Include from same directory as header.jinja #}
{% include '../../shared/common.jinja' %}       {# Include from templates/shared/ directory #}
```

### 11.3. Benefits

1. **Portability**: Templates can be moved together with their dependencies without breaking include paths
2. **Clarity**: Path relationships between templates are explicit in the include statements
3. **Modularity**: Templates can be organized in subdirectories and included from anywhere using relative paths
4. **Maintenance**: Directory restructuring is easier as related templates maintain their relative relationships
5. **Isolation**: Template modules can be developed independently in their own directories

### 11.4. Implementation Details

The relative include system uses a custom Jinja `FileSystemLoader` that:

1. Detects includes starting with `.` or `..` 
2. Resolves the path relative to the including template's directory
3. Falls back to standard Jinja include behavior for non-relative paths
4. Maintains full compatibility with existing templates
