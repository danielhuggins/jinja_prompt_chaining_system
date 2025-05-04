# Jinja Prompt Chaining System Specification

## 1. Objectives

1. Provide a simple Jinja‑based prompt‑chaining engine with an `{% llmquery … %}` / `{% endllmquery %}` tag.
2. Expose a single CLI (`jinja-run`) that takes:
   * **template** path (`.jinja`)
   * **context** YAML (`--context ctx.yaml`)
   * optional **output** path (`--out out.txt`)
   * optional **log directory** (`--logdir logs/`)
3. Let tag parameters be arbitrary Jinja expressions (no `with` keyword), where separators between `name=expr` pairs can be **spaces**, **commas**, or **both**.
4. Default to streaming; allow `stream=false`.
5. Log each API interaction in a log file named `<template>_<timestamp>.log.yaml` within the log directory that exactly mirrors the OpenAI request and response structures:
   * For non-streaming requests: Dump the exact OpenAI request and response
   * For streaming requests: Dump the exact request and reconstruct the full response from streamed chunks
   * Include a `done: true` flag to indicate when streaming is complete
6. Support both synchronous and asynchronous template rendering:
   * Enable `{% llmquery %}` tag usage in both sync and async environments
   * Allow seamless integration with async web frameworks and applications
   * Provide efficient non-blocking I/O during LLM API calls in async mode

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
  --context ctx.yaml \
  [--out out.txt] \
  [--logdir logs/]
```

## 4. `{% llmquery %}` Tag Semantics

### 4.1. Syntax

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

### 4.2. Examples

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

### 4.3. Async Support

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

## 5. Logging Format

```yaml
request:
  model: gpt-4o-mini
  temperature: 0.7
  stream: true
  max_tokens: 400
  messages:
    - role: user
      content: |
        Summarise the plot of Hamlet.
  tools:
    - type: function
      function:
        name: extract_pdf_text
        parameters:
          type: object
          properties:
            url: {type: string}

response:
  id: chatcmpl-abc123
  model: gpt-4o-mini
  choices:
    - index: 0
      message:
        role: assistant
        content: |
          Sure—here is a concise summary…
          (streamed text grows here)

  finish_reason: stop
  usage:
    prompt_tokens: 47
    completion_tokens: 123
    total_tokens: 170
  done: true
```

## 6. Testing

* **Unit tests** with `pytest`.
* **CLI end-to-end tests** that invoke:

```bash
python -m jinja_prompt_chaining_system.cli <template.jinja> --context ctx.yaml --out out.txt --logdir logs/
```

and verify both output and `<template>_<timestamp>.log.yaml`.

Run all tests with:

```bash
pytest -n auto
```

## 7. Release

* **Publish** `jinja_prompt_chaining_system` wheel to PyPI.
* **Docs** (optional) via MkDocs.
