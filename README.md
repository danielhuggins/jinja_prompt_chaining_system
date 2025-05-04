# Jinja Prompt Chaining System

A simple Jinja-based prompt chaining engine for LLM interactions.

## Features

- Custom Jinja tag `{% llmquery %}` for LLM interactions
- Simple CLI interface
- YAML-based context and logging
- Support for streaming and non-streaming responses
- Flexible parameter syntax

## Installation

```bash
pip install jinja-prompt-chaining-system
```

## Usage

### Basic Usage

```bash
jinja-run template.jinja --context context.yaml
```

### With Output File

```bash
jinja-run template.jinja --context context.yaml --out output.txt
```

### With Logging

```bash
jinja-run template.jinja --context context.yaml --logdir logs/
```

### Template Example

```jinja
{% llmquery model="gpt-4" temperature=0.7 %}
Summarise the plot of {{ book }}.
{% endllmquery %}
```

## Development

### Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

```bash
pytest -n auto
```

## License

MIT License 