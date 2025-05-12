import yaml
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper

# Test data with different types of content
data = {
    "messages": [
        {"role": "user", "content": "Single line content"},
        {"role": "assistant", "content": "Very long content " + "that would trigger line breaks " * 5},
        {"role": "system", "content": "Multiline\ncontent\nwith\nnewlines"}
    ]
}

# Dump with the fixed dumper
with open("test_output.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, Dumper=ContentAwareYAMLDumper, default_flow_style=False, sort_keys=False)

print("Output saved to test_output.yaml") 