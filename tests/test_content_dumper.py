"""
Test and demonstrate how to improve the ContentAwareYAMLDumper to force pipe style for content fields
"""

import yaml
from pathlib import Path
from jinja_prompt_chaining_system.logger import ContentAwareYAMLDumper

# Create an improved version that forces pipe style for content fields
class ImprovedContentDumper(yaml.SafeDumper):
    """A custom YAML dumper that uses the pipe (|) style for all content fields and multiline strings."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register a special representer for dictionaries
        self.add_representer(dict, self.represent_dict_with_content_fields)
        
    def represent_dict_with_content_fields(self, data):
        """Custom representer for dictionaries that forces pipe style for content fields"""
        # Process the dictionary to find and mark content fields
        processed_data = {}
        for key, value in data.items():
            if key == "content" and isinstance(value, str):
                # Force newline at end to ensure pipe style is used
                if not value.endswith('\n'):
                    value = value + '\n'
            processed_data[key] = value
            
        # Use the default dictionary representer - u'tag:yaml.org,2002:map' is the standard tag for maps
        return self.represent_mapping('tag:yaml.org,2002:map', processed_data)
        
    def represent_scalar(self, tag, value, style=None):
        """Use pipe style for multiline strings"""
        if isinstance(value, str) and '\n' in value:
            style = '|'
        return super().represent_scalar(tag, value, style)

def main():
    print("\n=== TESTING IMPROVED CONTENT DUMPER ===\n")
    
    # Create test file
    test_file = Path("dumper_test.yaml")
    improved_file = Path("improved_dumper_test.yaml")
    
    # Test with different types of content
    test_data = {
        "request": {
            "model": "gpt-4o-mini",
            "messages": [
                # Test 1: Single line content
                {"role": "user", "content": "This is a simple one-line content field."},
                
                # Test 2: Long single line that would normally get line breaks with continuation markers
                {"role": "assistant", "content": "This is a very long line " + "that would normally get broken up with line continuation markers " * 3},
                
                # Test 3: Already multiline content
                {"role": "system", "content": "Line 1\nLine 2\nLine 3"}
            ]
        }
    }
    
    # Original dumper
    print("Using original ContentAwareYAMLDumper:")
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(test_data, f, Dumper=ContentAwareYAMLDumper, 
                 default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Read and print
    with open(test_file, "r", encoding="utf-8") as f:
        original_content = f.read()
        print(original_content)
    
    # Count pipe vs quoted styles
    pipe_count = original_content.count("content: |")
    quoted_count = original_content.count("content: '") + original_content.count('content: "')
    
    print(f"Original dumper: {pipe_count} pipe-style fields, {quoted_count} quoted-style fields")
    
    # Improved dumper
    print("\nUsing improved ImprovedContentDumper:")
    with open(improved_file, "w", encoding="utf-8") as f:
        yaml.dump(test_data, f, Dumper=ImprovedContentDumper, 
                 default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Read and print
    with open(improved_file, "r", encoding="utf-8") as f:
        improved_content = f.read()
        print(improved_content)
    
    # Count pipe vs quoted styles
    pipe_count = improved_content.count("content: |")
    quoted_count = improved_content.count("content: '") + improved_content.count('content: "')
    
    print(f"Improved dumper: {pipe_count} pipe-style fields, {quoted_count} quoted-style fields")
    
    # Clean up
    test_file.unlink()
    improved_file.unlink()
    
    print("\n=== SUGGESTED IMPLEMENTATION ===")
    print("""
class ContentAwareYAMLDumper(yaml.SafeDumper):
    \"\"\"A custom YAML dumper that uses the pipe (|) style for all content fields and multiline strings.\"\"\"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register representer for dictionaries
        self.add_representer(dict, self.represent_dict_with_content_fields)
        
    def represent_dict_with_content_fields(self, data):
        \"\"\"Force pipe style for content fields\"\"\"
        processed_data = {}
        for key, value in data.items():
            if key == "content" and isinstance(value, str):
                # Force newline at end to ensure pipe style is used
                if not value.endswith('\\n'):
                    value = value + '\\n'
            processed_data[key] = value
            
        return self.represent_mapping('tag:yaml.org,2002:map', processed_data)
        
    def represent_scalar(self, tag, value, style=None):
        \"\"\"Use pipe style for multiline strings\"\"\"
        if isinstance(value, str) and '\\n' in value:
            style = '|'
        return super().represent_scalar(tag, value, style)
""")

if __name__ == "__main__":
    main() 