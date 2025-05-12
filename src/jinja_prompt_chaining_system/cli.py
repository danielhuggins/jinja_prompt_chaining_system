import os
import sys
import yaml
import click
import asyncio
import re
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, TemplateError
from .parser import LLMQueryExtension
from .logger import RunLogger
from . import create_environment

def parse_key_value_arg(arg: str) -> tuple:
    """Parse a key=value argument into a tuple of (key, parsed_value).
    
    Special features:
    - Values starting with @ are treated as file references and the value becomes
      the content of the referenced file.
    - Other values are parsed as YAML.
    """
    if '=' not in arg:
        raise ValueError(f"Invalid key-value pair: {arg}. Format should be key=value")
    
    key, value_str = arg.split('=', 1)
    
    # Check if this is a file reference
    if value_str.startswith('@') and not (value_str.startswith("'@") or value_str.startswith('"@')):
        file_path = value_str[1:]  # Remove the @ symbol
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return key, f.read()
        except (IOError, FileNotFoundError) as e:
            raise IOError(f"Error reading file referenced by {key}=@{file_path}: {str(e)}")
    
    # Not a file reference, parse as YAML
    try:
        # Parse the value as YAML to handle different data types
        parsed_value = yaml.safe_load(value_str)
        return key, parsed_value
    except yaml.YAMLError:
        # If YAML parsing fails, treat it as a string
        return key, value_str

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('template', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--context', '-c', type=click.Path(exists=True, dir_okay=False, readable=True),
              help='YAML file containing template context (optional)')
@click.option('--out', '-o', type=click.Path(writable=True),
              help='Output file path (defaults to stdout)')
@click.option('--logdir', '-l', type=click.Path(file_okay=False),
              help='Directory for log files')
@click.option('--name', '-n', type=str,
              help='Optional name for the run')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output with additional processing information')
@click.option('--quiet', '-q', is_flag=True, default=False,
              help='Suppress all non-error console output')
@click.argument('key_value_pairs', nargs=-1, type=click.UNPROCESSED)
def main(template: str, context: Optional[str], out: Optional[str], logdir: Optional[str], 
         name: Optional[str], verbose: bool, quiet: bool, key_value_pairs: List[str]):
    """Process a Jinja template with LLM query support.
    
    TEMPLATE: Path to the Jinja template file to process.
    
    KEY_VALUE_PAIRS: Optional key=value pairs for template context.
    These must come before any options and will override values from the context file.
    Values are parsed as YAML (e.g., name=Alice, age=30, active=true).
    File references like message=@input.txt will load file contents as values.
    
    Examples:
      jinja-run template.jinja --context data.yaml
      jinja-run template.jinja name=Alice age=30 --out result.txt
      jinja-run template.jinja message=@input.txt --logdir logs/
    """
    # Check for incompatible flags
    if verbose and quiet:
        click.echo("Error: --verbose and --quiet options cannot be used together", err=True)
        sys.exit(1)
    
    # Helper function for verbose output
    def verbose_echo(message):
        if verbose and not quiet:
            click.echo(f"[INFO] {message}", err=True)
    
    try:
        # Parse inline key-value pairs
        ctx = {}
        if key_value_pairs:
            verbose_echo(f"Parsing {len(key_value_pairs)} key-value pairs")
            for kv_pair in key_value_pairs:
                try:
                    key, value = parse_key_value_arg(kv_pair)
                    ctx[key] = value
                    # For file references, show a more helpful message
                    if kv_pair.startswith(f"{key}=@") and not (
                        kv_pair.startswith(f"{key}='@") or kv_pair.startswith(f'{key}="@')):
                        verbose_echo(f"Added context from file: {key}=@{kv_pair.split('=@', 1)[1]}")
                    else:
                        verbose_echo(f"Added context: {key}={value}")
                except ValueError as e:
                    click.echo(f"Error: {str(e)}", err=True)
                    sys.exit(1)
                except yaml.YAMLError as e:
                    click.echo(f"Error: Invalid YAML in value for {kv_pair}: {str(e)}", err=True)
                    sys.exit(1)
                except IOError as e:
                    click.echo(f"Error: {str(e)}", err=True)
                    sys.exit(1)
        
        # Load context file if provided and merge with inline pairs
        if context:
            verbose_echo(f"Loading context from file: {context}")
            try:
                with open(context, 'r', encoding='utf-8') as f:
                    try:
                        file_ctx = yaml.safe_load(f)
                        if file_ctx is None:
                            file_ctx = {}  # Handle empty YAML files gracefully
                            verbose_echo("Context file is empty, using empty context")
                        
                        # Merge with inline context, giving preference to inline values
                        if file_ctx:
                            # First make a copy of file_ctx
                            merged_ctx = {**file_ctx}
                            # Then update with inline context
                            merged_ctx.update(ctx)
                            ctx = merged_ctx
                            verbose_echo(f"Merged context from file with {len(ctx)} total keys")
                        
                    except yaml.YAMLError as e:
                        click.echo(f"Error: Invalid YAML in context file: {str(e)}", err=True)
                        sys.exit(1)
            except IOError as e:
                click.echo(f"Error: Failed to read context file: {str(e)}", err=True)
                sys.exit(1)
        elif not key_value_pairs:
            # No context provided at all, use empty dict
            verbose_echo("No context provided, using empty context")
        
        # Validate output path
        if out:
            out_dir = os.path.dirname(os.path.abspath(out))
            try:
                os.makedirs(out_dir, exist_ok=True)
                # Test if we can write to the output file
                with open(out, 'a') as f:
                    pass
            except (IOError, PermissionError) as e:
                click.echo(f"Error: Cannot write to output file {out}: {str(e)}", err=True)
                sys.exit(1)
        
        # Setup Jinja environment
        verbose_echo(f"Processing template: {template}")
        template_dir = os.path.dirname(os.path.abspath(template))
        try:
            env = create_environment(template_dir)
        except Exception as e:
            click.echo(f"Error: Failed to create Jinja environment: {str(e)}", err=True)
            sys.exit(1)
        
        # Load and render template
        template_name = os.path.basename(template)
        try:
            template_obj = env.get_template(template_name)
        except TemplateError as e:
            click.echo(f"Error: Failed to load template: {str(e)}", err=True)
            sys.exit(1)
        
        # Get the extension instance and set template name
        extension = env.globals['extension']
        extension.set_template_name(template)
        
        # Setup run-based logging if logdir is provided
        run_id = None
        if logdir:
            verbose_echo(f"Setting up logging in directory: {logdir}")
            try:
                os.makedirs(logdir, exist_ok=True)
                run_logger = RunLogger(logdir)
                
                # Start a new run with template metadata and context
                run_metadata = {
                    "template": template,
                }
                
                # Add context file path to metadata if used
                if context:
                    run_metadata["context_file"] = context
                
                if name:
                    verbose_echo(f"Using run name: {name}")
                
                run_id = run_logger.start_run(metadata=run_metadata, context=ctx, name=name)
                
                # Get the LLM logger for this run
                llm_logger = run_logger.get_llm_logger(run_id)
                extension.logger = llm_logger
                verbose_echo(f"Created run with ID: {run_id}")
            except Exception as e:
                click.echo(f"Error: Failed to setup logging: {str(e)}", err=True)
                sys.exit(1)
        
        # Render template - use manual sync rendering to avoid async issues
        verbose_echo("Rendering template...")
        try:
            result = render_template_sync(template_obj, ctx)
        except Exception as e:
            click.echo(f"Error: Failed to render template: {str(e)}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            sys.exit(1)
        
        # End the run if we started one
        if logdir and run_id:
            try:
                run_logger.end_run()
                verbose_echo(f"Completed run: {run_id}")
            except Exception as e:
                click.echo(f"Warning: Failed to properly end the run: {str(e)}", err=True)
        
        # Handle output
        if out:
            verbose_echo(f"Writing output to: {out}")
            try:
                with open(out, 'w', encoding='utf-8') as f:
                    f.write(result)
                if verbose:
                    click.echo(f"Output successfully written to {out}", err=True)
            except IOError as e:
                click.echo(f"Error: Failed to write to output file: {str(e)}", err=True)
                sys.exit(1)
        else:
            # If not quiet mode, output the result to stdout
            if not quiet:
                if verbose:
                    click.echo("Template output:", err=True)
                click.echo(result)
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        click.echo(f"Error: An unexpected error occurred: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

def render_template_sync(template, context):
    """Render a Jinja template synchronously, handling async calls if necessary."""
    # First try in sync mode
    try:
        return template.render(**context)
    except RuntimeError as e:
        if "async" in str(e).lower():
            # Fall back to async rendering
            return asyncio.run(template.render_async(**context))
        else:
            # Re-raise other errors
            raise

if __name__ == '__main__':
    main() 