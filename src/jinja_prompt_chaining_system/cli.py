import os
import yaml
import click
import asyncio
from jinja2 import Environment, FileSystemLoader
from .parser import LLMQueryExtension
from .logger import RunLogger
from . import create_environment

@click.command()
@click.argument('template', type=click.Path(exists=True))
@click.option('--context', '-c', required=True, type=click.Path(exists=True),
              help='YAML file containing template context')
@click.option('--out', '-o', type=click.Path(),
              help='Output file path (defaults to stdout)')
@click.option('--logdir', '-l', type=click.Path(),
              help='Directory for log files')
def main(template: str, context: str, out: str, logdir: str):
    """Process a Jinja template with LLM query support."""
    try:
        # Load context
        with open(context, 'r') as f:
            try:
                ctx = yaml.safe_load(f)
            except yaml.YAMLError as e:
                click.echo(f"Error: Invalid YAML in context file: {str(e)}", err=True)
                raise click.Abort()
        
        # Setup Jinja environment
        template_dir = os.path.dirname(os.path.abspath(template))
        env = create_environment(template_dir)
        
        # Load and render template
        template_name = os.path.basename(template)
        template_obj = env.get_template(template_name)
        
        # Get the extension instance and set template name
        extension = env.globals['extension']
        extension.set_template_name(template)
        
        # Setup run-based logging if logdir is provided
        run_id = None
        if logdir:
            os.makedirs(logdir, exist_ok=True)
            run_logger = RunLogger(logdir)
            
            # Start a new run with template metadata and context
            run_metadata = {
                "template": template,
                "context_file": context
            }
            run_id = run_logger.start_run(metadata=run_metadata, context=ctx)
            
            # Get the LLM logger for this run
            llm_logger = run_logger.get_llm_logger(run_id)
            extension.logger = llm_logger
        
        # Render template - use manual sync rendering to avoid async issues
        result = render_template_sync(template_obj, ctx)
        
        # End the run if we started one
        if logdir and run_id:
            run_logger.end_run()
        
        # Handle output
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, 'w') as f:
                f.write(result)
        else:
            click.echo(result)
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

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