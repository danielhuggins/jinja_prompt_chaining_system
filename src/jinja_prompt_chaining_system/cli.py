"""CLI for the Jinja Prompt Chaining System."""

import os
import click
from . import render_prompt
from .logger import RunLogger

@click.command()
@click.argument('template', type=click.Path(exists=True))
@click.option('--context', '-c', required=True, type=click.Path(exists=True),
              help='YAML file containing template context')
@click.option('--out', '-o', type=click.Path(),
              help='Output file path (defaults to stdout)')
@click.option('--logdir', '-l', type=click.Path(),
              help='Directory for log files')
@click.option('--name', '-n', type=str,
              help='Optional name for the run')
@click.option('--disable-parallel', is_flag=True, default=False,
              help='Disable parallel execution of LLM queries')
@click.option('--max-concurrent', type=int, default=4,
              help='Maximum number of concurrent queries when parallel is enabled')
def main(template: str, context: str, out: str, logdir: str, name: str,
         disable_parallel: bool, max_concurrent: int):
    """Process a Jinja template with LLM query support."""
    try:
        # Use the render_prompt API which now has parallel rendering by default
        # Render with parallel by default, unless disabled
        result = render_prompt(
            template_path=template,
            context=context,
            out=out,
            logdir=logdir,
            name=name,
            enable_parallel=not disable_parallel,
            max_concurrent=max_concurrent
        )
        
        # If not writing to a file, output to stdout
        if not out:
            click.echo(result)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    main() 