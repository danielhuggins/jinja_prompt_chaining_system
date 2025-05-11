#!/bin/bash
# Demo script showing how to use the run naming feature with the CLI

# Create the logs directory if it doesn't exist
mkdir -p examples/logs

# Run the template with a named run
python -m src.jinja_prompt_chaining_system.cli \
  examples/named_run_demo.jinja \
  --context examples/named_run_demo_context.yaml \
  --logdir examples/logs \
  --name "cli-experiment-demo"

echo ""
echo "Created log directories:"
for dir in examples/logs/run_*cli-experiment-demo*; do
  echo "  - $(basename $dir)"
done

echo ""
echo "Done! Check the logs directory for the named run." 