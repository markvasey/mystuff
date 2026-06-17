#!/bin/bash
# Navigate to the script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Run getstories.py using the python interpreter inside the virtual environment
echo "Executing getstories.py in venv..."
./venv/bin/python getstories.py

echo "Execution completed successfully!"
