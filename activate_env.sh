#!/bin/bash
# Helper script to activate the virtual environment
# Usage: source activate_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/islamgpt_env"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "[PASSED] Virtual environment activated"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version)"
    echo ""
    echo "To deactivate, run: deactivate"
else
    echo "[FAILED] Virtual environment not found at: $VENV_PATH"
    echo "  Please run: python3 setup_venv.py"
    exit 1
fi
