#!/usr/bin/env bash
# Wrapper script for remote Marimo execution
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="python3"

# If scripts/marimo_exec.py exists in the same dir or workspace
if [ -f "$DIR/marimo_exec.py" ]; then
    SCRIPT_PATH="$DIR/marimo_exec.py"
elif [ -f "$DIR/scripts/marimo_exec.py" ]; then
    SCRIPT_PATH="$DIR/scripts/marimo_exec.py"
elif [ -f "/Volumes/WorkSpace/Project/Gym_Classification/scripts/marimo_exec.py" ]; then
    SCRIPT_PATH="/Volumes/WorkSpace/Project/Gym_Classification/scripts/marimo_exec.py"
else
    echo "Error: marimo_exec.py not found" >&2
    exit 1
fi

exec $PYTHON_EXEC "$SCRIPT_PATH" "$@"
