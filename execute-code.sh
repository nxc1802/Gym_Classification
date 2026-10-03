#!/usr/bin/env bash
# Wrapper script for remote Marimo execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi
exec "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/marimo_exec.py" "$@"
