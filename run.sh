#!/bin/bash
# Quick start for local development
set -e

echo "=== Gajahmada AI — Local Dev ==="

PYTHON_BIN="python"
if [ -x ".venv312/bin/python" ]; then
  PYTHON_BIN=".venv312/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

# Install deps if needed
if ! "$PYTHON_BIN" -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Run with hot-reload
echo "Starting server on http://localhost:8000 (using $PYTHON_BIN)"
echo "UI available at http://localhost:8000/ui"
echo ""
"$PYTHON_BIN" main.py
