#!/bin/bash
# Quick start for local development
set -e

echo "=== Gajahmada AI — Local Dev ==="

# Install deps if needed
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Run with hot-reload
echo "Starting server on http://localhost:8000"
echo "UI available at http://localhost:8000/ui"
echo ""
python main.py
