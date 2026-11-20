@echo off
echo === Gajahmada AI - Local Dev ===
python -c "import fastapi" 2>nul || pip install -r requirements.txt
echo Starting server on http://localhost:8000
echo UI available at http://localhost:8000/ui
python main.py
