"""
Gajahmada AI — Entry point

Vercel:    imports `app` from gajahmada_ai.api.server (serverless, no ML)
Localhost: run with `python main.py` or `gajahmada start`
"""

from gajahmada_ai.api.server import app  # noqa: F401 — Vercel picks this up

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gajahmada_ai.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # hot-reload for local dev
        log_level="info",
    )
