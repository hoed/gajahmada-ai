# GajahMada AI — Vercel Free Deployment Guide

> **Architectural Separation Principle:**
> **Vercel Free** deploys the lightweight GajahMada AI web application, Brain Observatory, Hybrid Knowledge UI, Model Registry, Model Factory metadata, and Nexus Prime control plane.
> **Large-scale GNNM model training (including TITANFORGE 70B) and FAISS CPU native binaries** are intentionally separated for local or external GPU infrastructure and are **never** executed inside Vercel Functions.

---

## 1. Vercel Plan & Limitation Notice
- **Vercel Hobby (Free)** is intended strictly for **personal and non-commercial use**. If GajahMada AI is operated as a commercial service, upgrade to a compliant Vercel plan or deploy on dedicated infrastructure.
- **Serverless Constraints Respected:**
  - Lambda size kept under 50 MB (`requirements.txt` contains only lightweight pure-Python/FastAPI packages — no `torch`, `faiss-cpu`, or `llama-cpp-python`).
  - No permanent background workers, no persistent TCP/P2P daemons, no persistent WebSocket servers.
  - Filesystem writes inside `/tmp` are **ephemeral** and do not persist across serverless cold starts.

---

## 2. Zero-Dependency Deployment Steps

1. **Connect Repository:** Import `https://github.com/hoed/gajahmada-ai` in the Vercel Dashboard.
2. **Framework Preset:** `Other` (configured automatically via `vercel.json`).
3. **Build Command:** None required (zero-build HTML5/CSS3/ES2022 frontend + `@vercel/python` `main.py` entry point + `@vercel/static` for `sitemap.xml` and `robots.txt`).
4. **Required Paid Infrastructure:** **None** (no Docker, no VPS, no GPU cluster, no commercial AI API keys).

---

## 3. Optional Environment Variables

| Variable | Required? | Purpose |
| :--- | :---: | :--- |
| `VERCEL` | Auto (`1`) | Enables serverless mode, disables FAISS/PyTorch execution, activates storage persistence banner. |
| `GAJAHMADA_ADMIN_KEY` | Recommended | Protects operator/admin mutating endpoints & full reasoning trace inspection. |
| `GAJAHMADA_ALLOWED_ORIGINS` | Optional | Comma-separated CORS allowlist (defaults to `localhost:8000` + `https://gajahmada.vercel.app`). |
| `SUPABASE_URL` / `SUPABASE_KEY` | Optional | Enables optional persistent pgvector storage (otherwise app runs with browser `localStorage`/`IndexedDB` + clear ephemeral warning). |
| `GAJAHMADA_SSE_PER_IP` / `GAJAHMADA_SSE_MAX` | Optional | Caps concurrent Observatory SSE connections (default: `3` per IP, `50` global). |

---

## 4. Runtime Behavior on Vercel Free
- **Native Model Inference:** Displays `NATIVE MODEL INFERENCE: UNAVAILABLE IN CURRENT RUNTIME` unless a compatible local/external runtime loads verified weights. Symbolic reasoning, Hybrid Knowledge, and curated offline facts remain 100% active.
- **FAISS CPU:** Gracefully reports `INCOMPATIBLE_RUNTIME` on Vercel Free while remaining preserved (`ACTIVE` / `AVAILABLE_LOCALLY`) for local development and server environments via `requirements-local.txt`.
- **Model Factory & TITANFORGE 70B:** Architecture editor, analytical parameter calculator, and theoretical GPU/VRAM resource estimator run live on Vercel Free with zero GPUs; training execution remains explicitly `DISABLED_ON_VERCEL_FREE`.
