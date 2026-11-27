# GajahMada GNNM BumbleCore 300M — Phase 0 Repository Audit & Baseline Report

**Audit Date:** 2026-11-28  
**Git Baseline:** Branch `main` @ commit `ec200f8` (clean working tree)  
**Local Runtime Hardware Audited:**
- Python (system): `3.14` (FastAPI, pytest — no `torch` installed)
- Python (ML venv `.venv312`): `3.12` with `torch 2.15.0+cpu`, `cuda: False`
- System RAM: `7.63 GB` total (`~4.2 GB` available), Free Disk: `79.75 GB`

---

## 1. Existing Architecture & Reusable Components

| Component | Location | Status | Reuse Strategy for BumbleCore 300M |
| :--- | :--- | :---: | :--- |
| **GNNM Config & Presets** | `gajahmada_ai/gnnm/config.py` | Operational | Extend via `BumbleCoreConfig` (`gajahmada_ai/gnnm/bumblecore/`); tune `BUMBLECORE-300M` from `258.3M` to exact `303,350,784` params (`300M ± 5%`). |
| **GNNM Transformer** | `gajahmada_ai/gnnm/model.py` | Operational | Reuse `RMSNorm`, `RoPE`, `CausalSelfAttention` (GQA/MHA), `SwiGLUFeedForward`, `GNNMTransformer` (incl. gradient checkpointing). |
| **Native BPE Tokenizer** | `gajahmada_ai/gnnm/tokenizer.py` | Operational | Preserve `GNNMBPETokenizer` (`gnnm-bpe-v1`, lossless UTF-8 byte-level BPE + 7 special tokens); add tokenizer benchmark & audit report. |
| **Corpus & SFT Packing** | `gajahmada_ai/gnnm/dataset.py` | Operational | Extend with streaming preprocessing, PII redaction, quality/language filters, leakage-free train/val/test splits, & dataset manifests. |
| **Training Loop** | `gajahmada_ai/gnnm/trainer.py` | Operational | Extend in-place with gradient accumulation, warmup, BF16/FP16 AMP, grad-norm logging, resumable optimizer/scheduler/RNG checkpoints, emergency save, and strict non-empty validation (fixing fake `val_loss=0.0, ppl=1.0` on empty val). |
| **Resource Estimator** | `gajahmada_ai/gnnm/resource_estimator.py` | Operational | Extend with preflight hardware probe (`torch`, CUDA, VRAM, BF16/FP16, RAM, disk, dataset, config) and attention temp-memory estimate. |
| **Inference & Registry** | `gajahmada_ai/gnnm/inference.py`, `learning/model_registry.py`, `platform/model_factory.py` | Operational | Reuse & wire BumbleCore artifacts, release gates A–G, and Safetensors/native export verification. |
| **Knowledge, Memory, Observatory, Vercel** | `gajahmada_ai/knowledge/`, `memory/`, `api/`, `platform/` | Operational | Preserve 100% Vercel Free compatibility (no `torch`/`faiss-cpu` in `requirements.txt`). |

---

## 2. Identified Gaps & Risks (Pre-Implementation)
1. **Parameter Count Drift:** Existing `BUMBLECORE-300M` preset (`n_layers=20`) had `258,253,824` params (~14% below 300M). Adjusted to `n_layers=24` → **`303,350,784` exact trainable parameters**.
2. **Empty Validation Bug (`trainer.py:94-95`):** `evaluate_loss([])` previously returned `{"val_loss": 0.0, "perplexity": 1.0}` (fake perfect metrics). Must return explicit `UNAVAILABLE` / raise on empty validation.
3. **Resumable Training Missing:** `GNNMTrainer` saved only `model.pt`, without optimizer state, scheduler step, RNG state, or atomic write/emergency checkpoint recovery.
4. **No Dedicated CLI Entrypoint (`python -m gajahmada_ai.gnnm.train`):** Needs `--config`, `--stage pretraining|sft`, `--preflight`, and `--resume auto`.
5. **No Release Gates A–G or Safetensors Export/Import Parity Test:** Required before any Hugging Face preparation.
6. **Hardware Constraint (Documented Honestly):** Current machine is CPU-only (`7.63 GB` RAM, `cuda=False`). Real 303M instantiation & forward pass can be verified locally in CPU BF16/FP32, while full multi-billion-token 300M pretraining requires external GPU hardware (≥16–24 GB VRAM) and must NOT be faked or launched automatically.

---

## 3. Baseline Test Status (Before Release 2.5 Changes)
- `python3 -m pytest -q`: **134 passed, 0 failed** (verified at commit `ec200f8`).
- `.venv312/bin/python` PyTorch smoke + gradient checkpointing: **Passed** (`actual == est == 106,816`).

---

## 4. Proposed Implementation Sequence
- **Batch 1 (Phases 0–1):** Baseline report + `gajahmada_ai/gnnm/bumblecore/architecture.py` + `configs/bumblecore_300m.yaml` + architecture acceptance tests.
- **Batch 2 (Phase 2):** Tokenizer audit + streaming/PII/split/manifest data pipeline + `docs/GNNM_BUMBLECORE_DATA_POLICY.md`.
- **Batch 3 (Phases 3–4):** Extended `GNNMTrainer` (grad accum, warmup, AMP, resume, non-empty val) + Preflight & Resource Plan (`docs/GNNM_BUMBLECORE_RESOURCE_PLAN.md`).
- **Batch 4 (Phase 5):** `python -m gajahmada_ai.gnnm.train` CLI (`--preflight`, `--stage pretraining`, `--resume auto`) + run manifest & smoke pretraining validation.
- **Batch 5 (Phases 6–7):** Multi-turn SFT pipeline + 16-dimension Evaluation Suite & Release Gates A–G.
- **Batch 6 (Phases 8–9):** Native + Safetensors export/import parity tests + Hugging Face model card & optional upload CLI (dry-run / auth-gated).
- **Batch 7 (Phases 10–11):** GajahMada Model Registry, Nexus Prime, Brain Observatory integration & Vercel Free boundary verification.
- **Batch 8 (Phases 12–13):** Complete Release 2.5 documentation suite, full test execution, and `docs/GNNM_BUMBLECORE_FINAL_AUDIT.md`.
