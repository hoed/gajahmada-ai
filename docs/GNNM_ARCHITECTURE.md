# GajahMada Native Neural Model (`GNNM`) — Architectural Blueprint

## 1. Strategic Direction
1. **Native Architecture:** Custom PyTorch Decoder-Only Transformer (**Pre-Norm RMSNorm + Rotary Positional Embeddings (RoPE) + Grouped/Multi-Head Causal Self-Attention + SwiGLU + Gradient Checkpointing + optional Weight Tying**).
2. **Language Identity:** Balanced multilingual Indonesian & English powered by a lossless pure-Python Byte-Level BPE tokenizer (`GNNMBPETokenizer`).
3. **Control-Plane / Training Separation:** Architecture specs, parameter calculation, and resource estimation run anywhere (including Vercel Free), while actual training and large-model inference execute exclusively on local or external GPU infrastructure.

---

## 2. GNNM Model Family Catalog (`gajahmada_ai/gnnm/config.py`)

> **Honesty Rule:** Architecture configurations are specifications, **not** claims that trained weights exist. Only models with verified on-disk checkpoints (`model.pt`) report available weights.

| Family | Preset Key | Type | Analytical Params | Layers / `d_model` | Heads (Q/KV) | Context | Current Verified Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BUMBLECORE** | `GNNM-01-Tiny` | Dense | `918K–1.0M` | 4 / 128 | 4 / 4 | 256 | ✅ **Trained & Verified Locally** (`models/gnnm-01-tiny/`) |
| **BUMBLECORE** | `BUMBLECORE-12M` (`GNNM-01-12M`) | Dense | `~11.2M–12.4M` | 6 / 384 | 6 / 6 | 512 | ✅ **Trained & Verified Locally** (`models/gnnm-01-12m/`) |
| **BUMBLECORE** | `BUMBLECORE-100M` | Dense | `~105M` | 12 / 768 | 12 / 4 | 2,048 | `ARCHITECTURE_DESIGNED` (GPU scale) |
| **BUMBLECORE** | `BUMBLECORE-300M` | Dense | `~320M` | 20 / 1,024 | 16 / 4 | 4,096 | `ARCHITECTURE_DESIGNED` (GPU scale) |
| **OPTICORE** | `OPTICORE-1B` / `3B` / `7B` | Dense | `1.1B / 3.4B / 7.2B` | 22–32 / 2048–4096 | 16–32 / 4–8 | 4K–8K | `ARCHITECTURE_DESIGNED` / `PLANNED` |
| **TITANFORGE** | `TITANFORGE-30B` | Dense | `~30.5B` | 56 / 6,144 | 48 / 8 | 8,192 | `PLANNED` (external GPU cluster) |
| **TITANFORGE** | **`TITANFORGE-70B`** | **Dense** | **`70,549,512,192` (~70.55B)** | **80 / 8,192** | **64 / 8 (GQA)** | **8,192** | **`ARCHITECTURE_DESIGNED`** (verified analytical spec & estimator) |
| **TITANFORGE** | `TITANFORGE-70B-MoE-Exp` | MoE (8E / Top-2) | Separate Total vs. Active | 48 / 5,120 | 40 / 8 | 8,192 | `PLANNED` (experimental MoE accounting) |
| **TITANFORGE** | `TITANFORGE-120B` | Dense | `~122B` | 88 / 10,240 | 80 / 8 | 8,192 | `PLANNED` (external GPU cluster) |

**Planned Specializations (Not Separate Trained Models):** `NEUROX` (reasoning), `CODESTRIKE` (coding), and `KNOWTRON` (knowledge/retrieval) remain explicitly `PLANNED` until distinct verified checkpoints and evaluations exist.

---

## 3. Package Modules (`gajahmada_ai/gnnm/`)
- `config.py`: Modular `GNNMConfig`, exact Dense vs. MoE `parameter_breakdown()`, and `GNNM_PRESETS`.
- `model.py`: `GNNMTransformer` with `RMSNorm`, `RoPE`, `CausalSelfAttention` (GQA/MHA), `SwiGLUFeedForward`, and `torch.utils.checkpoint` gradient checkpointing.
- `tokenizer.py`: Lossless pure-Python Byte-Level BPE (`GNNMBPETokenizer`).
- `dataset.py`, `trainer.py`, `inference.py`: Corpus cleaning/dedup, AdamW + cosine LR + SFT (`-100` masking), and CPU/CUDA native inference.
- `resource_estimator.py`: Pure-Python analytical VRAM/memory/FLOPs/GPU estimator across FP32, BF16, FP16, INT8, and INT4 (zero GPUs required).
- `distributed_prep.py`: Modular data pipeline spec and DDP/FSDP/TP/PP readiness validator (strictly refuses execution inside Vercel Free).
