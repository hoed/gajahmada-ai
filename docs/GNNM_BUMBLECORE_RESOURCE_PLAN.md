# GajahMada GNNM BumbleCore 300M — Resource Plan & Preflight Report

**Model Configuration:** `configs/bumblecore_300m.yaml`  
**Exact Trainable Parameters:** `303,350,784` (`~303.35M`, Dense Decoder-Only Transformer: 24 layers, `d_model=1024`, GQA 16:4, SwiGLU `d_ff=2816`, `vocab_size=32,000`, `max_seq_len=2,048`, tied embeddings).

---

## 1. Analytical Resource Estimates (`THEORETICAL_ANALYTICAL_ESTIMATE`)

| Resource Component | Analytical Estimate (BF16 Mixed Precision + AdamW) |
| :--- | :---: |
| **Weights (BF16)** | `0.565 GB` |
| **Gradients (BF16)** | `0.565 GB` |
| **AdamW Optimizer States (FP32 master + m + v)** | `3.390 GB` |
| **Activations (`micro_batch=2`, `seq_len=2048`, Gradient Checkpointing ON)** | `~0.442 GB` |
| **Total Sharded Training VRAM (approx.)** | **`~5.0–7.5 GB` (with CUDA allocator & workspace headroom)** |
| **Recommended Single-GPU Target for Full Pretraining** | **1× 16 GB–24 GB VRAM GPU (e.g., RTX 4090 / L4 / A10G / A100)** |
| **Inference-Only Weights (BF16 / INT8 / INT4)** | `0.565 GB` / `0.283 GB` / `0.155 GB` |
| **Full Resumable Checkpoint Size (Weights + Optimizer)** | `~3.955 GB` |
| **Chinchilla-Optimal Token Horizon (`20 × N`)** | `~6.07 Billion tokens` |

---

## 2. Measured Current Local Host Hardware (`MEASURED_HARDWARE`)
- **PyTorch:** `2.15.0+cpu` in `.venv312` (`CUDA: False`, zero GPUs)
- **System RAM:** `7.63 GB` total (`~4.2 GB` available)
- **Free Disk:** `79.75 GB`
- **Preflight Verdict for Full 300M Pretraining on Current Host:** **`BLOCKED_INSUFFICIENT_RESOURCES`** (`GPU_REQUIRED_FOR_FULL_TRAINING` — CPU-only host cannot execute 6B-token 300M pretraining; smoke training & 303M BF16 instantiation/forward pass ARE supported locally).
- **Preflight Verdict on Vercel Free:** **`VERCEL_FREE_FORBIDDEN`** (strictly disabled on serverless).

## 3. Running Preflight Yourself
```bash
.venv312/bin/python -m gajahmada_ai.gnnm.train --config configs/bumblecore_300m.yaml --stage pretraining --preflight
```
