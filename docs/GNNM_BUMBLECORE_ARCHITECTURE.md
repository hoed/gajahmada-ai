# GajahMada GNNM BumbleCore 300M — Architecture & Hardening Specification (Release 2.6)

- **Full Name:** GajahMada GNNM BumbleCore 300M (`GNNM-BumbleCore-300M`)
- **Release:** `2.6` — Training Integrity & Data Quality *(architecture preserved identically from Release 2.5)*
- **Current Pretraining Status:** `PRETRAINING_STATUS = NOT_STARTED` (`lifecycle_stage = CONFIGURED`, `blocked_at_gate = GATE_C_PRETRAINING_CHECKPOINT_VALIDATED`)
- **Implementation:** `gajahmada_ai/gnnm/bumblecore/architecture.py` (`BumbleCoreConfig`, `BumbleCoreModel`) extending native `gajahmada_ai/gnnm/model.py` (`GNNMTransformer`).
- **Architecture:** Dense Decoder-Only Autoregressive Transformer (`RMSNorm` + `RoPE` + `Grouped Query Attention 16:4` + `SwiGLU` + `Gradient Checkpointing` + `Tied Embeddings`).

## 1. Exact Parameter Accounting (Verified Against Instantiated PyTorch Model)
| Component | Formula | Exact Parameters |
| :--- | :--- | :---: |
| **Token Embeddings (`vocab=32000, d=1024`)** | `32000 × 1024` | `32,768,000` |
| **Output Projection (`lm_head`)** | Tied with `token_emb` | `0` |
| **Attention (24 layers, Q=16, KV=4, head_dim=64)** | `24 × (1024² + 2×1024×256 + 1024²)` | `62,914,560` |
| **SwiGLU FFN (24 layers, `d_ff=2816`)** | `24 × (3 × 1024 × 2816)` | `207,618,048` |
| **RMSNorm (48 layer norms + final norm)** | `(24 × 2 + 1) × 1024` | `50,176` |
| **Total Trainable Parameters** | **Verified `actual == analytical` (`diff = 0`)** | **`303,350,784` (`303.35M`)** |

Configurations outside `[285M, 315M]` are automatically rejected unless `allow_target_override=True` is passed explicitly. MoE is explicitly disallowed in BumbleCore v1.

## 2. Release 2.6 Training Integrity Module Map (`gajahmada_ai/gnnm/bumblecore/`)
| Module | Responsibility |
| :--- | :--- |
| `architecture.py` | Canonical `BumbleCoreConfig` (`303,350,784`), `BumbleCoreModel`, smoke config, & structural validation. |
| `checkpoint_trust.py` | 3-layer **Checkpoint Trust Layer**: PyTorch `weights_only=True` tensor/shape/dtype/NaN-Inf/param verification + live source SHA-256 provenance + Ed25519 manifest signature verification. |
| `dataset_governance.py` | Strict dataset separation, SQLite WAL bounded streaming preprocessor, safe path-traversal/symlink guards, mandatory split hashes, & cross-split leakage rejection. |
| `streaming_loader.py` | `StreamingPretrainDataset` & `StreamingSFTDataset` — line-by-line JSONL streaming, lazy tokenization, sequence packing, bounded shuffle buffer, cursor resume, & RSS/throughput telemetry. |
| `evaluation.py` | 3-level evaluation (`functional`, `capability`, `safety`), 15-case bilingual `INDEPENDENT_BENCHMARK_CASES`, dataset SHA-256, training-leakage detection (`BENCHMARK_LEAKAGE_DETECTED`), & output stability. |
| `export.py` | Hardened HF export with `classify_training_status`, `validate_hf_model_card`, `validate_export_bundle`, `require_trained` guard, NaN rejection, recursive safety scan, & tensor/dtype/logit/Safetensors parity. |
| `gate_evidence.py` | Zero-hardcoded evidence-based evaluation of all 7 Release Gates (`A`–`G`) and honest lifecycle transitions. |
