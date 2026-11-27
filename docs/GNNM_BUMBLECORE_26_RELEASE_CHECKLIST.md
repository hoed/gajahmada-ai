# GajahMada GNNM BumbleCore 300M — Release 2.6 Remediation Checklist

- [x] **1. Full pretraining rejects seed-only data** (`DATASET_GOVERNANCE_FAILED` when `--dataset` missing/unapproved).
- [x] **2. Full SFT rejects seed-only data** (`validate_sft_dataset` enforces external dataset, manifest, approved license, & minimum examples).
- [x] **3. Dataset manifests validated** (existence, readability, non-empty, approved license, SHA-256 split hashes, doc/token counts).
- [x] **4. Zero split leakage** (document-hash + `source_doc_id` pinning across train/val/test + SQLite WAL bounded streaming).
- [x] **5. True token accounting** (`raw_input_tokens`, `valid_training_tokens`, `loss_bearing_target_tokens`, `tokens_consumed`, `validation_tokens`, `target_tokens` stopping).
- [x] **6. Config consistency** (`2 × 16 × 2048 × 50,000 = 3,276,800,000` tokens — corrected from 6B).
- [x] **7. Tokenizer locked before model instantiation** (config rebuilt & validated against locked vocab size + `tokenizer_sha256`).
- [x] **8. SFT learning rate respects YAML (`5e-5`)** (removed hidden `max(..., 1e-4)` override).
- [x] **9. Exact training resume** (model, optimizer, scheduler horizon, cursor, token counters, CPU/CUDA/Python/NumPy RNG, GradScaler, & hash-mismatch guard).
- [x] **10. 3-level evaluation hardening** (`functional`, `capability`, `safety` + EM/F1/numerical/JSON + empty benchmark rejection).
- [x] **11. Recursive HF upload safety scan** (symlink rejection, explicit allowlist, SHA-256 upload manifest).
- [x] **12. Evidence-based Release Gates A–G** (smoke checkpoints & mere file existence rejected; `PRETRAINING_STATUS = NOT_STARTED`).
- [x] **13. Vercel Free compatibility preserved** (zero PyTorch/CUDA/FAISS in `requirements.txt`).
