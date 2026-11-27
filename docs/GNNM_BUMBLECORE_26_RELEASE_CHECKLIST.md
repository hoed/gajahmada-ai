# GajahMada GNNM BumbleCore 300M — Release 2.6 Remediation & Hardening Checklist

- [x] **1. Full pretraining rejects seed-only data** (`DATASET_GOVERNANCE_FAILED` when `--dataset` missing/unapproved).
- [x] **2. Full SFT rejects seed-only data** (`validate_sft_dataset` enforces external dataset, manifest, approved license, & minimum examples).
- [x] **3. Hardened dataset manifest verification** (schema fields, `manifest_checksum`, path-traversal & symlink escape protection, mandatory live `split_hashes`, `TOKEN_COUNT_MISMATCH`, `DOCUMENT_ID_COLLISION`).
- [x] **4. Zero split leakage & true streaming engine** (document-hash + `source_doc_id` pinning, SQLite WAL `BoundedStreamingPreprocessor`, `StreamingPretrainDataset` & `StreamingSFTDataset` with bounded shuffle buffer & cursor resume).
- [x] **5. Exact zero-overshoot token budget** (tail-padding + `-100` target masking ensures `tokens_consumed == target_tokens`; validation tokens excluded; silent budget mutation rejected on resume).
- [x] **6. Config consistency** (`2 × 16 × 2048 × 50,000 = 3,276,800,000` tokens in `configs/bumblecore_300m.yaml`, dynamically verified by Gate B).
- [x] **7. Tokenizer locked before model instantiation** (config rebuilt & validated against locked vocab size + `tokenizer_sha256`).
- [x] **8. SFT learning rate respects YAML (`5e-5`)** (no hidden override).
- [x] **9. 7-dimension exact training resume** (model, optimizer, scheduler state, cursor, token counters, CPU/CUDA/Python/NumPy RNG, GradScaler, hardware topology, & config/dataset/tokenizer/budget hash guards).
- [x] **10. Checkpoint Trust Layer** (PyTorch `weights_only=True` tensor/shape/dtype/param-count/NaN-Inf inspection + live source SHA-256 provenance + Ed25519 manifest signing).
- [x] **11. Independent 15-case bilingual benchmark suite** (`functional`/`capability`/`safety`, strict objective `token_f1 >= 0.45` without keyword fallback, `benchmark_dataset_sha256`, `BENCHMARK_LEAKAGE_DETECTED`, & output stability).
- [x] **12. Hardened HF export & parity verification** (`classify_training_status`, `validate_hf_model_card`, `validate_export_bundle`, `require_trained` guard, `NaN` weight rejection, recursive safety scan, & tensor/dtype/logit/Safetensors parity).
- [x] **13. Zero-hardcoded Evidence-Based Release Gates A–G** (all 7 gates computed dynamically; `PRETRAINING_STATUS = NOT_STARTED`, `blocked_at_gate = GATE_C_PRETRAINING_CHECKPOINT_VALIDATED`).
- [x] **14. Brain Observatory & Vercel Free compliance** (`/api/v2/platform/bumblecore` exposes honest `actual_telemetry`, `brain.js` renders live values with zero fabricated progress, `requirements.txt` free of torch/CUDA/FAISS, `configs/bumblecore_300m.yaml` bundled in `vercel.json`).
