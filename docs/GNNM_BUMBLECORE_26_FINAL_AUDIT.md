# GajahMada GNNM BumbleCore 300M — Release 2.6 Final Audit Report

**Audit Date:** 2026-11-28  
**Release:** `2.6` — Training Integrity & Data Quality *(Batches 1–9 Hardening Completed)*  
**Actual Pretraining Status:** `PRETRAINING_STATUS = NOT_STARTED` *(software pipeline remediated, hardened & verified; full 3.2768B-token GPU pretraining not yet executed; blocked at `GATE_C_PRETRAINING_CHECKPOINT_VALIDATED`)*

---

## 1. Six Hardened Training Integrity Pillars (Batches 1–9)

| Pillar | Modules | Hardening Implemented | Verification |
| :--- | :--- | :--- | :---: |
| **1. Checkpoint Trust Layer** | `bumblecore/checkpoint_trust.py`, `train.py` | 3-layer verification: magic-byte + `torch.load(weights_only=True)` + exact tensor key/shape/dtype/param-count/finite (`NaN`/`Inf`) check; live source-file SHA-256 verification (`model.pt`, `config.json`, `tokenizer.json`, `dataset_manifest.json`); Ed25519 manifest signing & verification via `NodeIdentity`. | ✅ 10/10 PyTorch tests |
| **2. Exact Token Budget & Reproducible Resume** | `trainer.py` | Zero-overshoot exact `target_tokens` enforcement via tail-padding + `-100` target masking; skips empty micro-batches; 7-dimension exact resume (weights, optimizer, scheduler state, cursor, token counters, RNGs, hardware topology); rejects silent `target_tokens` / config / dataset / tokenizer changes; reports `APPROXIMATE_ENVIRONMENT_CHANGED` honestly on hardware drift. | ✅ 7+7 PyTorch tests |
| **3. True Streaming Training Engine** | `bumblecore/streaming_loader.py`, `trainer.py`, `train.py` | `StreamingPretrainDataset` & `StreamingSFTDataset` stream JSONL line-by-line with lazy tokenization, streaming sequence packing, bounded shuffle buffer, `skip_sequences` cursor resume, and RSS/throughput telemetry without materializing full corpora in RAM. | ✅ PyTorch streaming suite |
| **4. Dataset Manifest Verification** | `bumblecore/dataset_governance.py` | Enriched `DatasetManifest` (`dataset_description`, `language_distribution`, `created_at`, `approval_evidence`, `manifest_checksum`), `_resolve_safe_split_path()` (blocks absolute paths, `../` traversal, symlink escapes), mandatory `split_hashes`, `TOKEN_COUNT_MISMATCH`, `DOCUMENT_ID_COLLISION`, cross-split `DOCUMENT_LEAKAGE`/`SOURCE_LEAKAGE`, & SQLite WAL `BoundedStreamingPreprocessor`. | ✅ 10/10 manifest tests |
| **5. Independent Benchmark Suite** | `bumblecore/evaluation.py` | 15-case bilingual (ID/EN) `INDEPENDENT_BENCHMARK_CASES` disjoint from seed corpus; objective metrics (`normalized_exact`, `token_f1 >= 0.45` with zero keyword fallback, `numerical_exact`, `json_valid`, `abstention`, `safety_forbidden`); `compute_benchmark_dataset_sha256()`; `check_benchmark_training_leakage()` (`BENCHMARK_LEAKAGE_DETECTED`); `measure_output_stability()`. | ✅ Benchmark suite passed |
| **6. Evidence-Based Release Gates A–G & HF Export Hardening** | `bumblecore/gate_evidence.py`, `bumblecore/export.py`, `platform_routes.py`, `brain.js`, `vercel.json` | Zero hardcoded booleans on Gates A–G (Gate A/B dynamically validate 303.35M architecture & YAML token budget; Gate C/D use Checkpoint Trust Layer; Gate E verifies held-out benchmark hash & clean leakage; Gate F verifies export manifest hashes & backing checkpoint; Gate G verifies platform integration). Hardened HF export rejects untrained/smoke live uploads, `NaN` weights, and forged Model Cards; verifies tensor/dtype/logit/Safetensors parity. Brain Observatory displays honest `actual_telemetry`. | ✅ Gates A–G + PyTorch parity + Vercel tests |

---

## 2. Actual Test & Verification Results
- **Full Repository Regression (`python3 -m pytest -q`, Python 3.14, no `torch`):** **166 passed, 0 failed, 6 skipped** (20.22s).
- **PyTorch Acceptance Suites (`.venv312`, `torch 2.15.0+cpu`):**
  - `tests/gnnm/test_checkpoint_trust_layer.py` → **PASSED** (`ALL_10_CHECKPOINT_TRUST_PYTORCH_TESTS_PASSED`).
  - `tests/gnnm/test_exact_token_budget.py` → **PASSED** (`ALL_7_EXACT_TOKEN_BUDGET_TESTS_PASSED`, zero overshoot `tokens_consumed == target_tokens`).
  - `tests/gnnm/test_streaming_training_engine.py` → **PASSED** (`ALL_STREAMING_ENGINE_TESTS_PASSED`).
  - `tests/gnnm/test_exact_resume_reproducibility.py` → **PASSED** (`ALL_7_DIM_EXACT_RESUME_TESTS_PASSED`, continuous 6-step vs. 3+3-step `diff = 0.0`).
  - `tests/gnnm/test_hf_export_hardening.py` → **PASSED** (`ALL_PYTORCH_HF_EXPORT_HARDENING_TESTS_PASSED`).
- **Frontend Syntax Check:** `node --check interface/html/brain.js` → **PASSED**.
- **Not Executed (Hardware Unavailable):** CUDA single-GPU and multi-GPU distributed tests were **not executed** (`cuda=False` on local host).
- **Vercel Free Compatibility:** Preserved 100% (zero PyTorch/CUDA/FAISS in `requirements.txt`; `configs/bumblecore_300m.yaml` included in `vercel.json` within `50mb` `maxLambdaSize`).
