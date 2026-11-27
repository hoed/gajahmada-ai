# GajahMada GNNM BumbleCore 300M — Release 2.6 Final Audit Report

**Audit Date:** 2026-11-28  
**Release:** 2.6 — Training Integrity & Data Quality  
**Actual Pretraining Status:** `PRETRAINING_STATUS = NOT_STARTED` *(software pipeline remediated & verified; full 3.2768B-token GPU pretraining not yet executed)*

---

## 1. All 10 Release 2.5 Findings Remediated & Verified

| # | Finding | Remediation Implemented | Verification |
| :-: | :--- | :--- | :---: |
| 1 | Seed corpus fallback in full pretraining | `train.py` rejects missing/unapproved `--dataset` (`DATASET_GOVERNANCE_FAILED`); seed allowed ONLY with `--smoke`. | ✅ Tested |
| 2 | 3-example SFT seed used for full SFT | `validate_sft_dataset()` enforces external SFT dataset, approved license, role format, & `min_sft_examples`. | ✅ Tested |
| 3 | Target token budget not enforced & 6B vs 3.2768B mismatch | `configs/bumblecore_300m.yaml` corrected to `3,276,800,000` (`65,536 × 50,000`); `GNNMTrainer` tracks `raw_input_tokens`, `valid_training_tokens`, `loss_bearing_target_tokens`, `tokens_consumed`, `validation_tokens`, & stops at `target_tokens`. | ✅ Tested in PyTorch |
| 4 | Tokenizer vocab drift after config creation | `train.py` freezes tokenizer BEFORE model instantiation, rebuilds/validates config against locked vocab, & records `tokenizer_sha256`. | ✅ Tested |
| 5 | SFT LR `5e-5` overridden up to `1e-4` | Removed hidden `max(self.lr * 0.25, 1e-4)` clamp in `trainer.py`; optimizer uses configured `5e-5` directly. | ✅ Tested in PyTorch |
| 6 | Incomplete resume state | `trainer.py` saves/restores model, optimizer, scheduler horizon, `dataset_cursor`, token counters, CPU/CUDA/Python/NumPy RNG, FP16 GradScaler, & validates dataset/tokenizer hashes on resume. | ✅ Bit-exact (`diff = 0.0`) |
| 7 | Unbounded RAM in streaming preprocessor | `BoundedStreamingPreprocessor` uses on-disk SQLite WAL (`fingerprints.db`) for exact SHA-256 + 64-bit SimHash near-dup & incremental split JSONL writes. | ✅ Tested |
| 8 | Keyword-only merged evaluation | `evaluation.py` separates `functional`, `capability`, & `safety` levels, adds EM/Normalized EM/Token F1/Numerical/JSON metrics, & fails empty benchmarks. | ✅ Tested |
| 9 | Non-recursive HF upload check | `scan_export_directory_recursive()` walks subdirs, rejects symlinks & forbidden secrets, enforces explicit allowlist, & emits SHA-256 upload manifest. | ✅ Tested |
| 10 | File-existence Release Gates | `gate_evidence.py` verifies live SHA-256, `smoke_mode == False`, `stage`, steps, tokens, val loss, & approved license; `/api/v2/platform/bumblecore` reports `NOT_STARTED`. | ✅ Tested |

---

## 2. Actual Test & Verification Results
- **Repository `pytest -q` (Python 3.14, no `torch`):** **150 passed, 0 failed, 1 skipped** (all 142 pre-existing tests preserved + 8 new Release 2.6 test suites).
- **PyTorch Integrity Verification (`.venv312`, `torch 2.15.0+cpu`):**
  - Token budget stopping (`target_tokens=50` → `tokens_consumed=60`, `raw_tokens=80`, `validation_tokens=24` excluded from training counter) → **PASSED**.
  - SFT learning rate exactness (`configured_sft_lr=5e-5`, `last_optimizer_lr=2.75e-5 ≤ 5e-5`) → **PASSED**.
  - Exact interruption recovery (continuous 4 steps vs. 2 + save + resume + 2 steps) → **PASSED (`resume_max_weight_diff = 0.0`)**.
  - Dataset hash mismatch on resume → **Rejected (`ValueError`)**.
- **Not Executed (Hardware Unavailable):** CUDA single-GPU and multi-GPU distributed tests were **not executed** (`cuda=False` on local host).
- **Vercel Free Compatibility:** Preserved 100% (zero PyTorch/CUDA/FAISS in `requirements.txt`).
