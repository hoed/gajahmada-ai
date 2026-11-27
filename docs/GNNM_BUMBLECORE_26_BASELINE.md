# GajahMada GNNM BumbleCore 300M — Release 2.6 Baseline Audit

**Audit Date:** 2026-11-28  
**Baseline Release:** Release 2.5 (`142 passed, 1 skipped` on Python 3.14; `1 + 6` real PyTorch checks passed on `.venv312` `torch 2.15.0+cpu`)

---

## 1. Verified Findings Against Current Source Code

| # | Finding | Verification Status | Exact Code Location |
| :-: | :--- | :---: | :--- |
| 1 | Seed corpus used as fallback during full training | **CONFIRMED** | `gajahmada_ai/gnnm/train.py` (`dataset_path` omitted fell back to `load_seed_multilingual_corpus()` even when `smoke=False`) |
| 2 | SFT seed corpus contains only 3 conversations | **CONFIRMED** | `gajahmada_ai/gnnm/train.py` (`_build_seed_multiturn_sft()` used for full SFT) |
| 3 | Target token count not enforced as stopping criterion & 6B vs 3.2768B mismatch | **CONFIRMED** | `configs/bumblecore_300m.yaml` (`2×16×2048×50000 = 3,276,800,000` ≠ `6B`); `trainer.py` did not stop at `target_tokens` or exclude padding tokens |
| 4 | Tokenizer vocabulary changes can silently drift 303M parameter count | **CONFIRMED** | `train.py` mutated `cfg.vocab_size = tokenizer.vocab_size` without freezing tokenizer hash or re-validating 303M tolerance |
| 5 | SFT learning rate overrides YAML `5e-5` up to `1e-4` | **CONFIRMED** | `trainer.py:266` (`max(self.lr * 0.25, 1e-4)` clamped `5e-5` upward when `sft_lr` was not explicitly passed) |
| 6 | Resume missing NumPy/CUDA RNG, FP16 GradScaler, cursor, & provenance hash checks | **CONFIRMED** | `trainer.py:303-336` saved only PyTorch/Python RNG & optimizer without cursor or manifest/tokenizer hash checks |
| 7 | Dataset streaming retains all records and shingles in RAM | **CONFIRMED** | `data_pipeline.py:239-240` appended all docs to `self.records` and shingles to `self._seen_shingles` |
| 8 | Evaluation relies heavily on simple keyword matching & merges levels | **CONFIRMED** | `evaluation.py` used `expected_any` without separating Functional / Capability / Safety levels or EM/F1/PPL metrics |
| 9 | Hugging Face upload safety check is non-recursive | **CONFIRMED** | `export.py:194` used `os.listdir(export_dir)` instead of recursive walk + symlink rejection + explicit allowlist |
| 10 | Release gates check `os.path.isfile` instead of verifiable provenance evidence | **CONFIRMED** | `platform_routes.py:190-198` checked file existence without validating hashes, non-smoke flag, steps, tokens, or val loss |

---

## 2. Files to Modify / Add (Backward-Compatible)
- **New:** `gajahmada_ai/gnnm/bumblecore/dataset_governance.py`, `gajahmada_ai/gnnm/bumblecore/gate_evidence.py`, `tests/gnnm/test_bumblecore_release_2_6.py`, and Release 2.6 docs (`docs/GNNM_BUMBLECORE_26_*.md`, `docs/GNNM_BUMBLECORE_DATA_GOVERNANCE.md`).
- **Modified:** `configs/bumblecore_300m.yaml`, `gajahmada_ai/gnnm/trainer.py`, `gajahmada_ai/gnnm/train.py`, `gajahmada_ai/gnnm/bumblecore/evaluation.py`, `gajahmada_ai/gnnm/bumblecore/export.py`, `gajahmada_ai/api/platform_routes.py`.
- **Preserved:** Existing `GNNMTransformer`, `GNNMBPETokenizer`, all 142 existing tests, and 100% Vercel Free compatibility.
