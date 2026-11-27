# GajahMada GNNM BumbleCore 300M — Release 2.5 Checklist

- [x] **Phase 0:** Repository audit & `docs/GNNM_BUMBLECORE_BASELINE.md` completed at commit `ec200f8`.
- [x] **Phase 1:** `BumbleCoreConfig` & `BumbleCoreModel` implemented; exact `303,350,784` params verified on instantiated PyTorch BF16 model (`difference = 0`).
- [x] **Phase 2:** `GNNMBPETokenizer` lossless audit (8/8 categories), streaming/PII/dedup/zero-leakage split pipeline, & `docs/GNNM_BUMBLECORE_DATA_POLICY.md`.
- [x] **Phase 3:** `GNNMTrainer` extended with grad accum, warmup, AMP, telemetry, atomic resumable checkpoints, & strict non-empty validation.
- [x] **Phase 4:** `run_bumblecore_preflight` & `docs/GNNM_BUMBLECORE_RESOURCE_PLAN.md`.
- [x] **Phase 5 & 6:** `python -m gajahmada_ai.gnnm.train` CLI (`--preflight`, `--stage pretraining|sft`, `--resume auto`, `--smoke`) & multi-turn SFT `-100` masking.
- [x] **Phase 7:** 16-dimension evaluation suite & 7 sequential Release Gates A–G (`evaluation.py`).
- [x] **Phase 8 & 9:** Inference artifact export, deterministic parity verification, Hugging Face model card, & dry-run/auth-gated upload helper (`export.py`).
- [x] **Phase 10 & 11:** `/api/v2/platform/bumblecore` ecosystem endpoint & preserved Vercel Free compatibility.
- [ ] **External GPU Milestone:** Execute full 6B-token pretraining & SFT for `GNNM-BumbleCore-300M` on external ≥16–24 GB VRAM GPU hardware to clear Gates C–G.
- [ ] **Public Release Approval:** Explicit license, safety, & model card review prior to public Hugging Face upload.
