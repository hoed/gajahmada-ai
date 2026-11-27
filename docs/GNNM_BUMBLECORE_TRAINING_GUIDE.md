# GajahMada GNNM BumbleCore 300M — Training & Operations Guide (Release 2.6)

All commands below match the actual repository modules and have been verified. Full 300M pretraining and SFT are **never** executed automatically and require an approved external dataset manifest + dedicated GPU infrastructure (`PRETRAINING_STATUS = NOT_STARTED`).

## 1. Environment Setup (Local PyTorch in `.venv312`)
```bash
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r requirements.txt
uv pip install --python .venv312/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```
*(For NVIDIA GPU training, replace `/whl/cpu` with `/whl/cu124`.)*

## 2. Resource Preflight (Does NOT Launch Training)
```bash
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining \
  --preflight
```

## 3. CPU Smoke Validation (Pretraining, Exact Resume & SFT)
Seed corpus is permitted **only** when `--smoke` is explicitly set:
```bash
# Smoke Pretraining
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining --smoke --steps 4

# Exact Resume Smoke Pretraining (restores weights, optimizer, scheduler, cursor, tokens, RNGs)
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining --smoke --resume auto --steps 2

# Separate Multi-Turn SFT Stage (uses configured YAML sft.lr = 5e-5)
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage sft --smoke --steps 4
```

## 4. Full 300M Pretraining & SFT (External GPU Host ≥ 16–24 GB VRAM)
Requires a validated dataset directory containing `dataset_manifest.json` (`approval_status == "approved"`, approved license, verified `split_hashes`, zero cross-split leakage). Training streams line-by-line via `StreamingPretrainDataset` / `StreamingSFTDataset`, enforces exact zero-overshoot `target_tokens` (`3,276,800,000` in `configs/bumblecore_300m.yaml`), and signs `run_metadata.json` via Ed25519 `NodeIdentity` when `node_key.pem` is present:
```bash
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining --dataset /path/to/approved_pretrain_dataset_dir --resume auto

.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage sft --dataset /path/to/approved_sft_dataset_dir --resume ./models/bumblecore-300m/pretraining
```

## 5. Checkpoint Trust Verification, Export & Hugging Face Release CLI
Once full 300M pretraining/SFT completes on your GPU host, run the fail-closed 5-step release CLI (`scripts/release_bumblecore_300m_hf.py`):
```bash
# 1. Dry-run (verifies Checkpoint Trust, exports inference bundle, checks parity & Model Card — does NOT upload)
.venv312/bin/python scripts/release_bumblecore_300m_hf.py \
  --checkpoint-dir ./models/bumblecore-300m/sft \
  --stage sft \
  --export-dir ./models/bumblecore-300m-export \
  --repo-id GajahMada/GNNM-BumbleCore-300M

# 2. Live Private Upload (requires HF_TOKEN and explicit --confirm-upload)
HF_TOKEN=hf_xxx .venv312/bin/python scripts/release_bumblecore_300m_hf.py \
  --checkpoint-dir ./models/bumblecore-300m/sft \
  --stage sft \
  --export-dir ./models/bumblecore-300m-export \
  --repo-id GajahMada/GNNM-BumbleCore-300M \
  --confirm-upload
```
