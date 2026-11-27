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

## 5. Checkpoint Trust Verification, Export & Parity Check
```python
from gajahmada_ai.gnnm.bumblecore.checkpoint_trust import verify_trusted_checkpoint
from gajahmada_ai.gnnm.bumblecore.export import (
    export_inference_artifact,
    validate_export_bundle,
    verify_export_import_parity,
)

# Verify real PyTorch tensors, live SHA-256 provenance, and Ed25519 signature
print(verify_trusted_checkpoint("./models/bumblecore-300m/pretraining", expected_stage="pretraining"))

# Export inference-only artifact (pass require_trained=True for production releases)
export_inference_artifact("./models/bumblecore-smoke/pretraining", "./models/bumblecore-smoke-export")
print(verify_export_import_parity("./models/bumblecore-smoke/pretraining", "./models/bumblecore-smoke-export"))
print(validate_export_bundle("./models/bumblecore-smoke-export"))
```
