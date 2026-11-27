# GajahMada GNNM BumbleCore 300M — Training & Operations Guide (Release 2.5)

All commands below match the actual repository modules and have been verified.

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

## 3. CPU Smoke Validation (Pretraining, Resume & SFT)
```bash
# Smoke Pretraining
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining --smoke --steps 4

# Resume Smoke Pretraining
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining --smoke --resume auto --steps 2

# Separate Multi-Turn SFT Stage
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage sft --smoke --steps 4
```

## 4. Full 300M Pretraining & SFT (External GPU Host ≥ 16–24 GB VRAM)
```bash
.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage pretraining --dataset /path/to/curated_id_en.jsonl --resume auto

.venv312/bin/python -m gajahmada_ai.gnnm.train \
  --config configs/bumblecore_300m.yaml \
  --stage sft --resume ./models/bumblecore-300m/pretraining
```

## 5. Checkpoint Export & Parity Verification
```python
from gajahmada_ai.gnnm.bumblecore.export import export_inference_artifact, verify_export_import_parity
export_inference_artifact("./models/bumblecore-smoke/pretraining", "./models/bumblecore-smoke-export")
print(verify_export_import_parity("./models/bumblecore-smoke/pretraining", "./models/bumblecore-smoke-export"))
```
