#!/usr/bin/env bash
# GajahMada GNNM BumbleCore 300M — Release 2.6 Cloud GPU Runner (Colab / RunPod / A100 / H100)
#
# Fail-closed end-to-end training & export-verification script:
#   1. Verifies NVIDIA GPU + PyTorch CUDA availability.
#   2. Runs Release 2.6 resource preflight (configs/bumblecore_300m.yaml).
#   3. Executes full 300M pretraining on approved external dataset (--resume auto).
#   4. Optionally executes separate SFT stage if SFT_DATASET_DIR is provided.
#   5. Runs dry-run Hugging Face release verification (NEVER uploads automatically).
#
# Example usage on RunPod / A100 / Colab Pro:
#   PRETRAIN_DATASET_DIR=/workspace/data/pretrain_v1 \
#   SFT_DATASET_DIR=/workspace/data/sft_v1 \
#   HF_REPO_ID=GajahMada/GNNM-BumbleCore-300M \
#   bash scripts/train_bumblecore_300m_gpu.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/bumblecore_300m.yaml}"
PRETRAIN_DATASET_DIR="${PRETRAIN_DATASET_DIR:-}"
SFT_DATASET_DIR="${SFT_DATASET_DIR:-}"
EXPORT_DIR="${EXPORT_DIR:-./models/bumblecore-300m-export}"
HF_REPO_ID="${HF_REPO_ID:-GajahMada/GNNM-BumbleCore-300M}"

echo "=================================================================="
echo " GajahMada GNNM BumbleCore 300M — Release 2.6 Cloud GPU Runner"
echo "=================================================================="

if [[ -z "${PRETRAIN_DATASET_DIR}" ]]; then
  echo "ERROR: PRETRAIN_DATASET_DIR is required (Release 2.6 refuses seed-only full pretraining)." >&2
  echo "Prepare an approved dataset first using: scripts/prepare_bumblecore_dataset.py" >&2
  exit 1
fi

if [[ ! -f "${PRETRAIN_DATASET_DIR}/dataset_manifest.json" ]]; then
  echo "ERROR: Missing ${PRETRAIN_DATASET_DIR}/dataset_manifest.json" >&2
  exit 1
fi

echo "[1/5] Checking NVIDIA GPU & PyTorch CUDA..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "ERROR: nvidia-smi not found. Full 300M pretraining requires a CUDA GPU host (>=16-24GB VRAM)." >&2
  exit 1
fi

"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"

echo "[2/5] Running Release 2.6 Resource Preflight..."
"${PYTHON_BIN}" -m gajahmada_ai.gnnm.train \
  --config "${CONFIG_PATH}" \
  --stage pretraining \
  --preflight

echo "[3/5] Launching BumbleCore 300M Full Pretraining (exact 3.2768B token budget, --resume auto)..."
"${PYTHON_BIN}" -m gajahmada_ai.gnnm.train \
  --config "${CONFIG_PATH}" \
  --stage pretraining \
  --dataset "${PRETRAIN_DATASET_DIR}" \
  --resume auto

FINAL_STAGE="pretraining"
FINAL_CKPT="./models/bumblecore-300m/pretraining"

if [[ -n "${SFT_DATASET_DIR}" ]]; then
  echo "[4/5] Launching BumbleCore 300M Separate SFT Stage (YAML lr=5e-5)..."
  "${PYTHON_BIN}" -m gajahmada_ai.gnnm.train \
    --config "${CONFIG_PATH}" \
    --stage sft \
    --dataset "${SFT_DATASET_DIR}" \
    --resume "${FINAL_CKPT}"
  FINAL_STAGE="sft"
  FINAL_CKPT="./models/bumblecore-300m/sft"
else
  echo "[4/5] SFT_DATASET_DIR not set — skipping SFT stage."
fi

echo "[5/5] Running Fail-Closed Hugging Face Release Dry-Run (NO live upload)..."
"${PYTHON_BIN}" scripts/release_bumblecore_300m_hf.py \
  --checkpoint-dir "${FINAL_CKPT}" \
  --stage "${FINAL_STAGE}" \
  --export-dir "${EXPORT_DIR}" \
  --repo-id "${HF_REPO_ID}"

echo ""
echo "DONE! Dry-run export verified at: ${EXPORT_DIR}"
echo "To upload to Hugging Face after reviewing the dry-run manifest, run explicitly:"
echo "  HF_TOKEN=hf_xxx ${PYTHON_BIN} scripts/release_bumblecore_300m_hf.py \\"
echo "    --checkpoint-dir ${FINAL_CKPT} --stage ${FINAL_STAGE} \\"
echo "    --export-dir ${EXPORT_DIR} --repo-id ${HF_REPO_ID} --confirm-upload"
