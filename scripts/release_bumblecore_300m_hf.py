#!/usr/bin/env python3
"""
GajahMada GNNM BumbleCore 300M — Release 2.6 Verified Hugging Face Release CLI.

Fail-closed 5-step release pipeline for GPU hosts after pretraining/SFT completes:
  1. Checkpoint Trust Layer verification (PyTorch tensors, live SHA-256 provenance, optional Ed25519).
  2. Inference-only export (`require_trained=True`, excludes optimizer/trainer_state.pt).
  3. Full export/import & Safetensors parity verification (shapes, dtypes, values, deterministic logits).
  4. Recursive safety scan, manifest hash check, config/tokenizer consistency, & Model Card validation.
  5. Dry-run manifest generation by default; live upload ONLY with `--confirm-upload` + `HF_TOKEN`.

Usage (Dry-Run — Safe Default):
  python scripts/release_bumblecore_300m_hf.py \
    --checkpoint-dir ./models/bumblecore-300m/sft \
    --stage sft \
    --export-dir ./models/bumblecore-300m-export \
    --repo-id <your-hf-org>/GNNM-BumbleCore-300M

Usage (Live Private Upload — Explicit Approval):
  HF_TOKEN=hf_xxx python scripts/release_bumblecore_300m_hf.py \
    --checkpoint-dir ./models/bumblecore-300m/sft \
    --stage sft \
    --export-dir ./models/bumblecore-300m-export \
    --repo-id <your-hf-org>/GNNM-BumbleCore-300M \
    --confirm-upload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path when invoked directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gajahmada_ai.gnnm.bumblecore.checkpoint_trust import verify_trusted_checkpoint
from gajahmada_ai.gnnm.bumblecore.export import (
    export_inference_artifact,
    prepare_hf_upload,
    validate_export_bundle,
    verify_export_import_parity,
)


def run_release_pipeline(
    checkpoint_dir: str,
    stage: str,
    export_dir: str,
    repo_id: str,
    eval_report_path: Optional[str] = None,
    private: bool = True,
    confirm_upload: bool = False,
    require_signature: bool = False,
    min_params: int = 285_000_000,
) -> Dict[str, Any]:
    """Execute the fail-closed 5-step BumbleCore 300M Hugging Face release pipeline."""
    # Step 1: Checkpoint Trust Layer (rejects missing, fake, smoke, wrong-stage, or tampered checkpoints)
    trust = verify_trusted_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_stage=stage,
        expected_min_params=min_params,
        allow_smoke=False,
        require_signature=require_signature,
    )
    if not trust.get("passed"):
        return {
            "status": "FAILED_AT_CHECKPOINT_TRUST",
            "step": 1,
            "trust_report": trust,
        }

    # Optional benchmark report attachment
    benchmark_report: Optional[Dict[str, Any]] = None
    if eval_report_path:
        if not os.path.isfile(eval_report_path):
            return {
                "status": "FAILED_EVAL_REPORT_MISSING",
                "step": 1,
                "error": f"Evaluation report not found: {eval_report_path}",
            }
        with open(eval_report_path, "r", encoding="utf-8") as f:
            benchmark_report = json.load(f)

    # Step 2: Inference-only export with strict require_trained=True guard
    manifest = export_inference_artifact(
        source_checkpoint_dir=checkpoint_dir,
        export_dir=export_dir,
        repo_id=repo_id,
        benchmark_report=benchmark_report,
        require_trained=True,
    )

    # Step 3: Exact tensor/dtype/value/logit/Safetensors parity check
    parity = verify_export_import_parity(checkpoint_dir, export_dir)
    if not parity.get("parity_verified"):
        return {
            "status": "FAILED_AT_EXPORT_PARITY",
            "step": 3,
            "parity": parity,
        }

    # Step 4: Recursive safety scan, live manifest SHA-256, vocab consistency, & Model Card validation
    bundle = validate_export_bundle(export_dir)
    if not bundle.get("valid"):
        return {
            "status": "FAILED_AT_BUNDLE_VALIDATION",
            "step": 4,
            "bundle": bundle,
        }

    # Step 5: Dry-run manifest (default) or explicit auth-gated upload
    upload = prepare_hf_upload(
        export_dir=export_dir,
        repo_id=repo_id,
        private=private,
        dry_run=not confirm_upload,
        confirm_upload=confirm_upload,
        allow_untrained=False,
    )

    return {
        "status": upload.get("status", "COMPLETED"),
        "release": "2.6",
        "checkpoint_sha256": trust.get("checkpoint_sha256"),
        "exact_parameters": manifest.get("exact_parameters"),
        "training_status": manifest.get("training_status"),
        "safetensors_exported": manifest.get("safetensors_exported"),
        "parity": parity,
        "bundle_valid": bundle.get("valid"),
        "upload": upload,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GajahMada GNNM BumbleCore 300M (Release 2.6) — Verified Hugging Face Release CLI"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./models/bumblecore-300m/sft",
        help="Path to verified non-smoke 300M checkpoint directory (pretraining or sft).",
    )
    parser.add_argument(
        "--stage",
        choices=("pretraining", "sft"),
        default="sft",
        help="Expected training stage of the checkpoint.",
    )
    parser.add_argument(
        "--export-dir",
        default="./models/bumblecore-300m-export",
        help="Destination directory for lightweight inference-only export artifacts.",
    )
    parser.add_argument(
        "--repo-id",
        default="GajahMada/GNNM-BumbleCore-300M",
        help="Target Hugging Face repository ID (e.g. username/GNNM-BumbleCore-300M).",
    )
    parser.add_argument(
        "--eval-report",
        default=None,
        help="Optional path to verified held-out evaluation_report.json to cite in README.md.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/upload as a public Hugging Face repo (defaults to private=True).",
    )
    parser.add_argument(
        "--confirm-upload",
        action="store_true",
        help="Execute live upload to Hugging Face (requires HF_TOKEN). Without this flag, runs dry-run only.",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Require valid Ed25519 manifest signature on run_metadata.json.",
    )
    parser.add_argument(
        "--min-params",
        type=int,
        default=285_000_000,
        help="Minimum trainable parameter count required (default: 285,000,000 for 300M tier).",
    )
    args = parser.parse_args(argv)

    result = run_release_pipeline(
        checkpoint_dir=args.checkpoint_dir,
        stage=args.stage,
        export_dir=args.export_dir,
        repo_id=args.repo_id,
        eval_report_path=args.eval_report,
        private=not args.public,
        confirm_upload=args.confirm_upload,
        require_signature=args.require_signature,
        min_params=args.min_params,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("DRY_RUN_ONLY", "UPLOADED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
