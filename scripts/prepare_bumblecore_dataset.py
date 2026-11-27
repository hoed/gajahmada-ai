#!/usr/bin/env python3
"""
GajahMada GNNM BumbleCore 300M — Release 2.6 Streaming Dataset Preparation CLI.

Converts a raw licensed JSONL corpus into a Release 2.6 governed dataset directory
(`train.jsonl`, `validation.jsonl`, `test.jsonl`, `dataset_manifest.json`, `fingerprints.db`)
using bounded-memory SQLite WAL deduplication and verifies the resulting manifest immediately.

Usage:
  python scripts/prepare_bumblecore_dataset.py \
    --input /data/raw_id_en.jsonl \
    --output-dir /data/bumblecore_pretrain_v1 \
    --dataset-id bumblecore-id-en-v1 \
    --source-name wikimedia-id-en \
    --license CC-BY-SA-4.0 \
    --approval-status approved
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gajahmada_ai.gnnm.bumblecore.dataset_governance import (
    APPROVED_LICENSES,
    BoundedStreamingPreprocessor,
    validate_dataset_and_manifest,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare & verify a Release 2.6 BumbleCore 300M streaming dataset directory."
    )
    parser.add_argument("--input", required=True, help="Path to raw input JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Destination dataset directory.")
    parser.add_argument("--dataset-id", required=True, help="Unique dataset identifier.")
    parser.add_argument("--dataset-version", default="1.0.0", help="Semantic dataset version.")
    parser.add_argument("--description", default="Curated GajahMada BumbleCore ID/EN corpus", help="Dataset description.")
    parser.add_argument("--source-name", required=True, help="Upstream corpus source name.")
    parser.add_argument("--source-url", default=None, help="Optional upstream source URL.")
    parser.add_argument(
        "--license",
        required=True,
        help=f"Verified SPDX license. Approved: {', '.join(sorted(APPROVED_LICENSES))}",
    )
    parser.add_argument(
        "--approval-status",
        choices=("pending_review", "approved"),
        default="pending_review",
        help="Set to 'approved' ONLY after human license & provenance review.",
    )
    parser.add_argument(
        "--reviewer",
        default=None,
        help="Optional reviewer name/handle recorded in approval_evidence.",
    )
    parser.add_argument(
        "--languages",
        default="id,en,mixed",
        help="Comma-separated allowed language codes (default: id,en,mixed).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(json.dumps({"status": "INPUT_NOT_FOUND", "input": args.input}, indent=2))
        return 1

    approval_evidence = (
        {"reviewed_by": args.reviewer or "operator", "license_verified": True}
        if args.approval_status == "approved"
        else None
    )
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    proc = BoundedStreamingPreprocessor(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        dataset_description=args.description,
        source_name=args.source_name,
        source_url=args.source_url,
        license_name=args.license,
        approval_status=args.approval_status,
        approval_evidence=approval_evidence,
        allowed_languages=langs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    manifest = proc.process_jsonl(args.input)
    validation = validate_dataset_and_manifest(
        args.output_dir,
        require_approved=(args.approval_status == "approved"),
    )

    out = {
        "status": "READY" if validation["valid"] else "VALIDATION_FAILED",
        "output_dir": args.output_dir,
        "document_count": manifest.get("document_count"),
        "token_count": manifest.get("token_count"),
        "manifest_checksum": manifest.get("manifest_checksum"),
        "validation": validation,
    }
    print(json.dumps(out, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
