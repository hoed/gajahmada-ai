"""
Release 2.6 Batch 4 — Dataset Manifest Verification Acceptance Tests (All 10 Cases).

1. Valid manifest accepted (with approval_evidence, split_hashes, manifest_checksum, token_count)
2. Missing split hash rejected
3. Split hash mismatch rejected
4. Missing split file rejected
5. Path traversal (`../escape.jsonl`) rejected
6. Symlink escape rejected
7. Unapproved dataset & missing approval_evidence rejected
8. Unknown license rejected
9. Token count mismatch rejected
10. Train/test leakage rejected
"""
from __future__ import annotations

import json
import os

from gajahmada_ai.gnnm.bumblecore.dataset_governance import (
    BoundedStreamingPreprocessor,
    sha256_file,
    validate_dataset_and_manifest,
)


def _build_valid_dataset(tmp_path):
    raw_p = tmp_path / "raw.jsonl"
    rows = [
        json.dumps({"text": f"Artikel riset ilmiah nomor {i} membahas arsitektur GNNM BumbleCore dan Nusantara.", "source_doc_id": f"doc-{i}", "language": "id"})
        for i in range(30)
    ]
    raw_p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / "ds"
    proc = BoundedStreamingPreprocessor(
        output_dir=str(out_dir),
        dataset_id="bumblecore-test-ds",
        dataset_version="1.0.0",
        dataset_description="Unit test curated dataset",
        source_name="gajahmada_unit_test",
        source_url="https://github.com/hoed/gajahmada-ai",
        license_name="MIT",
        approval_status="approved",
        approval_evidence={"approved_by": "hoed", "review_reference": "PR-2.6-DATA"},
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    manifest = proc.process_jsonl(str(raw_p))
    return out_dir, manifest


def test_all_10_dataset_manifest_verification_cases(tmp_path):
    # 1. Valid manifest accepted
    out_dir, manifest = _build_valid_dataset(tmp_path)
    res_ok = validate_dataset_and_manifest(str(out_dir), require_approved=True, require_approval_evidence=True)
    assert res_ok["valid"] is True, res_ok["errors"]
    assert res_ok["measured_tokens"] == manifest["token_count"]

    m_path = out_dir / "dataset_manifest.json"

    # 2. Missing split hash rejected
    m2 = dict(manifest, split_hashes={"train": manifest["split_hashes"]["train"]})
    m2.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m2), encoding="utf-8")
    assert any("MISSING_SPLIT_HASH" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # 3. Hash mismatch rejected
    m3 = dict(manifest, split_hashes={**manifest["split_hashes"], "train": "0" * 64})
    m3.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m3), encoding="utf-8")
    assert any("HASH_MISMATCH" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # 4. Missing split file rejected
    m4 = dict(manifest, test_split="does_not_exist.jsonl")
    m4.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m4), encoding="utf-8")
    assert any("MISSING_SPLIT_FILE" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # 5. Path traversal (`../escape.jsonl`) rejected
    m5 = dict(manifest, train_split="../raw.jsonl")
    m5.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m5), encoding="utf-8")
    assert any("PATH_TRAVERSAL_FORBIDDEN" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # 6. Symlink escape rejected
    sym_path = out_dir / "sym_train.jsonl"
    os.symlink(str(tmp_path / "raw.jsonl"), str(sym_path))
    m6 = dict(manifest, train_split="sym_train.jsonl")
    m6.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m6), encoding="utf-8")
    assert any("SYMLINK_ESCAPE_FORBIDDEN" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])
    sym_path.unlink()

    # 7. Unapproved dataset & missing approval_evidence rejected
    m7 = dict(manifest, approval_status="pending_review", approval_evidence=None)
    m7.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m7), encoding="utf-8")
    errs7 = validate_dataset_and_manifest(str(out_dir), require_approved=True, require_approval_evidence=True)["errors"]
    assert any("UNAPPROVED_DATASET" in e for e in errs7)
    assert any("MISSING_APPROVAL_EVIDENCE" in e for e in errs7)

    # 8. Unknown license rejected
    m8 = dict(manifest, license="PROPRIETARY_SCRAPED")
    m8.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m8), encoding="utf-8")
    assert any("UNAPPROVED_LICENSE" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # 9. Token count mismatch rejected
    m9 = dict(manifest, token_count=manifest["token_count"] + 999)
    m9.pop("manifest_checksum", None)
    m_path.write_text(json.dumps(m9), encoding="utf-8")
    assert any("TOKEN_COUNT_MISMATCH" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # 10. Train/test leakage rejected
    leak_dir = tmp_path / "leak_tt"
    leak_dir.mkdir()
    rec = json.dumps({"text": "shared doc", "document_hash": "hash-xyz", "source_doc_id": "src-xyz", "token_count": 5})
    (leak_dir / "train.jsonl").write_text(rec + "\n", encoding="utf-8")
    (leak_dir / "validation.jsonl").write_text("", encoding="utf-8")
    (leak_dir / "test.jsonl").write_text(rec + "\n", encoding="utf-8")
    (leak_dir / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "leak-tt",
        "source_name": "test",
        "license": "MIT",
        "approval_status": "approved",
        "document_count": 2,
        "token_count": 10,
        "split_hashes": {
            "train": sha256_file(str(leak_dir / "train.jsonl")),
            "validation": sha256_file(str(leak_dir / "validation.jsonl")),
            "test": sha256_file(str(leak_dir / "test.jsonl")),
        },
    }), encoding="utf-8")
    errs10 = validate_dataset_and_manifest(str(leak_dir))["errors"]
    assert any("DOCUMENT_LEAKAGE(train<->test)" in e for e in errs10)
    assert any("SOURCE_LEAKAGE(train<->test)" in e for e in errs10)
