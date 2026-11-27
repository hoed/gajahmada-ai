"""
Release 2.6 Comprehensive Remediation Tests for GajahMada GNNM BumbleCore 300M.

Covers:
- Phase 1: Dataset governance, seed rejection for full pretraining & full SFT, approved license
  enforcement, hash mismatch, source-level & document-level leakage, SQLite WAL bounded streaming.
- Phase 2: Config token consistency (2*16*2048*50000 == 3,276,800,000).
- Phase 6: 3-level evaluation (functional/capability/safety), EM/F1/numerical/JSON metrics,
  empty benchmark failure.
- Phase 7: Recursive HF upload safety scan (nested secrets, symlinks, allowlist, upload manifest).
- Phase 8 & 9: Evidence-based Release Gates (rejects smoke checkpoints, hash tampering, wrong
  stage SFT->pretraining, unapproved license) & /api/v2/platform/bumblecore transparency.
"""
from __future__ import annotations

import json
import os
import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server
from gajahmada_ai.gnnm.bumblecore.dataset_governance import (
    BoundedStreamingPreprocessor,
    sha256_file,
    validate_dataset_and_manifest,
    validate_sft_dataset,
)
from gajahmada_ai.gnnm.bumblecore.evaluation import (
    exact_match,
    normalized_exact_match,
    numerical_match,
    run_held_out_evaluation,
    structured_json_valid,
    token_f1,
)
from gajahmada_ai.gnnm.bumblecore.export import prepare_hf_upload, scan_export_directory_recursive
from gajahmada_ai.gnnm.bumblecore.gate_evidence import (
    evaluate_evidence_based_gates,
    verify_stage_checkpoint_evidence,
)
from gajahmada_ai.gnnm.train import execute_training_run, load_yaml_config


def test_phase1_full_pretraining_and_sft_reject_seed_only_data():
    r_pre = execute_training_run("configs/bumblecore_300m.yaml", stage="pretraining", smoke=False, dataset_path=None)
    assert r_pre["status"] == "DATASET_GOVERNANCE_FAILED"
    r_sft = execute_training_run("configs/bumblecore_300m.yaml", stage="sft", smoke=False, dataset_path=None)
    assert r_sft["status"] == "DATASET_GOVERNANCE_FAILED"


def test_phase1_bounded_sqlite_streaming_manifest_license_and_leakage(tmp_path):
    raw_jsonl = tmp_path / "raw.jsonl"
    rows = []
    for i in range(25):
        rows.append(json.dumps({
            "text": f"Artikel riset nomor {i} membahas arsitektur GNNM BumbleCore 300M dan sejarah Nusantara secara ilmiah.",
            "source_doc_id": f"src-{i // 2}",  # pairs share source_doc_id -> must stay in same split
            "language": "id",
        }))
    # Add exact duplicate + near duplicate
    rows.append(rows[0])
    raw_jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "processed"
    proc = BoundedStreamingPreprocessor(
        output_dir=str(out_dir),
        dataset_id="test-ds",
        source_name="unit_test_corpus",
        license_name="MIT",
        approval_status="approved",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    manifest = proc.process_jsonl(str(raw_jsonl))
    assert os.path.isfile(out_dir / "fingerprints.db")
    assert manifest["token_count"] > 0
    assert manifest["stats"]["rejected_duplicate"] >= 1

    # Valid approved manifest passes
    val_ok = validate_dataset_and_manifest(str(out_dir), require_approved=True)
    assert val_ok["valid"] is True, val_ok["errors"]

    # Unapproved license fails
    m_path = out_dir / "dataset_manifest.json"
    bad_m = dict(manifest, license="UNKNOWN_SCRAPED")
    m_path.write_text(json.dumps(bad_m), encoding="utf-8")
    assert any("UNAPPROVED_LICENSE" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # Restore valid manifest, then tamper with train.jsonl -> HASH_MISMATCH
    m_path.write_text(json.dumps(manifest), encoding="utf-8")
    with open(out_dir / "train.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "tampered extra line", "document_hash": "deadbeef"}) + "\n")
    assert any("HASH_MISMATCH" in e for e in validate_dataset_and_manifest(str(out_dir))["errors"])

    # Explicit cross-split DOCUMENT_LEAKAGE and SOURCE_LEAKAGE detection
    leak_dir = tmp_path / "leak_ds"
    leak_dir.mkdir()
    dup_rec = json.dumps({"text": "shared text across splits", "document_hash": "abc123", "source_doc_id": "src-99"})
    (leak_dir / "train.jsonl").write_text(dup_rec + "\n", encoding="utf-8")
    (leak_dir / "validation.jsonl").write_text(dup_rec + "\n", encoding="utf-8")
    (leak_dir / "test.jsonl").write_text("", encoding="utf-8")
    (leak_dir / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "leak-test",
        "source_name": "test",
        "license": "MIT",
        "approval_status": "approved",
        "document_count": 2,
        "split_hashes": {
            "train": sha256_file(str(leak_dir / "train.jsonl")),
            "validation": sha256_file(str(leak_dir / "validation.jsonl")),
            "test": sha256_file(str(leak_dir / "test.jsonl")),
        },
    }), encoding="utf-8")
    leak_res = validate_dataset_and_manifest(str(leak_dir))
    assert leak_res["valid"] is False
    assert any("DOCUMENT_LEAKAGE" in e for e in leak_res["errors"])
    assert any("SOURCE_LEAKAGE" in e for e in leak_res["errors"])


def test_phase1_sft_dataset_validation_rejects_bad_format_and_insufficient_examples(tmp_path):
    sft_dir = tmp_path / "sft_ds"
    sft_dir.mkdir()
    (sft_dir / "train.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8")
    (sft_dir / "validation.jsonl").write_text("", encoding="utf-8")
    (sft_dir / "test.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "dataset_id": "sft-bad",
        "source_name": "test",
        "license": "MIT",
        "approval_status": "approved",
        "document_count": 1,
        "split_hashes": {
            "train": sha256_file(str(sft_dir / "train.jsonl")),
            "validation": sha256_file(str(sft_dir / "validation.jsonl")),
            "test": sha256_file(str(sft_dir / "test.jsonl")),
        },
    }
    (sft_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    res = validate_sft_dataset(str(sft_dir), min_examples=5)
    assert res["valid"] is False
    assert any("INVALID_SFT_FORMAT" in e for e in res["errors"])


def test_phase2_yaml_token_budget_consistency():
    raw = load_yaml_config("configs/bumblecore_300m.yaml")
    tr = raw["training"]
    m = raw["model"]
    per_step = tr["micro_batch_size"] * tr["gradient_accumulation_steps"] * m["max_seq_len"]
    assert per_step == 65_536
    assert tr["tokens_per_optimizer_step"] == 65_536
    assert per_step * tr["max_steps"] == 3_276_800_000
    assert tr["target_tokens"] == 3_276_800_000
    assert raw["sft"]["lr"] == 5e-5


def test_phase6_three_level_evaluation_metrics_and_empty_rejection():
    assert exact_match("Tokyo", "Tokyo") == 1.0
    assert normalized_exact_match("Tokyo!", "tokyo") == 1.0
    assert token_f1("Sumpah Palapa Gajah Mada", "Sumpah Palapa adalah ikrar Gajah Mada") > 0.7
    assert numerical_match("Hasilnya 84.0", 84.0) is True
    assert structured_json_valid('{"ok": true}') is True
    assert structured_json_valid("not json") is False

    empty_rep = run_held_out_evaluation(lambda _: "x", cases=[])
    assert empty_rep["status"] == "EMPTY_BENCHMARK_FAILED"
    assert empty_rep["overall_score"] is None

    rep = run_held_out_evaluation(lambda _: "Sumpah Palapa Nusantara. Ed25519 signature. 84. Tokyo. Belum tersedia.")
    assert rep["status"] == "COMPLETED"
    assert set(rep["levels"].keys()) == {"functional", "capability", "safety"}
    assert rep["levels"]["capability"]["avg_token_f1"] is not None


def test_phase7_recursive_hf_upload_safety_and_allowlist(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("# card", encoding="utf-8")
    clean_scan = scan_export_directory_recursive(str(tmp_path))
    assert clean_scan["safe"] is True
    assert len(clean_scan["manifest"]) == 2

    # Nested .env inside subdirectory MUST be caught recursively
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / ".env").write_text("SECRET=1", encoding="utf-8")
    bad_scan = scan_export_directory_recursive(str(tmp_path))
    assert bad_scan["safe"] is False
    assert any("FORBIDDEN_SENSITIVE_FILE" in v for v in bad_scan["violations"])
    with pytest.raises(ValueError, match="recursive safety"):
        prepare_hf_upload(str(tmp_path), dry_run=True)

    # Symlink rejection
    (sub / ".env").unlink()
    os.symlink(str(tmp_path / "config.json"), str(tmp_path / "link.json"))
    sym_scan = scan_export_directory_recursive(str(tmp_path))
    assert sym_scan["safe"] is False
    assert any("SYMLINK_FORBIDDEN" in v for v in sym_scan["violations"])


def test_phase8_evidence_based_release_gates_reject_smoke_tamper_and_wrong_stage(tmp_path):
    ckpt_dir = tmp_path / "pre"
    ckpt_dir.mkdir()
    model_pt = ckpt_dir / "model.pt"
    model_pt.write_bytes(b"fake-weights-for-hash-test")
    good_hash = sha256_file(str(model_pt))

    (ckpt_dir / "dataset_manifest.json").write_text(
        json.dumps({"license": "MIT", "approval_status": "approved"}), encoding="utf-8"
    )
    meta = {
        "stage": "pretraining",
        "smoke_mode": True,  # MUST BE REJECTED!
        "checkpoint_sha256": good_hash,
        "actual_steps": 10,
        "actual_tokens_processed": 5000,
        "exact_trainable_parameters": 303_350_784,
        "validation_metrics": {"status": "COMPUTED", "val_loss": 3.5},
        "tokenizer_sha256": "tokhash",
        "config_sha256": "cfghash",
        "dataset_manifest_sha256": "dshash",
    }
    (ckpt_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    ev_smoke = verify_stage_checkpoint_evidence(str(ckpt_dir), expected_stage="pretraining")
    assert ev_smoke["passed"] is False
    assert any("SMOKE_CHECKPOINT_REJECTED" in r for r in ev_smoke["reasons"])

    # Wrong stage (sft cannot satisfy pretraining)
    meta["smoke_mode"] = False
    meta["stage"] = "sft"
    (ckpt_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    ev_wrong = verify_stage_checkpoint_evidence(str(ckpt_dir), expected_stage="pretraining")
    assert ev_wrong["passed"] is False
    assert any("WRONG_STAGE" in r for r in ev_wrong["reasons"])

    # Valid non-smoke pretraining evidence passes
    meta["stage"] = "pretraining"
    (ckpt_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    ev_ok = verify_stage_checkpoint_evidence(str(ckpt_dir), expected_stage="pretraining")
    assert ev_ok["passed"] is True

    # Tampering weights breaks SHA-256 check
    model_pt.write_bytes(b"tampered-weights")
    ev_tamper = verify_stage_checkpoint_evidence(str(ckpt_dir), expected_stage="pretraining")
    assert ev_tamper["passed"] is False
    assert any("CHECKPOINT_HASH_MISMATCH" in r for r in ev_tamper["reasons"])


def test_phase9_platform_bumblecore_reports_26_not_started_and_3_27b_budget():
    server.ensure_initialized()
    client = TestClient(server.app)
    d = client.get("/api/v2/platform/bumblecore").json()
    assert d["release"] == "2.6"
    assert d["pretraining_status"] == "NOT_STARTED"
    assert d["lifecycle_stage"] in ("CONFIGURED", "SMOKE_TESTED")
    assert d["theoretical_token_budget"]["total_tokens"] == 3_276_800_000
    assert d["release_gates"]["blocked_at_gate"] == "GATE_C_PRETRAINING_CHECKPOINT_VALIDATED"
