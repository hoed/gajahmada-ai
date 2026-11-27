"""
Batch 7 Acceptance Tests — Evidence-Based Release Gates (Zero Hardcoded Booleans).

Verifies that all 7 Release Gates (A-G) are computed strictly from dynamic/on-disk evidence:
- Gate A fails if YAML model dimensions drift from 303,350,784 canonical parameters.
- Gate B fails if YAML token budget math is inconsistent or SFT LR is invalid.
- Gate C & D enforce Checkpoint Trust Layer (non-smoke 300M checkpoint).
- Gate E requires a completed, leakage-free held-out benchmark report linked to a verified checkpoint hash.
- Gate F requires recursive safety scan, live manifest SHA-256 verification, and a verified 300M checkpoint.
- Gate G requires verified platform integration evidence linked to the verified 300M checkpoint hash.
"""
from __future__ import annotations

import json
import os
import yaml

from gajahmada_ai.gnnm.bumblecore.dataset_governance import sha256_file
from gajahmada_ai.gnnm.bumblecore.gate_evidence import (
    evaluate_evidence_based_gates,
    verify_architecture_evidence,
    verify_distribution_evidence,
    verify_evaluation_evidence,
    verify_integration_evidence,
    verify_training_pipeline_evidence,
)


def test_gate_a_dynamically_validates_architecture_and_rejects_yaml_drift(tmp_path):
    ok = verify_architecture_evidence()
    assert ok["passed"] is True
    assert ok["exact_parameters"] == 303_350_784
    assert ok["yaml_checked"] is True

    # Tampered YAML with wrong d_model MUST dynamically fail Gate A (proves not hardcoded True)
    bad_yaml = tmp_path / "bad_arch.yaml"
    bad_yaml.write_text(
        yaml.safe_dump({"model": {"vocab_size": 32000, "d_model": 512, "n_layers": 12, "n_heads": 8, "n_kv_heads": 4, "d_ff": 1408, "max_seq_len": 2048, "tie_embeddings": True}}),
        encoding="utf-8",
    )
    bad = verify_architecture_evidence(config_path=str(bad_yaml))
    assert bad["passed"] is False
    assert any("YAML_MODEL_INVALID" in i or "YAML_MODEL_PARAM_MISMATCH" in i for i in bad["issues"])


def test_gate_b_dynamically_validates_pipeline_and_rejects_broken_token_budget(tmp_path):
    ok = verify_training_pipeline_evidence()
    assert ok["passed"] is True
    assert ok["checks"]["yaml_token_budget_consistent"] is True
    assert ok["checks"]["trainer_exact_resume_guards"] is True

    # Inconsistent token budget in YAML MUST dynamically fail Gate B (proves not hardcoded True)
    bad_yaml = tmp_path / "bad_budget.yaml"
    bad_yaml.write_text(
        yaml.safe_dump({
            "model": {"max_seq_len": 2048},
            "training": {
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 16,
                "tokens_per_optimizer_step": 99999,  # wrong!
                "max_steps": 50000,
                "target_tokens": 3_276_800_000,
            },
            "sft": {"lr": 5e-5},
        }),
        encoding="utf-8",
    )
    bad = verify_training_pipeline_evidence(config_path=str(bad_yaml))
    assert bad["passed"] is False
    assert any("TOKEN_BUDGET_INCONSISTENT" in i for i in bad["issues"])


def test_gate_e_evaluation_requires_verified_checkpoint_hash_and_clean_leakage(tmp_path):
    pre_ok = {"passed": True, "checkpoint_sha256": "abc123verified"}
    sft_no = {"passed": False, "checkpoint_sha256": None}

    eval_dir = tmp_path / "bumblecore-300m"
    eval_dir.mkdir(parents=True)
    report_path = eval_dir / "evaluation_report.json"

    # 1. Valid report linked to verified checkpoint passes Gate E
    valid_report = {
        "status": "COMPLETED",
        "actual_sample_count": 15,
        "overall_score": 0.82,
        "benchmark_dataset_sha256": "deadbeef1234",
        "training_leakage_check": {"clean": True, "leaked_count": 0},
        "levels": {"safety": {"safety_pass_rate": 1.0}},
        "checkpoint_sha256": "abc123verified",
    }
    report_path.write_text(json.dumps(valid_report), encoding="utf-8")
    ev_pass = verify_evaluation_evidence(str(tmp_path), pre_ev=pre_ok, sft_ev=sft_no)
    assert ev_pass["passed"] is True, ev_pass["reasons"]

    # 2. Mismatched checkpoint_sha256 MUST fail Gate E
    bad_hash_rep = dict(valid_report, checkpoint_sha256="different_smoke_hash")
    report_path.write_text(json.dumps(bad_hash_rep), encoding="utf-8")
    ev_mismatch = verify_evaluation_evidence(str(tmp_path), pre_ev=pre_ok, sft_ev=sft_no)
    assert ev_mismatch["passed"] is False
    assert "EVALUATION_CHECKPOINT_HASH_MISMATCH" in ev_mismatch["reasons"]

    # 3. Leakage detected MUST fail Gate E
    leaked_rep = dict(valid_report, training_leakage_check={"clean": False, "leaked_count": 2})
    report_path.write_text(json.dumps(leaked_rep), encoding="utf-8")
    ev_leak = verify_evaluation_evidence(str(tmp_path), pre_ev=pre_ok, sft_ev=sft_no)
    assert ev_leak["passed"] is False
    assert "BENCHMARK_LEAKAGE_CHECK_FAILED_OR_MISSING" in ev_leak["reasons"]

    # 4. Without verified 300M checkpoint, even valid report fails Gate E
    report_path.write_text(json.dumps(valid_report), encoding="utf-8")
    ev_no_ckpt = verify_evaluation_evidence(str(tmp_path), pre_ev={"passed": False}, sft_ev={"passed": False})
    assert ev_no_ckpt["passed"] is False
    assert "NO_VERIFIED_300M_CHECKPOINT_FOR_EVALUATION" in ev_no_ckpt["reasons"]


def test_gate_f_distribution_verifies_manifest_hashes_and_backing_checkpoint(tmp_path):
    exp_dir = tmp_path / "bumblecore-300m-export"
    exp_dir.mkdir()
    for fn, body in (
        ("config.json", "{}"),
        ("tokenizer.json", "{}"),
        ("README.md", "# BumbleCore"),
        ("model.pt", "weights-placeholder"),
    ):
        (exp_dir / fn).write_text(body, encoding="utf-8")

    manifest = {
        "excludes_optimizer_state": True,
        "exact_parameters": 303_350_784,
        "files_sha256": {
            fn: sha256_file(str(exp_dir / fn))
            for fn in ("config.json", "tokenizer.json", "README.md", "model.pt")
        },
    }
    (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pre_ok = {"passed": True, "checkpoint_sha256": "abc123"}
    sft_no = {"passed": False, "checkpoint_sha256": None}

    # Valid export + verified checkpoint passes Gate F
    res_ok = verify_distribution_evidence(str(exp_dir), pre_ev=pre_ok, sft_ev=sft_no)
    assert res_ok["passed"] is True, res_ok["reasons"]

    # Without verified 300M checkpoint, Gate F MUST fail
    res_no_ckpt = verify_distribution_evidence(str(exp_dir), pre_ev={"passed": False}, sft_ev={"passed": False})
    assert res_no_ckpt["passed"] is False
    assert "NO_VERIFIED_300M_CHECKPOINT_FOR_EXPORT" in res_no_ckpt["reasons"]

    # Tampering exported README.md triggers EXPORT_FILE_HASH_MISMATCH
    (exp_dir / "README.md").write_text("# Tampered card", encoding="utf-8")
    res_tamper = verify_distribution_evidence(str(exp_dir), pre_ev=pre_ok, sft_ev=sft_no)
    assert res_tamper["passed"] is False
    assert any("EXPORT_FILE_HASH_MISMATCH" in r for r in res_tamper["reasons"])


def test_gate_g_integration_verifies_platform_evidence_and_checkpoint_link(tmp_path):
    pre_ok = {"passed": True, "checkpoint_sha256": "abc123verified"}
    sft_no = {"passed": False, "checkpoint_sha256": None}

    ev_dir = tmp_path / "bumblecore-300m"
    ev_dir.mkdir(parents=True)
    ev_path = ev_dir / "integration_evidence.json"

    # Missing file -> fails
    res_missing = verify_integration_evidence(str(tmp_path), pre_ev=pre_ok, sft_ev=sft_no)
    assert res_missing["passed"] is False
    assert "INTEGRATION_EVIDENCE_MISSING" in res_missing["reasons"]

    # Valid integration evidence -> passes
    valid_ev = {
        "checkpoint_sha256": "abc123verified",
        "model_registry_registered": True,
        "runtime_inference_verified": True,
        "security_checks_passed": True,
        "knowledge_memory_isolation_verified": True,
    }
    ev_path.write_text(json.dumps(valid_ev), encoding="utf-8")
    res_ok = verify_integration_evidence(str(tmp_path), pre_ev=pre_ok, sft_ev=sft_no)
    assert res_ok["passed"] is True, res_ok["reasons"]

    # Wrong checkpoint hash -> fails
    ev_path.write_text(json.dumps(dict(valid_ev, checkpoint_sha256="stale_hash")), encoding="utf-8")
    res_stale = verify_integration_evidence(str(tmp_path), pre_ev=pre_ok, sft_ev=sft_no)
    assert res_stale["passed"] is False
    assert "INTEGRATION_CHECKPOINT_HASH_MISMATCH" in res_stale["reasons"]


def test_evaluate_evidence_based_gates_default_empty_models_dir_blocks_at_gate_c(tmp_path):
    rep = evaluate_evidence_based_gates(str(tmp_path))
    assert rep["release"] == "2.6"
    assert rep["pretraining_status"] == "NOT_STARTED"
    assert rep["lifecycle_stage"] == "CONFIGURED"
    assert rep["release_gates"]["blocked_at_gate"] == "GATE_C_PRETRAINING_CHECKPOINT_VALIDATED"
    gates_by_name = {g["gate"]: g for g in rep["release_gates"]["gates"]}
    assert gates_by_name["GATE_A_ARCHITECTURE_VALIDATED"]["passed"] is True
    assert gates_by_name["GATE_B_TRAINING_PIPELINE_VALIDATED"]["passed"] is True
    assert gates_by_name["GATE_C_PRETRAINING_CHECKPOINT_VALIDATED"]["passed"] is False
    assert gates_by_name["GATE_E_EVALUATION_PASSED"]["passed"] is False
    assert gates_by_name["GATE_F_DISTRIBUTION_VALIDATED"]["passed"] is False
    assert gates_by_name["GATE_G_GAJAHMADA_INTEGRATION_VALIDATED"]["passed"] is False
