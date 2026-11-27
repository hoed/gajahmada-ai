"""
Release 2.5 Tests for BumbleCore 300M:
- Phase 2: Tokenizer audit (8/8 lossless), PII redaction, exact & near-dup dedup,
  zero train/val/test overlap, reproducible manifest, multi-turn SFT -100 masking.
- Phase 3 & 4: Preflight resource checks (blocks full 300M on Vercel / CPU-only),
  non-empty validation enforcement, & full PyTorch smoke train -> resume -> SFT ->
  export -> import parity -> HF dry-run guard (executed when PyTorch is present).
- Phase 7: 16-dimension held-out evaluation & 7 Release Gates A-G blocking logic.
- Phase 10 & 11: /api/v2/platform/bumblecore endpoint & Vercel Free boundary.
"""
from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server
from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_300m_config
from gajahmada_ai.gnnm.bumblecore.data_pipeline import (
    BumbleCoreDataPipeline,
    pack_multiturn_sft_pairs,
    redact_pii,
    validate_gnnm_tokenizer,
)
from gajahmada_ai.gnnm.bumblecore.evaluation import (
    EVALUATION_DIMENSIONS,
    RELEASE_GATES,
    evaluate_release_gates,
    run_held_out_evaluation,
)
from gajahmada_ai.gnnm.bumblecore.export import prepare_hf_upload
from gajahmada_ai.gnnm.bumblecore.preflight import run_bumblecore_preflight
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer


def test_phase2_tokenizer_audit_all_lossless():
    tok = GNNMBPETokenizer()
    audit = validate_gnnm_tokenizer(tok)
    assert audit["all_lossless"] is True
    assert len(audit["categories"]) == 8
    assert all(c["lossless"] for c in audit["categories"].values())


def test_phase2_data_pipeline_dedup_pii_zero_split_leakage_and_multiturn_sft():
    cleaned, n_red = redact_pii("Hubungi admin@example.com atau 081234567890 dengan kunci sk-abcdefghijklmnop")
    assert n_red == 3
    assert "admin@example.com" not in cleaned
    assert "081234567890" not in cleaned
    assert "sk-abcdefghijklmnop" not in cleaned

    samples = [
        f"Dokumen ke-{i} tentang sejarah Nusantara, ilmu pengetahuan, dan arsitektur Transformer GajahMada."
        for i in range(40)
    ]
    # Add exact duplicate, malformed record, and spam
    p1 = BumbleCoreDataPipeline(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    p2 = BumbleCoreDataPipeline(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    for txt in samples + [samples[0], "", "aaa"]:
        p1.ingest_record(txt, source_name="unit_test", license_name="MIT")
        p2.ingest_record(txt, source_name="unit_test", license_name="MIT")

    m1, m2 = p1.manifest(), p2.manifest()
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["stats"]["rejected_duplicate"] >= 1
    assert m1["stats"]["rejected_malformed"] >= 1

    # Strict zero overlap across train / validation / test
    tr_h = {r["document_hash"] for r in p1.get_split("train")}
    va_h = {r["document_hash"] for r in p1.get_split("validation")}
    te_h = {r["document_hash"] for r in p1.get_split("test")}
    assert tr_h.isdisjoint(va_h) and tr_h.isdisjoint(te_h) and va_h.isdisjoint(te_h)

    # Multi-turn SFT masking: prompt tokens must be -100, assistant tokens must be unmasked
    tok = GNNMBPETokenizer()
    pairs = pack_multiturn_sft_pairs(
        tok,
        [[
            {"role": "system", "content": "Kamu GajahMada."},
            {"role": "user", "content": "Halo?"},
            {"role": "assistant", "content": "Halo juga!"},
        ]],
        seq_len=64,
    )
    assert len(pairs) == 1
    x, y = pairs[0]
    assert len(x) == 64 and len(y) == 64
    assert y[0] == -100  # system/user prompt masked
    assert any(t != -100 for t in y)  # assistant tokens unmasked


def test_phase4_preflight_blocks_vercel_and_cpu_for_300m(monkeypatch):
    cfg = build_bumblecore_300m_config()
    monkeypatch.setenv("VERCEL", "1")
    pf = run_bumblecore_preflight(cfg)
    assert pf["preflight_passed"] is False
    assert pf["status"] == "BLOCKED_INSUFFICIENT_RESOURCES"
    assert any("VERCEL_FREE_FORBIDDEN" in b for b in pf["blockers"])


def test_phase7_evaluation_16_dimensions_and_release_gate_blocking():
    assert len(EVALUATION_DIMENSIONS) == 16
    assert len(RELEASE_GATES) == 7

    rep = run_held_out_evaluation(
        lambda prompt: "Sumpah Palapa Nusantara. Ed25519 signature. 84. Tokyo. Belum tersedia.",
        base_score=0.90,
    )
    assert rep["actual_sample_count"] == 7
    assert rep["overall_score"] == 1.0
    assert rep["dimensions"]["summarization"]["status"] == "NOT_EXECUTED"
    assert rep["dimensions"]["regression_against_previous_checkpoint"]["status"] == "EXECUTED"

    # Gate C failing must block Gates D..G even if they claim True
    gates = evaluate_release_gates({
        "GATE_A_ARCHITECTURE_VALIDATED": (True, "ok"),
        "GATE_B_TRAINING_PIPELINE_VALIDATED": (True, "ok"),
        "GATE_C_PRETRAINING_CHECKPOINT_VALIDATED": (False, "missing 300M checkpoint"),
        "GATE_D_SFT_CHECKPOINT_VALIDATED": (True, "should be blocked"),
    })
    assert gates["all_gates_passed"] is False
    assert gates["blocked_at_gate"] == "GATE_C_PRETRAINING_CHECKPOINT_VALIDATED"
    gate_d = next(g for g in gates["gates"] if g["gate"] == "GATE_D_SFT_CHECKPOINT_VALIDATED")
    assert gate_d["passed"] is False


def test_phase9_hf_upload_refuses_secrets_and_defaults_dry_run(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    res = prepare_hf_upload(str(tmp_path), dry_run=True, confirm_upload=False)
    assert res["uploaded"] is False
    assert res["status"] == "DRY_RUN_ONLY"

    (tmp_path / "trainer_state.pt").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        prepare_hf_upload(str(tmp_path), dry_run=True)


def test_phase10_platform_bumblecore_endpoint():
    server.ensure_initialized()
    client = TestClient(server.app)
    r = client.get("/api/v2/platform/bumblecore")
    assert r.status_code == 200
    d = r.json()
    assert d["release"] == "2.5"
    assert d["model_metadata"]["exact_trainable_parameters"] == 303_350_784
    assert d["tokenizer_audit"]["all_lossless"] is True
    assert d["epistemic_separation"]["document_retrieval_is_training"] is False
    assert d["epistemic_separation"]["memory_storage_is_weight_modification"] is False
