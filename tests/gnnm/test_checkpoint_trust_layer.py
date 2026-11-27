"""
Release 2.6 Batch 1 — Checkpoint Trust Layer Acceptance Tests (All 10 Required Cases).

1. Valid checkpoint accepted (with Ed25519 signature verification)
2. Fake text file rejected
3. Empty checkpoint file / empty state_dict rejected
4. Checkpoint with wrong tensor shape rejected
5. Checkpoint with parameter count mismatch rejected
6. Checkpoint with NaN / Inf tensor values rejected
7. Forged metadata hash rejected (checkpoint / config / tokenizer / dataset)
8. Smoke checkpoint rejected for pretraining gate
9. SFT checkpoint rejected for pretraining gate
10. Checkpoint modified after manifest creation/signing rejected
"""
from __future__ import annotations

import json
import os
try:
    import pytest
except ImportError:  # pragma: no cover - allows standalone execution in .venv312 without pytest
    pytest = None

from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_smoke_config
from gajahmada_ai.gnnm.bumblecore.checkpoint_trust import (
    canonical_json_sha256,
    compute_tokenizer_sha256,
    sign_training_manifest,
    verify_checkpoint_structure,
    verify_trusted_checkpoint,
)
from gajahmada_ai.gnnm.bumblecore.dataset_governance import sha256_file
from gajahmada_ai.gnnm.model import GNNMTransformer, HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.security.crypto import NodeIdentity


def _make_valid_checkpoint(tmp_path, stage: str = "pretraining", smoke_mode: bool = False):
    """Build a real PyTorch checkpoint with authentic provenance & Ed25519 signature."""
    tok = GNNMBPETokenizer()
    cfg = build_bumblecore_smoke_config(vocab_size=tok.vocab_size, max_seq_len=32)
    model = GNNMTransformer(cfg)

    ckpt_dir = tmp_path / f"ckpt_{stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(str(ckpt_dir / "config.json"))
    tok.save(str(ckpt_dir / "tokenizer.json"))
    torch.save(model.state_dict(), str(ckpt_dir / "model.pt"))

    manifest = {
        "dataset_id": "test-approved",
        "license": "MIT",
        "approval_status": "approved",
        "document_count": 100,
    }
    manifest_path = ckpt_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    meta = {
        "release": "bumblecore-2.6",
        "run_id": "bc-test-run-001",
        "git_commit": "674794f",
        "stage": stage,
        "smoke_mode": smoke_mode,
        "config_sha256": canonical_json_sha256(cfg.to_dict()),
        "tokenizer_sha256": compute_tokenizer_sha256(tok),
        "dataset_manifest_sha256": sha256_file(str(manifest_path)),
        "checkpoint_sha256": sha256_file(str(ckpt_dir / "model.pt")),
        "exact_trainable_parameters": cfg.estimate_parameters(),
        "actual_steps": 10,
        "actual_tokens_processed": 1024,
        "validation_metrics": {"status": "COMPUTED", "val_loss": 4.2, "perplexity": 66.7},
        "hardware": {"cuda_available": False},
        "started_at": "2026-11-28T00:00:00+00:00",
        "ended_at": "2026-11-28T00:01:00+00:00",
    }
    identity = NodeIdentity(key_path=str(tmp_path / "test_node_key.pem"))
    meta = sign_training_manifest(meta, identity)
    (ckpt_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ckpt_dir, cfg, meta, identity


def test_fake_text_file_and_empty_file_rejected_even_without_torch(tmp_path):
    # Case 2: Fake text file rejected
    d_fake = tmp_path / "fake_txt"
    d_fake.mkdir()
    (d_fake / "model.pt").write_text("this is a fake checkpoint text file", encoding="utf-8")
    r_fake = verify_checkpoint_structure(str(d_fake))
    assert r_fake["valid"] is False
    assert any("INVALID_PYTORCH_CHECKPOINT" in r for r in r_fake["reasons"])

    # Case 3: Empty 0-byte checkpoint file rejected
    d_empty = tmp_path / "empty_file"
    d_empty.mkdir()
    (d_empty / "model.pt").write_bytes(b"")
    r_empty = verify_checkpoint_structure(str(d_empty))
    assert r_empty["valid"] is False
    assert any("EMPTY_CHECKPOINT_FILE" in r for r in r_empty["reasons"])


def _skipif_no_torch(fn):
    return pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for tensor-level acceptance tests")(fn) if pytest else fn


@_skipif_no_torch
def test_all_10_checkpoint_trust_cases_with_pytorch(tmp_path):
    # 1. Valid checkpoint accepted (including Ed25519 signature check)
    ckpt_dir, cfg, meta, _ = _make_valid_checkpoint(tmp_path, stage="pretraining", smoke_mode=False)
    res_ok = verify_trusted_checkpoint(
        str(ckpt_dir),
        expected_stage="pretraining",
        expected_min_params=cfg.estimate_parameters(),
        require_signature=True,
    )
    assert res_ok["passed"] is True, res_ok["reasons"]
    assert res_ok["signature_status"] == "ED25519_VERIFIED"

    # 3b. Empty state_dict rejected
    d_esd = tmp_path / "empty_sd"
    d_esd.mkdir()
    cfg.save(str(d_esd / "config.json"))
    torch.save({}, str(d_esd / "model.pt"))
    assert any("EMPTY_STATE_DICT" in r for r in verify_checkpoint_structure(str(d_esd))["reasons"])

    # 4. Wrong tensor shape rejected
    sd_bad_shape = torch.load(str(ckpt_dir / "model.pt"), map_location="cpu", weights_only=True)
    sd_bad_shape["final_norm.weight"] = torch.ones(cfg.d_model + 4)
    d_shape = tmp_path / "bad_shape"
    d_shape.mkdir()
    cfg.save(str(d_shape / "config.json"))
    torch.save(sd_bad_shape, str(d_shape / "model.pt"))
    assert any("TENSOR_SHAPE_MISMATCH" in r for r in verify_checkpoint_structure(str(d_shape))["reasons"])

    # 5. Parameter count below target rejected
    res_p = verify_trusted_checkpoint(str(ckpt_dir), expected_stage="pretraining", expected_min_params=285_000_000)
    assert res_p["passed"] is False
    assert any("PARAMETER_COUNT_BELOW_TARGET" in r for r in res_p["reasons"])

    # 6. NaN / Inf tensor rejected
    sd_nan = torch.load(str(ckpt_dir / "model.pt"), map_location="cpu", weights_only=True)
    sd_nan["final_norm.weight"] = torch.full_like(sd_nan["final_norm.weight"], float("nan"))
    d_nan = tmp_path / "nan_ckpt"
    d_nan.mkdir()
    cfg.save(str(d_nan / "config.json"))
    torch.save(sd_nan, str(d_nan / "model.pt"))
    assert any("NON_FINITE_TENSOR_VALUES" in r for r in verify_checkpoint_structure(str(d_nan))["reasons"])

    # 7. Forged metadata hash rejected (config_sha256 tampered)
    bad_meta = dict(meta, config_sha256="0" * 64)
    (ckpt_dir / "run_metadata.json").write_text(json.dumps(bad_meta), encoding="utf-8")
    res_hash = verify_trusted_checkpoint(str(ckpt_dir), expected_stage="pretraining", expected_min_params=1)
    assert res_hash["passed"] is False
    assert any("CONFIG_HASH_MISMATCH" in r for r in res_hash["reasons"])
    assert any("INVALID_MANIFEST_SIGNATURE" in r for r in res_hash["reasons"])
    (ckpt_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # 8. Smoke checkpoint rejected for pretraining gate
    smoke_dir, s_cfg, _, _ = _make_valid_checkpoint(tmp_path / "smoke_case", stage="pretraining", smoke_mode=True)
    res_smoke = verify_trusted_checkpoint(str(smoke_dir), expected_stage="pretraining", expected_min_params=1)
    assert res_smoke["passed"] is False
    assert any("SMOKE_CHECKPOINT_REJECTED" in r for r in res_smoke["reasons"])

    # 9. SFT checkpoint rejected for pretraining gate
    sft_dir, _, _, _ = _make_valid_checkpoint(tmp_path / "sft_case", stage="sft", smoke_mode=False)
    res_sft = verify_trusted_checkpoint(str(sft_dir), expected_stage="pretraining", expected_min_params=1)
    assert res_sft["passed"] is False
    assert any("WRONG_STAGE" in r for r in res_sft["reasons"])

    # 10. Checkpoint modified after manifest creation rejected
    sd_mod = torch.load(str(ckpt_dir / "model.pt"), map_location="cpu", weights_only=True)
    sd_mod["final_norm.weight"] = sd_mod["final_norm.weight"] + 0.01
    torch.save(sd_mod, str(ckpt_dir / "model.pt"))
    res_mod = verify_trusted_checkpoint(str(ckpt_dir), expected_stage="pretraining", expected_min_params=1)
    assert res_mod["passed"] is False
    assert any("CHECKPOINT_HASH_MISMATCH" in r for r in res_mod["reasons"])
