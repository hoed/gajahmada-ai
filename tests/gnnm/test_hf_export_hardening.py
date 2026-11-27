"""
Batch 8 Acceptance Tests — Hugging Face Export Hardening.

Verifies:
1. Model card validation (frontmatter, license, required sections, rejects false transformers/pretrained claims).
2. Untrained / random-init / smoke checkpoint labeling protection & `require_trained=True` guard.
3. Config <-> Tokenizer <-> generation_config consistency verification.
4. Recursive export bundle validation & live upload rejection for untrained/tampered bundles.
5. PyTorch export/import parity (shapes, dtypes, numerical values, deterministic logits, Safetensors if installed).
"""
from __future__ import annotations

import json
import os
import tempfile

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_smoke_config
from gajahmada_ai.gnnm.bumblecore.export import (
    classify_training_status,
    export_inference_artifact,
    generate_hf_model_card,
    prepare_hf_upload,
    validate_export_bundle,
    validate_hf_model_card,
    verify_export_import_parity,
)
from gajahmada_ai.gnnm.model import HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer


def test_model_card_validation_and_honest_untrained_labeling():
    cfg = build_bumblecore_smoke_config(vocab_size=260)

    # Random-init / untrained -> MUST NOT claim pretrained
    assert classify_training_status(None) == "UNTRAINED_OR_RANDOM_INIT"
    assert classify_training_status({"actual_steps": 0, "actual_tokens_processed": 0}) == "UNTRAINED_OR_RANDOM_INIT"
    assert classify_training_status({"smoke_mode": True, "actual_steps": 5, "actual_tokens_processed": 100}) == "SMOKE_TEST_ONLY"
    assert classify_training_status({"smoke_mode": False, "actual_steps": 100, "actual_tokens_processed": 50000}) == "TRAINED_WITH_PROVENANCE"

    card_untrained = generate_hf_model_card(cfg, training_metadata=None)
    assert "UNTRAINED_OR_RANDOM_INIT" in card_untrained
    assert "NOT pretrained" in card_untrained
    res_ok = validate_hf_model_card(card_untrained, expected_license="mit", expected_training_status="UNTRAINED_OR_RANDOM_INIT")
    assert res_ok["valid"] is True, res_ok["issues"]

    # Wrong license MUST fail
    res_bad_lic = validate_hf_model_card(card_untrained, expected_license="apache-2.0")
    assert res_bad_lic["valid"] is False
    assert any("MODEL_CARD_LICENSE_MISMATCH" in i for i in res_bad_lic["issues"])

    # Forged pretrained claim on an untrained bundle MUST fail
    forged_card = card_untrained.replace("UNTRAINED_OR_RANDOM_INIT", "TRAINED_WITH_PROVENANCE")
    res_forged = validate_hf_model_card(forged_card, expected_training_status="UNTRAINED_OR_RANDOM_INIT")
    assert res_forged["valid"] is False
    assert "MODEL_CARD_FALSE_PRETRAINED_CLAIM" in res_forged["issues"]


def test_validate_export_bundle_catches_tampering_and_vocab_mismatch(tmp_path):
    tok = GNNMBPETokenizer()
    cfg = build_bumblecore_smoke_config(vocab_size=tok.vocab_size)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    cfg.save(str(exp_dir / "config.json"))
    tok.save(str(exp_dir / "tokenizer.json"))
    (exp_dir / "generation_config.json").write_text(
        json.dumps({"bos_token_id": tok.bos_id, "eos_token_id": tok.eos_id, "pad_token_id": tok.pad_id}),
        encoding="utf-8",
    )
    (exp_dir / "model.pt").write_bytes(b"placeholder")
    (exp_dir / "README.md").write_text(generate_hf_model_card(cfg, training_metadata=None), encoding="utf-8")

    from gajahmada_ai.gnnm.bumblecore.dataset_governance import sha256_file
    manifest = {
        "excludes_optimizer_state": True,
        "training_status": "UNTRAINED_OR_RANDOM_INIT",
        "files_sha256": {
            fn: sha256_file(str(exp_dir / fn))
            for fn in ("config.json", "tokenizer.json", "generation_config.json", "model.pt", "README.md")
        },
    }
    (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ok = validate_export_bundle(str(exp_dir))
    assert ok["valid"] is True, ok["issues"]

    # Live upload MUST refuse untrained bundle even with confirm_upload=True & dry_run=False
    with pytest.raises(ValueError, match="untrained/smoke"):
        prepare_hf_upload(str(exp_dir), dry_run=False, confirm_upload=True)

    # Tampering config.json vocab_size MUST trigger both hash mismatch & vocab mismatch
    cfg_bad = build_bumblecore_smoke_config(vocab_size=tok.vocab_size + 10)
    cfg_bad.save(str(exp_dir / "config.json"))
    bad = validate_export_bundle(str(exp_dir))
    assert bad["valid"] is False
    assert any("CONFIG_TOKENIZER_VOCAB_MISMATCH" in i for i in bad["issues"])
    assert any("MANIFEST_FILE_HASH_MISMATCH" in i for i in bad["issues"])


def _run_pytorch_export_parity_tests():
    from gajahmada_ai.gnnm.bumblecore.architecture import BumbleCoreModel
    assert HAS_TORCH and torch is not None

    with tempfile.TemporaryDirectory() as td:
        src_dir = os.path.join(td, "src")
        exp_dir = os.path.join(td, "exp")
        os.makedirs(src_dir)

        tok = GNNMBPETokenizer()
        cfg = build_bumblecore_smoke_config(vocab_size=tok.vocab_size)
        model = BumbleCoreModel(cfg)
        cfg.save(os.path.join(src_dir, "config.json"))
        tok.save(os.path.join(src_dir, "tokenizer.json"))
        torch.save(model.state_dict(), os.path.join(src_dir, "model.pt"))

        # 1. require_trained=True MUST reject untrained checkpoint
        rejected = False
        try:
            export_inference_artifact(src_dir, exp_dir, require_trained=True)
        except ValueError as exc:
            rejected = "REFUSING_EXPORT_UNTRAINED_OR_SMOKE_CHECKPOINT" in str(exc)
        assert rejected, "Expected require_trained=True to reject untrained checkpoint"

        # 2. Export with provenance metadata succeeds & passes full parity + bundle validation
        meta = {"smoke_mode": False, "actual_steps": 10, "actual_tokens_processed": 2048}
        manifest = export_inference_artifact(src_dir, exp_dir, training_metadata=meta, require_trained=True)
        assert manifest["training_status"] == "TRAINED_WITH_PROVENANCE"
        assert manifest["config_tokenizer_consistent"] is True

        parity = verify_export_import_parity(src_dir, exp_dir)
        assert parity["parity_verified"] is True, parity
        assert parity["shapes_match"] is True
        assert parity["dtypes_match"] is True
        assert parity["values_match"] is True
        assert parity["deterministic_outputs_match"] is True
        assert parity["max_logit_diff"] <= 1e-6

        bundle = validate_export_bundle(exp_dir)
        assert bundle["valid"] is True, bundle["issues"]

        # 3. NaN weight in source checkpoint MUST be rejected before export
        bad_sd = model.state_dict()
        first_key = next(iter(bad_sd.keys()))
        bad_sd[first_key] = bad_sd[first_key].clone()
        bad_sd[first_key].view(-1)[0] = float("nan")
        torch.save(bad_sd, os.path.join(src_dir, "model.pt"))
        nan_rejected = False
        try:
            export_inference_artifact(src_dir, os.path.join(td, "exp_nan"), training_metadata=meta)
        except ValueError as exc:
            nan_rejected = "NON_FINITE_TENSOR_IN_EXPORT_SOURCE" in str(exc)
        assert nan_rejected, "Expected NaN weights to be rejected during export"

    print("ALL_PYTORCH_HF_EXPORT_HARDENING_TESTS_PASSED")


if pytest is not None:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for export parity test")
    def test_pytorch_export_parity_and_guards():
        _run_pytorch_export_parity_tests()


if __name__ == "__main__":
    _run_pytorch_export_parity_tests()
