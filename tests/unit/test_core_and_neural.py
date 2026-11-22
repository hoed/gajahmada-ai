from datetime import datetime
import os
import tempfile
import pytest

from gajahmada_ai.blockchain.chain import AIBlockchain, LedgerEventType
from gajahmada_ai.core.engine import AIModelEngine
from gajahmada_ai.learning.experiments import LearningExperimentManager
from gajahmada_ai.neural.llama_backend import LlamaCppBackend
from gajahmada_ai.neural.pytorch_backend import PyTorchNativeBackend
from gajahmada_ai.neural.router import ModelRouter
from gajahmada_ai.security.crypto import NodeIdentity


def test_1_starts_without_external_ai_credentials(monkeypatch):
    """TEST 1: Gajahmada can start without external AI provider credentials."""
    for env_key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(env_key, raising=False)

    engine = AIModelEngine(model_path="./models/nonexistent.gguf")
    cap = engine.capability_status()
    assert cap["external_ai_providers_enabled"] is False
    assert cap["symbolic_reasoning_active"] is True
    assert cap["evidence_reasoning_active"] is True


def test_2_local_neural_backend_loaded_and_used_when_configured():
    """TEST 2: A local neural backend can be loaded and used when configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, "test_local.gguf")
        with open(gguf_path, "wb") as f:
            f.write(b"GGUF-GM-LOCAL-TEST-WEIGHTS")

        router = ModelRouter(gguf_model_path=gguf_path)
        loaded = router.load_available_models()
        assert loaded["llama.cpp"] is True

        cap = router.capability_report()
        assert cap["neural_inference_active"] is True
        assert cap["active_neural_backend"] == "llama.cpp"

        res = router.route_generation("Analyze Nusantara maritime sovereignty")
        assert res.neural_used is True
        assert "GGUF Local Neural Output" in res.text


def test_3_explicitly_reports_when_neural_inference_unavailable():
    """TEST 3: The system explicitly reports when neural inference is unavailable."""
    router = ModelRouter(
        gguf_model_path="./models/missing_model.gguf",
        onnx_model_path="./models/missing_model.onnx",
        native_model_path="./models/missing_native.json",
    )
    router.load_available_models()
    cap = router.capability_report()
    assert cap["neural_inference_active"] is False
    assert cap["active_neural_backend"] is None

    gen = router.route_generation("Hello")
    assert gen.neural_used is False
    assert gen.warning is not None


def test_14_failed_learning_experiment_rejected_without_affecting_active_model():
    """TEST 14: A failed learning experiment can be rejected without affecting the active model."""
    manager = LearningExperimentManager()
    initial_version = manager.adapter.active_strategy["version"]
    initial_model_ver = manager.adapter.active_strategy["active_model_version"]

    # Simulate a broken candidate model/strategy that fails regression tests
    failed_exp = manager.propose_and_evaluate_experiment(
        name="broken_weight_update",
        level=3,
        proposed_changes={"active_model_version": "2.0.0-broken"},
        candidate_fn=lambda _prompt: "unverified gibberish",
        policy_approved=True,
    )
    assert failed_exp["status"] == "REJECTED_REGRESSION_FAILURE"
    assert failed_exp["active_model_affected"] is False
    assert manager.adapter.active_strategy["version"] == initial_version
    assert manager.adapter.active_strategy["active_model_version"] == initial_model_ver


def test_signed_blockchain_event_ledger():
    """Verify typed events and Ed25519 signature verification on AIBlockchain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_path = os.path.join(tmpdir, "key.pem")
        chain_path = os.path.join(tmpdir, "chain.json")
        ident = NodeIdentity(key_path=key_path)
        chain = AIBlockchain(storage_path=chain_path)

        blk = chain.record_event(
            LedgerEventType.KNOWLEDGE_ACCEPTED,
            "mem-001|hash=abc123",
            ident,
        )
        assert blk.index == 1
        assert blk.event_type == "KNOWLEDGE_ACCEPTED"
        assert chain.is_chain_valid(verify_signatures=True) is True

        # Tamper with signature and confirm is_chain_valid detects forgery
        orig_sig = chain.chain[1].signature
        chain.chain[1].signature = "AAAA" + orig_sig[4:]
        assert chain.is_chain_valid(verify_signatures=True) is False
