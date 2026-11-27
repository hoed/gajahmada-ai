"""
Comprehensive tests for GajahMada Platform Control Plane, GNNM Model Family,
TITANFORGE 70B architecture & analytical parameter counting, Resource Estimator,
Distributed Training Preparation, FAISS CPU preservation & fallback providers,
Unified Model Registry, Nexus Prime, and all Phase 10 & Phase 14 routes.
"""
from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server
from gajahmada_ai.gnnm.config import GNNM_PRESETS, GNNMConfig
from gajahmada_ai.gnnm.distributed_prep import (
    DistributedTrainingConfig,
    inspect_distributed_runtime_readiness,
)
from gajahmada_ai.gnnm.resource_estimator import estimate_model_resources
from gajahmada_ai.knowledge.providers import (
    FaissArtifactProvider,
    FaissLocalProvider,
    LightweightRetrievalProvider,
    ProviderStatus,
    RetrievalProviderRegistry,
)
from gajahmada_ai.platform.model_factory import LIFECYCLE_STATES, build_unified_model_registry


@pytest.fixture(scope="module")
def client():
    server.ensure_initialized()
    return TestClient(server.app)


def test_titanforge_70b_dense_and_moe_analytical_parameter_accounting():
    tf70 = GNNM_PRESETS["TITANFORGE-70B"]
    params = tf70.estimate_parameters()
    # Standard 80L / d=8192 / GQA 64:8 / SwiGLU 28672 / vocab 128k untied ~= 70.55B
    assert 69_000_000_000 <= params <= 72_000_000_000
    assert tf70.estimate_active_parameters() == params
    assert tf70.architecture_type == "dense"
    assert tf70.attention_type == "gqa"
    assert tf70.gradient_checkpointing is True

    tf_moe = GNNM_PRESETS["TITANFORGE-70B-MoE-Exp"]
    assert tf_moe.architecture_type == "moe"
    assert tf_moe.estimate_parameters() > tf_moe.estimate_active_parameters()
    assert tf_moe.parameter_breakdown()["router_params"] > 0

    with pytest.raises(ValueError):
        GNNMConfig(d_model=100, n_heads=6, n_kv_heads=2)
    with pytest.raises(ValueError):
        GNNMConfig(architecture_type="moe", num_experts=1)


def test_resource_estimator_supports_all_precisions_without_gpu():
    tf70 = GNNM_PRESETS["TITANFORGE-70B"]
    est = estimate_model_resources(tf70)
    assert est["estimate_kind"] == "THEORETICAL_ANALYTICAL_ESTIMATE"
    assert est["measured_benchmark"] is False
    assert set(est["inference_memory_by_precision"].keys()) == {"fp32", "bf16", "fp16", "int8", "int4"}
    assert est["inference_memory_by_precision"]["fp32"]["weights_gb"] > est["inference_memory_by_precision"]["bf16"]["weights_gb"]
    assert est["inference_memory_by_precision"]["bf16"]["weights_gb"] > est["inference_memory_by_precision"]["int4"]["weights_gb"]
    assert est["training_memory_gb"]["activations_with_ckpt_gb"] < est["training_memory_gb"]["activations_no_ckpt_gb"]
    assert est["vercel_free_compatibility"]["training_allowed"] is False
    assert est["vercel_free_compatibility"]["status"] == "UNAVAILABLE_ON_VERCEL_FREE"


def test_distributed_prep_refuses_vercel_and_never_simulates_multigpu(monkeypatch):
    tf70 = GNNM_PRESETS["TITANFORGE-70B"]
    dist_cfg = DistributedTrainingConfig(model_name="TITANFORGE-70B")
    monkeypatch.setenv("VERCEL", "1")
    readiness = inspect_distributed_runtime_readiness(tf70, dist_cfg)
    assert readiness["execution_ready"] is False
    assert readiness["distributed_tests_executed"] is False
    assert readiness["status"] == "UNAVAILABLE_ON_VERCEL_FREE"
    assert any("VERCEL_RUNTIME_FORBIDDEN" in b for b in readiness["blockers"])


def test_faiss_cpu_preservation_and_vercel_lightweight_fallback():
    server.ensure_initialized()
    # On Vercel Free, FAISS must gracefully report INCOMPATIBLE_RUNTIME while Lightweight activates honestly
    reg_vercel = RetrievalProviderRegistry(rag_store=server.rag, is_vercel=True)
    rep_v = reg_vercel.report()
    assert rep_v["selected_provider"] == "LightweightRetrievalProvider"
    faiss_local_st = next(p for p in rep_v["providers"] if p["provider"] == "FaissLocalProvider")
    faiss_art_st = next(p for p in rep_v["providers"] if p["provider"] == "FaissArtifactProvider")
    light_st = next(p for p in rep_v["providers"] if p["provider"] == "LightweightRetrievalProvider")
    assert faiss_local_st["status"] == ProviderStatus.INCOMPATIBLE_RUNTIME.value
    assert faiss_art_st["status"] == ProviderStatus.INCOMPATIBLE_RUNTIME.value
    assert light_st["status"] == ProviderStatus.ACTIVE.value
    assert light_st["uses_faiss"] is False

    # Lightweight search works cleanly without claiming FAISS
    emb = server.engine.create_embedding("Sumpah Palapa")
    hits = reg_vercel.select_active_provider().search(emb, query_text="Sumpah Palapa", k=2)
    assert len(hits) >= 1
    assert hits[0]["uses_faiss"] is False

    # Artifact provider returns NOT_CONFIGURED when unconfigured locally
    art = FaissArtifactProvider(artifact_dir=None, is_vercel=False)
    assert art.status()["status"] in (ProviderStatus.NOT_CONFIGURED.value,)


def test_unified_model_registry_lifecycle_and_honest_specializations():
    reg = build_unified_model_registry(models_dir="/nonexistent_models_dir", is_vercel=True)
    assert len(reg["lifecycle_states"]) == 11
    assert reg["lifecycle_states"] == LIFECYCLE_STATES
    assert reg["training_execution_policy"]["vercel_free_training_enabled"] is False
    # Zero fake weights when models_dir has no verified checkpoints
    assert all(m["weight_available"] is False for m in reg["models"])
    # Specializations are explicitly marked as planned, not separate trained models
    for spec in ("neurox", "codestrike", "knowtron"):
        assert reg["families"][spec]["is_specialization"] is True
        assert reg["families"][spec]["status"] == "PLANNED"


def test_all_platform_api_endpoints_and_vercel_mode(client: TestClient, monkeypatch):
    # Test API status under simulated VERCEL=1 (missing weights, missing DB, missing FAISS)
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setattr(server, "supabase_store", type("_UnconfiguredSB", (), {"configured": False})())
    st = client.get("/api/v2/platform/status")
    assert st.status_code == 200
    sdata = st.json()
    assert sdata["is_vercel"] is True
    assert sdata["native_model_inference"]["active"] is False
    assert sdata["native_model_inference"]["status_banner"] == "NATIVE MODEL INFERENCE: UNAVAILABLE IN CURRENT RUNTIME"
    assert sdata["storage"]["warning"] is not None

    # Models & families
    assert client.get("/api/v2/platform/models").status_code == 200
    for fam in ("bumblecore", "opticore", "titanforge", "neurox", "codestrike", "knowtron"):
        r = client.get(f"/api/v2/platform/models/{fam}")
        assert r.status_code == 200

    # Model Factory estimator & custom validation
    est = client.get("/api/v2/platform/factory/estimator?preset=TITANFORGE-70B")
    assert est.status_code == 200
    assert est.json()["estimate"]["parameters"]["total_billions"] >= 69.0

    val = client.post(
        "/api/v2/platform/factory/validate",
        json={"name": "TEST-CUSTOM", "d_model": 512, "n_layers": 6, "n_heads": 8, "n_kv_heads": 2, "d_ff": 1536, "vocab_size": 8192},
    )
    assert val.status_code == 200
    assert val.json()["valid"] is True

    bad = client.post("/api/v2/platform/factory/validate", json={"d_model": 510, "n_heads": 8, "n_kv_heads": 2})
    assert bad.status_code == 422

    # Distributed plan, providers, nexus, benchmarks
    assert client.get("/api/v2/platform/factory/distributed-plan?preset=TITANFORGE-70B").status_code == 200
    assert client.get("/api/v2/platform/providers").status_code == 200
    nx = client.get("/api/v2/platform/nexus")
    assert nx.status_code == 200
    assert nx.json()["layer"] == "NEXUS PRIME"
    assert client.get("/api/v2/platform/benchmarks").status_code == 200


@pytest.mark.parametrize(
    "route",
    [
        "/",
        "/chat",
        "/brain",
        "/brain/activity",
        "/brain/knowledge",
        "/brain/memory",
        "/brain/reasoning",
        "/brain/learning",
        "/knowledge",
        "/memory",
        "/models",
        "/models/bumblecore",
        "/models/opticore",
        "/models/titanforge",
        "/models/neurox",
        "/models/codestrike",
        "/models/knowtron",
        "/model-factory",
        "/training",
        "/benchmarks",
        "/infrastructure",
        "/nexus",
        "/settings",
        "/documentation",
    ],
)
def test_all_required_application_routes_return_200_html(client: TestClient, route: str):
    resp = client.get(route)
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
