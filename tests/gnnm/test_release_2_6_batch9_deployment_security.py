"""
Batch 9 Acceptance Tests — Brain Observatory Telemetry, Vercel Free Compatibility & Security.
"""
from __future__ import annotations

import json
import os
from fastapi.testclient import TestClient

from gajahmada_ai.api import server


def test_bumblecore_platform_telemetry_is_honest_and_complete():
    server.ensure_initialized()
    client = TestClient(server.app)
    res = client.get("/api/v2/platform/bumblecore")
    assert res.status_code == 200
    d = res.json()

    assert d["release"] == "2.6"
    assert d["pretraining_status"] == "NOT_STARTED"
    assert d["lifecycle_stage"] in d["status_taxonomy"]
    assert d["lifecycle_stage"] in ("CONFIGURED", "SMOKE_TESTED")

    tel = d["actual_telemetry"]
    assert tel["actual_pretraining_tokens"] == 0
    assert tel["actual_pretraining_steps"] == 0
    assert tel["checkpoint_trust_verified"] is False
    assert tel["evaluation_passed"] is False
    assert tel["evaluation_score"] is None
    assert tel["blocked_at_gate"] == "GATE_C_PRETRAINING_CHECKPOINT_VALIDATED"


def test_observatory_learning_and_overview_enforce_rbac_and_no_fake_training():
    server.ensure_initialized()
    client = TestClient(server.app)

    ov = client.get("/api/v2/observatory/overview")
    assert ov.status_code == 200
    assert ov.json()["actor_role"] in ("anonymous", "viewer", "guest")

    lr = client.get("/api/v2/observatory/learning")
    assert lr.status_code == 200
    lr_data = lr.json()
    assert lr_data["knowledge_state_summary"]["TRAINED"] == "NOT_EXECUTED"


def test_vercel_free_requirements_have_no_heavy_gpu_or_faiss_dependencies():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    req_path = os.path.join(repo_root, "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        reqs = f.read().lower()

    for forbidden in ("torch", "cuda", "faiss", "transformers", "deepspeed", "triton", "bitsandbytes"):
        assert forbidden not in reqs, f"Heavy dependency '{forbidden}' must NOT be in Vercel requirements.txt"

    vercel_path = os.path.join(repo_root, "vercel.json")
    with open(vercel_path, "r", encoding="utf-8") as f:
        vcfg = json.load(f)
    py_builds = [b for b in vcfg.get("builds", []) if b.get("use") == "@vercel/python"]
    assert len(py_builds) == 1
    assert "configs/bumblecore_300m.yaml" in py_builds[0]["config"]["includeFiles"]
