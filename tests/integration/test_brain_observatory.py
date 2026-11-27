"""
Integration tests for GajahMada Brain & Knowledge Observatory (/brain and /api/v2/observatory/*).

Verifies:
1. /brain HTML route serves the real Observatory UI with official logo & all 7 tabs.
2. All observatory API endpoints return authentic backend telemetry (no mock/fake data).
3. Strict authorization:
   - Unauthenticated /api/v2/observatory/requests/{id}/trace returns 401; authenticated admin returns 200.
   - Unauthenticated /api/v2/observatory/memory NEVER leaks PRIVATE/LOCAL records belonging to another user.
   - Unauthenticated /api/v2/observatory/cognition redacts user prompts.
4. Live data flow:
   - Executing a real query via POST /ai/chat creates a real trace visible in /api/v2/observatory/cognition.
   - Ingesting a real document via POST /api/v2/knowledge/ingest appears in /api/v2/observatory/knowledge/graph.
5. Pagination on /knowledge/graph and /memory.
"""

from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server
from gajahmada_ai.memory.models import MemoryType, VisibilityPolicy


@pytest.fixture(scope="module")
def client():
    server.ensure_initialized()
    return TestClient(server.app)


def _admin_headers() -> dict:
    server.ensure_initialized()
    assert server.auth_manager is not None
    return {"x-api-key": server.auth_manager.admin_api_key}


def test_brain_route_serves_observatory_html(client: TestClient):
    resp = client.get("/brain")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    html = resp.text
    assert "/ui/logo.png" in html
    assert "GAJAHMADA BRAIN &amp; KNOWLEDGE OBSERVATORY" in html
    for tab_id in [
        "brain-overview",
        "cognitive-activity",
        "knowledge-universe",
        "memory-explorer",
        "learning-observatory",
        "agent-network",
        "distributed-nodes",
    ]:
        assert f'data-tab="{tab_id}"' in html
    assert "/ui/brain.js" in html
    assert "/ui/brain.css" in html


def test_observatory_overview_and_brain_graph(client: TestClient):
    ov = client.get("/api/v2/observatory/overview")
    assert ov.status_code == 200
    data = ov.json()
    assert data["identity_redacted"] is True
    assert data["node_id"] is None
    assert data["counts"]["ledger_height"] is None
    assert data["system_status"] in ("HEALTHY", "DEGRADED")
    admin = client.get(
        "/api/v2/observatory/overview",
        headers={"X-API-Key": server.auth_manager.admin_api_key},
    )
    assert admin.status_code == 200
    assert admin.json()["node_id"]
    assert admin.json()["counts"]["ledger_height"] is not None
    assert "counts" in data
    assert "subsystems" in data
    assert "active_runtime_model" in data

    br = client.get("/api/v2/observatory/brain")
    assert br.status_code == 200
    bdata = br.json()
    assert len(bdata["groups"]) == 6
    node_ids = {n["id"] for n in bdata["nodes"]}
    assert {
        "gnnm-core",
        "cog-perception",
        "know-controller",
        "mem-manager",
        "learn-trainer",
        "net-local",
    }.issubset(node_ids)
    assert len(bdata["edges"]) >= 15


def test_observatory_cognition_trace_and_rbac(client: TestClient):
    # Execute a real query first so at least one trace exists
    unique_prompt = f"Apa itu GajahMada AI observatory trace test {uuid.uuid4().hex[:6]}?"
    chat_resp = client.post("/ai/chat", json={"prompt": unique_prompt})
    assert chat_resp.status_code == 200

    # Unauthenticated cognition endpoint must redact raw prompts
    cog_anon = client.get("/api/v2/observatory/cognition?limit=10")
    assert cog_anon.status_code == 200
    cdata_anon = cog_anon.json()
    assert cdata_anon["authenticated_detail"] is False
    assert len(cdata_anon["pipeline_stages"]) == 11
    assert len(cdata_anon["recent_traces"]) >= 1
    assert "REDACTED" in cdata_anon["recent_traces"][0]["objective_preview"]

    trace_id = cdata_anon["recent_traces"][0]["request_id"]

    # Unauthenticated trace detail must be rejected (401)
    trace_unauth = client.get(f"/api/v2/observatory/requests/{trace_id}/trace")
    assert trace_unauth.status_code == 401

    # Authenticated admin can view full trace and unredacted prompt
    trace_auth = client.get(f"/api/v2/observatory/requests/{trace_id}/trace", headers=_admin_headers())
    assert trace_auth.status_code == 200
    tdata = trace_auth.json()
    assert tdata["reasoning_id"] == trace_id
    assert len(tdata["reasoning_steps"]) >= 1

    cog_admin = client.get("/api/v2/observatory/cognition?limit=10", headers=_admin_headers())
    assert cog_admin.status_code == 200
    assert cog_admin.json()["authenticated_detail"] is True
    previews = [t["objective_preview"] for t in cog_admin.json()["recent_traces"]]
    assert any("observatory trace test" in p for p in previews)


def test_observatory_memory_authorization_no_private_leak(client: TestClient):
    server.ensure_initialized()
    assert server.memory_manager is not None

    secret_token = f"TOP_SECRET_OBSERVATORY_{uuid.uuid4().hex[:8]}"
    public_token = f"PUBLIC_OBSERVATORY_{uuid.uuid4().hex[:8]}"

    server.memory_manager.store(
        content=f"Confidential private memory {secret_token}",
        memory_type=MemoryType.EPISODIC,
        source_type="unit_test_private",
        owner_actor_id="isolated_owner_xyz",
        visibility_policy=VisibilityPolicy.PRIVATE,
    )
    server.memory_manager.store(
        content=f"Public observatory memory {public_token}",
        memory_type=MemoryType.SEMANTIC,
        source_type="unit_test_public",
        owner_actor_id="system",
        visibility_policy=VisibilityPolicy.PUBLIC,
    )

    # Unauthenticated caller must NOT see the PRIVATE record
    resp_anon = client.get("/api/v2/observatory/memory?limit=100&offset=0")
    assert resp_anon.status_code == 200
    mdata = resp_anon.json()
    assert len(mdata["available_scopes"]) == 9
    assert len(mdata["accessible_scope_counts"]) == 9
    contents_anon = [r["content"] for r in mdata["records"]]
    assert not any(secret_token in c for c in contents_anon)
    assert any(public_token in c for c in contents_anon)

    # Pagination check
    resp_page = client.get("/api/v2/observatory/memory?limit=2&offset=0")
    assert resp_page.status_code == 200
    assert len(resp_page.json()["records"]) <= 2


def test_observatory_knowledge_graph_live_ingest_and_pagination(client: TestClient):
    unique_title = f"Observatory Test Doc {uuid.uuid4().hex[:6]}"
    ingest_resp = client.post(
        "/api/v2/knowledge/ingest",
        headers=_admin_headers(),
        json={
            "title": unique_title,
            "content": f"{unique_title} explains how GajahMada Brain Observatory visualizes real SQLite WAL knowledge nodes.",
            "source_type": "LOCAL_FILE",
            "visibility_policy": "PUBLIC",
        },
    )
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["status"] == "INGESTED"

    kg_resp = client.get(f"/api/v2/observatory/knowledge/graph?query={unique_title}&limit=20&offset=0")
    assert kg_resp.status_code == 200
    kg = kg_resp.json()
    assert kg["pagination"]["total"] >= 1
    assert any(
        unique_title in (n.get("label") or "") or unique_title in (n.get("content_preview") or "")
        for n in kg["nodes"]
    )


def test_observatory_learning_agents_nodes_events(client: TestClient):
    lr = client.get("/api/v2/observatory/learning")
    assert lr.status_code == 200
    ldata = lr.json()
    assert len(ldata["lifecycle_steps"]) == 9
    assert set(ldata["knowledge_state_summary"].keys()) == {
        "RETRIEVED",
        "STORED",
        "VERIFIED",
        "TRAINED",
        "INTERNALIZED",
    }

    md = client.get("/api/v2/observatory/models")
    assert md.status_code == 200
    assert "model_versions" in md.json()

    ag = client.get("/api/v2/observatory/agents")
    assert ag.status_code == 200
    assert "agents" in ag.json()
    assert "tasks" in ag.json()

    nd = client.get("/api/v2/observatory/nodes")
    assert nd.status_code == 200
    ndata = nd.json()
    assert ndata["redacted"] is True
    assert ndata["local_node"]["node_id"] is None
    assert ndata["peers"] == []
    admin_nd = client.get(
        "/api/v2/observatory/nodes",
        headers={"X-API-Key": server.auth_manager.admin_api_key},
    )
    assert admin_nd.json()["local_node"]["node_id"]
    assert "peers" in admin_nd.json()

    ev = client.get("/api/v2/observatory/events?limit=20")
    assert ev.status_code == 200
    assert "events" in ev.json()
