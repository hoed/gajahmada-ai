"""
Regression tests for the audit findings on unauthenticated legacy endpoints:
1. Anonymous /ai/chat must not write user Q&A into the shared legacy RAG store (poisoning).
2. /knowledge/stats and /knowledge/search must not expose other users' Q&A to anonymous callers.
3. /node/config and /node/logs must be redacted for anonymous callers (full view for admin).
4. /ai/embedding must bound input size for anonymous callers.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server


@pytest.fixture(scope="module")
def client():
    server.ensure_initialized()
    return TestClient(server.app)


def _admin() -> dict:
    return {"x-api-key": server.auth_manager.admin_api_key}


def test_anonymous_chat_does_not_poison_or_leak_shared_rag(client: TestClient):
    secret = f"rahasia-{uuid.uuid4().hex[:10]}"
    before = len(server.rag.doc_store)

    r = client.post("/ai/chat", json={"prompt": f"Apa itu Majapahit {secret}?"})
    assert r.status_code == 200

    # Shared RAG store must not grow with anonymous Q&A, nor contain the secret.
    assert not any(secret in d for d in server.rag.doc_store)
    assert len(server.rag.doc_store) == before

    stats = client.get("/knowledge/stats").json()
    assert not any(secret in d["preview"] for d in stats["docs"])

    search = client.post("/knowledge/search", json={"query": secret, "k": 5}).json()
    assert not any(secret in x for x in search["results"])


def test_anonymous_cannot_see_legacy_user_qa_but_admin_can(client: TestClient):
    marker = f"legacy-{uuid.uuid4().hex[:10]}"
    entry = f"Pertanyaan: {marker} | Jawaban: contoh"
    emb = server.engine.create_embedding(entry)
    server.rag.ingest(entry, emb)

    anon = client.get("/knowledge/stats").json()
    assert not any(marker in d["preview"] for d in anon["docs"])
    assert anon["user_qa_entries_hidden"] >= 1

    admin = client.get("/knowledge/stats", headers=_admin()).json()
    assert any(marker in d["preview"] for d in admin["docs"])
    assert admin["user_qa_entries_hidden"] == 0


def test_public_corpus_remains_available_to_anonymous(client: TestClient):
    stats = client.get("/knowledge/stats").json()
    assert stats["total_docs"] >= 15
    assert len(stats["docs"]) >= 1
    res = client.post("/knowledge/search", json={"query": "Sumpah Palapa Majapahit", "k": 2}).json()
    assert res["count"] >= 1


def test_node_config_redacted_for_anonymous(client: TestClient):
    anon = client.get("/node/config").json()
    assert anon["redacted"] is True
    assert "model_path" not in str(anon)
    assert "bootstrap_peers" not in str(anon)
    assert "host" not in anon.get("node", {})

    full = client.get("/node/config", headers=_admin()).json()
    assert "bootstrap_peers" in full["network"]
    assert "model_path" in full["ai"]


def test_node_logs_redact_user_fragments_for_anonymous(client: TestClient):
    secret = f"logsecret-{uuid.uuid4().hex[:8]}"
    client.post("/knowledge/search", json={"query": secret, "k": 1})

    anon_logs = client.get("/node/logs").json()["logs"]
    assert not any(secret in str(e["message"]) for e in anon_logs)

    admin_logs = client.get("/node/logs", headers=_admin()).json()["logs"]
    assert any(secret in str(e["message"]) for e in admin_logs)


def test_embedding_input_bounded_for_anonymous(client: TestClient):
    ok = client.post("/ai/embedding", json={"text": "Sumpah Palapa"})
    assert ok.status_code == 200
    big = client.post("/ai/embedding", json={"text": "x" * 5000})
    assert big.status_code == 413
    admin_big = client.post("/ai/embedding", json={"text": "x" * 5000}, headers=_admin())
    assert admin_big.status_code == 200
