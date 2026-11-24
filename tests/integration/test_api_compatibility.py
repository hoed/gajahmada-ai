from fastapi.testclient import TestClient

from gajahmada_ai.api.server import app, get_runtime_context


def test_15_existing_supported_api_behavior_remains_compatible():
    """TEST 15: Existing supported API behavior remains compatible."""
    client = TestClient(app)

    # 1. Root & Status
    r_root = client.get("/?format=json")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "online"
    assert "ml_available" in r_root.json()

    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    assert "system_time" in r_status.json()

    # 2. Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["ok"] is True

    # 3. AI Chat (with RAG + PoW block + v2 cognitive fields)
    r_chat = client.post("/ai/chat", json={"prompt": "Siapa Mahapatih Gajah Mada?", "max_tokens": 128})
    assert r_chat.status_code == 200
    chat_data = r_chat.json()
    for key in ("response", "time_taken", "block_index", "task_hash", "signature", "rag_matches", "reasoning_id", "verification_status"):
        assert key in chat_data
    assert "Gajah Mada" in chat_data["response"]

    # 4. AI Embedding & Knowledge Search
    r_emb = client.post("/ai/embedding", json={"text": "Sumpah Palapa 1336"})
    assert r_emb.status_code == 200
    assert r_emb.json()["dimension"] == 384
    assert len(r_emb.json()["embedding"]) == 384

    r_kstats = client.get("/knowledge/stats")
    assert r_kstats.status_code == 200
    assert r_kstats.json()["total_docs"] >= 15

    r_ksearch = client.post("/knowledge/search", json={"query": "Sumpah Palapa Majapahit", "k": 2})
    assert r_ksearch.status_code == 200
    assert r_ksearch.json()["count"] >= 1

    # 5. Node Info, Metrics, Chain, Sign-Verify, Config, Dependencies, Logs
    r_info = client.get("/node/info")
    assert r_info.status_code == 200
    assert "node_id" in r_info.json()

    r_metrics = client.get("/node/metrics")
    assert r_metrics.status_code == 200
    assert "cpu_usage" in r_metrics.json()

    r_chain = client.get("/node/chain")
    assert r_chain.status_code == 200
    assert r_chain.json()["valid"] is True

    r_verify = client.post("/node/chain/verify")
    assert r_verify.status_code == 200
    assert r_verify.json()["valid"] is True

    r_sig = client.post("/security/sign-verify", json={"message": "compat-check"})
    assert r_sig.status_code == 200
    assert r_sig.json()["verified"] is True

    # 6. Test all v2 API routes
    r_v2_diag = client.get("/api/v2/intelligence/diagnose")
    assert r_v2_diag.status_code == 200
    assert r_v2_diag.json()["version"] == "2.0.0"

    r_v2_ask = client.post("/api/v2/intelligence/ask", json={"objective": "Hitung 50 * 4 + 25"})
    assert r_v2_ask.status_code == 200
    assert "225" in r_v2_ask.json()["conclusion"]

    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key

    r_v2_agent = client.post(
        "/api/v2/agents/run",
        headers={"X-API-Key": admin_key},
        json={"objective": "Investigasi Sumpah Palapa 1336 M"},
    )
    assert r_v2_agent.status_code == 200
    assert r_v2_agent.json()["state"] == "COMPLETED"

    r_v2_sys = client.get("/api/v2/system/health")
    assert r_v2_sys.status_code == 200
    assert r_v2_sys.json()["status"] == "healthy"

    # Authenticated v2 memory write test
    r_v2_mem = client.post(
        "/api/v2/memory",
        headers={"X-API-Key": admin_key},
        json={"content": "Authenticated sovereign knowledge entry.", "memory_type": "semantic"},
    )
    assert r_v2_mem.status_code == 200
    assert r_v2_mem.json()["record_id"].startswith("mem-")
