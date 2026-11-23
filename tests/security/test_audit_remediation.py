import json
import os
import tempfile
from fastapi.testclient import TestClient
import pytest

from gajahmada_ai.api.server import app, get_runtime_context
from gajahmada_ai.blockchain.chain import AIBlockchain, LedgerCorruptedError, LedgerEventType
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.coordinator import CollectiveCoordinator, SovereignPeerNode
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType
from gajahmada_ai.security.auth import AuthManager
from gajahmada_ai.security.crypto import NodeIdentity


def test_p0_1_admin_key_is_random_and_rotatable(monkeypatch):
    """P0.1: Admin key must NEVER be derived from public node_id and must support rotation."""
    monkeypatch.delenv("GAJAHMADA_ADMIN_KEY", raising=False)
    ident = NodeIdentity(key_path=None)
    auth = AuthManager(identity=ident)

    assert ident.node_id[:12] not in auth.admin_api_key
    assert len(auth.admin_api_key) >= 40

    old_key = auth.admin_api_key
    new_key = auth.rotate_admin_key()
    assert new_key != old_key
    assert len(new_key) >= 40


def test_p0_2_v1_admin_routes_fail_closed_by_default(monkeypatch):
    """P0.2: All v1 administrative mutation routes must reject unauthenticated requests by default."""
    monkeypatch.delenv("GAJAHMADA_REQUIRE_AUTH", raising=False)
    client = TestClient(app)

    assert client.post("/node/config", json={"node_name": "hijacked"}).status_code == 401
    assert client.delete("/node/logs").status_code == 401
    assert client.post("/node/sync", json={"peers": ["http://127.0.0.1:8009"]}).status_code == 401
    assert client.delete("/node/peers?peer=http://127.0.0.1:8009").status_code == 401
    assert client.delete("/knowledge/docs/0").status_code == 401


def test_p0_3_private_memory_versions_and_contradictions_protected():
    """P0.3: PRIVATE memory records, their versions, relations, and contradictions must never leak."""
    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key

    # Create a PRIVATE memory record as admin
    r_priv = client.post(
        "/api/v2/memory",
        headers={"X-API-Key": admin_key},
        json={
            "content": "Project Garuda secret vault code is 998877.",
            "memory_type": "semantic",
            "visibility_policy": "PRIVATE",
            "verification_status": "VERIFIED",
        },
    )
    assert r_priv.status_code == 200
    priv_id = r_priv.json()["record_id"]

    # Create a contradicting PRIVATE record so a contradiction entry is generated
    r_priv2 = client.post(
        "/api/v2/memory",
        headers={"X-API-Key": admin_key},
        json={
            "content": "Project Garuda secret vault code is not 998877.",
            "memory_type": "semantic",
            "visibility_policy": "PRIVATE",
            "verification_status": "VERIFIED",
        },
    )
    assert r_priv2.status_code == 200

    # Unauthenticated read of /api/v2/memory must hide PRIVATE records AND their contradictions
    r_pub_mem = client.get("/api/v2/memory")
    assert r_pub_mem.status_code == 200
    pub_ids = {r["record_id"] for r in r_pub_mem.json()["records"]}
    assert priv_id not in pub_ids
    for c in r_pub_mem.json()["contradictions"]:
        assert c["record_a_id"] != priv_id and c["record_b_id"] != priv_id

    # Unauthenticated read of PRIVATE record versions must return 403
    r_pub_ver = client.get(f"/api/v2/memory/{priv_id}/versions")
    assert r_pub_ver.status_code == 403

    # Authenticated admin CAN read versions
    r_adm_ver = client.get(f"/api/v2/memory/{priv_id}/versions", headers={"X-API-Key": admin_key})
    assert r_adm_ver.status_code == 200
    assert len(r_adm_ver.json()["versions"]) >= 1

    # Unauthenticated /api/v2/knowledge must not leak PRIVATE relations or contradictions
    r_pub_kg = client.get("/api/v2/knowledge")
    assert r_pub_kg.status_code == 200
    for c in r_pub_kg.json()["contradictions"]:
        assert c["record_a_id"] != priv_id and c["record_b_id"] != priv_id


def test_p0_5_human_approval_cannot_be_spoofed():
    """P0.5: Submitting human_approved=true in /agents/run must be ignored; only authenticated /approve works."""
    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key

    # Attempt to spoof human_approved=True on public submission
    r_run = client.post(
        "/api/v2/agents/run",
        json={
            "objective": "Execute sensitive sovereign operation",
            "requires_human_approval": True,
            "human_approved": True,
        },
    )
    assert r_run.status_code == 200
    task = r_run.json()
    assert task["state"] == "WAITING"

    # Unauthenticated approve must be rejected (401)
    assert client.post(f"/api/v2/agents/tasks/{task['task_id']}/approve").status_code == 401

    # Authenticated approve succeeds and completes task
    r_app = client.post(
        f"/api/v2/agents/tasks/{task['task_id']}/approve",
        headers={"X-API-Key": admin_key},
    )
    assert r_app.status_code == 200
    assert r_app.json()["state"] == "COMPLETED"


def test_p1_blockchain_tamper_evident_metadata_and_corruption_recovery():
    """P1: Tampering with event_type/event_ref/metadata breaks hash & triggers LEDGER_CORRUPTED (no silent wipe)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_path = os.path.join(tmpdir, "key.pem")
        chain_path = os.path.join(tmpdir, "chain.json")
        ident = NodeIdentity(key_path=key_path)
        chain = AIBlockchain(storage_path=chain_path)

        blk = chain.record_event(
            LedgerEventType.TASK_COMPLETED,
            "task-123",
            ident,
            metadata={"verification": "FAILED"},
        )
        assert blk.schema_version == 2
        assert chain.is_chain_valid(verify_signatures=True) is True

        # Add a second event so .bak holds a valid snapshot of block #1
        chain.record_event(
            LedgerEventType.EVIDENCE_SUBMITTED,
            "task-123-ev",
            ident,
            metadata={"source": "symbolic"},
        )
        assert os.path.exists(f"{chain_path}.bak")

        # Tamper with metadata on disk (FAILED -> APPROVED)
        with open(chain_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw[1]["metadata"]["verification"] = "APPROVED"
        raw[1]["event_type"] = "RESULT_VERIFIED"
        with open(chain_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        # Reloading tampered chain MUST enter LEDGER_CORRUPTED (NOT silently recreate genesis!)
        reloaded = AIBlockchain(storage_path=chain_path)
        assert reloaded.status == "LEDGER_CORRUPTED"
        assert reloaded.is_chain_valid(verify_signatures=True) is False
        with pytest.raises(LedgerCorruptedError):
            reloaded.record_event(LedgerEventType.AUDIT_BLOCK, "should-fail", ident)

        # Explicit recovery from backup restores healthy ledger
        assert reloaded.recover_from_backup() is True
        assert reloaded.status == "HEALTHY"
        assert reloaded.is_chain_valid(verify_signatures=True) is True


def test_p2_and_p3_gip_full_handlers_and_knowledge_quarantine():
    """P2 & P3: /api/v2/network/gip handles IDENTITY, TASK, KNOWLEDGE (quarantined), VERIFY, CONSENSUS."""
    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key
    peer_id = NodeIdentity(key_path=None)

    # 1. GIP-IDENTITY handshake
    m_id = GIPMessage(
        message_type=GIPMessageType.GIP_IDENTITY,
        sender_node_id=peer_id.node_id,
        payload={"node_name": "peer-beta", "endpoint": "http://127.0.0.1:9102", "capabilities": ["symbolic"]},
    ).sign(peer_id)
    r_id = client.post("/api/v2/network/gip", json=m_id.model_dump())
    assert r_id.status_code == 200
    assert r_id.json()["accepted"] is True

    # 2. GIP-TASK execution
    m_task = GIPMessage(
        message_type=GIPMessageType.GIP_TASK,
        sender_node_id=peer_id.node_id,
        payload={"task_id": "t-gip-01", "objective": "Hitung 20 * 5 + 15"},
    ).sign(peer_id)
    r_task = client.post("/api/v2/network/gip", json=m_task.model_dump())
    assert r_task.status_code == 200
    assert "115" in r_task.json()["reply"]["payload"]["conclusion"]

    # 3. GIP-KNOWLEDGE enters LOCAL Quarantine (UNVERIFIED) — then promoted via authenticated /promote
    m_know = GIPMessage(
        message_type=GIPMessageType.GIP_KNOWLEDGE,
        sender_node_id=peer_id.node_id,
        payload={"content": "Peer Beta observed ocean buoy temperature 28.4C at station-07.", "confidence": 0.99},
    ).sign(peer_id)
    r_know = client.post("/api/v2/network/gip", json=m_know.model_dump())
    assert r_know.status_code == 200
    assert r_know.json()["status"] == "QUARANTINED_UNVERIFIED"
    q_rec_id = r_know.json()["quarantined_record_id"]

    rec_before = rt["memory"].storage.get_record(q_rec_id)
    assert rec_before["verification_status"] == "UNVERIFIED"
    assert rec_before["visibility_policy"] == "LOCAL"

    # Promote after verification
    r_prom = client.post(f"/api/v2/knowledge/promote/{q_rec_id}", headers={"X-API-Key": admin_key})
    assert r_prom.status_code == 200
    assert r_prom.json()["record"]["verification_status"] == "VERIFIED"
    assert r_prom.json()["record"]["visibility_policy"] == "COLLECTIVE"


def test_p5_three_independent_nodes_end_to_end_and_restart_consistency():
    """P5: 3 independent nodes coordinate, verify with signed v2 ledger, and remain consistent after restart."""
    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "node_a")
        dir_b = os.path.join(base_dir, "node_b")
        dir_c = os.path.join(base_dir, "node_c")

        node_a = SovereignPeerNode("alpha", "http://127.0.0.1:9401", storage_dir=dir_a)
        node_b = SovereignPeerNode("beta", "http://127.0.0.1:9402", storage_dir=dir_b)
        node_c = SovereignPeerNode("gamma", "http://127.0.0.1:9403", storage_dir=dir_c)

        coord = CollectiveCoordinator(node_a)
        outcome = coord.coordinate_distributed_task("Hitung 36 * 3 + 12", peers=[node_b, node_c], quorum_size=2)
        assert outcome["quorum_reached"] is True
        assert "120" in outcome["selected_conclusion"]
        assert outcome["collective_memory_record_id"].startswith("mem-")

        # Record IDs & heights before restart
        id_a, id_b, id_c = node_a.identity.node_id, node_b.identity.node_id, node_c.identity.node_id
        h_a, h_b, h_c = len(node_a.blockchain.chain), len(node_b.blockchain.chain), len(node_c.blockchain.chain)

        # Simulate full process restart of all 3 independent nodes from disk
        node_a2 = SovereignPeerNode("alpha", "http://127.0.0.1:9401", storage_dir=dir_a)
        node_b2 = SovereignPeerNode("beta", "http://127.0.0.1:9402", storage_dir=dir_b)
        node_c2 = SovereignPeerNode("gamma", "http://127.0.0.1:9403", storage_dir=dir_c)

        assert node_a2.identity.node_id == id_a
        assert node_b2.identity.node_id == id_b
        assert node_c2.identity.node_id == id_c
        assert node_a2.blockchain.is_chain_valid(verify_signatures=True) is True
        assert node_b2.blockchain.is_chain_valid(verify_signatures=True) is True
        assert node_c2.blockchain.is_chain_valid(verify_signatures=True) is True
        assert len(node_a2.blockchain.chain) >= h_a
        assert len(node_b2.blockchain.chain) >= h_b
        assert len(node_c2.blockchain.chain) >= h_c
        assert node_a2.memory.storage.get_record(outcome["collective_memory_record_id"]) is not None
