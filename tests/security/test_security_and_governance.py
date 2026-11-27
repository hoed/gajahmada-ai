import socket
import tempfile
from fastapi.testclient import TestClient
import pytest

from gajahmada_ai.api.server import app
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.coordinator import SovereignPeerNode
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType
from gajahmada_ai.security.crypto import NodeIdentity
from gajahmada_ai.security.ssrf import validate_peer_url


def test_12_unauthenticated_administrative_requests_rejected():
    """TEST 12: Unauthenticated administrative requests are rejected."""
    client = TestClient(app)

    # Attempt unauthenticated write to /api/v2/memory
    res_mem = client.post("/api/v2/memory", json={"content": "Unauthorized injection"})
    assert res_mem.status_code == 401

    # Attempt unauthenticated key rotation
    res_rot = client.post("/api/v2/system/rotate-key")
    assert res_rot.status_code == 401

    # Attempt unauthenticated learning experiment
    res_exp = client.post("/api/v2/system/learning/experiment", json={"name": "unauth_exp"})
    assert res_exp.status_code == 401

    # Attempt unauthenticated memory backup
    res_bak = client.post("/api/v2/memory/backup")
    assert res_bak.status_code == 401


def test_13_malicious_peer_cannot_execute_commands_or_alter_protected_memory():
    """TEST 13: A malicious peer cannot execute commands or alter protected memory."""
    victim_node = SovereignPeerNode(node_name="victim", endpoint="http://127.0.0.1:9301")
    attacker_node = SovereignPeerNode(node_name="attacker", endpoint="http://127.0.0.1:9302")

    # Store a PRIVATE memory record on victim node
    priv_rec = victim_node.memory.store(
        content="Classified sovereign key shard metadata",
        memory_type=MemoryType.SEMANTIC,
        visibility_policy=VisibilityPolicy.PRIVATE,
        verification_status=VerificationStatus.VERIFIED,
    )

    # 1. Attacker sends signed GIP message attempting shell command execution
    cmd_msg = GIPMessage(
        message_type=GIPMessageType.GIP_TASK,
        sender_node_id=attacker_node.identity.node_id,
        recipient_node_id=victim_node.identity.node_id,
        payload={"action": "exec_shell", "command": "rm -rf /"},
    ).sign(attacker_node.identity)

    resp1 = victim_node.receive_gip_message(cmd_msg)
    assert resp1["accepted"] is False
    assert "Forbidden remote peer action" in resp1["reason"]

    # 2. Attacker attempts to forge victim's signature to modify memory
    forged_msg = GIPMessage(
        message_type=GIPMessageType.GIP_SYNC,
        sender_node_id=victim_node.identity.node_id,
        recipient_node_id=victim_node.identity.node_id,
        payload={"records": [{"content": "Tampered record"}]},
        signature=attacker_node.identity.sign_message(b"forged"),
    )
    resp2 = victim_node.receive_gip_message(forged_msg)
    assert resp2["accepted"] is False
    assert "signature verification failed" in resp2["reason"].lower()

    # 3. Victim's PRIVATE memory is never exported during synchronization
    sync_out = victim_node.synchronizer.build_sync_message()
    exported_ids = [r["record_id"] for r in sync_out.payload.get("records", [])]
    assert priv_rec["record_id"] not in exported_ids

    # 4. Sandbox blocks unrestricted tool execution (e.g., shell_command without approval)
    task = victim_node.scheduler.submit_objective(
        objective="Run shell diagnostic",
        requested_tools=["shell_command"],
        human_approved=False,
    )
    assert task.state.value == "WAITING"


def test_ssrf_and_predictable_seed_prevention(monkeypatch):
    """Verify SSRF protection and rejection of predictable Ed25519 seeds."""
    # Cloud metadata IP must be blocked
    ok_meta, _ = validate_peer_url("http://169.254.169.254/latest/meta-data")
    assert ok_meta is False

    # Internal PostgreSQL / Redis ports must be blocked
    ok_pg, _ = validate_peer_url("http://127.0.0.1:5432")
    assert ok_pg is False
    ok_redis, _ = validate_peer_url("http://127.0.0.1:6379")
    assert ok_redis is False

    ok_local, _ = validate_peer_url("http://127.0.0.1:8001", allow_local=True)
    assert ok_local is True
    ok_strict, reason_strict = validate_peer_url("http://127.0.0.1:8001", allow_local=False)
    assert ok_strict is False
    assert "Private/loopback" in reason_strict

    def _resolve_metadata(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]

    monkeypatch.setattr("gajahmada_ai.security.ssrf.socket.getaddrinfo", _resolve_metadata)
    ok_dns, reason_dns = validate_peer_url("http://looks-public.example/latest")
    assert ok_dns is False
    assert "metadata" in reason_dns.lower() or "Link-local" in reason_dns

    # Predictable default Vercel seed must be rejected if passed via env
    monkeypatch.setenv("GAJAHMADA_NODE_SEED", "gajahmada-sovereign-ai-vercel-node-v2")
    with pytest.raises(ValueError, match="Predictable or low-entropy"):
        NodeIdentity(key_path=None, allow_env_seed=True)
