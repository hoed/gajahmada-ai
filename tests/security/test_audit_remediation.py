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

    # Unauthenticated /agents/run must be rejected (401)
    assert client.post("/api/v2/agents/run", json={"objective": "unauth"}).status_code == 401

    # Attempt to spoof human_approved=True on authenticated submission
    r_run = client.post(
        "/api/v2/agents/run",
        headers={"X-API-Key": admin_key},
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
    """P1/P2/P3: GIP trust lifecycle (PENDING->TRUSTED), SSRF guard, Quarantine, and source-corroborated promotion."""
    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key
    peer_id = NodeIdentity(key_path=None)

    # 0. GIP-DISCOVERY with SSRF metadata URL must be blocked (400)
    m_ssrf = GIPMessage(
        message_type=GIPMessageType.GIP_DISCOVERY,
        sender_node_id=peer_id.node_id,
        payload={"node_name": "evil", "endpoint": "http://169.254.169.254/latest/meta-data"},
    ).sign(peer_id)
    assert client.post("/api/v2/network/gip", json=m_ssrf.model_dump()).status_code == 400

    # 1. GIP-IDENTITY handshake registers peer as PENDING (NOT auto-TRUSTED!)
    m_id = GIPMessage(
        message_type=GIPMessageType.GIP_IDENTITY,
        sender_node_id=peer_id.node_id,
        payload={"node_name": "peer-beta", "endpoint": "http://127.0.0.1:9102", "capabilities": ["symbolic"]},
    ).sign(peer_id)
    r_id = client.post("/api/v2/network/gip", json=m_id.model_dump())
    assert r_id.status_code == 200
    assert r_id.json()["trust_state"] == "PENDING"

    # 1B. PENDING peer MUST be rejected from GIP-TASK (403)
    m_task_blocked = GIPMessage(
        message_type=GIPMessageType.GIP_TASK,
        sender_node_id=peer_id.node_id,
        payload={"task_id": "t-gip-00", "objective": "Hitung 2 + 2"},
    ).sign(peer_id)
    assert client.post("/api/v2/network/gip", json=m_task_blocked.model_dump()).status_code == 403

    # 1C. Admin promotes peer to TRUSTED via /api/v2/nodes/{id}/trust
    r_trust = client.post(
        f"/api/v2/nodes/{peer_id.node_id}/trust",
        headers={"X-API-Key": admin_key},
        json={"trust_state": "TRUSTED"},
    )
    assert r_trust.status_code == 200
    assert r_trust.json()["trust_state"] == "TRUSTED"

    # 2. Now TRUSTED peer can execute GIP-TASK
    m_task = GIPMessage(
        message_type=GIPMessageType.GIP_TASK,
        sender_node_id=peer_id.node_id,
        payload={"task_id": "t-gip-01", "objective": "Hitung 20 * 5 + 15"},
    ).sign(peer_id)
    r_task = client.post("/api/v2/network/gip", json=m_task.model_dump())
    assert r_task.status_code == 200
    assert "115" in r_task.json()["reply"]["payload"]["conclusion"]

    # 3. GIP-KNOWLEDGE enters LOCAL Quarantine (UNVERIFIED) — promotion requires independent_sources
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

    # Promotion without independent sources MUST fail (422)
    assert client.post(
        f"/api/v2/knowledge/promote/{q_rec_id}",
        headers={"X-API-Key": admin_key},
        json={"independent_sources": []},
    ).status_code == 422

    # Round 3 Finding A: independent_node_verification WITHOUT cryptographic attestations MUST fail (400)
    assert client.post(
        f"/api/v2/knowledge/promote/{q_rec_id}",
        headers={"X-API-Key": admin_key},
        json={
            "independent_sources": ["https://bmkg.go.id/buoy-07", "node:verifier-gamma"],
            "verification_method": "independent_node_verification",
        },
    ).status_code == 400

    # Plain strings alone can AT MOST achieve SOURCE_CORROBORATED (NEVER INDEPENDENTLY_VERIFIED)
    r_corrob = client.post(
        f"/api/v2/knowledge/promote/{q_rec_id}",
        headers={"X-API-Key": admin_key},
        json={
            "independent_sources": ["https://bmkg.go.id/buoy-07", "https://noaa.gov/buoy-07"],
            "verification_method": "source_corroboration",
        },
    )
    assert r_corrob.status_code == 200
    assert r_corrob.json()["epistemic_status"] == "SOURCE_CORROBORATED"

    # Promotion WITH >= 2 cryptographically signed Ed25519 attestations from distinct TRUSTED verifiers
    # and distinct evidence hashes achieves INDEPENDENTLY_VERIFIED
    from gajahmada_ai.security.crypto import canonical_json_bytes
    local_ident = rt["identity"]
    rec_hash = rt["memory"].storage.get_record(q_rec_id)["integrity_hash"]

    def make_attestation(verifier_ident, ev_hash, src_ref):
        payload = canonical_json_bytes({
            "record_id": q_rec_id,
            "record_hash": rec_hash,
            "evidence_hash": ev_hash,
            "source_ref": src_ref,
            "verifier_node_id": verifier_ident.node_id,
        })
        return {
            "node_id": verifier_ident.node_id,
            "signature": verifier_ident.sign_message(payload),
            "evidence_hash": ev_hash,
            "source_ref": src_ref,
        }

    r_prom = client.post(
        f"/api/v2/knowledge/promote/{q_rec_id}",
        headers={"X-API-Key": admin_key},
        json={
            "independent_sources": ["https://bmkg.go.id/buoy-07", "https://noaa.gov/buoy-07"],
            "verification_method": "independent_node_verification",
            "node_attestations": [
                make_attestation(local_ident, "sha256-bmkg-sensor-07", "https://bmkg.go.id/buoy-07"),
                make_attestation(peer_id, "sha256-noaa-buoy-07", "https://noaa.gov/buoy-07"),
            ],
        },
    )
    assert r_prom.status_code == 200
    assert r_prom.json()["epistemic_status"] == "INDEPENDENTLY_VERIFIED"
    assert r_prom.json()["verified_attestations"] == 2
    assert r_prom.json()["record"]["visibility_policy"] == "COLLECTIVE"


def test_p0_signing_oracle_eliminated_and_mutation_endpoints_authenticated():
    """P0: /security/sign-verify cannot act as an admin signing oracle; all mutation endpoints require auth."""
    client = TestClient(app)
    rt = get_runtime_context()
    node_id = rt["identity"].node_id

    # 1. /security/sign-verify must use isolated ephemeral key (NOT node private key)
    r_sv = client.post("/security/sign-verify", json={"message": "POST|/api/v2/system/rotate-key|9999999999"})
    assert r_sv.status_code == 200
    sv_data = r_sv.json()
    assert sv_data["oracle_protected"] is True
    assert sv_data["selftest_ephemeral_key_id"] != node_id

    # Attempt to replay that signature as X-Node-Id admin auth MUST FAIL (401)
    r_replay = client.post(
        "/api/v2/system/rotate-key",
        headers={
            "X-Node-Id": node_id,
            "X-Signature": sv_data["signature"],
            "X-Timestamp": "9999999999",
        },
    )
    assert r_replay.status_code == 401

    # 2. All state-mutating endpoints reject unauthenticated requests (401)
    assert client.post("/ai/ingest", json={"text": "unauth injection"}).status_code == 401
    assert client.post("/node/chain/mine", json={"note": "unauth mine"}).status_code == 401
    assert client.post("/node/broadcast", json={"message": "unauth broadcast"}).status_code == 401
    assert client.post("/knowledge/crawl", json={"query": "jakarta"}).status_code == 401


def test_p0_blockchain_v2_rejects_legacy_signature_and_single_field_tamper():
    """P0: Schema v2 blocks strictly reject legacy task_hash-only sigs AND any single-field tamper."""
    from gajahmada_ai.blockchain.chain import Block

    with tempfile.TemporaryDirectory() as tmpdir:
        ident = NodeIdentity(key_path=os.path.join(tmpdir, "k.pem"))
        chain = AIBlockchain(storage_path=os.path.join(tmpdir, "c.json"))

        # Legacy task_hash-only signature on schema_version=2 block MUST be rejected
        bad_v2 = Block(
            index=1,
            timestamp=1700000000.0,
            task_hash="abc123",
            node_id=ident.node_id,
            previous_hash=chain.get_latest_block().hash,
            signature=ident.sign_message(b"abc123"),
            schema_version=2,
        )
        assert chain.verify_block_signature(bad_v2) is False

        # Valid v2 block via record_event() passes; tampering ANY single field breaks validation
        blk = chain.record_event(
            LedgerEventType.TASK_COMPLETED, "ref-01", ident, metadata={"verification": "FAILED"}
        )
        assert chain.is_chain_valid(verify_signatures=True) is True

        for attr, mutated in [
            ("metadata", {"verification": "APPROVED"}),
            ("event_type", "RESULT_VERIFIED"),
            ("event_ref", "ref-99"),
            ("nonce", blk.nonce + 1),
        ]:
            orig = getattr(blk, attr)
            setattr(blk, attr, mutated)
            assert chain.is_chain_valid(verify_signatures=True) is False, f"Tampering {attr} was not detected!"
            setattr(blk, attr, orig)
        assert chain.is_chain_valid(verify_signatures=True) is True


def test_p1_gip_consensus_rejects_forged_submissions_and_enforces_policy_quorum():
    """P1: GIP-CONSENSUS ignores caller signature_verified=True, rejects duplicates, & enforces policy quorum >= 2."""
    from gajahmada_ai.security.crypto import canonical_json_bytes

    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key
    peer_a = NodeIdentity(key_path=None)
    peer_b = NodeIdentity(key_path=None)

    # Register & trust peer_a
    m_id = GIPMessage(
        message_type=GIPMessageType.GIP_IDENTITY,
        sender_node_id=peer_a.node_id,
        payload={"node_name": "cons-a", "endpoint": "http://127.0.0.1:9111"},
    ).sign(peer_a)
    assert client.post("/api/v2/network/gip", json=m_id.model_dump()).status_code == 200
    assert client.post(
        f"/api/v2/nodes/{peer_a.node_id}/trust",
        headers={"X-API-Key": admin_key},
        json={"trust_state": "TRUSTED"},
    ).status_code == 200

    # 1. Forged payload claiming signature_verified=True & quorum_size=1 MUST be rejected
    m_forged = GIPMessage(
        message_type=GIPMessageType.GIP_CONSENSUS,
        sender_node_id=peer_a.node_id,
        payload={
            "task_id": "cons-01",
            "quorum_size": 1,
            "submissions": [
                {"node_id": peer_a.node_id, "signature_verified": True, "conclusion": "fake", "confidence": 0.99},
                {"node_id": peer_b.node_id, "signature_verified": True, "conclusion": "fake", "confidence": 0.99},
            ],
        },
    ).sign(peer_a)
    r_forged = client.post("/api/v2/network/gip", json=m_forged.model_dump())
    assert r_forged.status_code == 200
    c_forged = r_forged.json()["consensus"]
    assert c_forged["quorum_reached"] is False
    assert c_forged["rejected_submissions"] == 2
    assert c_forged["enforced_quorum_size"] >= 2

    # 2. Two genuinely signed submissions from 2 distinct nodes succeed
    def make_signed_sub(node_ident, conclusion):
        ev = [{"provenance_hash": f"prov-{node_ident.node_id[:8]}"}]
        sig = node_ident.sign_message(canonical_json_bytes({
            "task_id": "cons-02",
            "node_id": node_ident.node_id,
            "conclusion": conclusion,
            "confidence": 0.92,
            "evidence": ev,
        }))
        return {
            "node_id": node_ident.node_id,
            "conclusion": conclusion,
            "confidence": 0.92,
            "symbolic_verified": True,
            "evidence": ev,
            "signature": sig,
        }

    m_valid = GIPMessage(
        message_type=GIPMessageType.GIP_CONSENSUS,
        sender_node_id=peer_a.node_id,
        payload={
            "task_id": "cons-02",
            "submissions": [make_signed_sub(peer_a, "120"), make_signed_sub(peer_b, "120")],
        },
    ).sign(peer_a)
    r_valid = client.post("/api/v2/network/gip", json=m_valid.model_dump())
    assert r_valid.status_code == 200
    assert r_valid.json()["consensus"]["quorum_reached"] is True
    assert r_valid.json()["consensus"]["participating_nodes"] == 2


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


# ─── Round 3 Audit Remediation Regression Tests (Findings A, B, C, D, E, F) ───

def test_round3_a_consensus_verifies_signatures_directly_and_checks_evidence_independence():
    """Finding A: QuorumConsensusManager ignores signature_verified=True and downgrades single-source echoes."""
    from gajahmada_ai.network.consensus import QuorumConsensusManager
    from gajahmada_ai.security.crypto import canonical_json_bytes

    mgr = QuorumConsensusManager()
    id_a = NodeIdentity(key_path=None)
    id_b = NodeIdentity(key_path=None)

    # 1. Spoofed signature_verified=True without real Ed25519 signature MUST be rejected
    res_spoof = mgr.evaluate_submissions(
        task_id="r3-a-01",
        submissions=[
            {"node_id": id_a.node_id, "signature_verified": True, "conclusion": "fake", "confidence": 0.99},
            {"node_id": id_b.node_id, "signature_verified": True, "conclusion": "fake", "confidence": 0.99},
        ],
        quorum_size=2,
    )
    assert res_spoof["quorum_reached"] is False
    assert res_spoof["rejected_submissions"] == 2
    assert res_spoof["status"] == "INSUFFICIENT_VERIFIED_SUBMISSIONS"

    # 2. Two signed nodes echoing the exact same single provenance hash cannot reach VERIFIED_BY_EVIDENCE_AND_QUORUM
    def sign_sub(ident, task_id, conclusion, ev):
        sig = ident.sign_message(canonical_json_bytes({
            "task_id": task_id,
            "node_id": ident.node_id,
            "conclusion": conclusion,
            "confidence": 0.9,
            "evidence": ev,
        }))
        return {
            "node_id": ident.node_id,
            "conclusion": conclusion,
            "confidence": 0.9,
            "evidence": ev,
            "signature": sig,
        }

    same_ev = [{"provenance_hash": "shared-single-source-hash"}]
    res_single = mgr.evaluate_submissions(
        task_id="r3-a-02",
        submissions=[
            sign_sub(id_a, "r3-a-02", "Result X", same_ev),
            sign_sub(id_b, "r3-a-02", "Result X", same_ev),
        ],
        quorum_size=2,
    )
    assert res_single["quorum_reached"] is True
    assert res_single["independent_corroboration"] is False
    assert res_single["status"] == "CORROBORATED_SINGLE_SOURCE"

    # 3. Two signed nodes with DISTINCT provenance hashes reach VERIFIED_BY_EVIDENCE_AND_QUORUM
    res_indep = mgr.evaluate_submissions(
        task_id="r3-a-03",
        submissions=[
            sign_sub(id_a, "r3-a-03", "Result X", [{"provenance_hash": "source-hash-1"}]),
            sign_sub(id_b, "r3-a-03", "Result X", [{"provenance_hash": "source-hash-2"}]),
        ],
        quorum_size=2,
    )
    assert res_indep["quorum_reached"] is True
    assert res_indep["independent_corroboration"] is True
    assert res_indep["status"] == "VERIFIED_BY_EVIDENCE_AND_QUORUM"


def test_round3_b_local_and_shared_visibility_isolated_from_anonymous():
    """Finding B: LOCAL is restricted to admin/operator; SHARED requires auth; only COLLECTIVE/PUBLIC are anonymous-readable."""
    from gajahmada_ai.memory.authorization import MemoryAuthorizationService

    authz = MemoryAuthorizationService()
    assert authz.can_read_visibility("PRIVATE", "anonymous") is False
    assert authz.can_read_visibility("LOCAL", "anonymous") is False
    assert authz.can_read_visibility("LOCAL", "peer") is False
    assert authz.can_read_visibility("LOCAL", "operator") is True
    assert authz.can_read_visibility("LOCAL", "admin") is True
    assert authz.can_read_visibility("SHARED", "anonymous") is False
    assert authz.can_read_visibility("SHARED", "peer") is True
    assert authz.can_read_visibility("COLLECTIVE", "anonymous") is True
    assert authz.can_read_visibility("PUBLIC", "anonymous") is True
    assert authz.can_read_visibility("UNKNOWN_POLICY", "admin") is False

    # End-to-end API check: LOCAL record created by admin must NOT leak to anonymous /api/v2/memory or /versions
    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key
    r_create = client.post(
        "/api/v2/memory",
        headers={"X-API-Key": admin_key},
        json={
            "content": "Internal node-local quarantine note 445566.",
            "visibility_policy": "LOCAL",
        },
    )
    assert r_create.status_code == 200
    local_id = r_create.json()["record_id"]

    r_anon = client.get("/api/v2/memory")
    assert r_anon.status_code == 200
    assert local_id not in {r["record_id"] for r in r_anon.json()["records"]}
    assert client.get(f"/api/v2/memory/{local_id}/versions").status_code == 403

    r_adm = client.get("/api/v2/memory", headers={"X-API-Key": admin_key})
    assert local_id in {r["record_id"] for r in r_adm.json()["records"]}


def test_round3_c_internal_reasoning_and_agent_endpoints_require_auth():
    """Finding C: /api/v2/reasoning, /api/v2/reasoning/{id}, /api/v2/agents require auth; health hides recent_events."""
    client = TestClient(app)
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key

    assert client.get("/api/v2/reasoning").status_code == 401
    assert client.get("/api/v2/reasoning/nonexistent-id").status_code == 401
    assert client.get("/api/v2/agents").status_code == 401

    # Authenticated admin can read internal operational endpoints
    assert client.get("/api/v2/reasoning", headers={"X-API-Key": admin_key}).status_code == 200
    assert client.get("/api/v2/agents", headers={"X-API-Key": admin_key}).status_code == 200

    # Public /api/v2/system/health omits detailed security_audit.recent_events
    r_pub_health = client.get("/api/v2/system/health")
    assert r_pub_health.status_code == 200
    assert "recent_events" not in r_pub_health.json()["security_audit"]
    assert r_pub_health.json()["ledger"]["type"] == "tamper_evident_local_signed_event_ledger"

    r_adm_health = client.get("/api/v2/system/health", headers={"X-API-Key": admin_key})
    assert "recent_events" in r_adm_health.json()["security_audit"]


def test_round3_d_gip_replay_protection_persists_across_restarts():
    """Finding D: GIP nonce replay protection is SQLite-backed and survives process restart / multiple validators."""
    from gajahmada_ai.network.protocol import GIPProtocolValidator

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "gip_replay.db")
        local_id = NodeIdentity(key_path=os.path.join(tmpdir, "local.pem"))
        peer_id = NodeIdentity(key_path=None)

        msg = GIPMessage(
            message_type=GIPMessageType.GIP_PING if hasattr(GIPMessageType, "GIP_PING") else GIPMessageType.GIP_CAPABILITY,
            sender_node_id=peer_id.node_id,
            recipient_node_id=local_id.node_id,
            payload={"endpoint": "http://127.0.0.1:9999"},
        ).sign(peer_id)

        v1 = GIPProtocolValidator(local_identity=local_id, replay_db_path=db_path)
        ok1, reason1 = v1.validate(msg)
        assert ok1 is True and reason1 == "VALID"

        # Simulate process restart / second worker reading the same replay DB
        v2 = GIPProtocolValidator(local_identity=local_id, replay_db_path=db_path)
        ok2, reason2 = v2.validate(msg)
        assert ok2 is False
        assert "Replay attack" in reason2


def test_round3_e_key_persistence_fails_closed_and_persists_rotations_and_revocations():
    """Finding E: Key write failure raises KeyPersistenceError; rotations & revocations persist across restarts."""
    from gajahmada_ai.security.crypto import KeyPersistenceError

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Unwritable path (directory path used as file) MUST raise KeyPersistenceError (fail-closed)
        bad_path = os.path.join(tmpdir, "as_dir")
        os.makedirs(bad_path, exist_ok=True)
        with pytest.raises(KeyPersistenceError):
            NodeIdentity(key_path=bad_path)

        # 2. Rotation certificates & peer revocations persist across restarts and verify dual signatures
        key_path = os.path.join(tmpdir, "node_key.pem")
        ident1 = NodeIdentity(key_path=key_path)
        orig_id = ident1.node_id
        revoked_peer = NodeIdentity(key_path=None).node_id

        ident1.revoke_peer(revoked_peer, reason="compromised_key")
        cert = ident1.rotate_key()
        assert NodeIdentity.verify_rotation_certificate(cert) is True
        assert ident1.node_id != orig_id

        # Reload from disk (restart simulation)
        ident2 = NodeIdentity(key_path=key_path)
        assert ident2.node_id == ident1.node_id
        assert ident2.key_version == 2
        assert ident2.is_revoked(orig_id) is True
        assert ident2.is_revoked(revoked_peer) is True
        assert len(ident2.rotation_history) == 1
        assert len(ident2.revocation_records) == 1

        # Independent peer can verify & apply the dual-signed rotation certificate
        peer_ident = NodeIdentity(key_path=os.path.join(tmpdir, "peer.pem"))
        assert peer_ident.apply_peer_rotation_certificate(cert) is True
        assert peer_ident.is_revoked(orig_id) is True


def test_round3_f_ledger_rejects_unsigned_or_spoofed_genesis_non_genesis_blocks():
    """Finding F: Non-genesis blocks can NEVER bypass Ed25519 verification via empty signature or Genesis Node ID."""
    from gajahmada_ai.blockchain.chain import Block

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = AIBlockchain(storage_path=os.path.join(tmpdir, "chain.json"))
        prev_hash = chain.get_latest_block().hash

        unsigned_blk = Block(
            index=1,
            timestamp=1700000001.0,
            task_hash="unsigned-attempt",
            node_id="some-node-id",
            previous_hash=prev_hash,
            signature="",
            schema_version=2,
        )
        assert chain.verify_block_signature(unsigned_blk) is False

        spoofed_genesis_blk = Block(
            index=1,
            timestamp=1700000002.0,
            task_hash="spoofed-genesis",
            node_id="Genesis Node",
            previous_hash=prev_hash,
            signature="",
            schema_version=2,
        )
        assert chain.verify_block_signature(spoofed_genesis_blk) is False
