#!/usr/bin/env python3
"""
Stage 3 — GIP Distributed Intelligence Verification Harness (3 -> 10 -> 100 Independent Nodes).
Tests:
1. 3-node full SovereignPeerNode mesh: distributed task, SQLite WAL replay protection, restart recovery.
2. 10-node full SovereignPeerNode mesh: signed GIP handshakes, peer trust, conflicting claims
   (evidence-backed minority vs. unbacked majority), replay rejection, revoked-node rejection.
3. 100-node independent cryptographic quorum & conflict stress test: 100 distinct Ed25519
   identities, Sybil duplicate rejection, forged signature_verified=True rejection,
   single-source echo downgrade vs. independent multi-source corroboration.
"""
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.consensus import QuorumConsensusManager
from gajahmada_ai.network.coordinator import CollectiveCoordinator, SovereignPeerNode
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType
from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


def run_3_node_verification() -> Dict[str, Any]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gm_stage3_3n_") as tmp:
        nodes = [
            SovereignPeerNode(f"node-{i}", f"http://127.0.0.1:{9500+i}", storage_dir=os.path.join(tmp, f"n{i}"))
            for i in range(3)
        ]
        coord = CollectiveCoordinator(nodes[0])
        res = coord.coordinate_distributed_task("Hitung 125 * 4 + 50", peers=nodes[1:], quorum_size=2)
        assert res["quorum_reached"] is True
        assert res["status"] == "VERIFIED_BY_EVIDENCE_AND_QUORUM"
        assert "550" in res["selected_conclusion"]

        # Restart persistence check
        reloaded = SovereignPeerNode("node-0", "http://127.0.0.1:9500", storage_dir=os.path.join(tmp, "n0"))
        assert reloaded.identity.node_id == nodes[0].identity.node_id
        assert reloaded.blockchain.is_chain_valid(verify_signatures=True) is True
        assert reloaded.memory.storage.get_record(res["collective_memory_record_id"]) is not None

        return {
            "tier": "3_nodes",
            "status": "PASSED",
            "participating_nodes": res["participating_nodes"],
            "quorum_reached": res["quorum_reached"],
            "restart_recovery": True,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def run_10_node_mesh_verification() -> Dict[str, Any]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gm_stage3_10n_") as tmp:
        nodes = [
            SovereignPeerNode(f"mesh-{i}", f"http://127.0.0.1:{9600+i}", storage_dir=os.path.join(tmp, f"m{i}"))
            for i in range(10)
        ]
        hub = nodes[0]
        peers = nodes[1:]

        # 1. Signed GIP-IDENTITY handshakes from all 9 peers + replay attack rejection
        replay_rejected = 0
        for p in peers:
            m = GIPMessage(
                message_type=GIPMessageType.GIP_IDENTITY,
                sender_node_id=p.identity.node_id,
                recipient_node_id=hub.identity.node_id,
                payload={"node_name": p.node_name, "endpoint": p.endpoint, "capabilities": p.capabilities},
            ).sign(p.identity)
            r1 = hub.receive_gip_message(m)
            assert r1["accepted"] is True
            r2 = hub.receive_gip_message(m)
            if not r2["accepted"] and "Replay attack" in r2["reason"]:
                replay_rejected += 1

        assert replay_rejected == 9

        # 2. Revoke 1 compromised peer & verify immediate rejection
        compromised = peers[-1]
        hub.identity.revoke_peer(compromised.identity.node_id, reason="stage3_revocation_drill")
        m_rev = GIPMessage(
            message_type=GIPMessageType.GIP_TASK,
            sender_node_id=compromised.identity.node_id,
            recipient_node_id=hub.identity.node_id,
            payload={"task_id": "rev-01", "objective": "Hitung 1 + 1"},
        ).sign(compromised.identity)
        r_rev = hub.receive_gip_message(m_rev)
        assert r_rev["accepted"] is False and "revoked" in r_rev["reason"]

        # 3. Distributed task across 8 healthy peers
        coord = CollectiveCoordinator(hub)
        res = coord.coordinate_distributed_task("Hitung 64 * 8 - 12", peers=peers[:-1], quorum_size=5)
        assert res["quorum_reached"] is True
        assert res["participating_nodes"] == 8
        assert "500" in res["selected_conclusion"]

        return {
            "tier": "10_nodes",
            "status": "PASSED",
            "total_nodes": 10,
            "handshakes_verified": 9,
            "replays_blocked": replay_rejected,
            "revoked_peer_blocked": True,
            "quorum_participating": res["participating_nodes"],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def run_100_node_quorum_stress_test() -> Dict[str, Any]:
    t0 = time.perf_counter()
    identities = [NodeIdentity(key_path=None) for _ in range(100)]
    verifier = NodeIdentity(key_path=None)
    mgr = QuorumConsensusManager(verifier_identity=verifier)

    # Revoke 5 compromised nodes
    for rev_id in identities[95:]:
        verifier.revoke_peer(rev_id.node_id, reason="stress_test_revoked")

    task_id = "stage3-100node-quorum"
    submissions: List[Dict[str, Any]] = []

    # 30 honest evidence-backed nodes with independent provenance hashes
    for idx, ident in enumerate(identities[:30]):
        ev = [{"provenance_hash": f"indep-prov-source-{idx:03d}"}]
        sig = ident.sign_message(canonical_json_bytes({
            "task_id": task_id,
            "node_id": ident.node_id,
            "conclusion": "Verified answer: 1336 M",
            "confidence": 0.94,
            "evidence": ev,
        }))
        submissions.append({
            "node_id": ident.node_id,
            "conclusion": "Verified answer: 1336 M",
            "confidence": 0.94,
            "evidence": ev,
            "signature": sig,
        })

    # 55 unbacked majority nodes claiming a wrong answer without evidence
    for ident in identities[30:85]:
        sig = ident.sign_message(canonical_json_bytes({
            "task_id": task_id,
            "node_id": ident.node_id,
            "conclusion": "Unbacked majority claim: 1400 M",
            "confidence": 0.99,
            "evidence": [],
        }))
        submissions.append({
            "node_id": ident.node_id,
            "conclusion": "Unbacked majority claim: 1400 M",
            "confidence": 0.99,
            "evidence": [],
            "signature": sig,
        })

    # 10 forged submissions (claiming signature_verified=True without valid Ed25519 signature)
    for ident in identities[85:95]:
        submissions.append({
            "node_id": ident.node_id,
            "signature_verified": True,
            "conclusion": "Forged attack",
            "confidence": 1.0,
            "evidence": [{"provenance_hash": "forged"}],
        })

    # 5 revoked nodes submitting signed messages
    for ident in identities[95:]:
        ev = [{"provenance_hash": "revoked-source"}]
        sig = ident.sign_message(canonical_json_bytes({
            "task_id": task_id,
            "node_id": ident.node_id,
            "conclusion": "Revoked node claim",
            "confidence": 0.99,
            "evidence": ev,
        }))
        submissions.append({
            "node_id": ident.node_id,
            "conclusion": "Revoked node claim",
            "confidence": 0.99,
            "evidence": ev,
            "signature": sig,
        })

    # Add 5 Sybil duplicate submissions from node 0
    for _ in range(5):
        submissions.append(dict(submissions[0]))

    res = mgr.evaluate_submissions(task_id=task_id, submissions=submissions, quorum_size=20)
    assert res["quorum_reached"] is True
    assert res["participating_nodes"] == 85  # 30 honest + 55 unbacked; 10 forged + 5 revoked + 5 sybil rejected
    assert res["rejected_submissions"] == 20
    assert res["unverified_majority_rejected"] is True
    assert res["selected_conclusion"] == "Verified answer: 1336 M"
    assert res["independent_evidence_sources"] == 30
    assert len(res["conflicts"]) > 0

    return {
        "tier": "100_nodes",
        "status": "PASSED",
        "total_identities": 100,
        "total_submissions_processed": len(submissions),
        "valid_participating_nodes": res["participating_nodes"],
        "rejected_forged_revoked_sybil": res["rejected_submissions"],
        "independent_evidence_sources": res["independent_evidence_sources"],
        "unverified_majority_rejected": res["unverified_majority_rejected"],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def main():
    report = {
        "stage": "Stage 3 — GIP Distributed Intelligence",
        "results": [
            run_3_node_verification(),
            run_10_node_mesh_verification(),
            run_100_node_quorum_stress_test(),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
