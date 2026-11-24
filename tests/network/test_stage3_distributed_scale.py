"""
Stage 3 Automated Tests: 3-node, 10-node, and 100-node GIP Distributed Intelligence.
"""
from scripts.run_gip_network_simulation import (
    run_3_node_verification,
    run_10_node_mesh_verification,
    run_100_node_quorum_stress_test,
)


def test_stage3_3_node_distributed_and_restart_recovery():
    res = run_3_node_verification()
    assert res["status"] == "PASSED"
    assert res["quorum_reached"] is True
    assert res["restart_recovery"] is True


def test_stage3_10_node_mesh_security_replay_and_revocation():
    res = run_10_node_mesh_verification()
    assert res["status"] == "PASSED"
    assert res["replays_blocked"] == 9
    assert res["revoked_peer_blocked"] is True
    assert res["quorum_participating"] == 8


def test_stage3_100_node_cryptographic_quorum_and_sybil_resistance():
    res = run_100_node_quorum_stress_test()
    assert res["status"] == "PASSED"
    assert res["valid_participating_nodes"] == 85
    assert res["rejected_forged_revoked_sybil"] == 20
    assert res["unverified_majority_rejected"] is True
    assert res["independent_evidence_sources"] == 30
