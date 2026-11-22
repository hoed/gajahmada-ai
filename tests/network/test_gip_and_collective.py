from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.coordinator import CollectiveCoordinator, SovereignPeerNode
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType


def test_9_node_operates_locally_when_disconnected_from_network():
    """TEST 9: A node operates locally when disconnected from the network."""
    node = SovereignPeerNode(node_name="isolated-alpha", endpoint="http://127.0.0.1:9001")
    node.memory.store(
        content="Deklarasi Djuanda 13 Desember 1957 menyatukan laut antarpulau NKRI.",
        memory_type=MemoryType.SEMANTIC,
        verification_status=VerificationStatus.VERIFIED,
    )
    # Run autonomous task scheduler with zero network peers connected
    task = node.scheduler.submit_objective("Jelaskan Deklarasi Djuanda 1957")
    assert task.state.value == "COMPLETED"
    assert "1957" in task.final_report
    assert task.confidence >= 0.70


def test_10_two_independent_nodes_authenticate_and_exchange_signed_messages():
    """TEST 10: Two independently configured nodes can authenticate and exchange signed messages."""
    node_a = SovereignPeerNode(node_name="node-a", endpoint="http://127.0.0.1:9101")
    node_b = SovereignPeerNode(node_name="node-b", endpoint="http://127.0.0.1:9102")

    assert node_a.identity.node_id != node_b.identity.node_id

    msg = GIPMessage(
        message_type=GIPMessageType.GIP_IDENTITY,
        sender_node_id=node_a.identity.node_id,
        recipient_node_id=node_b.identity.node_id,
        payload={
            "node_name": node_a.node_name,
            "endpoint": node_a.endpoint,
            "capabilities": node_a.capabilities,
        },
    ).sign(node_a.identity)

    resp = node_b.receive_gip_message(msg)
    assert resp["accepted"] is True
    reply_msg = GIPMessage(**resp["reply"])

    # Node A validates Node B's signed reply
    valid, reason = node_a.validator.validate(reply_msg)
    assert valid is True
    assert reason == "VALID"

    # Replaying the exact same message to Node B must be rejected
    replay_resp = node_b.receive_gip_message(msg)
    assert replay_resp["accepted"] is False
    assert "Replay attack" in replay_resp["reason"]


def test_11_three_independent_nodes_coordinate_and_verify_distributed_task():
    """TEST 11: Three independent nodes can coordinate and verify a distributed task."""
    node_a = SovereignPeerNode(node_name="coordinator-a", endpoint="http://127.0.0.1:9201")
    node_b = SovereignPeerNode(node_name="historical-b", endpoint="http://127.0.0.1:9202")
    node_c = SovereignPeerNode(node_name="verifier-c", endpoint="http://127.0.0.1:9203")

    # Seed verified historical evidence in Node B and Node C
    evidence_text = "Sumpah Palapa diucapkan oleh Mahapatih Gajah Mada pada tahun 1336 M untuk menyatukan Nusantara."
    for peer in (node_b, node_c):
        peer.memory.store(
            content=evidence_text,
            memory_type=MemoryType.SEMANTIC,
            origin_node_id=peer.identity.node_id,
            source_type="sovereign_core_corpus",
            confidence=0.92,
            verification_status=VerificationStatus.VERIFIED,
            visibility_policy=VisibilityPolicy.COLLECTIVE,
        )

    coordinator = CollectiveCoordinator(node_a)
    result = coordinator.coordinate_distributed_task(
        objective="Kapan Sumpah Palapa diucapkan oleh Mahapatih Gajah Mada?",
        peers=[node_b, node_c],
        quorum_size=2,
    )

    assert result["quorum_reached"] is True
    assert result["participating_nodes"] == 2
    assert result["status"] == "VERIFIED_BY_EVIDENCE_AND_QUORUM"
    assert result["confidence"] >= 0.75
    assert result["collective_memory_record_id"] is not None

    # Verify collective memory in Node A preserved original peer attribution
    stored_rec = node_a.memory.storage.get_record(result["collective_memory_record_id"])
    assert stored_rec is not None
    assert stored_rec["memory_type"] == "collective"
    assert stored_rec["source_provenance"]["origin_node_id"] in (
        node_b.identity.node_id,
        node_c.identity.node_id,
    )
