import tempfile
from typing import Any, Dict, List, Optional

from gajahmada_ai.agents.scheduler import AutonomousTaskScheduler
from gajahmada_ai.blockchain.chain import AIBlockchain, LedgerEventType
from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.consensus import QuorumConsensusManager
from gajahmada_ai.network.discovery import PeerDiscoveryRegistry
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType, GIPProtocolValidator
from gajahmada_ai.network.synchronization import KnowledgeSynchronizer
from gajahmada_ai.security.audit import SecurityAuditLogger
from gajahmada_ai.security.crypto import NodeIdentity
from gajahmada_ai.security.policies import GovernancePolicy


class SovereignPeerNode:
    """
    Self-contained Sovereign GajahMada 2.0 Node instance (Layer 7).
    Maintains its own independent:
    - Cryptographic Ed25519 identity (`identity`)
    - Persistent memory store (`memory`)
    - Hybrid cognitive reasoning pipeline (`cognition`)
    - Autonomous task scheduler (`scheduler`)
    - Signed event blockchain ledger (`blockchain`)
    - Governance policies & GIP protocol validator (`validator`)
    """

    def __init__(
        self,
        node_name: str,
        endpoint: str,
        storage_dir: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ):
        self.node_name = node_name
        self.endpoint = endpoint.strip().rstrip("/")
        self.storage_dir = storage_dir or tempfile.mkdtemp(prefix=f"gm_{node_name}_")
        self.capabilities = capabilities or ["research", "symbolic", "evidence", "verification", "synthesis"]

        self.identity = NodeIdentity(key_path=f"{self.storage_dir}/node_key.pem")
        self.audit_logger = SecurityAuditLogger(storage_path=f"{self.storage_dir}/security_audit.json")
        self.policy = GovernancePolicy()
        self.blockchain = AIBlockchain(storage_path=f"{self.storage_dir}/chain.json")
        self.memory = MemoryManager(
            sqlite_path=f"{self.storage_dir}/gajahmada_v2.db",
            backup_dir=f"{self.storage_dir}/objects",
        )
        self.cognition = CognitiveReasoningPipeline()
        self.scheduler = AutonomousTaskScheduler(
            memory_manager=self.memory,
            cognitive_pipeline=self.cognition,
            policy=self.policy,
            audit_logger=self.audit_logger,
        )
        self.validator = GIPProtocolValidator(local_identity=self.identity)
        self.discovery = PeerDiscoveryRegistry()
        self.synchronizer = KnowledgeSynchronizer(self.memory, self.identity)

        self.blockchain.record_event(
            LedgerEventType.NODE_REGISTERED,
            f"{self.node_name}|{self.identity.node_id}",
            self.identity,
        )

    def receive_gip_message(self, msg: GIPMessage) -> Dict[str, Any]:
        valid, reason = self.validator.validate(msg)
        if not valid:
            self.audit_logger.record(
                event_type="GIP_MESSAGE_REJECTED",
                severity="HIGH",
                actor=msg.sender_node_id or "unknown",
                action=f"receive:{msg.message_type}",
                outcome="REJECTED",
                details={"reason": reason, "message_id": msg.message_id},
            )
            return {
                "accepted": False,
                "node_id": self.identity.node_id,
                "reason": reason,
            }

        if msg.message_type == GIPMessageType.GIP_IDENTITY or msg.message_type == GIPMessageType.GIP_DISCOVERY:
            ep = msg.payload.get("endpoint", "http://127.0.0.1:8000")
            caps = msg.payload.get("capabilities", ["symbolic", "evidence"])
            self.discovery.register_peer(
                node_id=msg.sender_node_id,
                endpoint=ep,
                node_name=msg.payload.get("node_name", "peer"),
                capabilities=caps,
            )
            reply = GIPMessage(
                message_type=GIPMessageType.GIP_CAPABILITY,
                sender_node_id=self.identity.node_id,
                recipient_node_id=msg.sender_node_id,
                payload={
                    "node_name": self.node_name,
                    "endpoint": self.endpoint,
                    "capabilities": self.capabilities,
                },
            ).sign(self.identity)
            return {"accepted": True, "node_id": self.identity.node_id, "reply": reply.model_dump()}

        if msg.message_type == GIPMessageType.GIP_TASK:
            objective = str(msg.payload.get("objective", "")).strip()
            task_id = str(msg.payload.get("task_id", msg.message_id))
            if not objective:
                return {"accepted": False, "node_id": self.identity.node_id, "reason": "Empty task objective"}

            self.blockchain.record_event(
                LedgerEventType.TASK_ACCEPTED,
                f"{task_id}|{objective[:64]}",
                self.identity,
            )

            # Retrieve local memory & execute cognitive pipeline
            records = self.memory.retrieve(query=objective, top_k=4)
            res = self.cognition.execute(objective=objective, memory_records=records)

            self.blockchain.record_event(
                LedgerEventType.EVIDENCE_SUBMITTED,
                f"{task_id}|ev={len(res.evidence)}|conf={res.confidence}",
                self.identity,
            )

            reply = GIPMessage(
                message_type=GIPMessageType.GIP_VERIFY,
                sender_node_id=self.identity.node_id,
                recipient_node_id=msg.sender_node_id,
                payload={
                    "task_id": task_id,
                    "node_name": self.node_name,
                    "conclusion": res.conclusion,
                    "confidence": res.confidence,
                    "verification_status": res.verification_status,
                    "symbolic_verified": bool(res.capabilities_used.get("symbolic_used_in_this_call")),
                    "evidence": [e.model_dump() for e in res.evidence],
                },
            ).sign(self.identity)
            return {"accepted": True, "node_id": self.identity.node_id, "reply": reply.model_dump()}

        if msg.message_type == GIPMessageType.GIP_SYNC:
            imported = self.synchronizer.ingest_sync_message(msg)
            return {
                "accepted": True,
                "node_id": self.identity.node_id,
                "imported_count": len(imported),
            }

        return {"accepted": True, "node_id": self.identity.node_id, "status": "acknowledged"}


class CollectiveCoordinator:
    """
    Layer 7 Distributed Task Coordinator.
    Coordinates multiple independent SovereignPeerNodes through the 11-step workflow:
    1. Receives objective -> 2. Decomposes -> 3. Identifies peers ->
    4. Sends signed GIP-TASK requests -> 5. Peers independently accept/reject ->
    6. Peers execute task -> 7. Peers submit signed GIP-VERIFY evidence & results ->
    8. Coordinator verifies signatures & compares submissions ->
    9. Conflicts trigger evaluation -> 10. Verified findings assembled ->
    11. Authorized findings enter collective memory & signed ledger.
    """

    def __init__(self, coordinator_node: SovereignPeerNode):
        self.coordinator = coordinator_node
        self.consensus = QuorumConsensusManager()

    def coordinate_distributed_task(
        self,
        objective: str,
        peers: List[SovereignPeerNode],
        quorum_size: int = 2,
    ) -> Dict[str, Any]:
        task_id = f"dist-tsk-{self.coordinator.identity.node_id[:8]}"
        self.coordinator.blockchain.record_event(
            LedgerEventType.TASK_CREATED,
            f"{task_id}|{objective[:80]}",
            self.coordinator.identity,
        )

        # Step 1-4: Send signed GIP-TASK messages to each peer
        submissions: List[Dict[str, Any]] = []
        for peer in peers:
            task_msg = GIPMessage(
                message_type=GIPMessageType.GIP_TASK,
                sender_node_id=self.coordinator.identity.node_id,
                recipient_node_id=peer.identity.node_id,
                payload={
                    "task_id": task_id,
                    "objective": objective,
                },
            ).sign(self.coordinator.identity)

            resp = peer.receive_gip_message(task_msg)
            if not resp.get("accepted") or "reply" not in resp:
                continue

            reply_msg = GIPMessage(**resp["reply"])
            # Coordinator independently verifies peer's signed GIP-VERIFY reply
            valid, _ = self.coordinator.validator.validate(reply_msg)
            if valid:
                submissions.append({
                    "node_id": reply_msg.sender_node_id,
                    "node_name": reply_msg.payload.get("node_name", "peer"),
                    "signature_verified": True,
                    "conclusion": reply_msg.payload.get("conclusion", ""),
                    "confidence": reply_msg.payload.get("confidence", 0.5),
                    "verification_status": reply_msg.payload.get("verification_status", "UNVERIFIED"),
                    "symbolic_verified": reply_msg.payload.get("symbolic_verified", False),
                    "evidence": reply_msg.payload.get("evidence", []),
                })

        # Step 8-10: Evaluate quorum & evidence independence
        evaluation = self.consensus.evaluate_submissions(
            task_id=task_id,
            submissions=submissions,
            quorum_size=quorum_size,
        )

        self.coordinator.blockchain.record_event(
            LedgerEventType.RESULT_VERIFIED,
            f"{task_id}|status={evaluation['status']}|conf={evaluation['confidence']}",
            self.coordinator.identity,
        )

        # Step 11: Authorized findings enter collective memory if confidence >= policy threshold
        stored_collective = None
        if (
            evaluation["quorum_reached"]
            and evaluation["confidence"] >= self.coordinator.policy.min_confidence_for_collective_memory
            and evaluation["selected_conclusion"]
        ):
            stored_collective = self.coordinator.memory.store(
                content=evaluation["selected_conclusion"],
                memory_type=MemoryType.COLLECTIVE,
                origin_node_id=evaluation["selected_node_id"],
                source_type="collective_quorum_verified",
                author=f"node:{evaluation['selected_node_id'][:12]}",
                confidence=evaluation["confidence"],
                verification_status=VerificationStatus.VERIFIED,
                visibility_policy=VisibilityPolicy.COLLECTIVE,
            )
            self.coordinator.blockchain.record_event(
                LedgerEventType.KNOWLEDGE_ACCEPTED,
                f"{task_id}|rec={stored_collective['record_id']}",
                self.coordinator.identity,
            )
        else:
            self.coordinator.blockchain.record_event(
                LedgerEventType.KNOWLEDGE_REJECTED,
                f"{task_id}|reason={evaluation['status']}",
                self.coordinator.identity,
            )

        evaluation["collective_memory_record_id"] = stored_collective["record_id"] if stored_collective else None
        return evaluation
