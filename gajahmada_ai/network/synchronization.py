from typing import Any, Dict, List

from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType
from gajahmada_ai.security.crypto import NodeIdentity


class KnowledgeSynchronizer:
    """
    Permission-Aware Collective Memory Synchronizer (Layer 4 & Layer 7).
    - Exports only `SHARED` or `COLLECTIVE` records (never `PRIVATE` or `LOCAL`).
    - Preserves original `origin_node_id` and signature when importing peer knowledge.
    """

    def __init__(self, memory: MemoryManager, identity: NodeIdentity):
        self.memory = memory
        self.identity = identity

    def build_sync_message(self, recipient_node_id: str = "*") -> GIPMessage:
        shareable = self.memory.get_shareable_records(limit=50)
        msg = GIPMessage(
            message_type=GIPMessageType.GIP_SYNC,
            sender_node_id=self.identity.node_id,
            recipient_node_id=recipient_node_id,
            payload={
                "records": [
                    {
                        "record_id": r["record_id"],
                        "content": r["content"],
                        "origin_node_id": r.get("source_provenance", {}).get("origin_node_id", self.identity.node_id),
                        "confidence": r.get("confidence", 0.75),
                        "verification_status": r.get("verification_status", "UNVERIFIED"),
                        "integrity_hash": r["integrity_hash"],
                    }
                    for r in shareable
                ]
            },
        )
        return msg.sign(self.identity)

    def ingest_sync_message(self, msg: GIPMessage) -> List[Dict[str, Any]]:
        imported = []
        for item in msg.payload.get("records", []):
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            origin_id = str(item.get("origin_node_id") or msg.sender_node_id)
            rec = self.memory.store(
                content=content,
                memory_type=MemoryType.COLLECTIVE,
                origin_node_id=origin_id,
                source_type="collective_peer_sync",
                author=f"peer:{msg.sender_node_id[:12]}",
                signature=msg.signature,
                confidence=float(item.get("confidence", 0.7)),
                verification_status=VerificationStatus.VERIFIED if item.get("verification_status") == "VERIFIED" else VerificationStatus.UNVERIFIED,
                visibility_policy=VisibilityPolicy.COLLECTIVE,
            )
            imported.append(rec)
        return imported
