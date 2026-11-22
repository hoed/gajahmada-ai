from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set


@dataclass
class PeerDescriptor:
    node_id: str
    endpoint: str
    node_name: str = "gajahmada-peer"
    capabilities: List[str] = field(default_factory=lambda: ["symbolic", "evidence", "memory"])
    reputation_advisory: float = 0.75  # Advisory metadata only, never treated as truth guarantee
    last_seen: float = field(default_factory=time.time)
    revoked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "node_name": self.node_name,
            "capabilities": self.capabilities,
            "reputation_advisory": round(self.reputation_advisory, 3),
            "last_seen": self.last_seen,
            "revoked": self.revoked,
        }


class PeerDiscoveryRegistry:
    """
    Layer 2 & Layer 7 Peer Discovery, Capability Registry, and Advisory Reputation Tracker.
    Note: Peer reputation is stored strictly as advisory routing metadata—never used
    to bypass cryptographic or evidence verification.
    """

    def __init__(self):
        self._peers_by_id: Dict[str, PeerDescriptor] = {}
        self._revoked_ids: Set[str] = set()

    def register_peer(
        self,
        node_id: str,
        endpoint: str,
        node_name: str = "gajahmada-peer",
        capabilities: Optional[List[str]] = None,
    ) -> PeerDescriptor:
        if node_id in self._revoked_ids:
            raise PermissionError(f"Cannot register revoked peer node_id '{node_id[:12]}...'")
        desc = PeerDescriptor(
            node_id=node_id,
            endpoint=endpoint.strip().rstrip("/"),
            node_name=node_name,
            capabilities=capabilities or ["symbolic", "evidence", "memory"],
            last_seen=time.time(),
            revoked=False,
        )
        self._peers_by_id[node_id] = desc
        return desc

    def revoke_peer(self, node_id: str):
        self._revoked_ids.add(node_id)
        if node_id in self._peers_by_id:
            self._peers_by_id[node_id].revoked = True

    def find_peers_with_capability(self, capability: str) -> List[PeerDescriptor]:
        return [
            p for p in self._peers_by_id.values()
            if not p.revoked and capability in p.capabilities
        ]

    def list_active_peers(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._peers_by_id.values() if not p.revoked]
