from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set


class PeerTrustState(str, Enum):
    """Explicit peer trust lifecycle (P1 Remediation): Ed25519 signature != network trust."""
    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    AUTHENTICATED = "AUTHENTICATED"
    TRUSTED = "TRUSTED"
    REVOKED = "REVOKED"


@dataclass
class PeerDescriptor:
    node_id: str
    endpoint: str
    node_name: str = "gajahmada-peer"
    capabilities: List[str] = field(default_factory=lambda: ["symbolic", "evidence", "memory"])
    reputation_advisory: float = 0.75  # Advisory metadata only, never treated as truth guarantee
    trust_state: PeerTrustState = PeerTrustState.PENDING
    last_seen: float = field(default_factory=time.time)
    revoked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "node_name": self.node_name,
            "capabilities": self.capabilities,
            "reputation_advisory": round(self.reputation_advisory, 3),
            "trust_state": self.trust_state.value if isinstance(self.trust_state, PeerTrustState) else str(self.trust_state),
            "last_seen": self.last_seen,
            "revoked": self.revoked,
        }


class PeerDiscoveryRegistry:
    """
    Layer 2 & Layer 7 Peer Discovery, Trust State Machine, and Capability Registry.
    Enforces: UNKNOWN -> PENDING -> AUTHENTICATED -> TRUSTED -> REVOKED.
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
        trust_state: PeerTrustState = PeerTrustState.PENDING,
    ) -> PeerDescriptor:
        if node_id in self._revoked_ids:
            raise PermissionError(f"Cannot register revoked peer node_id '{node_id[:12]}...'")
        existing = self._peers_by_id.get(node_id)
        # Do not silently elevate an existing PENDING peer unless explicitly requested as TRUSTED
        effective_trust = existing.trust_state if (existing and trust_state == PeerTrustState.PENDING) else trust_state
        desc = PeerDescriptor(
            node_id=node_id,
            endpoint=endpoint.strip().rstrip("/"),
            node_name=node_name,
            capabilities=capabilities or ["symbolic", "evidence", "memory"],
            trust_state=effective_trust,
            last_seen=time.time(),
            revoked=False,
        )
        self._peers_by_id[node_id] = desc
        return desc

    def set_trust_state(self, node_id: str, state: PeerTrustState) -> PeerDescriptor:
        if node_id not in self._peers_by_id:
            raise KeyError(f"Peer '{node_id[:12]}...' not found in discovery registry.")
        if state == PeerTrustState.REVOKED:
            self.revoke_peer(node_id)
        else:
            self._peers_by_id[node_id].trust_state = state
            self._peers_by_id[node_id].revoked = False
        return self._peers_by_id[node_id]

    def get_trust_state(self, node_id: str) -> PeerTrustState:
        if node_id in self._revoked_ids:
            return PeerTrustState.REVOKED
        peer = self._peers_by_id.get(node_id)
        return peer.trust_state if peer else PeerTrustState.UNKNOWN

    def is_trusted(self, node_id: str) -> bool:
        return self.get_trust_state(node_id) == PeerTrustState.TRUSTED

    def revoke_peer(self, node_id: str):
        self._revoked_ids.add(node_id)
        if node_id in self._peers_by_id:
            self._peers_by_id[node_id].revoked = True
            self._peers_by_id[node_id].trust_state = PeerTrustState.REVOKED

    def find_peers_with_capability(self, capability: str) -> List[PeerDescriptor]:
        return [
            p for p in self._peers_by_id.values()
            if not p.revoked and p.trust_state == PeerTrustState.TRUSTED and capability in p.capabilities
        ]

    def list_active_peers(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._peers_by_id.values() if not p.revoked]
