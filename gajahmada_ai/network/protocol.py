from collections import deque
from enum import Enum
import secrets
import time
from typing import Any, Deque, Dict, Optional, Set, Tuple
import uuid
from pydantic import BaseModel, Field

from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


class GIPMessageType(str, Enum):
    GIP_IDENTITY = "GIP-IDENTITY"
    GIP_DISCOVERY = "GIP-DISCOVERY"
    GIP_CAPABILITY = "GIP-CAPABILITY"
    GIP_TASK = "GIP-TASK"
    GIP_KNOWLEDGE = "GIP-KNOWLEDGE"
    GIP_VERIFY = "GIP-VERIFY"
    GIP_CONSENSUS = "GIP-CONSENSUS"
    GIP_SYNC = "GIP-SYNC"


class GIPMessage(BaseModel):
    protocol_version: str = "GIP/2.0"
    message_id: str = Field(default_factory=lambda: f"gip-{uuid.uuid4().hex[:14]}")
    message_type: GIPMessageType
    sender_node_id: str
    recipient_node_id: str = "*"
    timestamp: float = Field(default_factory=lambda: round(time.time(), 4))
    nonce: str = Field(default_factory=lambda: secrets.token_hex(12))
    payload: Dict[str, Any] = Field(default_factory=dict)
    signature: str = ""

    def signable_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "message_id": self.message_id,
            "message_type": self.message_type.value if isinstance(self.message_type, GIPMessageType) else str(self.message_type),
            "sender_node_id": self.sender_node_id,
            "recipient_node_id": self.recipient_node_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "payload": self.payload,
        }

    def sign(self, identity: NodeIdentity) -> "GIPMessage":
        self.sender_node_id = identity.node_id
        self.signature = identity.sign_message(canonical_json_bytes(self.signable_dict()))
        return self


class GIPProtocolValidator:
    """
    Validates incoming GajahMada Intelligence Protocol (GIP/2.0) messages:
    - Verifies protocol version (`GIP/2.0`).
    - Checks peer revocation and optional peer allowlist.
    - Enforces timestamp freshness window (+/- 300 seconds).
    - Enforces nonce replay protection via a bounded sliding cache.
    - Verifies Ed25519 signature over canonical message envelope.
    - Rejects any payload attempting command injection or policy override.
    """

    FORBIDDEN_PAYLOAD_ACTIONS = {
        "exec_shell",
        "shell_command",
        "override_policy",
        "export_private_key",
        "delete_private_memory",
        "modify_config",
    }

    def __init__(
        self,
        local_identity: NodeIdentity,
        max_clock_skew_sec: float = 300.0,
        replay_cache_size: int = 4096,
    ):
        self.local_identity = local_identity
        self.max_clock_skew_sec = max_clock_skew_sec
        self._seen_nonces: Set[Tuple[str, str]] = set()
        self._nonce_order: Deque[Tuple[str, str]] = deque(maxlen=replay_cache_size)
        self.allowed_peers: Optional[Set[str]] = None

    def validate(self, msg: GIPMessage) -> Tuple[bool, str]:
        if msg.protocol_version != "GIP/2.0":
            return False, f"Unsupported GIP protocol version: {msg.protocol_version}"

        if not msg.sender_node_id or not msg.signature or not msg.nonce:
            return False, "Missing mandatory cryptographic fields (sender_node_id, signature, or nonce)"

        if self.local_identity.is_revoked(msg.sender_node_id):
            return False, f"Sender node '{msg.sender_node_id[:12]}...' is revoked"

        if self.allowed_peers is not None and msg.sender_node_id not in self.allowed_peers:
            return False, f"Sender node '{msg.sender_node_id[:12]}...' not in peer allowlist"

        if msg.recipient_node_id not in ("*", self.local_identity.node_id):
            return False, f"Message recipient mismatch (intended for '{msg.recipient_node_id[:12]}...')"

        now = time.time()
        if abs(now - msg.timestamp) > self.max_clock_skew_sec:
            return False, f"Message timestamp outside allowed clock skew window ({abs(now - msg.timestamp):.1f}s)"

        nonce_key = (msg.sender_node_id, msg.nonce)
        if nonce_key in self._seen_nonces:
            return False, "Replay attack detected: duplicate (sender_node_id, nonce)"

        # Verify Ed25519 signature over canonical envelope
        if not self.local_identity.verify_message(
            canonical_json_bytes(msg.signable_dict()),
            msg.signature,
            msg.sender_node_id,
        ):
            return False, "Ed25519 signature verification failed"

        # Inspect payload for malicious command/policy override attempts
        requested_action = str(msg.payload.get("action", "")).lower().strip()
        if requested_action in self.FORBIDDEN_PAYLOAD_ACTIONS:
            return False, f"Forbidden remote peer action rejected: '{requested_action}'"

        if len(self._nonce_order) == self._nonce_order.maxlen:
            oldest = self._nonce_order.popleft()
            self._seen_nonces.discard(oldest)
        self._nonce_order.append(nonce_key)
        self._seen_nonces.add(nonce_key)

        return True, "VALID"
