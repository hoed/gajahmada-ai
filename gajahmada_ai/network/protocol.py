from enum import Enum
import os
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple
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


class PersistentNonceStore:
    """
    Atomic, restart-persistent, multi-worker-safe GIP replay nonce store (Round 3 Remediation).
    - Uses SQLite WAL mode with PRIMARY KEY (sender_node_id, nonce) and atomic INSERT OR IGNORE
      so concurrent workers and restarted processes share the exact same replay window.
    - Automatically evicts expired nonces whose TTL (> 2 * max_clock_skew_sec) has elapsed.
    - When `db_path` is None, falls back to a thread-safe in-memory TTL store (for ephemeral tests).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._mem_nonces: Dict[Tuple[str, str], float] = {}
        if self.db_path:
            parent = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(parent, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gip_replay_nonces (
                        sender_node_id TEXT NOT NULL,
                        nonce TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        PRIMARY KEY (sender_node_id, nonce)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gip_nonces_expires ON gip_replay_nonces (expires_at)"
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)

    def check_and_record(self, sender_node_id: str, nonce: str, ttl_seconds: float) -> bool:
        """
        Atomically check if (sender_node_id, nonce) has already been seen within the active window.
        Returns True if the nonce is fresh and was recorded; False if it is a replay.
        """
        now = time.time()
        expires_at = now + max(1.0, float(ttl_seconds))

        if not self.db_path:
            with self._lock:
                # Prune expired entries
                expired = [k for k, exp in self._mem_nonces.items() if exp <= now]
                for k in expired:
                    self._mem_nonces.pop(k, None)
                key = (sender_node_id, nonce)
                if key in self._mem_nonces:
                    return False
                self._mem_nonces[key] = expires_at
                return True

        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    conn.execute("DELETE FROM gip_replay_nonces WHERE expires_at <= ?", (now,))
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO gip_replay_nonces (sender_node_id, nonce, expires_at) VALUES (?, ?, ?)",
                        (sender_node_id, nonce, expires_at),
                    )
                    inserted = cur.rowcount == 1
                    conn.execute("COMMIT;")
                    return inserted
                except Exception:
                    conn.execute("ROLLBACK;")
                    raise


class GIPProtocolValidator:
    """
    Validates incoming GajahMada Intelligence Protocol (GIP/2.0) messages:
    - Verifies protocol version (`GIP/2.0`).
    - Checks peer revocation and optional peer allowlist.
    - Enforces timestamp freshness window (+/- 300 seconds).
    - Enforces restart-persistent, multi-worker-safe atomic nonce replay protection.
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
        replay_db_path: Optional[str] = None,
        nonce_store: Optional[PersistentNonceStore] = None,
    ):
        self.local_identity = local_identity
        self.max_clock_skew_sec = max_clock_skew_sec
        self.replay_cache_size = replay_cache_size
        self.nonce_store = nonce_store or PersistentNonceStore(db_path=replay_db_path)
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

        # Verify Ed25519 signature over canonical envelope BEFORE consuming nonce slot
        # (prevents unauthenticated nonce-table flooding)
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

        # Atomic, persistent replay check with TTL aligned to clock-skew window
        nonce_ttl = max(60.0, self.max_clock_skew_sec * 2.0)
        if not self.nonce_store.check_and_record(msg.sender_node_id, msg.nonce, ttl_seconds=nonce_ttl):
            return False, "Replay attack detected: duplicate (sender_node_id, nonce)"

        return True, "VALID"
