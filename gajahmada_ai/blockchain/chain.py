from enum import Enum
import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


class LedgerEventType(str, Enum):
    GENESIS = "GENESIS"
    NODE_REGISTERED = "NODE_REGISTERED"
    TASK_CREATED = "TASK_CREATED"
    TASK_ACCEPTED = "TASK_ACCEPTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    EVIDENCE_SUBMITTED = "EVIDENCE_SUBMITTED"
    RESULT_VERIFIED = "RESULT_VERIFIED"
    KNOWLEDGE_PROPOSED = "KNOWLEDGE_PROPOSED"
    KNOWLEDGE_ACCEPTED = "KNOWLEDGE_ACCEPTED"
    KNOWLEDGE_REJECTED = "KNOWLEDGE_REJECTED"
    MODEL_VERSION_CHANGED = "MODEL_VERSION_CHANGED"
    AUDIT_BLOCK = "AUDIT_BLOCK"


class LedgerCorruptedError(RuntimeError):
    """Raised when attempting to append to a ledger in LEDGER_CORRUPTED state."""


class Block:
    """
    Tamper-evident Signed Event Block (Schema v2 with legacy v1 verification support).
    - Schema v2 hashes & signs the FULL canonical event envelope including
      `event_type`, `event_ref`, `metadata`, `timestamp`, `node_id`, and `previous_hash`.
    - Legacy schema v1 blocks (without `schema_version >= 2`) remain verifiable under
      their original hash/signature formula for backward compatibility.
    """

    def __init__(
        self,
        index: int,
        timestamp: float,
        task_hash: str,
        node_id: str,
        previous_hash: str,
        signature: str = "",
        nonce: int = 0,
        block_hash: Optional[str] = None,
        event_type: str = LedgerEventType.TASK_COMPLETED.value,
        event_ref: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        schema_version: int = 2,
    ):
        self.schema_version = int(schema_version)
        self.index = index
        self.timestamp = timestamp
        self.task_hash = task_hash
        self.node_id = node_id
        self.previous_hash = previous_hash
        self.signature = signature
        self.nonce = nonce
        self.event_type = event_type
        self.event_ref = event_ref
        self.metadata = metadata or {}
        self.hash = block_hash if block_hash else self.calculate_hash()

    def signable_envelope_bytes(self) -> bytes:
        """Canonical byte serialization of the event payload signed by Ed25519."""
        envelope = {
            "schema_version": self.schema_version,
            "index": self.index,
            "timestamp": self.timestamp,
            "task_hash": self.task_hash,
            "node_id": self.node_id,
            "previous_hash": self.previous_hash,
            "event_type": self.event_type,
            "event_ref": self.event_ref,
            "metadata": self.metadata,
        }
        return canonical_json_bytes(envelope)

    def calculate_hash(self) -> str:
        if self.schema_version < 2:
            # Historical v1 block verification formula
            block_string = (
                f"{self.index}|{self.timestamp}|{self.task_hash}|"
                f"{self.node_id}|{self.previous_hash}|{self.signature}|{self.nonce}"
            )
            return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

        # Schema v2: Canonical hash covering EVERY block & event metadata field
        canonical_payload = {
            "schema_version": self.schema_version,
            "index": self.index,
            "timestamp": self.timestamp,
            "task_hash": self.task_hash,
            "node_id": self.node_id,
            "previous_hash": self.previous_hash,
            "event_type": self.event_type,
            "event_ref": self.event_ref,
            "metadata": self.metadata,
            "signature": self.signature,
            "nonce": self.nonce,
        }
        return hashlib.sha256(canonical_json_bytes(canonical_payload)).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "index": self.index,
            "timestamp": self.timestamp,
            "task_hash": self.task_hash,
            "node_id": self.node_id,
            "previous_hash": self.previous_hash,
            "signature": self.signature,
            "nonce": self.nonce,
            "hash": self.hash,
            "event_type": self.event_type,
        }
        if self.event_ref:
            d["event_ref"] = self.event_ref
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        # Legacy blocks serialized before P1 lack `schema_version` -> verify as v1
        sv = int(data.get("schema_version", 1))
        return cls(
            index=data["index"],
            timestamp=data["timestamp"],
            task_hash=data["task_hash"],
            node_id=data["node_id"],
            previous_hash=data["previous_hash"],
            signature=data.get("signature", ""),
            nonce=data.get("nonce", 0),
            block_hash=data.get("hash"),
            event_type=data.get(
                "event_type",
                LedgerEventType.GENESIS.value if data["index"] == 0 else LedgerEventType.TASK_COMPLETED.value,
            ),
            event_ref=data.get("event_ref", ""),
            metadata=data.get("metadata", {}),
            schema_version=sv,
        )


class AIBlockchain:
    """
    Tamper-Evident Local Signed Event & Cryptographic Audit Ledger (P1 + Round 3 Remediated).
    Note: This is a local tamper-evident signed event ledger (hash-chained, Ed25519-signed,
    lightweight PoW integrity seal), NOT a Byzantine-fault-tolerant public distributed
    consensus blockchain.
    - Canonical JSON block hashing covering event_type, event_ref, and metadata.
    - Strict Ed25519 signature verification on ALL non-genesis blocks (`index > 0`);
      unsigned blocks and `Genesis Node` exemptions are strictly forbidden for `index > 0`.
    - Atomic persistence (write temp + fsync + backup + os.replace).
    - Explicit LEDGER_CORRUPTED mode on integrity failure (never silently wipes chain).
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.difficulty = 2
        self.storage_path = storage_path
        self.chain: List[Block] = []
        self.status: str = "HEALTHY"
        self.corruption_error: Optional[str] = None
        self._verifier = NodeIdentity(key_path=None)

        if self.storage_path and os.path.exists(self.storage_path):
            self.load()
        elif not self.chain:
            self.chain = [self.create_genesis_block()]
            self.save()

    def create_genesis_block(self) -> Block:
        return Block(
            0,
            time.time(),
            "Genesis Task",
            "Genesis Node",
            "0",
            event_type=LedgerEventType.GENESIS.value,
            schema_version=2,
        )

    def get_latest_block(self) -> Block:
        if not self.chain:
            raise LedgerCorruptedError(self.corruption_error or "Ledger is empty or corrupted.")
        return self.chain[-1]

    def add_block(self, new_block: Block) -> Block:
        if self.status == "LEDGER_CORRUPTED":
            raise LedgerCorruptedError(
                f"Cannot append block while ledger is in LEDGER_CORRUPTED state: {self.corruption_error}"
            )
        new_block.index = len(self.chain)
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        target_prefix = "0" * self.difficulty
        while not new_block.hash.startswith(target_prefix):
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        self.save()
        return new_block

    def record_event(
        self,
        event_type: LedgerEventType,
        content_reference: str,
        identity: NodeIdentity,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Block:
        """Record a typed, full-envelope Ed25519 signed audit event on the ledger."""
        if self.status == "LEDGER_CORRUPTED":
            raise LedgerCorruptedError(
                f"Cannot record event while ledger is in LEDGER_CORRUPTED state: {self.corruption_error}"
            )
        t0 = time.time()
        ev_name = event_type.value if isinstance(event_type, LedgerEventType) else str(event_type)
        meta = metadata or {}
        next_idx = len(self.chain)
        prev_hash = self.get_latest_block().hash
        task_hash = hashlib.sha256(
            canonical_json_bytes({"event_type": ev_name, "ref": content_reference, "ts": t0, "metadata": meta})
        ).hexdigest()

        block = Block(
            index=next_idx,
            timestamp=t0,
            task_hash=task_hash,
            node_id=identity.node_id,
            previous_hash=prev_hash,
            signature="",
            event_type=ev_name,
            event_ref=content_reference[:128],
            metadata=meta,
            schema_version=2,
        )
        block.signature = identity.sign_message(block.signable_envelope_bytes())
        return self.add_block(block)

    def _is_valid_genesis_block(self, block: Block) -> bool:
        return (
            block.index == 0
            and block.node_id == "Genesis Node"
            and block.previous_hash == "0"
            and block.event_type == LedgerEventType.GENESIS.value
            and block.hash == block.calculate_hash()
        )

    def verify_block_signature(self, block: Block) -> bool:
        # Round 3 Hardening: ONLY the authentic genesis block (index == 0) is exempt
        # from carrying an Ed25519 signature. Non-genesis blocks can NEVER bypass
        # signature verification via empty signature or spoofed "Genesis Node" node_id.
        if block.index == 0:
            return self._is_valid_genesis_block(block)
        if not block.signature or not block.node_id or block.node_id == "Genesis Node":
            return False
        if block.schema_version >= 2:
            # P0 Strict Enforcement: Schema v2 blocks MUST carry a valid full-envelope Ed25519 signature.
            # Legacy task_hash-only fallback is strictly prohibited on schema_version >= 2.
            return self._verifier.verify_message(block.signable_envelope_bytes(), block.signature, block.node_id)
        # Historical schema v1 blocks only
        return self._verifier.verify_message(block.task_hash.encode("utf-8"), block.signature, block.node_id)

    def is_chain_valid(self, verify_signatures: bool = True) -> bool:
        if self.status == "LEDGER_CORRUPTED" or not self.chain:
            return False
        if not self._is_valid_genesis_block(self.chain[0]):
            return False
        target_prefix = "0" * self.difficulty
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.index != previous.index + 1:
                return False
            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
            if not current.hash.startswith(target_prefix):
                return False
            if verify_signatures and not self.verify_block_signature(current):
                return False
        return True

    def save(self):
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        tmp_path = f"{self.storage_path}.tmp"
        bak_path = f"{self.storage_path}.bak"
        if os.path.exists(self.storage_path) and self.status == "HEALTHY":
            try:
                shutil.copy2(self.storage_path, bak_path)
            except Exception:
                pass
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.storage_path)

    def load(self):
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list) or not raw:
                raise ValueError("Ledger JSON is empty or not a list of blocks.")
            self.chain = [Block.from_dict(item) for item in raw]
            if not self.is_chain_valid(verify_signatures=True):
                self.status = "LEDGER_CORRUPTED"
                self.corruption_error = "Loaded chain failed hash linkage, PoW, or Ed25519 signature verification."
        except Exception as e:
            self.status = "LEDGER_CORRUPTED"
            self.corruption_error = f"Failed to load ledger: {e}"
            self.chain = []

    def recover_from_backup(self) -> bool:
        """Explicit operator recovery from `.bak` snapshot when in LEDGER_CORRUPTED state."""
        if not self.storage_path:
            return False
        bak_path = f"{self.storage_path}.bak"
        if not os.path.exists(bak_path):
            return False
        shutil.copy2(bak_path, self.storage_path)
        self.status = "HEALTHY"
        self.corruption_error = None
        self.load()
        return self.status == "HEALTHY"
