from enum import Enum
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.security.crypto import NodeIdentity


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


class Block:
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
    ):
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

    def calculate_hash(self) -> str:
        # Backward compatible hash formula so existing chain.json blocks remain 100% valid
        block_string = (
            f"{self.index}|{self.timestamp}|{self.task_hash}|"
            f"{self.node_id}|{self.previous_hash}|{self.signature}|{self.nonce}"
        )
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = {
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
        return cls(
            index=data["index"],
            timestamp=data["timestamp"],
            task_hash=data["task_hash"],
            node_id=data["node_id"],
            previous_hash=data["previous_hash"],
            signature=data.get("signature", ""),
            nonce=data.get("nonce", 0),
            block_hash=data.get("hash"),
            event_type=data.get("event_type", LedgerEventType.GENESIS.value if data["index"] == 0 else LedgerEventType.TASK_COMPLETED.value),
            event_ref=data.get("event_ref", ""),
            metadata=data.get("metadata", {}),
        )


class AIBlockchain:
    """
    Signed Event & Cryptographic Audit Ledger.
    Preserves SHA-256 Proof-of-Work hash linkage AND enforces Ed25519 signature
    verification on signed event blocks.
    Never stores raw bulk documents or neural weight binaries on-chain—only
    SHA-256 content hashes and compact event references.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.difficulty = 2
        self.storage_path = storage_path
        self.chain: List[Block] = []
        self._verifier = NodeIdentity(key_path=None)
        if self.storage_path and os.path.exists(self.storage_path):
            self.load()
        if not self.chain:
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
        )

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, new_block: Block) -> Block:
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
        """Record a typed, signed audit event on the ledger using content hashes."""
        t0 = time.time()
        ev_name = event_type.value if isinstance(event_type, LedgerEventType) else str(event_type)
        task_hash = hashlib.sha256(f"{ev_name}|{content_reference}|{t0}".encode("utf-8")).hexdigest()
        signature = identity.sign_message(task_hash.encode("utf-8"))
        block = Block(
            index=len(self.chain),
            timestamp=t0,
            task_hash=task_hash,
            node_id=identity.node_id,
            previous_hash=self.get_latest_block().hash,
            signature=signature,
            event_type=ev_name,
            event_ref=content_reference[:128],
            metadata=metadata or {},
        )
        return self.add_block(block)

    def is_chain_valid(self, verify_signatures: bool = True) -> bool:
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
            if verify_signatures and current.signature and current.node_id != "Genesis Node":
                if not self._verifier.verify_message(
                    current.task_hash.encode("utf-8"),
                    current.signature,
                    current.node_id,
                ):
                    return False
        return True

    def save(self):
        if not self.storage_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([b.to_dict() for b in self.chain], f, indent=2)
        except Exception:
            pass

    def load(self):
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.chain = [Block.from_dict(item) for item in raw]
            if not self.is_chain_valid(verify_signatures=False):
                self.chain = []
        except Exception:
            self.chain = []
