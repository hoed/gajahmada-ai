import hashlib
import json
import os
import time
from typing import List, Optional


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
    ):
        self.index = index
        self.timestamp = timestamp
        self.task_hash = task_hash
        self.node_id = node_id
        self.previous_hash = previous_hash
        self.signature = signature
        self.nonce = nonce
        self.hash = block_hash if block_hash else self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = (
            f"{self.index}|{self.timestamp}|{self.task_hash}|"
            f"{self.node_id}|{self.previous_hash}|{self.signature}|{self.nonce}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "task_hash": self.task_hash,
            "node_id": self.node_id,
            "previous_hash": self.previous_hash,
            "signature": self.signature,
            "nonce": self.nonce,
            "hash": self.hash,
        }

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
        )


class AIBlockchain:
    def __init__(self, storage_path: Optional[str] = None):
        self.difficulty = 2
        self.storage_path = storage_path
        self.chain: List[Block] = []
        if self.storage_path and os.path.exists(self.storage_path):
            self.load()
        if not self.chain:
            self.chain = [self.create_genesis_block()]
            self.save()

    def create_genesis_block(self) -> Block:
        return Block(0, time.time(), "Genesis Task", "Genesis Node", "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, new_block: Block):
        new_block.index = len(self.chain)
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        target_prefix = "0" * self.difficulty
        while not new_block.hash.startswith(target_prefix):
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        self.save()

    def is_chain_valid(self) -> bool:
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
            if not self.is_chain_valid():
                self.chain = []
        except Exception:
            self.chain = []
