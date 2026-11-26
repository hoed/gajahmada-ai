from enum import Enum
import hashlib
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    COLLECTIVE = "collective"


class VerificationStatus(str, Enum):
    # Explicit 6-State Epistemic Model (P2 Remediation) + backward-compatible aliases
    UNVERIFIED = "UNVERIFIED"
    MODEL_ASSESSED = "MODEL_ASSESSED"
    SOURCE_CORROBORATED = "SOURCE_CORROBORATED"
    INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"
    # Backward-compatible aliases
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    CONTRADICTED = "CONTRADICTED"


class VisibilityPolicy(str, Enum):
    PRIVATE = "PRIVATE"
    SESSION = "SESSION"
    AGENT = "AGENT"
    ORGANIZATION = "ORGANIZATION"
    LOCAL = "LOCAL"
    SHARED = "SHARED"
    COLLECTIVE = "COLLECTIVE"
    PUBLIC = "PUBLIC"
    DISTRIBUTED_SHARED = "DISTRIBUTED_SHARED"


class SourceProvenance(BaseModel):
    origin_node_id: str = "local"
    source_type: str = "operator_ingested"
    author: str = "local_operator"
    reference_uri: str = ""
    signature: str = ""
    recorded_at: float = Field(default_factory=time.time)


class MemoryRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: f"mem-{uuid.uuid4().hex[:12]}")
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    entity_references: List[str] = Field(default_factory=list)
    source_provenance: SourceProvenance = Field(default_factory=SourceProvenance)
    created_at: float = Field(default_factory=time.time)
    modified_at: float = Field(default_factory=time.time)
    version: int = 1
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    visibility_policy: VisibilityPolicy = VisibilityPolicy.LOCAL
    integrity_hash: str = ""
    embedding: List[float] = Field(default_factory=list)
    superseded_by: Optional[str] = None
    is_deleted: bool = False

    def compute_integrity_hash(self) -> str:
        payload = (
            f"{self.record_id}|{self.content}|{self.memory_type.value}|"
            f"{self.version}|{self.source_provenance.origin_node_id}|{self.verification_status.value}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ensure_hash(self) -> "MemoryRecord":
        self.integrity_hash = self.compute_integrity_hash()
        return self

    def to_storage_dict(self) -> Dict[str, Any]:
        self.ensure_hash()
        return {
            "record_id": self.record_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "entity_references": self.entity_references,
            "source_provenance": self.source_provenance.model_dump(),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "version": self.version,
            "confidence": self.confidence,
            "verification_status": self.verification_status.value,
            "visibility_policy": self.visibility_policy.value,
            "integrity_hash": self.integrity_hash,
            "embedding": self.embedding,
            "superseded_by": self.superseded_by,
            "is_deleted": self.is_deleted,
        }
