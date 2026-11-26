"""
GajahMada AI — Hybrid Knowledge Architecture Interfaces & Feature Flags.

Defines stable, typed contracts for:
- KnowledgeSource
- KnowledgeRecord
- RetrievalResult
- KnowledgeAccessContext
- HybridKnowledgeConfig (backward-compatible feature flags)
- EmbeddingProvider (ABC)
- KnowledgeRetriever (ABC)
- KnowledgeController (ABC)

Designed to be 100% additive and compatible with existing `MemoryRecord`,
`MemoryAuthorizationService`, `ModelRouter`, and `CognitiveReasoningPipeline`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from gajahmada_ai.memory.models import VerificationStatus, VisibilityPolicy


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass
class HybridKnowledgeConfig:
    """Backward-compatible configuration and feature flags for GNNM Hybrid Knowledge."""

    enabled: bool = field(default_factory=lambda: _env_bool("GM_HYBRID_KNOWLEDGE", True))
    external_retrieval: bool = field(default_factory=lambda: _env_bool("GM_EXTERNAL_RETRIEVAL", True))
    semantic_embedding: bool = field(default_factory=lambda: _env_bool("GM_SEMANTIC_EMBEDDING", False))
    hash_embedding_fallback: bool = field(default_factory=lambda: _env_bool("GM_HASH_EMBEDDING_FALLBACK", True))
    knowledge_training: bool = field(default_factory=lambda: _env_bool("GM_KNOWLEDGE_TRAINING", False))
    verification_required: bool = field(default_factory=lambda: _env_bool("GM_KNOWLEDGE_VERIFICATION_REQUIRED", True))
    max_results: int = field(default_factory=lambda: _env_int("GM_KNOWLEDGE_MAX_RESULTS", 5))
    max_context_tokens: int = field(default_factory=lambda: _env_int("GM_KNOWLEDGE_MAX_CONTEXT_TOKENS", 1024))
    enable_bm25: bool = field(default_factory=lambda: _env_bool("GM_KNOWLEDGE_ENABLE_BM25", True))
    enable_vector_search: bool = field(default_factory=lambda: _env_bool("GM_KNOWLEDGE_ENABLE_VECTOR_SEARCH", True))
    enable_graph_search: bool = field(default_factory=lambda: _env_bool("GM_KNOWLEDGE_ENABLE_GRAPH_SEARCH", True))
    enable_distributed_sources: bool = field(
        default_factory=lambda: _env_bool("GM_KNOWLEDGE_ENABLE_DISTRIBUTED_SOURCES", False)
    )


@dataclass
class KnowledgeAccessContext:
    """Actor, visibility, agent, session, organization, and node authorization context enforced at retrieval boundaries."""

    actor_id: str = "anonymous"
    actor_role: str = "anonymous"
    agent_id: Optional[str] = None
    node_id: Optional[str] = None
    org_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}")
    allowed_source_types: Optional[List[str]] = None
    allowed_visibilities: Optional[List[str]] = None
    include_external: bool = True
    include_memory: bool = True
    include_distributed: bool = False
    max_results: int = 5

    @property
    def organization_id(self) -> Optional[str]:
        return self.org_id


@dataclass
class KnowledgeSource:
    """Provenance and governance metadata for an external or federated knowledge source."""

    source_id: str = field(default_factory=lambda: f"src-{uuid.uuid4().hex[:12]}")
    source_type: str = "document"  # txt | markdown | pdf | html | docx | structured | web | memory | distributed
    location: str = ""
    title: str = ""
    author: str = "unknown"
    organization: str = ""
    license: str = "unspecified"
    usage_restrictions: str = ""
    created_at: float = field(default_factory=time.time)
    collected_at: float = field(default_factory=time.time)
    document_hash: str = ""
    verification_status: str = VerificationStatus.UNVERIFIED.value
    visibility_policy: str = VisibilityPolicy.LOCAL.value
    provenance: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    parent_source_id: Optional[str] = None
    owner_node_id: str = "local"
    training_approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "location": self.location,
            "title": self.title,
            "author": self.author,
            "organization": self.organization,
            "license": self.license,
            "usage_restrictions": self.usage_restrictions,
            "created_at": self.created_at,
            "collected_at": self.collected_at,
            "document_hash": self.document_hash,
            "verification_status": self.verification_status,
            "visibility_policy": self.visibility_policy,
            "provenance": dict(self.provenance),
            "version": self.version,
            "parent_source_id": self.parent_source_id,
            "owner_node_id": self.owner_node_id,
            "training_approved": self.training_approved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeSource":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeRecord:
    """
    Unified record representation across External Knowledge chunks and Persistent Memory records.
    Maintains clear separation via `knowledge_layer` (`external` | `memory` | `distributed`).
    """

    record_id: str = field(default_factory=lambda: f"kr-{uuid.uuid4().hex[:12]}")
    content: str = ""
    source_id: str = ""
    source_type: str = "document"
    knowledge_layer: str = "external"  # external | memory | distributed
    visibility_policy: str = VisibilityPolicy.LOCAL.value
    verification_status: str = VerificationStatus.UNVERIFIED.value
    confidence: float = 0.75
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    embedding_encoder_id: str = ""
    embedding_version: str = ""
    chunk_index: int = 0
    page_number: Optional[int] = None
    heading: str = ""
    owner_node_id: str = "local"
    owner_agent_id: Optional[str] = None

    def to_memory_compatible_dict(self) -> Dict[str, Any]:
        """Convert to a dict compatible with existing `EvidenceReasoner` & `CognitiveReasoningPipeline`."""
        return {
            "record_id": self.record_id,
            "content": self.content,
            "memory_type": self.metadata.get("memory_type", self.knowledge_layer),
            "visibility_policy": self.visibility_policy,
            "verification_status": self.verification_status,
            "confidence": self.confidence,
            "created_at": self.timestamp,
            "integrity_hash": self.provenance.get("integrity_hash", self.provenance.get("document_hash", "")),
            "source_provenance": {
                "origin_node_id": self.owner_node_id,
                "source_type": self.source_type,
                "source_id": self.source_id,
                "knowledge_layer": self.knowledge_layer,
                **self.provenance,
            },
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Single ranked retrieval hit with explicit provenance and scoring breakdown."""

    record: KnowledgeRecord
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    graph_score: float = 0.0
    retriever_source: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record.record_id,
            "source_id": self.record.source_id,
            "source_type": self.record.source_type,
            "knowledge_layer": self.record.knowledge_layer,
            "content": self.record.content,
            "score": round(self.score, 5),
            "semantic_score": round(self.semantic_score, 5),
            "lexical_score": round(self.lexical_score, 5),
            "graph_score": round(self.graph_score, 5),
            "verification_status": self.record.verification_status,
            "visibility_policy": self.record.visibility_policy,
            "timestamp": self.record.timestamp,
            "heading": self.record.heading,
            "page_number": self.record.page_number,
            "provenance": self.record.provenance,
            "retriever_source": self.retriever_source,
        }


class EmbeddingProvider(ABC):
    """Pluggable offline-capable embedding provider interface with explicit encoder identity."""

    @property
    @abstractmethod
    def encoder_id(self) -> str:
        ...

    @property
    @abstractmethod
    def encoder_version(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def identity_metadata(self) -> Dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "dimension": self.dimension,
            "offline_capable": True,
        }

    def is_compatible_with(self, encoder_id: str, encoder_version: str, vector: List[float]) -> bool:
        return (
            encoder_id == self.encoder_id
            and encoder_version == self.encoder_version
            and len(vector) == self.dimension
        )


class KnowledgeRetriever(ABC):
    """Abstract hybrid retriever interface."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        access_context: KnowledgeAccessContext,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        ...


class KnowledgeController(ABC):
    """Abstract Hybrid Knowledge Controller coordinating Internal, External, and Persistent Memory."""

    @abstractmethod
    def should_retrieve(self, query: str, access_context: KnowledgeAccessContext) -> Tuple[bool, str]:
        ...

    @abstractmethod
    def query(
        self,
        query: str,
        access_context: KnowledgeAccessContext,
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...
