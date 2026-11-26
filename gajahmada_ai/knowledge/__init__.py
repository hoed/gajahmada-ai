"""
GajahMada AI — Hybrid Knowledge Package.
Preserves existing `LocalKnowledge` while exporting the new GNNM Hybrid Knowledge modules.
"""
from gajahmada_ai.knowledge.access_policy import KnowledgeAccessPolicy
from gajahmada_ai.knowledge.chunker import DocumentChunk, DocumentChunker
from gajahmada_ai.knowledge.context_builder import BuiltContext, TokenBudgetContextBuilder
from gajahmada_ai.knowledge.controller import GNNMKnowledgeAdapter, HybridKnowledgeController
from gajahmada_ai.knowledge.embedding import (
    EmbeddingMigrationManager,
    FutureGNNMEmbeddingProvider,
    HashEmbeddingProvider,
    NativeSemanticEmbeddingProvider,
    cosine_similarity,
)
from gajahmada_ai.knowledge.ingestion import KnowledgeIngestionPipeline
from gajahmada_ai.knowledge.interfaces import (
    EmbeddingProvider,
    HybridKnowledgeConfig,
    KnowledgeAccessContext,
    KnowledgeController,
    KnowledgeRecord,
    KnowledgeRetriever,
    KnowledgeSource,
    RetrievalResult,
)
from gajahmada_ai.knowledge.rag import LocalKnowledge
from gajahmada_ai.knowledge.retriever import BM25Index, HybridKnowledgeRetriever
from gajahmada_ai.knowledge.source_registry import SourceRegistry

__all__ = [
    "LocalKnowledge",
    "HybridKnowledgeConfig",
    "KnowledgeAccessContext",
    "KnowledgeSource",
    "KnowledgeRecord",
    "RetrievalResult",
    "EmbeddingProvider",
    "KnowledgeRetriever",
    "KnowledgeController",
    "KnowledgeAccessPolicy",
    "SourceRegistry",
    "HashEmbeddingProvider",
    "NativeSemanticEmbeddingProvider",
    "FutureGNNMEmbeddingProvider",
    "EmbeddingMigrationManager",
    "cosine_similarity",
    "DocumentChunk",
    "DocumentChunker",
    "BuiltContext",
    "TokenBudgetContextBuilder",
    "BM25Index",
    "HybridKnowledgeRetriever",
    "KnowledgeIngestionPipeline",
    "HybridKnowledgeController",
    "GNNMKnowledgeAdapter",
]
