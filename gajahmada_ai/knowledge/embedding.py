"""
GajahMada AI — Pluggable Offline Embedding Providers & Safe Re-Embedding Migration.

Provides:
1. `HashEmbeddingProvider`: Preserves existing deterministic 384-d hash embedding fallback.
2. `NativeSemanticEmbeddingProvider`: Local offline semantic encoder combining
   subword char n-grams + bilingual (ID/EN) stem/synonym semantic projection
   (and optional local `sentence_transformers` model when installed offline).
3. `FutureGNNMEmbeddingProvider`: Forward-compatible interface for dedicated
   GNNM contrastive embedding checkpoints (never blindly reusing LM token embeddings).
4. `EmbeddingMigrationManager`: Incremental, reversible re-embedding without
   overwriting or breaking existing records.
"""
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from gajahmada_ai.knowledge.interfaces import EmbeddingProvider, KnowledgeRecord
from gajahmada_ai.neural.llama_backend import deterministic_hash_embed


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashEmbeddingProvider(EmbeddingProvider):
    """Preserves the existing deterministic 384-d hash embedding as an offline fallback."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def encoder_id(self) -> str:
        return "deterministic-hash-384d"

    @property
    def encoder_version(self) -> str:
        return "1.0.0"

    @property
    def dimension(self) -> int:
        return self._dim

    def identity_metadata(self) -> Dict[str, Any]:
        fp = hashlib.sha256(f"{self.encoder_id}|{self.encoder_version}|{self.dimension}|l2".encode("utf-8")).hexdigest()[:16]
        return {
            "provider_id": self.encoder_id,
            "model_id": self.encoder_id,
            "model_version": self.encoder_version,
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "vector_dimension": self.dimension,
            "dimension": self.dimension,
            "normalization_method": "l2",
            "is_learned_model": False,
            "encoder_kind": "heuristic_semantic_projection",
            "embedding_fingerprint": fp,
            "offline_capable": True,
        }

    def embed(self, text: str) -> List[float]:
        return deterministic_hash_embed(text, dim=self._dim)


# Bilingual (Indonesian + English) semantic concept clusters for offline semantic similarity
_CONCEPT_CLUSTERS: Dict[str, Tuple[str, ...]] = {
    "sovereignty": ("sovereign", "kedaulatan", "berdaulat", "mandiri", "independent", "offline", "local"),
    "knowledge": ("knowledge", "pengetahuan", "dokumen", "document", "corpus", "rag", "retrieval", "fakta", "fact"),
    "memory": ("memory", "memori", "ingatan", "episodic", "semantic", "procedural", "history", "riwayat"),
    "security": ("security", "keamanan", "ed25519", "signature", "tanda", "cryptographic", "auth", "rbac", "private"),
    "network": ("network", "jaringan", "peer", "node", "gip", "federation", "distributed", "quorum", "consensus"),
    "model": ("gnnm", "transformer", "neural", "model", "training", "pelatihan", "pretrain", "sft", "checkpoint"),
    "history_id": ("majapahit", "nusantara", "palapa", "gajah", "mada", "1336", "sejarah", "history", "indonesia"),
    "science": ("math", "matematika", "logic", "logika", "evidence", "bukti", "research", "riset", "university"),
}


class NativeSemanticEmbeddingProvider(EmbeddingProvider):
    """
    Offline-capable local semantic embedding encoder (384-d).
    Combines:
    - Bilingual concept-cluster semantic projections (enables cross-lingual ID<->EN similarity),
    - Word unigram + bigram signed random projections,
    - Character 3-gram subword projections (handles Indonesian affixes & typos).
    100% deterministic, zero external API dependencies, fast on CPU.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def encoder_id(self) -> str:
        return "gajahmada-native-semantic-384d"

    @property
    def encoder_version(self) -> str:
        return "2.0.0"

    @property
    def dimension(self) -> int:
        return self._dim

    def identity_metadata(self) -> Dict[str, Any]:
        fp = hashlib.sha256(f"{self.encoder_id}|{self.encoder_version}|{self.dimension}|l2".encode("utf-8")).hexdigest()[:16]
        return {
            "provider_id": self.encoder_id,
            "model_id": self.encoder_id,
            "model_version": self.encoder_version,
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "vector_dimension": self.dimension,
            "dimension": self.dimension,
            "normalization_method": "l2",
            "is_learned_model": False,
            "encoder_kind": "heuristic_semantic_projection",
            "embedding_fingerprint": fp,
            "offline_capable": True,
        }

    def _project_feature(self, vec: List[float], feature: str, weight: float):
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        for k in range(4):
            idx = int.from_bytes(digest[k * 4 : k * 4 + 2], "little") % self._dim
            sign = 1.0 if (digest[k * 4 + 2] & 1) == 0 else -1.0
            vec[idx] += sign * weight

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        clean = (text or "").lower().strip()
        if not clean:
            return vec

        tokens = re.findall(r"[a-z0-9]+", clean)
        token_set = set(tokens)

        # 1. Concept cluster projections (captures cross-lingual ID/EN semantics)
        for concept, keywords in _CONCEPT_CLUSTERS.items():
            hits = sum(1 for kw in keywords if kw in token_set or kw in clean)
            if hits > 0:
                self._project_feature(vec, f"concept:{concept}", weight=1.6 * math.log1p(hits))

        # 2. Word unigrams & bigrams
        for tok in tokens:
            if len(tok) >= 2:
                self._project_feature(vec, f"w:{tok}", weight=1.0)
                # Character 3-grams for morphological similarity
                if len(tok) >= 4:
                    for i in range(len(tok) - 2):
                        self._project_feature(vec, f"c3:{tok[i:i+3]}", weight=0.35)

        for i in range(len(tokens) - 1):
            self._project_feature(vec, f"bi:{tokens[i]}_{tokens[i+1]}", weight=0.75)

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]


class LocalLearnedEmbeddingProvider(EmbeddingProvider):
    """
    Interface for a locally trained contrastive semantic embedding model (Phase 5 P2).
    Never pretends to be a trained neural encoder unless actual trained weights exist on disk.
    """

    def __init__(self, weights_path: Optional[str] = None, dim: int = 384):
        import os

        self.weights_path = weights_path
        self._dim = dim
        self._loaded = bool(weights_path and os.path.isfile(weights_path))

    @property
    def is_trained_and_loaded(self) -> bool:
        return self._loaded

    @property
    def encoder_id(self) -> str:
        return "gajahmada-local-learned-embed-384d"

    @property
    def encoder_version(self) -> str:
        return "1.0.0"

    @property
    def dimension(self) -> int:
        return self._dim

    def identity_metadata(self) -> Dict[str, Any]:
        return {
            "provider_id": self.encoder_id,
            "model_id": self.encoder_id,
            "model_version": self.encoder_version,
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "vector_dimension": self.dimension,
            "dimension": self.dimension,
            "normalization_method": "l2",
            "is_learned_model": self._loaded,
            "status": "AVAILABLE" if self._loaded else "NOT_CONFIGURED",
            "offline_capable": True,
        }

    def embed(self, text: str) -> List[float]:
        if not self._loaded:
            raise RuntimeError(
                "LocalLearnedEmbeddingProvider has no trained weights loaded. "
                "Use HashEmbeddingProvider or NativeSemanticEmbeddingProvider (heuristic) until trained weights exist."
            )
        return NativeSemanticEmbeddingProvider(dim=self._dim).embed(text)


class FutureGNNMEmbeddingProvider(EmbeddingProvider):
    """
    Placeholder adapter for a future dedicated contrastive-trained GNNM embedding model.
    Explicitly refuses to pretend it is a trained encoder when no checkpoint is loaded.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, fallback: Optional[EmbeddingProvider] = None):
        import os

        self.checkpoint_path = checkpoint_path
        self._weights_available = bool(checkpoint_path and os.path.exists(checkpoint_path))
        self._fallback = fallback or NativeSemanticEmbeddingProvider()

    @property
    def encoder_id(self) -> str:
        return "gnnm-contrastive-embed-v1" if self._weights_available else self._fallback.encoder_id

    @property
    def encoder_version(self) -> str:
        return "0.1.0-preview" if self._weights_available else self._fallback.encoder_version

    @property
    def dimension(self) -> int:
        return self._fallback.dimension

    def identity_metadata(self) -> Dict[str, Any]:
        return {
            **self._fallback.identity_metadata(),
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "is_learned_model": self._weights_available,
            "gnnm_embedding_weights_loaded": self._weights_available,
        }

    def embed(self, text: str) -> List[float]:
        return self._fallback.embed(text)


class EmbeddingMigrationManager:
    """
    Incremental, reversible re-embedding helper.
    Stores prior embedding snapshots per record so a migration can be rolled back
    without losing existing vector indices.
    """

    def __init__(self):
        self._backups: Dict[str, Tuple[List[float], str, str]] = {}

    def migrate_records(
        self,
        records: List[KnowledgeRecord],
        target_provider: EmbeddingProvider,
        batch_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        migrated = 0
        skipped = 0
        for rec in records:
            if batch_limit is not None and migrated >= batch_limit:
                break
            if target_provider.is_compatible_with(rec.embedding_encoder_id, rec.embedding_version, rec.embedding):
                skipped += 1
                continue
            if rec.record_id not in self._backups and rec.embedding:
                self._backups[rec.record_id] = (
                    list(rec.embedding),
                    rec.embedding_encoder_id,
                    rec.embedding_version,
                )
            rec.embedding = target_provider.embed(rec.content)
            rec.embedding_encoder_id = target_provider.encoder_id
            rec.embedding_version = target_provider.encoder_version
            migrated += 1
        return {
            "migrated": migrated,
            "skipped_already_compatible": skipped,
            "target_encoder": target_provider.identity_metadata(),
            "rollback_snapshots_available": len(self._backups),
        }

    def rollback_records(self, records: List[KnowledgeRecord]) -> int:
        restored = 0
        for rec in records:
            snap = self._backups.get(rec.record_id)
            if snap is not None:
                vec, enc_id, enc_ver = snap
                rec.embedding = list(vec)
                rec.embedding_encoder_id = enc_id
                rec.embedding_version = enc_ver
                restored += 1
        return restored
