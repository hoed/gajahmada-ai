"""
GajahMada AI — Hybrid Knowledge Retriever (Semantic + BM25 + Metadata + KG + Persistent Memory + Distributed).

Combines:
1. External indexed document chunks (`KnowledgeRecord` store with encoder version metadata)
2. Vector cosine similarity (enforcing strict encoder compatibility before comparison!)
3. BM25 lexical scoring (pure Python, zero mandatory external packages)
4. Metadata filtering (source_type, verification_status, heading, custom keys)
5. Knowledge Graph lookup via existing `MemoryManager.knowledge_graph`
6. Persistent Memory retrieval via existing `MemoryManager.retrieve()`
7. Optional authorized distributed node retrieval callback (opt-in via feature flag)
8. Fail-closed retrieval-boundary authorization via `KnowledgeAccessPolicy`
"""
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from typing import Any, Callable, Dict, List, Optional

from gajahmada_ai.knowledge.access_policy import KnowledgeAccessPolicy
from gajahmada_ai.knowledge.embedding import (
    HashEmbeddingProvider,
    NativeSemanticEmbeddingProvider,
    cosine_similarity,
)
from gajahmada_ai.knowledge.interfaces import (
    EmbeddingProvider,
    HybridKnowledgeConfig,
    KnowledgeAccessContext,
    KnowledgeRecord,
    KnowledgeRetriever,
    RetrievalResult,
)
from gajahmada_ai.memory.models import VisibilityPolicy


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


class BM25Index:
    """Lightweight, deterministic Okapi BM25 lexical ranker."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_all(self, query: str, docs: List[KnowledgeRecord]) -> Dict[str, float]:
        q_terms = _tokenize(query)
        if not q_terms or not docs:
            return {}

        tokenized_docs = [_tokenize(d.content) for d in docs]
        n_docs = len(docs)
        avgdl = sum(len(td) for td in tokenized_docs) / max(1, n_docs)

        df: Dict[str, int] = {}
        for td in tokenized_docs:
            for term in set(td):
                df[term] = df.get(term, 0) + 1

        scores: Dict[str, float] = {}
        for rec, td in zip(docs, tokenized_docs):
            dl = len(td)
            if dl == 0:
                scores[rec.record_id] = 0.0
                continue
            tf_map: Dict[str, int] = {}
            for t in td:
                tf_map[t] = tf_map.get(t, 0) + 1

            s = 0.0
            for qt in q_terms:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                n_q = df.get(qt, 0)
                idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, avgdl)))
                s += idf * ((tf * (self.k1 + 1.0)) / denom)
            scores[rec.record_id] = s
        return scores


class HybridKnowledgeRetriever(KnowledgeRetriever):
    """
    Unified Hybrid Retriever coordinating External Knowledge, Persistent Memory,
    Knowledge Graph, and optional Distributed Nodes with strict access control.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        fallback_embedding_provider: Optional[EmbeddingProvider] = None,
        memory_manager: Optional[Any] = None,
        access_policy: Optional[KnowledgeAccessPolicy] = None,
        config: Optional[HybridKnowledgeConfig] = None,
        distributed_provider: Optional[Callable[[str, KnowledgeAccessContext, int], List[KnowledgeRecord]]] = None,
        storage_path: Optional[str] = None,
    ):
        self.config = config or HybridKnowledgeConfig()
        self.fallback_embedding_provider = fallback_embedding_provider or HashEmbeddingProvider()
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        elif self.config.semantic_embedding:
            self.embedding_provider = NativeSemanticEmbeddingProvider()
        else:
            self.embedding_provider = self.fallback_embedding_provider

        self.memory_manager = memory_manager
        self.access_policy = access_policy or KnowledgeAccessPolicy()
        self.distributed_provider = distributed_provider
        self.storage_path = storage_path
        self._db_lock = threading.Lock()
        self._bm25 = BM25Index()
        self._external_records: Dict[str, KnowledgeRecord] = {}
        if self.storage_path:
            self._init_sqlite_store()
            self._load_from_sqlite()

    def _init_sqlite_store(self):
        parent = os.path.dirname(os.path.abspath(self.storage_path))
        os.makedirs(parent, exist_ok=True)
        with self._db_lock, sqlite3.connect(self.storage_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_knowledge_chunks (
                    record_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    knowledge_layer TEXT NOT NULL,
                    visibility_policy TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_encoder_id TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    heading TEXT NOT NULL,
                    owner_node_id TEXT NOT NULL,
                    owner_agent_id TEXT
                )
                """
            )
            conn.commit()

    def _persist_record_sqlite(self, rec: KnowledgeRecord):
        if not self.storage_path:
            return
        with self._db_lock, sqlite3.connect(self.storage_path, timeout=10.0) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO external_knowledge_chunks (
                    record_id, source_id, source_type, knowledge_layer, visibility_policy,
                    verification_status, confidence, timestamp, content, metadata_json,
                    provenance_json, embedding_json, embedding_encoder_id, embedding_version,
                    chunk_index, page_number, heading, owner_node_id, owner_agent_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.record_id,
                    rec.source_id,
                    rec.source_type,
                    rec.knowledge_layer,
                    rec.visibility_policy,
                    rec.verification_status,
                    float(rec.confidence),
                    float(rec.timestamp),
                    rec.content,
                    json.dumps(rec.metadata, ensure_ascii=False),
                    json.dumps(rec.provenance, ensure_ascii=False),
                    json.dumps(rec.embedding),
                    rec.embedding_encoder_id,
                    rec.embedding_version,
                    int(rec.chunk_index),
                    rec.page_number,
                    rec.heading,
                    rec.owner_node_id,
                    rec.owner_agent_id,
                ),
            )
            conn.commit()

    def _load_from_sqlite(self):
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        with self._db_lock, sqlite3.connect(self.storage_path, timeout=10.0) as conn:
            rows = conn.execute(
                """
                SELECT record_id, source_id, source_type, knowledge_layer, visibility_policy,
                       verification_status, confidence, timestamp, content, metadata_json,
                       provenance_json, embedding_json, embedding_encoder_id, embedding_version,
                       chunk_index, page_number, heading, owner_node_id, owner_agent_id
                FROM external_knowledge_chunks
                """
            ).fetchall()
        for row in rows:
            rec = KnowledgeRecord(
                record_id=row[0],
                source_id=row[1],
                source_type=row[2],
                knowledge_layer=row[3],
                visibility_policy=row[4],
                verification_status=row[5],
                confidence=float(row[6]),
                timestamp=float(row[7]),
                content=row[8],
                metadata=json.loads(row[9] or "{}"),
                provenance=json.loads(row[10] or "{}"),
                embedding=json.loads(row[11] or "[]"),
                embedding_encoder_id=row[12] or "",
                embedding_version=row[13] or "",
                chunk_index=int(row[14]),
                page_number=row[15],
                heading=row[16] or "",
                owner_node_id=row[17] or "local",
                owner_agent_id=row[18],
            )
            self._external_records[rec.record_id] = rec

    def index_record(self, record: KnowledgeRecord) -> KnowledgeRecord:
        if not record.embedding:
            record.embedding = self.embedding_provider.embed(record.content)
            record.embedding_encoder_id = self.embedding_provider.encoder_id
            record.embedding_version = self.embedding_provider.encoder_version
        self._external_records[record.record_id] = record
        self._persist_record_sqlite(record)
        return record

    def list_external_records(self) -> List[KnowledgeRecord]:
        return list(self._external_records.values())

    def _matches_metadata(self, rec: KnowledgeRecord, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if k == "source_type" and rec.source_type != v:
                return False
            elif k == "source_id" and rec.source_id != v:
                return False
            elif k == "knowledge_layer" and rec.knowledge_layer != v:
                return False
            elif k == "verification_status" and rec.verification_status != v:
                return False
            elif k == "visibility_policy" and rec.visibility_policy != v:
                return False
            elif k in rec.metadata and rec.metadata[k] != v:
                return False
            elif k not in ("source_type", "source_id", "knowledge_layer", "verification_status", "visibility_policy") and k not in rec.metadata:
                return False
        return True

    def _collect_memory_records(self, query: str, top_k: int) -> List[KnowledgeRecord]:
        if self.memory_manager is None:
            return []
        try:
            raw_recs = self.memory_manager.retrieve(query=query, top_k=max(top_k * 2, 6))
        except Exception:
            return []

        converted: List[KnowledgeRecord] = []
        for r in raw_recs:
            prov = r.get("source_provenance") if isinstance(r.get("source_provenance"), dict) else {}
            converted.append(
                KnowledgeRecord(
                    record_id=str(r.get("record_id", "")),
                    content=str(r.get("content", "")),
                    source_id=str(prov.get("source_id") or r.get("record_id", "")),
                    source_type=str(prov.get("source_type", "persistent_memory")),
                    knowledge_layer="memory",
                    visibility_policy=str(r.get("visibility_policy", VisibilityPolicy.LOCAL.value)),
                    verification_status=str(r.get("verification_status", "UNVERIFIED")),
                    confidence=float(r.get("confidence", 0.75)),
                    timestamp=float(r.get("created_at", 0.0) or 0.0),
                    metadata={"memory_type": r.get("memory_type", "semantic")},
                    provenance={**prov, "integrity_hash": r.get("integrity_hash", "")},
                    embedding=list(r.get("embedding") or []),
                    embedding_encoder_id=self.fallback_embedding_provider.encoder_id if r.get("embedding") else "",
                    embedding_version=self.fallback_embedding_provider.encoder_version if r.get("embedding") else "",
                    owner_node_id=str(prov.get("origin_node_id", "local")),
                )
            )
        return converted

    def _collect_graph_boosts(self, query: str) -> Dict[str, float]:
        if not self.config.enable_graph_search or self.memory_manager is None:
            return {}
        kg = getattr(self.memory_manager, "knowledge_graph", None)
        if kg is None:
            return {}
        boosts: Dict[str, float] = {}
        for term in _tokenize(query)[:6]:
            if len(term) < 3:
                continue
            try:
                rels = kg.query_entity(term)
            except Exception:
                rels = []
            for rel in rels:
                rid = rel.get("record_id")
                if rid:
                    boosts[rid] = min(1.0, boosts.get(rid, 0.0) + 0.35)
        return boosts

    def _safe_vector_similarity(self, query: str, rec: KnowledgeRecord, query_vec_cache: Dict[str, List[float]]) -> float:
        if not self.config.enable_vector_search or not rec.embedding:
            return 0.0

        # Strict encoder compatibility check before comparing vectors!
        if self.embedding_provider.is_compatible_with(rec.embedding_encoder_id, rec.embedding_version, rec.embedding):
            q_key = f"{self.embedding_provider.encoder_id}:{self.embedding_provider.encoder_version}"
            if q_key not in query_vec_cache:
                query_vec_cache[q_key] = self.embedding_provider.embed(query)
            return max(0.0, cosine_similarity(query_vec_cache[q_key], rec.embedding))

        if (
            self.config.hash_embedding_fallback
            and self.fallback_embedding_provider.is_compatible_with(
                rec.embedding_encoder_id, rec.embedding_version, rec.embedding
            )
        ):
            q_key = f"{self.fallback_embedding_provider.encoder_id}:{self.fallback_embedding_provider.encoder_version}"
            if q_key not in query_vec_cache:
                query_vec_cache[q_key] = self.fallback_embedding_provider.embed(query)
            return max(0.0, cosine_similarity(query_vec_cache[q_key], rec.embedding))

        # Incompatible encoder versions are NEVER blindly compared!
        return 0.0

    def retrieve(
        self,
        query: str,
        access_context: KnowledgeAccessContext,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if access_context is None:
            return []

        candidates: List[KnowledgeRecord] = []
        if self.config.external_retrieval and access_context.include_external:
            candidates.extend(self._external_records.values())
        if access_context.include_memory:
            candidates.extend(self._collect_memory_records(query, top_k=top_k))
        if (
            self.config.enable_distributed_sources
            and access_context.include_distributed
            and self.distributed_provider is not None
        ):
            try:
                candidates.extend(self.distributed_provider(query, access_context, top_k))
            except Exception:
                pass

        # Enforce retrieval-boundary authorization & metadata filtering BEFORE scoring
        authorized = [
            rec
            for rec in self.access_policy.filter_records(candidates, access_context)
            if self._matches_metadata(rec, metadata_filters)
        ]
        if not authorized:
            return []

        # Deduplicate by normalized content hash (preferring higher-verification or external provenance)
        dedup_map: Dict[str, KnowledgeRecord] = {}
        for rec in authorized:
            c_hash = hashlib.sha256(rec.content.strip().lower().encode("utf-8")).hexdigest()[:16]
            if c_hash not in dedup_map or rec.confidence > dedup_map[c_hash].confidence:
                dedup_map[c_hash] = rec
        unique_docs = list(dedup_map.values())

        bm25_scores = self._bm25.score_all(query, unique_docs) if self.config.enable_bm25 else {}
        max_bm25 = max(bm25_scores.values(), default=0.0) or 1.0
        graph_boosts = self._collect_graph_boosts(query)
        q_vec_cache: Dict[str, List[float]] = {}

        ranked: List[RetrievalResult] = []
        for rec in unique_docs:
            sem = self._safe_vector_similarity(query, rec, q_vec_cache)
            lex = (bm25_scores.get(rec.record_id, 0.0) / max_bm25) if bm25_scores else 0.0
            g_boost = graph_boosts.get(rec.record_id, 0.0)

            # Skip completely unrelated records (zero lexical and near-zero semantic overlap)
            if lex <= 0.0 and sem < 0.12 and g_boost <= 0.0:
                continue

            # Provenance / verification bonus
            ver_bonus = 0.08 if rec.verification_status in ("INDEPENDENTLY_VERIFIED", "VERIFIED", "SOURCE_CORROBORATED") else 0.0
            combined = (0.50 * sem) + (0.42 * lex) + (0.15 * g_boost) + ver_bonus
            ranked.append(
                RetrievalResult(
                    record=rec,
                    score=combined,
                    semantic_score=sem,
                    lexical_score=lex,
                    graph_score=g_boost,
                    retriever_source=f"{rec.knowledge_layer}:hybrid",
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(1, int(top_k))]
