"""
GajahMada AI — Unified Learning Event Tracker & Temporal Knowledge Store.

Distinguishes the four mandatory knowledge states:
  1. RETRIEVED_KNOWLEDGE    (obtained dynamically via RAG/external retrieval)
  2. STORED_KNOWLEDGE       (ingested and persisted in knowledge base or memory)
  3. INTERNALIZED_KNOWLEDGE (demonstrated to be incorporated into GNNM weights via real training)
  4. VERIFIED_KNOWLEDGE     (evidence & provenance passed verification)

Persists real timestamped learning events to SQLite WAL (`learning_events.db`),
enforces `MemoryAuthorizationService` on every temporal query, and NEVER fabricates
timestamps or learning history.
"""
from dataclasses import dataclass, field
from enum import Enum
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import VisibilityPolicy


class KnowledgeStateType(str, Enum):
    RETRIEVED_KNOWLEDGE = "RETRIEVED_KNOWLEDGE"
    STORED_KNOWLEDGE = "STORED_KNOWLEDGE"
    INTERNALIZED_KNOWLEDGE = "INTERNALIZED_KNOWLEDGE"
    VERIFIED_KNOWLEDGE = "VERIFIED_KNOWLEDGE"


class LearningEventType(str, Enum):
    DOCUMENT_INGESTED = "DOCUMENT_INGESTED"
    KNOWLEDGE_STORED = "KNOWLEDGE_STORED"
    KNOWLEDGE_VERIFIED = "KNOWLEDGE_VERIFIED"
    KNOWLEDGE_RETRIEVED = "KNOWLEDGE_RETRIEVED"
    TRAINING_STARTED = "TRAINING_STARTED"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    TRAINING_REJECTED = "TRAINING_REJECTED"
    MODEL_PROMOTED = "MODEL_PROMOTED"


@dataclass
class LearningEvent:
    event_id: str = field(default_factory=lambda: f"lev-{uuid.uuid4().hex[:12]}")
    event_type: str = LearningEventType.DOCUMENT_INGESTED.value
    timestamp: float = field(default_factory=lambda: round(time.time(), 3))
    actor_id: str = "system"
    visibility_policy: str = VisibilityPolicy.COLLECTIVE.value
    organization_id: Optional[str] = None
    source_id: str = ""
    source_type: str = ""
    source_title: str = ""
    knowledge_domain: str = "general_knowledge"
    summary: str = ""
    content_hash: str = ""
    verification_status: str = "UNVERIFIED"
    embedding_provider: str = ""
    model_version_before: Optional[str] = None
    model_version_after: Optional[str] = None
    training_checkpoint: Optional[str] = None
    training_status: str = "NOT_PERFORMED"
    internalization_status: str = "NOT_DEMONSTRATED"  # NOT_DEMONSTRATED | RETRIEVED_ONLY | STORED_ONLY | INTERNALIZED_IN_GNNM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "visibility_policy": self.visibility_policy,
            "organization_id": self.organization_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "knowledge_domain": self.knowledge_domain,
            "summary": self.summary,
            "content_hash": self.content_hash,
            "verification_status": self.verification_status,
            "embedding_provider": self.embedding_provider,
            "model_version_before": self.model_version_before,
            "model_version_after": self.model_version_after,
            "training_checkpoint": self.training_checkpoint,
            "training_status": self.training_status,
            "internalization_status": self.internalization_status,
        }


class LearningEventTracker:
    """SQLite WAL-backed temporal learning event log with authorization filtering."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        authorization: Optional[MemoryAuthorizationService] = None,
    ):
        self.storage_path = storage_path
        self.authorization = authorization or MemoryAuthorizationService()
        self._lock = threading.Lock()
        self._events: List[LearningEvent] = []
        if self.storage_path:
            self._init_db()
            self._load_db()

    def _init_db(self):
        parent = os.path.dirname(os.path.abspath(self.storage_path))
        os.makedirs(parent, exist_ok=True)
        with self._lock, sqlite3.connect(self.storage_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    actor_id TEXT NOT NULL,
                    visibility_policy TEXT NOT NULL,
                    organization_id TEXT,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_title TEXT NOT NULL,
                    knowledge_domain TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    model_version_before TEXT,
                    model_version_after TEXT,
                    training_checkpoint TEXT,
                    training_status TEXT NOT NULL,
                    internalization_status TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_le_ts ON learning_events (timestamp DESC)")
            conn.commit()

    def _load_db(self):
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        with self._lock, sqlite3.connect(self.storage_path, timeout=10.0) as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, timestamp, actor_id, visibility_policy,
                       organization_id, source_id, source_type, source_title, knowledge_domain,
                       summary, content_hash, verification_status, embedding_provider,
                       model_version_before, model_version_after, training_checkpoint,
                       training_status, internalization_status
                FROM learning_events ORDER BY timestamp ASC
                """
            ).fetchall()
        self._events = [
            LearningEvent(
                event_id=r[0],
                event_type=r[1],
                timestamp=float(r[2]),
                actor_id=r[3],
                visibility_policy=r[4],
                organization_id=r[5],
                source_id=r[6],
                source_type=r[7],
                source_title=r[8],
                knowledge_domain=r[9],
                summary=r[10],
                content_hash=r[11],
                verification_status=r[12],
                embedding_provider=r[13],
                model_version_before=r[14],
                model_version_after=r[15],
                training_checkpoint=r[16],
                training_status=r[17],
                internalization_status=r[18],
            )
            for r in rows
        ]

    def record_event(self, event: LearningEvent) -> LearningEvent:
        with self._lock:
            self._events.append(event)
            if self.storage_path:
                with sqlite3.connect(self.storage_path, timeout=10.0) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO learning_events (
                            event_id, event_type, timestamp, actor_id, visibility_policy,
                            organization_id, source_id, source_type, source_title, knowledge_domain,
                            summary, content_hash, verification_status, embedding_provider,
                            model_version_before, model_version_after, training_checkpoint,
                            training_status, internalization_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.event_type,
                            float(event.timestamp),
                            event.actor_id,
                            event.visibility_policy,
                            event.organization_id,
                            event.source_id,
                            event.source_type,
                            event.source_title,
                            event.knowledge_domain,
                            event.summary,
                            event.content_hash,
                            event.verification_status,
                            event.embedding_provider,
                            event.model_version_before,
                            event.model_version_after,
                            event.training_checkpoint,
                            event.training_status,
                            event.internalization_status,
                        ),
                    )
                    conn.commit()
        return event

    def _is_authorized(self, ev: LearningEvent, access_context: Optional[Any]) -> bool:
        pseudo_record = {
            "record_id": ev.event_id,
            "visibility_policy": ev.visibility_policy,
            "source_provenance": {
                "author": ev.actor_id,
                "organization_id": ev.organization_id or "",
            },
            "metadata": {
                "owner_actor_id": ev.actor_id,
                "organization_id": ev.organization_id or "",
            },
        }
        if access_context is None:
            # Fail-closed for PRIVATE/LOCAL/ORGANIZATION/SESSION/AGENT when no context is given
            return str(ev.visibility_policy).upper() in (
                VisibilityPolicy.COLLECTIVE.value,
                VisibilityPolicy.PUBLIC.value,
            )
        return self.authorization.can_read_record_with_context(pseudo_record, access_context)

    def query_temporal(
        self,
        access_context: Optional[Any] = None,
        since_timestamp: Optional[float] = None,
        until_timestamp: Optional[float] = None,
        event_types: Optional[List[str]] = None,
        topic_keyword: Optional[str] = None,
        limit: int = 20,
    ) -> List[LearningEvent]:
        """
        Temporal retrieval ordered strictly by event timestamp DESC (newest first).
        Enforces actor/visibility authorization before returning any event.
        """
        with self._lock:
            snapshot = list(self._events)

        type_set = set(event_types) if event_types else None
        kw = (topic_keyword or "").strip().lower()

        matched: List[LearningEvent] = []
        for ev in snapshot:
            if not self._is_authorized(ev, access_context):
                continue
            if since_timestamp is not None and ev.timestamp < since_timestamp:
                continue
            if until_timestamp is not None and ev.timestamp > until_timestamp:
                continue
            if type_set is not None and ev.event_type not in type_set:
                continue
            if kw:
                hay = f"{ev.source_title} {ev.knowledge_domain} {ev.summary} {ev.source_id}".lower()
                if kw not in hay:
                    continue
            matched.append(ev)

        matched.sort(key=lambda e: e.timestamp, reverse=True)
        return matched[: max(1, int(limit))]
