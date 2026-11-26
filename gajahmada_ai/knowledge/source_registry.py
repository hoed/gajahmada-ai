"""
GajahMada AI — Persistent Knowledge Source Registry.

Tracks full provenance, authorship, license/usage restrictions, SHA-256 content
hashes, verification status, versioning, and parent lineage for all ingested
external and federated knowledge sources.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.knowledge.interfaces import KnowledgeSource


class SourceRegistry:
    """Persistent, atomic JSON-backed registry for external & federated knowledge sources."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._sources: Dict[str, KnowledgeSource] = {}
        self._hash_index: Dict[str, str] = {}
        if self.storage_path and os.path.exists(self.storage_path):
            self._load()

    def register(self, source: KnowledgeSource) -> KnowledgeSource:
        existing = self._sources.get(source.source_id)
        if existing is not None:
            source.version = existing.version + 1
            if not source.parent_source_id:
                source.parent_source_id = existing.source_id
        self._sources[source.source_id] = source
        if source.document_hash:
            self._hash_index[source.document_hash] = source.source_id
        self._save()
        return source

    def get(self, source_id: str) -> Optional[KnowledgeSource]:
        return self._sources.get(source_id)

    def find_by_hash(self, document_hash: str) -> Optional[KnowledgeSource]:
        sid = self._hash_index.get(document_hash)
        return self._sources.get(sid) if sid else None

    def list_sources(
        self,
        source_type: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self._sources.values():
            if source_type and s.source_type != source_type:
                continue
            if verification_status and s.verification_status != verification_status:
                continue
            out.append(s.to_dict())
        return out

    def update_verification(
        self,
        source_id: str,
        verification_status: str,
        training_approved: Optional[bool] = None,
        verifier_note: str = "",
    ) -> KnowledgeSource:
        src = self._sources.get(source_id)
        if src is None:
            raise KeyError(f"Source '{source_id}' not found in SourceRegistry.")
        src.verification_status = verification_status
        if training_approved is not None:
            src.training_approved = training_approved
        if verifier_note:
            history = src.provenance.setdefault("verification_history", [])
            history.append({"status": verification_status, "note": verifier_note, "ts": round(time.time(), 3)})
        self._save()
        return src

    def _save(self):
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        tmp = f"{self.storage_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({sid: s.to_dict() for sid, s in self._sources.items()}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.storage_path)

    def _load(self):
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for sid, item in raw.items():
                    s = KnowledgeSource.from_dict(item)
                    self._sources[sid] = s
                    if s.document_hash:
                        self._hash_index[s.document_hash] = sid
        except Exception:
            self._sources = {}
            self._hash_index = {}
