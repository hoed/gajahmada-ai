import time
from typing import Any, Callable, Dict, List, Optional

from gajahmada_ai.cognition.perception import PerceptionModule
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.collective import CollectiveMemory
from gajahmada_ai.memory.consolidation import MemoryConsolidator
from gajahmada_ai.memory.episodic import EpisodicMemory
from gajahmada_ai.memory.knowledge_graph import KnowledgeGraphManager
from gajahmada_ai.memory.models import (
    MemoryRecord,
    MemoryType,
    SourceProvenance,
    VerificationStatus,
    VisibilityPolicy,
)
from gajahmada_ai.memory.procedural import ProceduralMemory
from gajahmada_ai.memory.provenance import ProvenanceValidator
from gajahmada_ai.memory.retrieval import HybridMemoryRetriever
from gajahmada_ai.memory.semantic import SemanticMemory
from gajahmada_ai.memory.versioning import MemoryVersionController
from gajahmada_ai.memory.working import WorkingMemory
from gajahmada_ai.neural.llama_backend import deterministic_hash_embed
from gajahmada_ai.reasoning.contradictions import ContradictionDetector
from gajahmada_ai.storage.interfaces import StorageBackend
from gajahmada_ai.storage.object_store import ContentAddressedObjectStore
from gajahmada_ai.storage.postgres import RelationalMemoryStore


class MemoryManager:
    """
    Unified 5-Tier Memory & Knowledge Engine (Layer 4).
    Enforces all 6 GajahMada Memory Rules:
    1. Unverified knowledge remains unverified.
    2. Contradictions are preserved and investigated (never silently overwritten).
    3. New evidence may supersede older conclusions via `correct_memory()`.
    4. Previous versions remain auditable via `MemoryVersionController`.
    5. Collective knowledge preserves its original `origin_node_id` & signature.
    6. Memory synchronization respects visibility policies (`PRIVATE`, `LOCAL`, `SHARED`, `COLLECTIVE`).
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        sqlite_path: str = "./knowledge_store/gajahmada_v2.db",
        backup_dir: str = "./knowledge_store/objects",
    ):
        self.storage = storage or RelationalMemoryStore(sqlite_path=sqlite_path)
        self.embed_fn = embed_fn or deterministic_hash_embed
        self.working = WorkingMemory(capacity=64)
        self.semantic = SemanticMemory(self.storage)
        self.episodic = EpisodicMemory(self.storage)
        self.procedural = ProceduralMemory(self.storage)
        self.collective = CollectiveMemory(self.storage)
        self.knowledge_graph = KnowledgeGraphManager(self.storage)
        self.provenance = ProvenanceValidator()
        self.versioning = MemoryVersionController(self.storage)
        self.retriever = HybridMemoryRetriever(self.storage, self.embed_fn)
        self.consolidator = MemoryConsolidator()
        self.contradiction_detector = ContradictionDetector()
        self.perception = PerceptionModule()
        self.object_store = ContentAddressedObjectStore(base_dir=backup_dir)
        self.authorization = MemoryAuthorizationService()

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        origin_node_id: str = "local",
        source_type: str = "operator_ingested",
        author: str = "local_operator",
        signature: str = "",
        confidence: float = 0.75,
        verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
        visibility_policy: VisibilityPolicy = VisibilityPolicy.LOCAL,
        record_id: Optional[str] = None,
        owner_actor_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty")

        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type.lower())
        if isinstance(verification_status, str):
            verification_status = VerificationStatus(verification_status.upper())
        if isinstance(visibility_policy, str):
            visibility_policy = VisibilityPolicy(visibility_policy.upper())

        # Rule 1: Unverified sources remain UNVERIFIED
        if source_type in ("unverified_peer", "llm_hypothesis") and verification_status == VerificationStatus.VERIFIED:
            verification_status = VerificationStatus.UNVERIFIED

        perceived = self.perception.perceive(clean_content)
        embedding = self.embed_fn(clean_content)

        prov = SourceProvenance(
            origin_node_id=origin_node_id,
            source_type=source_type,
            author=owner_actor_id or author,
            signature=signature,
        )
        scope_meta: Dict[str, Any] = {}
        if owner_actor_id:
            scope_meta["owner_actor_id"] = owner_actor_id
        if organization_id:
            scope_meta["organization_id"] = organization_id
        if agent_id:
            scope_meta["agent_id"] = agent_id
        if session_id:
            scope_meta["session_id"] = session_id
        rec_kwargs: Dict[str, Any] = {
            "content": clean_content,
            "memory_type": memory_type,
            "entity_references": perceived.entities,
            "source_provenance": prov,
            "confidence": confidence,
            "verification_status": verification_status,
            "visibility_policy": visibility_policy,
            "embedding": embedding,
        }
        if record_id:
            rec_kwargs["record_id"] = record_id

        record = MemoryRecord(**rec_kwargs).ensure_hash()

        if memory_type == MemoryType.WORKING:
            d = record.to_storage_dict()
            self.working.push(d)
            return d

        # Rule 2: Scan existing semantic/collective memories for contradictions and preserve both!
        existing_records = self.storage.list_records(memory_type=memory_type.value, limit=250)
        detected_contradictions = []
        for ex in existing_records:
            if ex["record_id"] == record.record_id:
                continue
            contra = self.contradiction_detector.check_pair(
                ex["record_id"],
                ex["content"],
                record.record_id,
                record.content,
            )
            if contra is not None:
                c_dict = {
                    "contradiction_id": contra.contradiction_id,
                    "record_a_id": contra.record_a_id,
                    "record_b_id": contra.record_b_id,
                    "subject": contra.subject,
                    "claim_a": contra.claim_a,
                    "claim_b": contra.claim_b,
                    "reason": contra.reason,
                    "severity": contra.severity,
                    "resolution_status": "OPEN",
                    "detected_at": time.time(),
                }
                self.storage.save_contradiction(c_dict)
                detected_contradictions.append(c_dict)

        rec_dict = record.to_storage_dict()
        if scope_meta:
            rec_dict["source_provenance"] = {**(rec_dict.get("source_provenance") or {}), **scope_meta}
        self.storage.save_record(rec_dict)
        self.versioning.record_version_snapshot(rec_dict, modified_by=author, reason="initial_store")
        self.knowledge_graph.extract_and_store_relations(
            record.record_id,
            clean_content,
            perceived.entities,
            confidence=confidence,
        )
        rec_dict["detected_contradictions"] = detected_contradictions
        return rec_dict

    def correct_memory(
        self,
        record_id: str,
        new_content: str,
        modified_by: str = "local_operator",
        reason: str = "evidence_supersession",
        new_confidence: Optional[float] = None,
        new_verification_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Rule 3 & Rule 4: Update/supersede a memory record while preserving previous
        versions in the immutable version table.
        """
        existing = self.storage.get_record(record_id)
        if not existing:
            raise KeyError(f"Memory record '{record_id}' not found")

        new_ver = int(existing.get("version", 1)) + 1
        conf = float(new_confidence if new_confidence is not None else existing.get("confidence", 0.75))
        v_status = new_verification_status or existing.get("verification_status", "UNVERIFIED")
        perceived = self.perception.perceive(new_content)
        emb = self.embed_fn(new_content)

        prov_dict = existing.get("source_provenance", {})
        rec = MemoryRecord(
            record_id=record_id,
            content=new_content.strip(),
            memory_type=MemoryType(existing["memory_type"]),
            entity_references=perceived.entities,
            source_provenance=SourceProvenance(**prov_dict),
            created_at=float(existing["created_at"]),
            modified_at=time.time(),
            version=new_ver,
            confidence=conf,
            verification_status=VerificationStatus(v_status),
            visibility_policy=VisibilityPolicy(existing.get("visibility_policy", "LOCAL")),
            embedding=emb,
        ).ensure_hash()

        updated_dict = rec.to_storage_dict()
        self.storage.save_record(updated_dict)
        self.versioning.record_version_snapshot(updated_dict, modified_by=modified_by, reason=reason)
        return updated_dict

    def delete_memory(self, record_id: str, hard: bool = False) -> bool:
        return self.storage.delete_record(record_id, hard=hard)

    def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        only_verified: bool = False,
        access_context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memory records with direct memory-boundary authorization (Phase 1 P0).
        - When `access_context` is provided, strictly enforces `PRIVATE`, `SESSION`, `AGENT`,
          `ORGANIZATION`, `LOCAL`, `SHARED`, `COLLECTIVE`, `PUBLIC`, and `DISTRIBUTED_SHARED`.
        - When `access_context` is None (legacy internal callers), NEVER returns strictly scoped
          records (`PRIVATE`, `SESSION`, `AGENT`, `ORGANIZATION`) — fail-closed by default.
        """
        fetch_k = max(top_k * 3, 15)
        raw = self.retriever.retrieve(
            query=query,
            memory_type=memory_type,
            top_k=fetch_k,
            only_verified=only_verified,
        )
        if access_context is not None:
            authorized = self.authorization.filter_records_with_context(raw, access_context)
            return authorized[:top_k]

        restricted_without_ctx = {
            VisibilityPolicy.PRIVATE.value,
            VisibilityPolicy.SESSION.value,
            VisibilityPolicy.AGENT.value,
            VisibilityPolicy.ORGANIZATION.value,
        }
        safe = [
            r for r in raw
            if str(r.get("visibility_policy", "LOCAL")).upper() not in restricted_without_ctx
        ]
        return safe[:top_k]

    def get_shareable_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Rule 6: Only export records whose visibility policy permits sharing."""
        all_recs = self.storage.list_records(include_deleted=False, limit=limit)
        return [r for r in all_recs if self.provenance.can_share_externally(r)]

    def export_backup(self) -> Dict[str, Any]:
        records = self.storage.list_records(include_deleted=False, limit=5000)
        contradictions = self.storage.list_contradictions()
        relations = self.storage.list_relations()
        payload = {
            "schema_version": "2.0.0",
            "exported_at": round(time.time(), 3),
            "records_count": len(records),
            "records": records,
            "contradictions": contradictions,
            "relations": relations,
        }
        meta = self.object_store.put_json(payload)
        return {
            "backup_hash": meta["object_hash"],
            "backup_path": meta["path"],
            "size_bytes": meta["size_bytes"],
            "records_count": len(records),
            "contradictions_count": len(contradictions),
            "timestamp": meta["timestamp"],
        }

    def stats(self) -> Dict[str, Any]:
        h = self.storage.health()
        by_type = {}
        for mt in MemoryType:
            if mt == MemoryType.WORKING:
                by_type[mt.value] = self.working.count()
            else:
                by_type[mt.value] = len(self.storage.list_records(memory_type=mt.value, limit=1000))
        return {
            "storage_health": h,
            "tiers": by_type,
            "total_persistent_records": h["active_records"],
            "contradictions_count": h["contradictions_count"],
            "relations_count": h["relations_count"],
            "versions_count": h["versions_count"],
        }
