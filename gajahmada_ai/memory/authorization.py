"""
Centralized Memory Authorization Service (P0 Security Remediation).
Enforces visibility policies (`PRIVATE`, `LOCAL`, `SHARED`, `COLLECTIVE`) consistently
across every read path: record listing, version history, contradictions, entity relations,
and semantic search.
"""

from typing import Any, Dict, List, Optional, Set
from fastapi import HTTPException

from gajahmada_ai.memory.models import VisibilityPolicy


class MemoryAuthorizationService:
    PRIVILEGED_ROLES = {"admin", "operator"}

    def can_read_visibility(self, visibility: str, actor_role: str = "anonymous") -> bool:
        vis = str(visibility or VisibilityPolicy.LOCAL.value).upper()
        if vis == VisibilityPolicy.PRIVATE.value:
            return actor_role in self.PRIVILEGED_ROLES
        return True

    def can_read_record(self, record: Optional[Dict[str, Any]], actor_role: str = "anonymous") -> bool:
        if not record:
            return False
        return self.can_read_visibility(record.get("visibility_policy", "LOCAL"), actor_role)

    def require_record_read(self, record: Optional[Dict[str, Any]], actor_role: str = "anonymous") -> Dict[str, Any]:
        if not record:
            raise HTTPException(status_code=404, detail="Memory record not found.")
        if not self.can_read_record(record, actor_role):
            raise HTTPException(
                status_code=403,
                detail="Access denied: PRIVATE memory record requires authenticated admin/operator privileges.",
            )
        return record

    def filter_records(self, records: List[Dict[str, Any]], actor_role: str = "anonymous") -> List[Dict[str, Any]]:
        return [r for r in records if self.can_read_record(r, actor_role)]

    def visible_record_ids(self, storage: Any, actor_role: str = "anonymous", limit: int = 1000) -> Set[str]:
        all_recs = storage.list_records(limit=limit)
        return {r["record_id"] for r in all_recs if self.can_read_record(r, actor_role)}

    def filter_contradictions(
        self,
        contradictions: List[Dict[str, Any]],
        allowed_ids: Set[str],
        actor_role: str = "anonymous",
    ) -> List[Dict[str, Any]]:
        if actor_role in self.PRIVILEGED_ROLES:
            return contradictions
        return [
            c for c in contradictions
            if c.get("record_a_id") in allowed_ids and c.get("record_b_id") in allowed_ids
        ]

    def filter_relations(
        self,
        relations: List[Dict[str, Any]],
        allowed_ids: Set[str],
        actor_role: str = "anonymous",
    ) -> List[Dict[str, Any]]:
        if actor_role in self.PRIVILEGED_ROLES:
            return relations
        return [
            rel for rel in relations
            if not rel.get("record_id") or rel.get("record_id") in allowed_ids
        ]
