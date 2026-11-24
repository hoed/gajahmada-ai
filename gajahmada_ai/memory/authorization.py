"""
Centralized Memory Authorization Service (Round 3 Hardened).
Enforces an explicit, fail-closed access matrix across every visibility policy:
- PRIVATE    : admin, operator only (strictly isolated owner/node secret)
- LOCAL      : admin, operator only (local node working/quarantine data; NEVER exposed to anonymous or external peers)
- SHARED     : admin, operator, peer, reader (authenticated actors only; NEVER anonymous)
- COLLECTIVE : admin, operator, peer, reader, anonymous (verified collective knowledge)
- PUBLIC     : admin, operator, peer, reader, anonymous (explicitly public knowledge)

Applied consistently across record listing, version history, contradictions,
entity relations, and semantic search.
"""

from typing import Any, Dict, List, Optional, Set
from fastapi import HTTPException

from gajahmada_ai.memory.models import VisibilityPolicy


class MemoryAuthorizationService:
    PRIVILEGED_ROLES: Set[str] = {"admin", "operator"}

    VISIBILITY_ACCESS_MATRIX: Dict[str, Set[str]] = {
        VisibilityPolicy.PRIVATE.value: {"admin", "operator"},
        VisibilityPolicy.LOCAL.value: {"admin", "operator"},
        VisibilityPolicy.SHARED.value: {"admin", "operator", "peer", "reader"},
        VisibilityPolicy.COLLECTIVE.value: {"admin", "operator", "peer", "reader", "anonymous"},
        VisibilityPolicy.PUBLIC.value: {"admin", "operator", "peer", "reader", "anonymous"},
    }

    def can_read_visibility(self, visibility: str, actor_role: str = "anonymous") -> bool:
        vis = str(visibility or VisibilityPolicy.LOCAL.value).upper().strip()
        role = str(actor_role or "anonymous").lower().strip()
        allowed_roles = self.VISIBILITY_ACCESS_MATRIX.get(vis)
        if allowed_roles is None:
            # Fail-closed on unknown visibility policies
            return False
        return role in allowed_roles

    def can_read_record(self, record: Optional[Dict[str, Any]], actor_role: str = "anonymous") -> bool:
        if not record:
            return False
        return self.can_read_visibility(record.get("visibility_policy", VisibilityPolicy.LOCAL.value), actor_role)

    def require_record_read(self, record: Optional[Dict[str, Any]], actor_role: str = "anonymous") -> Dict[str, Any]:
        if not record:
            raise HTTPException(status_code=404, detail="Memory record not found.")
        if not self.can_read_record(record, actor_role):
            vis = str(record.get("visibility_policy", VisibilityPolicy.LOCAL.value)).upper()
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: {vis} memory record is not readable by role '{actor_role}'.",
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
