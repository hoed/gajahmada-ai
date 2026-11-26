"""
Centralized Memory Authorization Service (Round 3 + Master Phase 1 P0 Hardened).
Enforces an explicit, fail-closed access matrix across ALL visibility scopes:
- PRIVATE            : owner actor or admin only
- SESSION            : matching session_id or admin only
- AGENT              : matching agent_id or admin only
- ORGANIZATION       : matching organization_id or admin only
- LOCAL              : admin, operator on local node only
- SHARED             : authenticated actors (admin, operator, peer, reader)
- DISTRIBUTED_SHARED : authenticated actors & trusted peers
- COLLECTIVE         : public / all roles
- PUBLIC             : public / all roles

Records access-denied events via optional `SecurityAuditLogger` WITHOUT exposing
private content in logs or error messages.
"""

from typing import Any, Dict, List, Optional, Set
from fastapi import HTTPException

from gajahmada_ai.memory.models import VisibilityPolicy


class MemoryAuthorizationService:
    PRIVILEGED_ROLES: Set[str] = {"admin", "operator"}

    VISIBILITY_ACCESS_MATRIX: Dict[str, Set[str]] = {
        VisibilityPolicy.PRIVATE.value: {"admin", "operator"},
        VisibilityPolicy.SESSION.value: {"admin", "operator", "reader"},
        VisibilityPolicy.AGENT.value: {"admin", "operator"},
        VisibilityPolicy.ORGANIZATION.value: {"admin", "operator", "reader"},
        VisibilityPolicy.LOCAL.value: {"admin", "operator"},
        VisibilityPolicy.SHARED.value: {"admin", "operator", "peer", "reader"},
        VisibilityPolicy.DISTRIBUTED_SHARED.value: {"admin", "operator", "peer", "reader"},
        VisibilityPolicy.COLLECTIVE.value: {"admin", "operator", "peer", "reader", "anonymous"},
        VisibilityPolicy.PUBLIC.value: {"admin", "operator", "peer", "reader", "anonymous"},
    }

    def __init__(self, audit_logger: Optional[Any] = None):
        self.audit_logger = audit_logger

    def _log_denial(self, record_id: str, visibility: str, actor_id: str, reason: str, request_id: str = ""):
        if self.audit_logger is not None:
            try:
                self.audit_logger.record(
                    event_type="MEMORY_ACCESS_DENIED",
                    severity="WARN",
                    actor=actor_id or "anonymous",
                    action=f"memory:read:{record_id}",
                    outcome="DENIED",
                    # NEVER log private record content!
                    details={"visibility": visibility, "reason": reason, "request_id": request_id},
                )
            except Exception:
                pass

    def can_read_visibility(self, visibility: str, actor_role: str = "anonymous") -> bool:
        vis = str(visibility or VisibilityPolicy.LOCAL.value).upper().strip()
        role = str(actor_role or "anonymous").lower().strip()
        allowed_roles = self.VISIBILITY_ACCESS_MATRIX.get(vis)
        if allowed_roles is None:
            return False
        return role in allowed_roles

    def can_read_record(self, record: Optional[Dict[str, Any]], actor_role: str = "anonymous") -> bool:
        if not record:
            return False
        return self.can_read_visibility(record.get("visibility_policy", VisibilityPolicy.LOCAL.value), actor_role)

    def can_read_record_with_context(
        self,
        record: Optional[Dict[str, Any]],
        access_context: Optional[Any],
    ) -> bool:
        """
        Strict scope-aware authorization using canonical `AccessContext`.
        Fail-closed when `access_context` is None.
        """
        if not record or access_context is None:
            return False

        role = str(getattr(access_context, "actor_role", "anonymous") or "anonymous").lower().strip()
        actor_id = str(getattr(access_context, "actor_id", "anonymous") or "anonymous").strip()
        req_id = str(getattr(access_context, "request_id", "") or "")
        rec_id = str(record.get("record_id", "unknown"))
        vis = str(record.get("visibility_policy", VisibilityPolicy.LOCAL.value)).upper().strip()

        if not self.can_read_visibility(vis, role):
            self._log_denial(rec_id, vis, actor_id, f"role_{role}_not_allowed_for_{vis}", req_id)
            return False

        # Admin bypasses owner/scope match only if explicitly role == "admin"
        if role == "admin":
            return True

        prov = record.get("source_provenance") if isinstance(record.get("source_provenance"), dict) else {}
        meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

        owner_actor = str(meta.get("owner_actor_id") or prov.get("author") or "").strip()
        owner_org = str(meta.get("organization_id") or prov.get("organization_id") or "").strip()
        owner_agent = str(meta.get("agent_id") or prov.get("agent_id") or "").strip()
        owner_session = str(meta.get("session_id") or prov.get("session_id") or "").strip()

        if vis == VisibilityPolicy.PRIVATE.value:
            if owner_actor and owner_actor != actor_id:
                self._log_denial(rec_id, vis, actor_id, "cross_user_private_denied", req_id)
                return False

        elif vis == VisibilityPolicy.SESSION.value:
            ctx_session = str(getattr(access_context, "session_id", "") or "").strip()
            if not ctx_session or (owner_session and owner_session != ctx_session):
                self._log_denial(rec_id, vis, actor_id, "cross_session_denied", req_id)
                return False

        elif vis == VisibilityPolicy.AGENT.value:
            ctx_agent = str(getattr(access_context, "agent_id", "") or "").strip()
            if not ctx_agent or (owner_agent and owner_agent != ctx_agent):
                self._log_denial(rec_id, vis, actor_id, "cross_agent_denied", req_id)
                return False

        elif vis == VisibilityPolicy.ORGANIZATION.value:
            ctx_org = str(getattr(access_context, "organization_id", "") or "").strip()
            if not ctx_org or (owner_org and owner_org != ctx_org):
                self._log_denial(rec_id, vis, actor_id, "cross_organization_denied", req_id)
                return False

        return True

    def filter_records_with_context(
        self,
        records: List[Dict[str, Any]],
        access_context: Optional[Any],
    ) -> List[Dict[str, Any]]:
        if access_context is None:
            return []
        return [r for r in records if self.can_read_record_with_context(r, access_context)]

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
