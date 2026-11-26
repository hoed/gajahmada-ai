"""
GajahMada AI — Knowledge Access Policy (Retrieval-Boundary Authorization).

Integrates directly with existing `MemoryAuthorizationService` to enforce
visibility policies (`PRIVATE`, `LOCAL`, `SHARED`, `COLLECTIVE`, `PUBLIC`),
cross-agent isolation, and cross-node isolation at retrieval time — NEVER
relying merely on prompt instructions.

Default behavior: FAIL-CLOSED (deny when permissions are missing or ambiguous).
"""
from typing import List, Optional

from gajahmada_ai.knowledge.interfaces import KnowledgeAccessContext, KnowledgeRecord
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import VisibilityPolicy


class KnowledgeAccessPolicy:
    """Fail-closed retrieval-boundary authorization guard."""

    def __init__(self, memory_auth: Optional[MemoryAuthorizationService] = None):
        self.memory_auth = memory_auth or MemoryAuthorizationService()

    def can_access_record(
        self,
        record: Optional[KnowledgeRecord],
        ctx: Optional[KnowledgeAccessContext],
    ) -> bool:
        if record is None or ctx is None:
            return False

        role = str(ctx.actor_role or "").lower().strip()
        if not role:
            return False

        vis = str(record.visibility_policy or "").upper().strip()
        if not vis:
            return False

        # 1. Delegate base visibility matrix check to existing MemoryAuthorizationService
        if not self.memory_auth.can_read_visibility(vis, role):
            return False

        # 2. Enforce caller-scoped visibility allowlist if provided
        if ctx.allowed_visibilities is not None:
            allowed_vis = {v.upper().strip() for v in ctx.allowed_visibilities}
            if vis not in allowed_vis:
                return False

        # 3. Enforce caller-scoped source_type allowlist if provided
        if ctx.allowed_source_types is not None:
            if record.source_type not in ctx.allowed_source_types:
                return False

        # 4. Enforce layer toggles
        if record.knowledge_layer == "external" and not ctx.include_external:
            return False
        if record.knowledge_layer == "memory" and not ctx.include_memory:
            return False
        if record.knowledge_layer == "distributed" and not ctx.include_distributed:
            return False

        # 5. Cross-agent isolation for PRIVATE agent-scoped records
        if (
            vis == VisibilityPolicy.PRIVATE.value
            and record.owner_agent_id
            and ctx.agent_id
            and record.owner_agent_id != ctx.agent_id
            and role != "admin"
        ):
            return False

        # 6. Cross-node isolation for LOCAL/PRIVATE records
        if (
            vis in (VisibilityPolicy.PRIVATE.value, VisibilityPolicy.LOCAL.value)
            and record.owner_node_id
            and record.owner_node_id != "local"
            and ctx.node_id
            and record.owner_node_id != ctx.node_id
            and role != "admin"
        ):
            return False

        return True

    def filter_records(
        self,
        records: List[KnowledgeRecord],
        ctx: Optional[KnowledgeAccessContext],
    ) -> List[KnowledgeRecord]:
        if ctx is None:
            return []
        return [r for r in records if self.can_access_record(r, ctx)]
