"""
GajahMada AI — Canonical Request & Actor Access Context (Phase 1 P0).

Single unified `AccessContext` propagated consistently across:
Incoming Request -> AuthManager -> CognitiveReasoningPipeline ->
HybridKnowledgeController -> HybridKnowledgeRetriever -> MemoryManager -> Response.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import uuid


@dataclass(frozen=True)
class AccessContext:
    """Immutable canonical actor & authorization context."""

    actor_id: str = "anonymous"
    actor_role: str = "anonymous"
    organization_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    access_level: str = "public"  # public | reader | peer | operator | admin
    permissions: frozenset = field(default_factory=frozenset)
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}")

    @classmethod
    def from_actor_dict(
        cls,
        actor_dict: Optional[Dict[str, Any]],
        organization_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        permissions: Optional[Set[str]] = None,
        request_id: Optional[str] = None,
    ) -> "AccessContext":
        info = actor_dict or {}
        role = str(info.get("role", "anonymous")).lower().strip() or "anonymous"
        actor = str(info.get("actor", "anonymous")).strip() or "anonymous"
        level_map = {
            "admin": "admin",
            "operator": "operator",
            "peer": "peer",
            "reader": "reader",
        }
        return cls(
            actor_id=actor,
            actor_role=role,
            organization_id=organization_id,
            agent_id=agent_id,
            session_id=session_id,
            node_id=node_id,
            access_level=level_map.get(role, "public"),
            permissions=frozenset(permissions or ()),
            request_id=request_id or f"req-{uuid.uuid4().hex[:12]}",
        )

    def to_knowledge_access_context(
        self,
        include_external: bool = True,
        include_memory: bool = True,
        include_distributed: bool = False,
        max_results: int = 5,
        allowed_source_types: Optional[List[str]] = None,
        allowed_visibilities: Optional[List[str]] = None,
    ):
        from gajahmada_ai.knowledge.interfaces import KnowledgeAccessContext

        return KnowledgeAccessContext(
            actor_id=self.actor_id,
            actor_role=self.actor_role,
            agent_id=self.agent_id,
            node_id=self.node_id,
            org_id=self.organization_id,
            session_id=self.session_id,
            request_id=self.request_id,
            allowed_source_types=allowed_source_types,
            allowed_visibilities=allowed_visibilities,
            include_external=include_external,
            include_memory=include_memory,
            include_distributed=include_distributed,
            max_results=max_results,
        )
