"""
GajahMada AI — Privacy-Safe Structured Observability & Subsystem Health (Phase 5 P2).

Records structured lifecycle metrics WITHOUT storing private prompts or private memory
content in logs, and provides honest per-subsystem health classification:
  HEALTHY | DEGRADED | UNAVAILABLE | NOT_CONFIGURED
"""
import hashlib
import time
from typing import Any, Dict, List, Optional


class LifecycleObservabilityLogger:
    """In-memory bounded ring buffer of privacy-safe structured lifecycle events."""

    def __init__(self, max_events: int = 500):
        self.max_events = max_events
        self._events: List[Dict[str, Any]] = []

    @staticmethod
    def _privacy_actor_tag(actor_id: str) -> str:
        clean = (actor_id or "anonymous").strip()
        if clean in ("anonymous", "admin", "operator") or clean.startswith("role:"):
            return clean
        return f"actor:{hashlib.sha256(clean.encode('utf-8')).hexdigest()[:12]}"

    def record(
        self,
        event_name: str,
        request_id: str,
        actor_id: str = "anonymous",
        active_model_version: Optional[str] = None,
        retrieval_duration_ms: Optional[float] = None,
        retrieved_count: Optional[int] = None,
        embedding_provider: Optional[str] = None,
        context_tokens_used: Optional[int] = None,
        inference_duration_ms: Optional[float] = None,
        status: str = "OK",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Explicitly scrub any accidental prompt/content fields from `extra`
        safe_extra = {
            k: v for k, v in (extra or {}).items()
            if k not in ("prompt", "content", "private_memory", "secret", "api_key")
        }
        entry = {
            "timestamp": round(time.time(), 3),
            "event_name": event_name,
            "request_id": request_id,
            "actor_tag": self._privacy_actor_tag(actor_id),
            "active_model_version": active_model_version,
            "retrieval_duration_ms": retrieval_duration_ms,
            "retrieved_count": retrieved_count,
            "embedding_provider": embedding_provider,
            "context_tokens_used": context_tokens_used,
            "inference_duration_ms": inference_duration_ms,
            "status": status,
            "extra": safe_extra,
        }
        self._events.append(entry)
        if len(self._events) > self.max_events:
            del self._events[: len(self._events) - self.max_events]
        return entry

    def recent(self, limit: int = 25) -> List[Dict[str, Any]]:
        return self._events[-limit:]


def build_subsystem_health_report(rt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Honest subsystem health classification distinguishing:
    HEALTHY | DEGRADED | UNAVAILABLE | NOT_CONFIGURED.
    """
    engine = rt.get("engine")
    cap = engine.capability_status() if engine else {}
    memory = rt.get("memory")
    blockchain = rt.get("blockchain")
    retriever = rt.get("hybrid_retriever")
    reg = rt.get("gnnm_registry")
    supabase = rt.get("supabase")
    p2p = rt.get("p2p")

    from gajahmada_ai.gnnm.model import HAS_TORCH

    gnnm_status = (
        "HEALTHY" if cap.get("neural_inference_active")
        else ("DEGRADED" if HAS_TORCH else "UNAVAILABLE")
    )
    knowledge_status = "HEALTHY" if retriever is not None else "NOT_CONFIGURED"
    memory_status = "HEALTHY" if memory is not None else "UNAVAILABLE"
    ledger_status = (
        "HEALTHY" if (blockchain and blockchain.status == "HEALTHY" and blockchain.is_chain_valid())
        else "DEGRADED"
    )
    learning_status = "HEALTHY" if HAS_TORCH else "UNAVAILABLE"
    registry_status = "HEALTHY" if reg is not None else "NOT_CONFIGURED"
    distributed_status = "HEALTHY" if (p2p is not None and len(getattr(p2p, "peers", [])) > 0) else "NOT_CONFIGURED"
    supabase_status = "HEALTHY" if (supabase and getattr(supabase, "configured", False)) else "NOT_CONFIGURED"

    return {
        "api": {"status": "HEALTHY", "version": "2.0.0"},
        "gnnm_runtime": {
            "status": gnnm_status,
            "neural_inference_active": bool(cap.get("neural_inference_active", False)),
            "active_backend": cap.get("active_neural_backend"),
            "active_model_id": cap.get("active_model_id"),
            "pytorch_available": bool(HAS_TORCH),
        },
        "knowledge_subsystem": {
            "status": knowledge_status,
            "external_chunks": len(retriever.list_external_records()) if retriever else 0,
            "embedding_provider": retriever.embedding_provider.identity_metadata() if retriever else None,
        },
        "memory_subsystem": {
            "status": memory_status,
            "storage": memory.storage.health() if memory else None,
        },
        "learning_pipeline": {
            "status": learning_status,
            "pytorch_training_capable": bool(HAS_TORCH),
        },
        "model_registry": {
            "status": registry_status,
            **(reg.runtime_status() if reg else {}),
        },
        "audit_ledger": {
            "status": ledger_status,
            "type": "tamper_evident_local_signed_event_ledger",
            "height": len(blockchain.chain) if blockchain else 0,
        },
        "distributed_network": {
            "status": distributed_status,
            "peers_count": len(getattr(p2p, "peers", [])) if p2p else 0,
        },
        "supabase_remote_store": {
            "status": supabase_status,
        },
    }
