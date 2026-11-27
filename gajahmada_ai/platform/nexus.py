"""
NEXUS PRIME — Lightweight Orchestration & Coordination Layer (Phase 12).

Within Vercel Free:
- Request routing & model capability selection
- Honest native inference reporting (`NATIVE MODEL INFERENCE: UNAVAILABLE IN CURRENT RUNTIME`)
- Knowledge retrieval provider selection (via RetrievalProviderRegistry)
- Lightweight task coordination & execution status metadata

Future distributed capabilities (permanent P2P nodes, federated learning,
distributed inference/training, multi-node orchestration) are preserved as
clean documented interfaces explicitly marked `PLANNED` on Vercel Free.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.knowledge.providers import RetrievalProviderRegistry


class NexusPrimeOrchestrator:
    """Lightweight, stateless-friendly request & capability orchestrator."""

    def __init__(
        self,
        model_router: Optional[Any] = None,
        provider_registry: Optional[RetrievalProviderRegistry] = None,
        scheduler: Optional[Any] = None,
        is_vercel: Optional[bool] = None,
    ):
        self.model_router = model_router
        self.provider_registry = provider_registry
        self.scheduler = scheduler
        self.is_vercel = (os.environ.get("VERCEL") == "1") if is_vercel is None else is_vercel
        self._recent_routes: List[Dict[str, Any]] = []

    def select_execution_plan(self, task_type: str = "general", requires_retrieval: bool = True) -> Dict[str, Any]:
        cap = self.model_router.capability_report() if self.model_router else {}
        neural_active = bool(cap.get("neural_inference_active", False)) and not self.is_vercel
        selected_provider = (
            self.provider_registry.select_active_provider().name
            if self.provider_registry
            else "NOT_CONFIGURED"
        )
        plan = {
            "timestamp": round(time.time(), 3),
            "task_type": task_type,
            "runtime_mode": "VERCEL_FREE_SERVERLESS" if self.is_vercel else "LOCAL_NODE",
            "native_inference_status": (
                f"ACTIVE ({cap.get('active_model_id')})"
                if neural_active
                else "NATIVE MODEL INFERENCE: UNAVAILABLE IN CURRENT RUNTIME"
            ),
            "symbolic_reasoning": "ACTIVE",
            "retrieval_provider": selected_provider if requires_retrieval else "SKIPPED",
            "persistent_background_workers": "DISABLED" if self.is_vercel else "AVAILABLE_LOCALLY",
        }
        self._recent_routes.append(plan)
        if len(self._recent_routes) > 30:
            self._recent_routes = self._recent_routes[-30:]
        return plan

    def status_report(self) -> Dict[str, Any]:
        plan = self.select_execution_plan(task_type="status_probe")
        providers = self.provider_registry.report() if self.provider_registry else {"providers": []}
        tasks_total = len(self.scheduler.tasks) if self.scheduler else 0
        agents_count = len(self.scheduler.list_agents()) if self.scheduler else 0

        return {
            "layer": "NEXUS PRIME",
            "runtime_mode": "VERCEL_FREE_SERVERLESS" if self.is_vercel else "LOCAL_NODE",
            "native_inference_banner": plan["native_inference_status"],
            "active_capabilities": [
                {"name": "Request & Capability Routing", "status": "ACTIVE"},
                {"name": "Symbolic & Epistemic Reasoning", "status": "ACTIVE"},
                {"name": "Hybrid Knowledge Retrieval Coordination", "status": "ACTIVE", "provider": providers.get("selected_provider")},
                {"name": "Lightweight Task Metadata", "status": "ACTIVE", "registered_agents": agents_count, "recorded_tasks": tasks_total},
                {"name": "Model Factory & Registry Control Plane", "status": "ACTIVE"},
            ],
            "future_distributed_roadmap": [
                {"name": "Permanent Multi-Node P2P Mesh", "status": "UNAVAILABLE ON VERCEL FREE" if self.is_vercel else "EXPERIMENTAL", "note": "No persistent TCP/WebSocket daemons inside Vercel Functions."},
                {"name": "Distributed Multi-GPU Training (TITANFORGE 70B)", "status": "PLANNED", "note": "Requires external FSDP/TP GPU cluster."},
                {"name": "Distributed Inference & High-Availability Serving", "status": "PLANNED", "note": "Future external model serving endpoints."},
                {"name": "Federated Learning & Knowledge Federation", "status": "PLANNED", "note": "Interfaces preserved in gajahmada_ai/network/federation.py."},
            ],
            "retrieval_providers": providers,
            "recent_routing_decisions": self._recent_routes[-10:],
        }
