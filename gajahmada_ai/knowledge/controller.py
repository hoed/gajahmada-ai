"""
GajahMada AI — Hybrid Knowledge Controller & GNNM Knowledge Adapter.

Coordinates all three knowledge systems without tightly coupling them to any single LLM backend:
1. Internal Knowledge (GNNM parameters via `ModelRouter`)
2. External Knowledge (documents, structured data, KG, distributed nodes via `HybridKnowledgeRetriever`)
3. Persistent Memory (via existing `MemoryManager`)

Also provides `GNNMKnowledgeAdapter` to connect token-budget-aware retrieved context
into `ModelRouter.route_generation(...)` while preserving all existing cognitive stages.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from gajahmada_ai.knowledge.context_builder import BuiltContext, TokenBudgetContextBuilder
from gajahmada_ai.knowledge.interfaces import (
    HybridKnowledgeConfig,
    KnowledgeAccessContext,
    KnowledgeController,
    RetrievalResult,
)
from gajahmada_ai.knowledge.retriever import HybridKnowledgeRetriever
from gajahmada_ai.knowledge.source_registry import SourceRegistry
from gajahmada_ai.reasoning.contradictions import ContradictionDetector


_PURE_MATH_RE = re.compile(r"^\s*(hitung|berapa|calculate|compute)?\s*[\d\s\+\-\*\/\(\)\.\^]+\??\s*$", re.IGNORECASE)
_SMALLTALK_RE = re.compile(r"^\s*(halo|hai|hello|hi|selamat pagi|selamat malam|terima kasih|thanks?)\W*$", re.IGNORECASE)


class HybridKnowledgeController(KnowledgeController):
    """Unified controller orchestrating Internal, External, and Persistent Memory access."""

    def __init__(
        self,
        retriever: HybridKnowledgeRetriever,
        source_registry: Optional[SourceRegistry] = None,
        context_builder: Optional[TokenBudgetContextBuilder] = None,
        config: Optional[HybridKnowledgeConfig] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
    ):
        self.retriever = retriever
        self.source_registry = source_registry or SourceRegistry()
        self.config = config or HybridKnowledgeConfig()
        self.context_builder = context_builder or TokenBudgetContextBuilder(
            max_seq_len=max(256, self.config.max_context_tokens)
        )
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self._agent_policies: Dict[str, Dict[str, Any]] = {}

    def register_agent_policy(self, agent_id: str, policy: Dict[str, Any]):
        """Configure agent-specific retrieval boundaries (allowed layers, source types, max_results)."""
        self._agent_policies[agent_id] = dict(policy)

    def _apply_agent_policy(self, ctx: KnowledgeAccessContext) -> KnowledgeAccessContext:
        if not ctx.agent_id or ctx.agent_id not in self._agent_policies:
            return ctx
        pol = self._agent_policies[ctx.agent_id]
        return KnowledgeAccessContext(
            actor_id=ctx.actor_id,
            actor_role=ctx.actor_role,
            agent_id=ctx.agent_id,
            node_id=ctx.node_id,
            org_id=ctx.org_id,
            allowed_source_types=pol.get("allowed_source_types", ctx.allowed_source_types),
            allowed_visibilities=pol.get("allowed_visibilities", ctx.allowed_visibilities),
            include_external=pol.get("include_external", ctx.include_external),
            include_memory=pol.get("include_memory", ctx.include_memory),
            include_distributed=pol.get("include_distributed", ctx.include_distributed),
            max_results=int(pol.get("max_results", ctx.max_results)),
        )

    def should_retrieve(self, query: str, access_context: KnowledgeAccessContext) -> Tuple[bool, str]:
        if not self.config.enabled:
            return False, "hybrid_knowledge_disabled"
        clean = (query or "").strip()
        if not clean:
            return False, "empty_query"
        if _PURE_MATH_RE.match(clean):
            return False, "deterministic_math_no_retrieval_needed"
        if _SMALLTALK_RE.match(clean):
            return False, "conversational_greeting_internal_only"
        return True, "retrieval_recommended"

    def detect_conflicts(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        if len(results) < 2:
            return []
        out: List[Dict[str, Any]] = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                ra = results[i].record
                rb = results[j].record
                found = self.contradiction_detector.check_pair(
                    ra.record_id, ra.content, rb.record_id, rb.content
                )
                if found is not None:
                    out.append(dict(found.__dict__))
        return out

    def query(
        self,
        query: str,
        access_context: KnowledgeAccessContext,
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        effective_ctx = self._apply_agent_policy(access_context)
        need_retrieval, reason = self.should_retrieve(query, effective_ctx)
        if not need_retrieval:
            return {
                "retrieval_executed": False,
                "reason": reason,
                "results": [],
                "conflicts": [],
                "memory_compatible_records": [],
                "knowledge_layers_used": {"external": 0, "memory": 0, "distributed": 0},
            }

        k = top_k if top_k is not None else min(effective_ctx.max_results, self.config.max_results)
        hits = self.retriever.retrieve(
            query=query,
            access_context=effective_ctx,
            top_k=max(1, int(k)),
            metadata_filters=metadata_filters,
        )
        conflicts = self.detect_conflicts(hits)

        layers = {"external": 0, "memory": 0, "distributed": 0}
        for h in hits:
            layer = h.record.knowledge_layer
            if layer in layers:
                layers[layer] += 1

        return {
            "retrieval_executed": True,
            "reason": reason,
            "results": hits,
            "conflicts": conflicts,
            "memory_compatible_records": [h.record.to_memory_compatible_dict() for h in hits],
            "knowledge_layers_used": layers,
        }

    def build_context(
        self,
        question: str,
        results: List[RetrievalResult],
        max_context_tokens: Optional[int] = None,
    ) -> BuiltContext:
        return self.context_builder.build(
            question=question,
            results=results,
            max_context_tokens=max_context_tokens or self.config.max_context_tokens,
        )


class GNNMKnowledgeAdapter:
    """
    Non-invasive adapter bridging `HybridKnowledgeController` and existing `ModelRouter`.
    Does NOT replace `ModelRouter` or bypass `CognitiveReasoningPipeline`.
    """

    def __init__(
        self,
        model_router: Any,
        knowledge_controller: HybridKnowledgeController,
    ):
        self.model_router = model_router
        self.knowledge_controller = knowledge_controller

    def generate(
        self,
        question: str,
        actor_context: KnowledgeAccessContext,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        opts = options or {}
        top_k = opts.get("top_k")
        max_tokens = int(opts.get("max_tokens", 128))
        temperature = float(opts.get("temperature", 0.6))
        metadata_filters = opts.get("metadata_filters")

        retrieval_pkg = self.knowledge_controller.query(
            query=question,
            access_context=actor_context,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        hits: List[RetrievalResult] = retrieval_pkg["results"]

        # Respect active GNNM checkpoint max_seq_len if available
        active_backend = self.model_router.get_active_neural_backend() if self.model_router else None
        max_ctx_tokens = opts.get("max_context_tokens", self.knowledge_controller.config.max_context_tokens)
        if active_backend is not None:
            backend_ctx = getattr(getattr(active_backend, "metadata", None), "context_window", None)
            if isinstance(backend_ctx, int) and backend_ctx > 64:
                self.knowledge_controller.context_builder.max_seq_len = backend_ctx
                # Attach GNNM tokenizer if loaded for exact token budgeting
                gnnm_eng = getattr(active_backend, "_gnnm_engine", None)
                if gnnm_eng is not None and hasattr(gnnm_eng, "tokenizer"):
                    self.knowledge_controller.context_builder.tokenizer = gnnm_eng.tokenizer

        built = self.knowledge_controller.build_context(
            question=question,
            results=hits,
            max_context_tokens=max_ctx_tokens,
        )

        gen_res = self.model_router.route_generation(
            prompt=question,
            context=built.context_block,
            task_type=opts.get("task_type", "general"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "generation": gen_res,
            "text": gen_res.text,
            "neural_used": gen_res.neural_used,
            "backend_name": gen_res.backend_name,
            "model_id": gen_res.model_id,
            "retrieval_executed": retrieval_pkg["retrieval_executed"],
            "sources": built.included_sources,
            "conflicts": retrieval_pkg["conflicts"],
            "knowledge_layers_used": retrieval_pkg["knowledge_layers_used"],
            "token_budget": built.token_budget,
            "memory_compatible_records": retrieval_pkg["memory_compatible_records"],
        }
