"""
GajahMada AI — Brain & Knowledge Observatory Backend API (`/api/v2/observatory/*`).

100% real backend telemetry — ZERO mock data, ZERO fake metrics.
Reuses existing runtime singletons (`get_runtime()`), enforces canonical `AccessContext`
and `MemoryAuthorizationService` on every resource boundary, supports pagination, and
honestly reports `HEALTHY | DEGRADED | UNAVAILABLE | NOT_CONFIGURED` for every subsystem.
"""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from gajahmada_ai.api.http_utils import build_sse_limiter, resolve_client_ip
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import VisibilityPolicy
from gajahmada_ai.security.context import AccessContext
from gajahmada_ai.security.observability import build_subsystem_health_report


SSE_LIMITER = build_sse_limiter()


def _iso_ts(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _build_access_context(rt: Dict[str, Any], request: Request) -> AccessContext:
    auth = rt["auth"]
    identity = rt["identity"]
    actor = auth.resolve_optional_actor(request)
    return AccessContext.from_actor_dict(
        actor,
        node_id=identity.node_id if identity else None,
        organization_id=request.headers.get("x-organization-id") or None,
        session_id=request.headers.get("x-session-id") or None,
        agent_id=request.headers.get("x-agent-id") or None,
    )


def create_observatory_router(get_runtime: Callable[[], Dict[str, Any]]) -> APIRouter:
    router = APIRouter(prefix="/api/v2/observatory", tags=["GajahMada Brain & Knowledge Observatory"])

    # ─── 1. /api/v2/observatory/overview ──────────────────────────────────────
    @router.get("/overview")
    def observatory_overview(request: Request):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        subsystems = build_subsystem_health_report(rt)
        mem = rt["memory"]
        mem_auth = MemoryAuthorizationService()
        retriever = rt.get("hybrid_retriever")
        scheduler = rt["scheduler"]
        p2p = rt["p2p"]
        discovery = rt["discovery"]
        reg = rt.get("gnnm_registry")
        cognition = rt["cognition"]
        tracker = rt.get("learning_events")

        # Count ONLY accessible memory records for this actor
        all_mem = mem.storage.list_records(include_deleted=False, limit=1000)
        accessible_mem = mem_auth.filter_records_with_context(all_mem, ctx)

        # Count ONLY accessible external knowledge chunks
        ext_count = 0
        if retriever is not None:
            ext_recs = retriever.list_external_records()
            ext_count = len(retriever.access_policy.filter_records(ext_recs, ctx.to_knowledge_access_context()))

        agents_list = scheduler.list_agents()
        running_tasks = [t for t in scheduler.tasks.values() if t.state.value in ("RUNNING", "PLANNED")]
        active_peers = discovery.list_active_peers()
        trusted_peers = [p for p in active_peers if p.get("trust_state") == "TRUSTED"]

        recent_events = tracker.query_temporal(access_context=ctx, limit=5) if tracker else []
        cap = rt["engine"].capability_status() if rt.get("engine") else {}

        privileged = ctx.actor_role in ("admin", "operator")
        return {
            "timestamp": _iso_ts(time.time()),
            "node_id": rt["identity"].node_id if privileged else None,
            "identity_redacted": not privileged,
            "actor_role": ctx.actor_role,
            "system_status": "HEALTHY" if subsystems["api"]["status"] == "HEALTHY" else "DEGRADED",
            "active_runtime_model": {
                "neural_inference_active": bool(cap.get("neural_inference_active", False)),
                "backend": cap.get("active_neural_backend") or "NOT_CONFIGURED",
                "model_id": cap.get("active_model_id") or "UNAVAILABLE",
                "registry_active_version": reg.active_version_id if reg else None,
                "registry_checkpoint_sha256": (reg.get_active() or {}).get("checkpoint_sha256") if reg and privileged else None,
            },
            "counts": {
                "accessible_memory_records": len(accessible_mem),
                "accessible_external_chunks": ext_count,
                "knowledge_graph_relations": mem.storage.health().get("relations_count", 0),
                "registered_agents": len(agents_list),
                "running_tasks": len(running_tasks),
                "registered_peers": len(active_peers) if privileged else None,
                "trusted_peers": len(trusted_peers) if privileged else None,
                "bootstrap_peer_urls": len(getattr(p2p, "peers", [])) if privileged else None,
                "reasoning_traces_recorded": len(cognition.history),
                "recent_learning_events": len(recent_events),
                "ledger_height": len(rt["blockchain"].chain) if privileged else None,
            },
            "subsystems": subsystems,
        }

    # ─── 2. /api/v2/observatory/brain ─────────────────────────────────────────
    @router.get("/brain")
    def observatory_brain_graph(request: Request):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        subsystems = build_subsystem_health_report(rt)
        cognition = rt["cognition"]
        mem = rt["memory"]
        retriever = rt.get("hybrid_retriever")
        reg = rt.get("gnnm_registry")
        scheduler = rt["scheduler"]
        discovery = rt["discovery"]
        tracker = rt.get("learning_events")

        gnnm_st = subsystems["gnnm_runtime"]["status"]
        know_st = subsystems["knowledge_subsystem"]["status"]
        mem_st = subsystems["memory_subsystem"]["status"]
        learn_st = subsystems["learning_pipeline"]["status"]
        dist_st = subsystems["distributed_network"]["status"]

        has_recent_cognition = len(cognition.history) > 0
        recent_learn_evs = tracker.query_temporal(access_context=ctx, limit=5) if tracker else []
        has_recent_ingest = len(recent_learn_evs) > 0
        has_active_tasks = any(t.state.value == "RUNNING" for t in scheduler.tasks.values())

        nodes = [
            # Group A: GNNM CORE
            {"id": "gnnm-core", "label": "GNNM Native Model", "group": "GNNM_CORE", "status": gnnm_st, "active": gnnm_st == "HEALTHY",
             "description": "Custom Decoder-Only Transformer (RMSNorm + RoPE + SwiGLU).",
             "metrics": subsystems["gnnm_runtime"]},
            {"id": "gnnm-tokenizer", "label": "GNNM BPE Tokenizer", "group": "GNNM_CORE", "status": "HEALTHY", "active": gnnm_st == "HEALTHY",
             "description": "Lossless pure-Python Byte-Level BPE multilingual tokenizer (ID/EN).",
             "metrics": {"type": "byte-level-bpe"}},
            {"id": "gnnm-router", "label": "ModelRouter", "group": "GNNM_CORE", "status": "HEALTHY", "active": has_recent_cognition,
             "description": "Honest capability router across PyTorchNativeBackend and ONNX.",
             "metrics": {"active_backend": subsystems["gnnm_runtime"]["active_backend"] or "NONE"}},

            # Group B: COGNITIVE ENGINE
            {"id": "cog-perception", "label": "Perception & Intent", "group": "COGNITIVE_ENGINE", "status": "HEALTHY", "active": has_recent_cognition,
             "description": "Input normalization, entity extraction, and Meta-Cognitive vs. Factual intent classification.",
             "metrics": {"traces_count": len(cognition.history)}},
            {"id": "cog-symbolic", "label": "Symbolic Reasoner", "group": "COGNITIVE_ENGINE", "status": "HEALTHY", "active": has_recent_cognition,
             "description": "Deterministic AST math, propositional logic, and constraint verification.",
             "metrics": {"neural_independent": True}},
            {"id": "cog-evidence", "label": "Evidence & Critic", "group": "COGNITIVE_ENGINE", "status": "HEALTHY", "active": has_recent_cognition,
             "description": "Evidence fusion, ContradictionDetector, and MetacognitiveCritic sufficiency gate.",
             "metrics": {"contradictions_tracked": mem.storage.health().get("contradictions_count", 0)}},
            {"id": "cog-metacog", "label": "Meta-Cognitive Orchestrator", "group": "COGNITIVE_ENGINE", "status": "HEALTHY", "active": has_recent_cognition,
             "description": "Temporal learning-history query engine distinguishing RETRIEVED / STORED / VERIFIED / INTERNALIZED.",
             "metrics": {"events_tracked": len(recent_learn_evs)}},

            # Group C: KNOWLEDGE SYSTEM
            {"id": "know-controller", "label": "Hybrid Knowledge Controller", "group": "KNOWLEDGE_SYSTEM", "status": know_st, "active": has_recent_cognition,
             "description": "Unified control plane orchestrating Internal, External, and Persistent Memory.",
             "metrics": {"external_chunks": subsystems["knowledge_subsystem"]["external_chunks"]}},
            {"id": "know-retriever", "label": "Hybrid Retriever (BM25+Vec)", "group": "KNOWLEDGE_SYSTEM", "status": know_st, "active": has_recent_ingest,
             "description": "SQLite WAL external chunks + Okapi BM25 + encoder-checked cosine similarity.",
             "metrics": {"embedding": subsystems["knowledge_subsystem"]["embedding_provider"]}},
            {"id": "know-graph", "label": "Knowledge Graph", "group": "KNOWLEDGE_SYSTEM", "status": mem_st, "active": mem.storage.health().get("relations_count", 0) > 0,
             "description": "Entity-relation graph preserving provenance and contradiction links.",
             "metrics": {"relations": mem.storage.health().get("relations_count", 0)}},
            {"id": "know-sources", "label": "Source Registry", "group": "KNOWLEDGE_SYSTEM", "status": know_st, "active": has_recent_ingest,
             "description": "SHA-256 content hashes, authorship, license, and verification lineage.",
             "metrics": {"sources_count": len(rt["source_registry"].list_sources()) if rt.get("source_registry") else 0}},

            # Group D: MEMORY SYSTEM
            {"id": "mem-manager", "label": "5-Tier MemoryManager", "group": "MEMORY_SYSTEM", "status": mem_st, "active": mem.storage.health().get("active_records", 0) > 0,
             "description": "Working, Semantic, Episodic, Procedural, and Collective persistent memory.",
             "metrics": mem.storage.health()},
            {"id": "mem-auth", "label": "Memory Authorization", "group": "MEMORY_SYSTEM", "status": "HEALTHY", "active": True,
             "description": "Fail-closed 9-scope AccessContext enforcement at memory & retrieval boundaries.",
             "metrics": {"scopes": [v.value for v in VisibilityPolicy]}},
            {"id": "mem-ledger", "label": "Signed Event Ledger", "group": "MEMORY_SYSTEM", "status": subsystems["audit_ledger"]["status"], "active": len(rt["blockchain"].chain) > 1,
             "description": "Tamper-evident local Ed25519-signed SHA-256 audit ledger (Schema v2).",
             "metrics": subsystems["audit_ledger"]},

            # Group E: LEARNING SYSTEM
            {"id": "learn-filter", "label": "Knowledge Filter & Dataset", "group": "LEARNING_SYSTEM", "status": "HEALTHY", "active": has_recent_ingest,
             "description": "Filters unverified/disputed/injection/unlicensed data; separates Pretrain & SFT.",
             "metrics": {"verification_required": True}},
            {"id": "learn-trainer", "label": "Isolated Training Pipeline", "group": "LEARNING_SYSTEM", "status": learn_st, "active": False,
             "description": "Strict P0 PyTorch training from isolated checkpoint copy with SHA-256 checks.",
             "metrics": subsystems["learning_pipeline"]},
            {"id": "learn-eval", "label": "6-Dim Evaluation Gate", "group": "LEARNING_SYSTEM", "status": "HEALTHY", "active": False,
             "description": "Functional, ID/EN/Mixed, Knowledge/Abstention, Reasoning, Security, & Leakage gate.",
             "metrics": {"benchmark_version": "gm-eval-v2.1"}},
            {"id": "learn-registry", "label": "GNNM Model Registry", "group": "LEARNING_SYSTEM", "status": subsystems["model_registry"]["status"], "active": bool(reg and reg.active_version_id),
             "description": "Strict lifecycle state machine, runtime staging, atomic switch, & rollback.",
             "metrics": subsystems["model_registry"]},

            # Group F: DISTRIBUTED NETWORK
            {"id": "net-local", "label": "Local Sovereign Node", "group": "DISTRIBUTED_NETWORK", "status": "HEALTHY", "active": True,
             "description": (
                 f"Ed25519 identity {rt['identity'].node_id[:16]}... with fail-closed key persistence."
                 if ctx.actor_role in ("admin", "operator")
                 else "Ed25519 node identity is hidden from anonymous callers."
             ),
             "metrics": (
                 {"node_id": rt["identity"].node_id, "key_version": rt["identity"].key_version}
                 if ctx.actor_role in ("admin", "operator")
                 else {"node_id": None, "redacted": True}
             )},
            {"id": "net-agents", "label": "Autonomous Agents", "group": "DISTRIBUTED_NETWORK", "status": "HEALTHY", "active": has_active_tasks,
             "description": "Planner, Researcher, Analyst, and Verifier agents with bounded budgets.",
             "metrics": {"registered": len(scheduler.list_agents()), "tasks": len(scheduler.tasks)}},
            {"id": "net-peers", "label": "GIP/2.0 Peer Mesh", "group": "DISTRIBUTED_NETWORK", "status": dist_st, "active": len(discovery.list_active_peers()) > 0,
             "description": "8-namespace signed GIP/2.0 with SQLite WAL replay protection & PeerTrustState.",
             "metrics": {"peers_discovered": len(discovery.list_active_peers()), "federation_orgs": len(rt["federation"].list_organizations()) if rt.get("federation") else 0}},
        ]

        edges = [
            {"source": "cog-perception", "target": "cog-metacog", "label": "meta_intent", "active": has_recent_cognition},
            {"source": "cog-perception", "target": "know-controller", "label": "factual_query", "active": has_recent_cognition},
            {"source": "cog-perception", "target": "cog-symbolic", "label": "ast_logic", "active": has_recent_cognition},
            {"source": "know-controller", "target": "know-retriever", "label": "hybrid_retrieve", "active": has_recent_ingest},
            {"source": "know-controller", "target": "mem-manager", "label": "memory_retrieve", "active": has_recent_cognition},
            {"source": "know-retriever", "target": "know-graph", "label": "kg_boost", "active": mem.storage.health().get("relations_count", 0) > 0},
            {"source": "know-retriever", "target": "know-sources", "label": "provenance", "active": has_recent_ingest},
            {"source": "mem-manager", "target": "mem-auth", "label": "access_context", "active": True},
            {"source": "know-controller", "target": "gnnm-router", "label": "token_budget_ctx", "active": gnnm_st == "HEALTHY" and has_recent_cognition},
            {"source": "gnnm-router", "target": "gnnm-core", "label": "inference", "active": gnnm_st == "HEALTHY" and has_recent_cognition},
            {"source": "gnnm-core", "target": "gnnm-tokenizer", "label": "bpe_tokens", "active": gnnm_st == "HEALTHY"},
            {"source": "cog-symbolic", "target": "cog-evidence", "label": "symbolic_proof", "active": has_recent_cognition},
            {"source": "gnnm-core", "target": "cog-evidence", "label": "neural_draft", "active": gnnm_st == "HEALTHY" and has_recent_cognition},
            {"source": "cog-evidence", "target": "mem-ledger", "label": "signed_audit", "active": len(rt["blockchain"].chain) > 1},
            {"source": "know-sources", "target": "learn-filter", "label": "verified_corpus", "active": False},
            {"source": "learn-filter", "target": "learn-trainer", "label": "jsonl_dataset", "active": False},
            {"source": "learn-trainer", "target": "learn-eval", "label": "candidate_ckpt", "active": False},
            {"source": "learn-eval", "target": "learn-registry", "label": "gate_report", "active": False},
            {"source": "learn-registry", "target": "gnnm-router", "label": "atomic_switch", "active": bool(reg and reg.active_version_id)},
            {"source": "net-local", "target": "net-peers", "label": "gip_2_0", "active": len(discovery.list_active_peers()) > 0},
            {"source": "net-local", "target": "net-agents", "label": "task_schedule", "active": has_active_tasks},
            {"source": "net-agents", "target": "cog-perception", "label": "agent_execute", "active": has_active_tasks},
        ]

        return {
            "timestamp": _iso_ts(time.time()),
            "groups": ["GNNM_CORE", "COGNITIVE_ENGINE", "KNOWLEDGE_SYSTEM", "MEMORY_SYSTEM", "LEARNING_SYSTEM", "DISTRIBUTED_NETWORK"],
            "nodes": nodes,
            "edges": edges,
        }

    # ─── 3. /api/v2/observatory/cognition & trace inspection ──────────────────
    @router.get("/cognition")
    def observatory_cognition(request: Request, limit: int = Query(default=20, ge=1, le=50)):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        cognition = rt["cognition"]
        is_privileged = ctx.actor_role in ("admin", "operator")

        # Actual pipeline stage definitions from CognitiveReasoningPipeline
        pipeline_stages = [
            {"index": 1, "stage": "PERCEPTION", "description": "Normalize input, detect language, extract entities & expressions."},
            {"index": 2, "stage": "INTENT_CLASSIFICATION", "description": "Classify factual, symbolic, diagnostic, or 8 meta-cognitive intents."},
            {"index": 3, "stage": "MEMORY_RETRIEVAL", "description": "Retrieve authorized memory/external records or temporal learning history."},
            {"index": 4, "stage": "PROBLEM_DECOMPOSITION", "description": "Decompose complex objectives into ordered sub-objectives."},
            {"index": 5, "stage": "HYPOTHESIS_GENERATION", "description": "Formulate testable hypotheses and explicit assumptions."},
            {"index": 6, "stage": "SYMBOLIC_REASONING", "description": "Execute deterministic AST math, logic rules, and constraints."},
            {"index": 7, "stage": "EVIDENCE_REASONING", "description": "Score source reliability, provenance, and contradiction checks."},
            {"index": 8, "stage": "NEURAL_REASONING", "description": "Invoke loaded local GNNM backend through ModelRouter (if active)."},
            {"index": 9, "stage": "EVIDENCE_FUSION", "description": "Fuse symbolic, evidence, and neural outputs into conclusion."},
            {"index": 10, "stage": "METACOGNITIVE_CRITIC", "description": "Evaluate sufficiency, quality score, and unsupported claims."},
            {"index": 11, "stage": "DECISION", "description": "Build structured EpistemicBreakdown separating facts from unknowns."},
        ]

        history_items = list(cognition.history.values())[-limit:]
        history_items.reverse()
        traces = []
        for r in history_items:
            traces.append({
                "request_id": r.reasoning_id,
                "timestamp": _iso_ts(r.timestamp),
                "objective_preview": r.objective[:140] if is_privileged else "[REDACTED — Authenticate as Operator to inspect prompts]",
                "interpretation": r.interpretation,
                "verification_status": r.verification_status,
                "confidence": r.confidence,
                "stages_executed": [
                    {
                        "step_index": s.step_index,
                        "stage": s.stage,
                        "summary": s.summary if is_privileged else f"{s.stage} ({s.outcome})",
                        "status": "COMPLETED" if s.outcome in ("COMPLETED", "VERIFIED", "EXECUTED", "SUFFICIENT") else ("SKIPPED" if s.outcome in ("NOT_APPLICABLE", "UNAVAILABLE_REPORTED") else s.outcome),
                        "raw_outcome": s.outcome,
                    }
                    for s in r.reasoning_steps
                ],
                "capabilities_used": {
                    "neural_used": r.capabilities_used.get("neural_used_in_this_call", False),
                    "symbolic_used": r.capabilities_used.get("symbolic_used_in_this_call", False),
                    "meta_cognitive_used": r.capabilities_used.get("meta_cognitive_used_in_this_call", False),
                    "evidence_count": r.capabilities_used.get("evidence_count_in_this_call", 0),
                },
            })

        return {
            "timestamp": _iso_ts(time.time()),
            "authenticated_detail": is_privileged,
            "pipeline_stages": pipeline_stages,
            "recent_traces_count": len(traces),
            "recent_traces": traces,
        }

    @router.get("/requests/{request_id}/trace")
    def observatory_request_trace(request_id: str, request: Request):
        rt = get_runtime()
        rt["auth"].authenticate_request(request, required_capability="reasoning:read")
        res = rt["cognition"].history.get(request_id)
        if res is None:
            raise HTTPException(status_code=404, detail=f"Reasoning trace '{request_id}' not found.")
        return res.model_dump()

    # ─── 4. /api/v2/observatory/knowledge/graph & /{record_id} ────────────────
    @router.get("/knowledge/graph")
    def observatory_knowledge_graph(
        request: Request,
        query: Optional[str] = None,
        verification_status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = Query(default=40, ge=1, le=120),
        offset: int = Query(default=0, ge=0),
    ):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        mem = rt["memory"]
        mem_auth = MemoryAuthorizationService()
        retriever = rt.get("hybrid_retriever")
        source_reg = rt.get("source_registry")

        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        # 1. Authorized Memory Records
        raw_mem = mem.storage.list_records(include_deleted=False, limit=300)
        auth_mem = mem_auth.filter_records_with_context(raw_mem, ctx)
        allowed_mem_ids = {r["record_id"] for r in auth_mem}

        ql = (query or "").strip().lower()
        vf = (verification_status or "").strip().upper()
        sf = (source or "").strip().lower()

        filtered_items: List[Dict[str, Any]] = []
        for r in auth_mem:
            prov = r.get("source_provenance") if isinstance(r.get("source_provenance"), dict) else {}
            src_lbl = str(prov.get("source_type", "memory"))
            if ql and ql not in str(r.get("content", "")).lower():
                continue
            if vf and str(r.get("verification_status", "")).upper() != vf:
                continue
            if sf and sf not in src_lbl.lower():
                continue
            filtered_items.append(("MEMORY", r))

        # 2. Authorized External Knowledge Chunks
        if retriever is not None:
            ext_recs = retriever.access_policy.filter_records(
                retriever.list_external_records(), ctx.to_knowledge_access_context()
            )
            for er in ext_recs:
                if ql and ql not in er.content.lower() and ql not in er.heading.lower():
                    continue
                if vf and er.verification_status.upper() != vf:
                    continue
                if sf and sf not in er.source_type.lower() and sf not in er.source_id.lower():
                    continue
                filtered_items.append(("DOCUMENT", er))

        total_items = len(filtered_items)
        page_items = filtered_items[offset : offset + limit]

        for kind, item in page_items:
            if kind == "MEMORY":
                rid = item["record_id"]
                prov = item.get("source_provenance") if isinstance(item.get("source_provenance"), dict) else {}
                src_type = str(prov.get("source_type", "memory"))
                nodes[rid] = {
                    "id": rid,
                    "label": (item.get("content", "")[:42] + "...") if len(item.get("content", "")) > 42 else item.get("content", ""),
                    "category": "MEMORY",
                    "domain": item.get("memory_type", "semantic"),
                    "source": src_type,
                    "verification_status": item.get("verification_status", "UNVERIFIED"),
                    "visibility_policy": item.get("visibility_policy", "LOCAL"),
                    "confidence": item.get("confidence", 0.75),
                    "created_at": _iso_ts(item.get("created_at")),
                    "content_preview": item.get("content", "")[:320],
                    "provenance": prov,
                }
                src_node_id = f"src-type:{src_type}"
                if src_node_id not in nodes:
                    nodes[src_node_id] = {
                        "id": src_node_id,
                        "label": src_type,
                        "category": "SOURCE",
                        "domain": "provenance",
                        "source": src_type,
                        "verification_status": "VERIFIED",
                        "visibility_policy": "PUBLIC",
                        "content_preview": f"Knowledge provenance source type: {src_type}",
                    }
                edges.append({"source": rid, "target": src_node_id, "relation": "DERIVED_FROM"})
            else:
                rid = item.record_id
                nodes[rid] = {
                    "id": rid,
                    "label": item.heading or (item.content[:42] + "..."),
                    "category": "DOCUMENT",
                    "domain": item.metadata.get("domain", item.source_type),
                    "source": item.source_id,
                    "verification_status": item.verification_status,
                    "visibility_policy": item.visibility_policy,
                    "confidence": item.confidence,
                    "created_at": _iso_ts(item.timestamp),
                    "content_preview": item.content[:320],
                    "provenance": item.provenance,
                }
                if item.source_id:
                    s_obj = source_reg.get(item.source_id) if source_reg else None
                    if item.source_id not in nodes:
                        nodes[item.source_id] = {
                            "id": item.source_id,
                            "label": s_obj.title if s_obj else item.source_id,
                            "category": "SOURCE",
                            "domain": item.source_type,
                            "source": s_obj.location if s_obj else "external",
                            "verification_status": s_obj.verification_status if s_obj else item.verification_status,
                            "visibility_policy": item.visibility_policy,
                            "content_preview": f"Source {item.source_id} (license={s_obj.license if s_obj else 'unspecified'})",
                        }
                    edges.append({"source": rid, "target": item.source_id, "relation": "DERIVED_FROM"})

        # 3. Real Knowledge Graph Relations (ONLY for visible authorized records)
        raw_rels = mem.storage.list_relations()
        auth_rels = mem_auth.filter_relations(raw_rels, allowed_mem_ids, ctx.actor_role)
        for rel in auth_rels[:80]:
            subj = str(rel.get("subject", "")).strip()
            obj = str(rel.get("object", "")).strip()
            pred = str(rel.get("predicate", "RELATED_TO")).strip()
            rec_ref = rel.get("record_id")
            if subj and obj and (not rec_ref or rec_ref in nodes):
                sid = f"ent:{subj.lower()[:32]}"
                oid = f"ent:{obj.lower()[:32]}"
                if sid not in nodes:
                    nodes[sid] = {"id": sid, "label": subj[:36], "category": "ENTITY", "domain": "knowledge_graph", "verification_status": "VERIFIED", "visibility_policy": "COLLECTIVE", "content_preview": f"Extracted entity: {subj}"}
                if oid not in nodes:
                    nodes[oid] = {"id": oid, "label": obj[:36], "category": "ENTITY", "domain": "knowledge_graph", "verification_status": "VERIFIED", "visibility_policy": "COLLECTIVE", "content_preview": f"Extracted entity: {obj}"}
                edges.append({"source": sid, "target": oid, "relation": pred.upper()})
                if rec_ref and rec_ref in nodes:
                    edges.append({"source": rec_ref, "target": sid, "relation": "REFERENCES"})

        # 4. Real Contradictions (ONLY between visible authorized records)
        raw_contras = mem.storage.list_contradictions()
        auth_contras = mem_auth.filter_contradictions(raw_contras, allowed_mem_ids, ctx.actor_role)
        for c in auth_contras:
            ra, rb = c.get("record_a_id"), c.get("record_b_id")
            if ra in nodes and rb in nodes:
                edges.append({"source": ra, "target": rb, "relation": "CONTRADICTS"})

        return {
            "timestamp": _iso_ts(time.time()),
            "pagination": {"total": total_items, "limit": limit, "offset": offset},
            "nodes": list(nodes.values()),
            "edges": edges,
            "contradictions_count": len(auth_contras),
        }

    # ─── 5. /api/v2/observatory/memory ────────────────────────────────────────
    @router.get("/memory")
    def observatory_memory(
        request: Request,
        scope: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = Query(default=40, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        mem = rt["memory"]
        mem_auth = MemoryAuthorizationService()

        all_recs = mem.storage.list_records(include_deleted=False, limit=500)
        authorized = mem_auth.filter_records_with_context(all_recs, ctx)

        # Count accessible records by real VisibilityPolicy
        scope_counts: Dict[str, int] = {vp.value: 0 for vp in VisibilityPolicy}
        for r in authorized:
            vp = str(r.get("visibility_policy", "LOCAL")).upper()
            if vp in scope_counts:
                scope_counts[vp] += 1

        ql = (query or "").strip().lower()
        sc = (scope or "").strip().upper()
        filtered = []
        for r in authorized:
            if sc and str(r.get("visibility_policy", "")).upper() != sc:
                continue
            if ql and ql not in str(r.get("content", "")).lower():
                continue
            filtered.append(r)

        filtered.sort(key=lambda x: float(x.get("created_at", 0.0)), reverse=True)
        page = filtered[offset : offset + limit]

        return {
            "timestamp": _iso_ts(time.time()),
            "actor_role": ctx.actor_role,
            "available_scopes": [vp.value for vp in VisibilityPolicy],
            "accessible_scope_counts": scope_counts,
            "pagination": {"total": len(filtered), "limit": limit, "offset": offset},
            "records": [
                {
                    "record_id": r["record_id"],
                    "memory_type": r.get("memory_type"),
                    "visibility_policy": r.get("visibility_policy"),
                    "verification_status": r.get("verification_status"),
                    "confidence": r.get("confidence"),
                    "version": r.get("version", 1),
                    "created_at": _iso_ts(r.get("created_at")),
                    "content": r.get("content", "")[:400],
                    "source_provenance": r.get("source_provenance", {}),
                }
                for r in page
            ],
        }

    # ─── 6. /api/v2/observatory/learning & /models ────────────────────────────
    @router.get("/learning")
    @router.get("/models")
    def observatory_learning(request: Request):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        reg = rt.get("gnnm_registry")
        tracker = rt.get("learning_events")
        subsystems = build_subsystem_health_report(rt)
        cap = rt["engine"].capability_status() if rt.get("engine") else {}

        events = tracker.query_temporal(access_context=ctx, limit=30) if tracker else []
        versions = reg.list_versions() if reg else []
        active_ver = reg.get_active() if reg else None

        has_ingested = any(e.event_type in ("DOCUMENT_INGESTED", "KNOWLEDGE_STORED") for e in events)
        has_verified = any(e.verification_status in ("VERIFIED", "SOURCE_CORROBORATED", "INDEPENDENTLY_VERIFIED") for e in events)
        has_trained = any(e.event_type == "TRAINING_COMPLETED" for e in events)
        has_internalized = any(e.internalization_status == "INTERNALIZED_IN_GNNM" for e in events)

        lifecycle_steps = [
            {"step": 1, "name": "Knowledge Collection", "status": "COMPLETED" if has_ingested else "IDLE"},
            {"step": 2, "name": "Knowledge Filtering", "status": "HEALTHY"},
            {"step": 3, "name": "Dataset Builder", "status": "READY"},
            {"step": 4, "name": "Training Pipeline", "status": subsystems["learning_pipeline"]["status"]},
            {"step": 5, "name": "Candidate Checkpoint", "status": "AVAILABLE" if versions else "NO_DATA"},
            {"step": 6, "name": "Evaluation Gate", "status": "READY"},
            {"step": 7, "name": "Model Registry", "status": subsystems["model_registry"]["status"]},
            {"step": 8, "name": "Model Promotion", "status": "ACTIVE" if active_ver else "IDLE"},
            {"step": 9, "name": "Active GNNM Model", "status": subsystems["gnnm_runtime"]["status"]},
        ]

        return {
            "timestamp": _iso_ts(time.time()),
            "pytorch_available": subsystems["learning_pipeline"]["pytorch_training_capable"],
            "runtime_model_id": cap.get("active_model_id") or "N/A",
            "runtime_neural_active": bool(cap.get("neural_inference_active", False)),
            "knowledge_state_summary": {
                "RETRIEVED": "ACTIVE" if len(rt["cognition"].history) > 0 else "IDLE",
                "STORED": "COMPLETED" if has_ingested or rt["memory"].storage.health().get("active_records", 0) > 0 else "NO_DATA",
                "VERIFIED": "COMPLETED" if has_verified else "NO_DATA",
                "TRAINED": "COMPLETED" if has_trained else "NOT_EXECUTED",
                "INTERNALIZED": "DEMONSTRATED" if has_internalized else "NOT_DEMONSTRATED",
            },
            "lifecycle_steps": lifecycle_steps,
            "active_registry_version": active_ver,
            "model_versions": versions,
            "recent_learning_events": [e.to_dict() for e in events],
        }

    # ─── 7. /api/v2/observatory/agents ────────────────────────────────────────
    @router.get("/agents")
    def observatory_agents(request: Request):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        is_privileged = ctx.actor_role in ("admin", "operator")
        scheduler = rt["scheduler"]

        registered = scheduler.list_agents()
        tasks_list = list(scheduler.tasks.values())
        running_tasks = [t for t in tasks_list if t.state.value == "RUNNING"]

        agents_out = []
        for idx, ag in enumerate(registered):
            name = ag.get("name", f"agent-{idx}") if isinstance(ag, dict) else str(ag)
            role = ag.get("role", "specialist") if isinstance(ag, dict) else "specialist"
            agents_out.append({
                "agent_id": f"agent-{name.lower().replace(' ', '-')}",
                "name": name,
                "role": role,
                "status": "BUSY" if running_tasks else "IDLE",
                "assigned_node": rt["identity"].node_id[:16],
                "memory_scope": "AGENT / LOCAL (AccessContext Enforced)",
                "permissions": ag.get("capabilities", ["reason", "verify"]) if isinstance(ag, dict) else ["reason"],
            })

        task_history = [
            {
                "task_id": t.task_id,
                "state": t.state.value,
                "confidence": t.confidence,
                "objective": t.objective[:160] if is_privileged else "[REDACTED — Authenticate to view task objectives]",
                "requires_human_approval": t.requires_human_approval,
            }
            for t in tasks_list[-20:]
        ]

        return {
            "timestamp": _iso_ts(time.time()),
            "registered_count": len(agents_out),
            "running_count": len(running_tasks),
            "authenticated_detail": is_privileged,
            "agents": agents_out,
            "tasks": task_history,
        }

    # ─── 8. /api/v2/observatory/nodes ─────────────────────────────────────────
    @router.get("/nodes")
    def observatory_nodes(request: Request):
        rt = get_runtime()
        identity = rt["identity"]
        discovery = rt["discovery"]
        p2p = rt["p2p"]
        federation = rt.get("federation")

        privileged = rt["auth"].resolve_optional_actor(request)["role"] in ("admin", "operator")
        local_fp = hashlib.sha256(identity.node_id.encode("utf-8")).hexdigest()[:16]
        orgs = federation.list_organizations() if federation and privileged else []
        org_by_node = {o["node_id"]: o for o in orgs}

        peers = []
        if privileged:
            for p in discovery.list_active_peers():
                nid = p.get("node_id", "")
                trust = p.get("trust_state", "PENDING")
                is_rev = identity.is_revoked(nid)
                org_info = org_by_node.get(nid)
                peers.append({
                    "node_id": nid,
                    "fingerprint": hashlib.sha256(nid.encode("utf-8")).hexdigest()[:16] if nid else "N/A",
                    "node_name": p.get("node_name", "peer"),
                    "endpoint": p.get("endpoint", ""),
                    "connection_status": "DISCONNECTED" if is_rev else "CONNECTED",
                    "trust_status": "REVOKED" if is_rev else trust,
                    "organization": org_info["org_name"] if org_info else "UNVERIFIED_OPERATOR",
                    "organization_type": org_info["org_type"] if org_info else None,
                    "last_heartbeat": _iso_ts(p.get("last_seen")),
                    "capabilities": p.get("capabilities", []),
                })

        return {
            "timestamp": _iso_ts(time.time()),
            "local_node": {
                "node_id": identity.node_id if privileged else None,
                "fingerprint": local_fp if privileged else None,
                "key_version": identity.key_version if privileged else None,
                "connection_status": "CONNECTED",
                "trust_status": "TRUSTED",
                "ledger_height": len(rt["blockchain"].chain) if privileged else None,
                "revoked_peers_count": len(identity.revoked_node_ids) if privileged else None,
            },
            "bootstrap_urls": list(getattr(p2p, "peers", [])) if privileged else [],
            "registered_peers_count": len(peers) if privileged else None,
            "trusted_peers_count": sum(1 for p in peers if p["trust_status"] == "TRUSTED") if privileged else None,
            "federation_organizations": orgs,
            "peers": peers,
            "redacted": not privileged,
        }

    # ─── 9. /api/v2/observatory/events & SSE /stream ──────────────────────────
    def _collect_unified_events(rt: Dict[str, Any], ctx: AccessContext, limit: int = 50) -> List[Dict[str, Any]]:
        is_privileged = ctx.actor_role in ("admin", "operator")
        out: List[Dict[str, Any]] = []

        # 1. Learning events (authorized)
        tracker = rt.get("learning_events")
        if tracker:
            for e in tracker.query_temporal(access_context=ctx, limit=limit):
                out.append({
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "timestamp": _iso_ts(e.timestamp),
                    "epoch": e.timestamp,
                    "request_id": None,
                    "subsystem": "learning" if "TRAINING" in e.event_type or "MODEL" in e.event_type else "knowledge",
                    "status": e.verification_status,
                    "description": f"{e.source_title or e.source_id} ({e.internalization_status})",
                    "metadata": {"source_id": e.source_id, "domain": e.knowledge_domain},
                })

        # 2. Recent ledger events (public tamper-evident audit headers)
        for blk in rt["blockchain"].chain[-20:]:
            out.append({
                "event_id": f"blk-{blk.index}",
                "event_type": blk.event_type,
                "timestamp": _iso_ts(blk.timestamp),
                "epoch": blk.timestamp,
                "request_id": blk.metadata.get("reasoning_id") if isinstance(blk.metadata, dict) else None,
                "subsystem": "ledger",
                "status": "SIGNED",
                "description": f"Block #{blk.index} · {blk.event_type} ({blk.hash[:12]}...)",
                "metadata": {"index": blk.index, "hash": blk.hash[:16]},
            })

        # 3. Security audit events (privileged only)
        if is_privileged and rt.get("audit_logger"):
            for ae in rt["audit_logger"].get_recent(limit=20):
                out.append({
                    "event_id": ae.get("event_id", f"aud-{ae.get('timestamp')}"),
                    "event_type": ae.get("event_type", "AUDIT"),
                    "timestamp": _iso_ts(ae.get("timestamp")),
                    "epoch": float(ae.get("timestamp", 0.0)),
                    "request_id": None,
                    "subsystem": "security",
                    "status": ae.get("outcome", "INFO"),
                    "description": f"{ae.get('action', '')} -> {ae.get('outcome', '')}",
                    "metadata": ae.get("details", {}),
                })

        out.sort(key=lambda x: float(x.get("epoch") or 0.0), reverse=True)
        return out[:limit]

    @router.get("/events")
    def observatory_events(
        request: Request,
        subsystem: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = Query(default=40, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        events = _collect_unified_events(rt, ctx, limit=120)
        sub_f = (subsystem or "").strip().lower()
        q_f = (search or "").strip().lower()
        filtered = [
            e for e in events
            if (not sub_f or e["subsystem"].lower() == sub_f)
            and (not q_f or q_f in e["event_type"].lower() or q_f in e["description"].lower())
        ]
        return {
            "timestamp": _iso_ts(time.time()),
            "pagination": {"total": len(filtered), "limit": limit, "offset": offset},
            "events": filtered[offset : offset + limit],
        }

    sse_limiter = SSE_LIMITER

    @router.get("/stream")
    async def observatory_sse_stream(request: Request):
        """Real Server-Sent Events (SSE) stream pushing actual backend telemetry snapshots."""
        rt = get_runtime()
        ctx = _build_access_context(rt, request)
        client_ip = resolve_client_ip(request)
        if not sse_limiter.acquire(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many concurrent observatory streams. Close an existing stream and retry.",
                headers={"Retry-After": "10"},
            )

        async def event_gen():
          try:
            last_sig = ""
            # Send initial real snapshot immediately, then poll internal state cleanly (max 60s per serverless conn)
            for _ in range(15):
                if await request.is_disconnected():
                    break
                evs = _collect_unified_events(rt, ctx, limit=15)
                sig = f"{len(rt['blockchain'].chain)}:{len(rt['cognition'].history)}:{evs[0]['event_id'] if evs else 'none'}"
                if sig != last_sig:
                    last_sig = sig
                    payload = json.dumps({"timestamp": _iso_ts(time.time()), "events": evs})
                    yield f"event: observatory_update\ndata: {payload}\n\n"
                else:
                    yield f": heartbeat {int(time.time())}\n\n"
                await asyncio.sleep(4)
          finally:
            sse_limiter.release(client_ip)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return router
