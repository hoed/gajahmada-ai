from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from gajahmada_ai.blockchain.chain import LedgerEventType
from gajahmada_ai.knowledge.interfaces import KnowledgeAccessContext
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.consensus import QuorumConsensusManager
from gajahmada_ai.network.discovery import PeerTrustState
from gajahmada_ai.network.federation import OrganizationProfile
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType
from gajahmada_ai.security.crypto import canonical_json_bytes
from gajahmada_ai.security.ssrf import validate_peer_url


class V2AskRequest(BaseModel):
    objective: str = Field(..., min_length=1, max_length=16384)
    symbolic_facts: List[str] = Field(default_factory=list)
    max_tokens: int = Field(default=256, ge=32, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class V2MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32768)
    memory_type: str = "semantic"
    source_type: str = "operator_ingested"
    author: str = "operator"
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    verification_status: str = "VERIFIED"
    visibility_policy: str = "LOCAL"


class V2MemoryUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32768)
    reason: str = "evidence_supersession"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verification_status: Optional[str] = None


class V2AgentRunRequest(BaseModel):
    """P0 Remediation: `human_approved` removed from public request payload so approval cannot be spoofed."""
    objective: str = Field(..., min_length=1, max_length=16384)
    requires_human_approval: bool = False
    max_iterations: int = Field(default=8, ge=1, le=20)
    requested_tools: List[str] = Field(default_factory=list)


class V2RegisterPeerRequest(BaseModel):
    node_id: str = Field(..., min_length=10, max_length=128)
    endpoint: str = Field(..., min_length=7, max_length=256)
    node_name: str = "gajahmada-peer"
    capabilities: List[str] = Field(default_factory=lambda: ["symbolic", "evidence", "memory"])


class V2LearningExperimentRequest(BaseModel):
    name: str
    level: int = Field(default=2, ge=1, le=3)
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)
    simulate_failure: bool = False


class NodeAttestation(BaseModel):
    """Cryptographically signed independent verifier attestation for knowledge promotion."""
    node_id: str = Field(..., min_length=16, max_length=128)
    signature: str = Field(..., min_length=32, max_length=256)
    evidence_hash: str = Field(..., min_length=8, max_length=128)
    source_ref: str = Field(..., min_length=4, max_length=512)


class V2PromoteKnowledgeRequest(BaseModel):
    """
    P2 + Round 3 Remediation:
    - Plain `independent_sources` strings can at most promote to `SOURCE_CORROBORATED`.
    - `INDEPENDENTLY_VERIFIED` strictly requires >= 2 cryptographically verified
      Ed25519 `node_attestations` from distinct trusted verifiers with distinct evidence hashes.
    """
    independent_sources: List[str] = Field(..., min_length=1)
    verification_method: str = Field(default="source_corroboration", min_length=3, max_length=64)
    verifier_notes: str = Field(default="", max_length=1024)
    node_attestations: List[NodeAttestation] = Field(default_factory=list)


class V2PeerTrustRequest(BaseModel):
    trust_state: str = Field(..., description="PENDING | AUTHENTICATED | TRUSTED | REVOKED")


class V2HybridIngestRequest(BaseModel):
    content: str = Field(..., min_length=4, max_length=200000)
    source_type: str = Field(default="txt", max_length=32)
    title: str = Field(default="", max_length=256)
    location: str = Field(default="api_upload", max_length=512)
    author: str = Field(default="operator", max_length=128)
    organization: str = Field(default="", max_length=128)
    license: str = Field(default="unspecified", max_length=64)
    visibility_policy: str = Field(default="LOCAL", max_length=32)
    verification_status: str = Field(default="UNVERIFIED", max_length=32)
    training_approved: bool = False


class V2HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    include_external: bool = True
    include_memory: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None


class V2KnowledgeTrainRequest(BaseModel):
    mode: str = Field(default="continued_pretraining", description="continued_pretraining | sft")
    base_checkpoint_dir: str = Field(default="./models/gnnm-01-tiny", max_length=256)
    seed: int = Field(default=42, ge=0, le=999999)
    steps: int = Field(default=4, ge=1, le=500)
    lr: float = Field(default=1e-4, gt=0.0, le=0.1)
    sft_examples: List[Dict[str, Any]] = Field(default_factory=list)


class V2PromoteModelRequest(BaseModel):
    operator_approved: bool = Field(default=False)
    notes: str = Field(default="", max_length=512)


def create_v2_router(get_runtime: Callable[[], Dict[str, Any]]) -> APIRouter:
    router = APIRouter(prefix="/api/v2", tags=["GajahMada 2.0 Sovereign Intelligence API"])
    mem_auth = MemoryAuthorizationService()
    consensus_mgr = QuorumConsensusManager()

    # ─── 1. /api/v2/intelligence ──────────────────────────────────────────────
    @router.get("/intelligence")
    @router.get("/intelligence/diagnose")
    def intelligence_diagnose():
        rt = get_runtime()
        engine = rt["engine"]
        memory = rt["memory"]
        scheduler = rt["scheduler"]
        identity = rt["identity"]
        return {
            "version": "2.0.0",
            "node_id": identity.node_id,
            "key_version": identity.key_version,
            "capabilities": engine.capability_status(),
            "memory_summary": memory.stats(),
            "agents_registered": len(scheduler.list_agents()),
            "offline_capable": True,
            "external_ai_providers_mandatory": False,
        }

    @router.post("/intelligence/ask")
    def intelligence_ask(req: V2AskRequest, request: Request):
        rt = get_runtime()
        cognition = rt["cognition"]
        memory = rt["memory"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]
        auth = rt["auth"]

        actor = auth.resolve_optional_actor(request)
        can_write_state = actor["role"] in ("admin", "operator")

        raw_retrieved = memory.retrieve(query=req.objective, top_k=4)
        retrieved = mem_auth.filter_records(raw_retrieved, actor["role"])
        result = cognition.execute(
            objective=req.objective,
            memory_records=retrieved,
            symbolic_facts=req.symbolic_facts,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        # P0 + P2: Public unauthenticated /ask is read-only (cannot mutate memory or ledger).
        # Authenticated operators persist into LOCAL Quarantine (MODEL_ASSESSED / UNVERIFIED),
        # NEVER auto-promoting model output to COLLECTIVE without independent verification.
        if can_write_state:
            clean_ans = result.conclusion.split("\n\n---\n📚")[0].strip()
            qa_entry = f"Pertanyaan: {req.objective.strip()} | Jawaban: {clean_ans}"
            is_generic = (
                "Permintaan telah diproses oleh **GajahMada Hybrid Cognitive Engine**" in result.conclusion
                or "Belum ada data spesifik untuk" in result.conclusion
            )
            epistemic_status = (
                VerificationStatus.DISPUTED if result.contradictions
                else (VerificationStatus.MODEL_ASSESSED if not is_generic else VerificationStatus.UNVERIFIED)
            )
            memory.store(
                content=qa_entry,
                memory_type=MemoryType.EPISODIC if is_generic else MemoryType.SEMANTIC,
                origin_node_id=identity.node_id,
                source_type="cognitive_brain_qa_update",
                author=actor["actor"],
                confidence=result.confidence,
                verification_status=epistemic_status,
                visibility_policy=VisibilityPolicy.LOCAL,
            )
            memory.store(
                content=f"Q: {req.objective[:160]} -> A ({epistemic_status.value}, conf={result.confidence})",
                memory_type=MemoryType.WORKING,
                origin_node_id=identity.node_id,
                source_type="cognitive_pipeline",
                confidence=result.confidence,
                verification_status=epistemic_status,
            )
            blockchain.record_event(
                LedgerEventType.TASK_COMPLETED,
                f"{result.reasoning_id}|status={epistemic_status.value}|conf={result.confidence}",
                identity,
                metadata={"reasoning_id": result.reasoning_id, "epistemic_status": epistemic_status.value},
            )
        return result.model_dump()

    # ─── 2. /api/v2/reasoning (Internal Operational Endpoints — Authenticated) ──
    @router.get("/reasoning")
    def list_reasoning_history(request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        cognition = rt["cognition"]
        auth.authenticate_request(request, required_capability="reasoning:read")
        items = list(cognition.history.values())[-25:]
        return {
            "count": len(items),
            "results": [r.model_dump() for r in items],
        }

    @router.post("/reasoning")
    def run_reasoning(req: V2AskRequest, request: Request):
        return intelligence_ask(req, request)

    @router.get("/reasoning/{reasoning_id}")
    def inspect_reasoning(reasoning_id: str, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        cognition = rt["cognition"]
        auth.authenticate_request(request, required_capability="reasoning:read")
        res = cognition.history.get(reasoning_id)
        if not res:
            raise HTTPException(status_code=404, detail=f"Reasoning trace '{reasoning_id}' not found.")
        return res.model_dump()

    # ─── 3. /api/v2/memory & /api/v2/knowledge (Protected by MemoryAuthorizationService) ──
    @router.get("/memory")
    def inspect_memory(request: Request, memory_type: Optional[str] = None, limit: int = 50):
        rt = get_runtime()
        memory = rt["memory"]
        auth = rt["auth"]
        actor = auth.resolve_optional_actor(request)

        records = memory.storage.list_records(memory_type=memory_type, limit=limit)
        visible = mem_auth.filter_records(records, actor["role"])
        allowed_ids = mem_auth.visible_record_ids(memory.storage, actor["role"])
        contradictions = mem_auth.filter_contradictions(
            memory.storage.list_contradictions(), allowed_ids, actor["role"]
        )
        return {
            "stats": memory.stats(),
            "records": visible,
            "contradictions": contradictions,
        }

    @router.post("/memory")
    def create_memory_record(req: V2MemoryCreateRequest, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        memory = rt["memory"]
        identity = rt["identity"]
        blockchain = rt["blockchain"]

        actor_info = auth.authenticate_request(request, required_capability="memory:write")
        rec = memory.store(
            content=req.content,
            memory_type=MemoryType(req.memory_type.lower()),
            origin_node_id=identity.node_id,
            source_type=req.source_type,
            author=actor_info["actor"],
            confidence=req.confidence,
            verification_status=VerificationStatus(req.verification_status.upper()),
            visibility_policy=VisibilityPolicy(req.visibility_policy.upper()),
        )
        blockchain.record_event(
            LedgerEventType.KNOWLEDGE_PROPOSED,
            f"{rec['record_id']}|hash={rec['integrity_hash'][:16]}",
            identity,
            metadata={"record_id": rec["record_id"], "visibility": rec["visibility_policy"]},
        )
        return rec

    @router.put("/memory/{record_id}")
    def update_memory_record(record_id: str, req: V2MemoryUpdateRequest, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        memory = rt["memory"]
        actor_info = auth.authenticate_request(request, required_capability="memory:write")
        try:
            return memory.correct_memory(
                record_id=record_id,
                new_content=req.content,
                modified_by=actor_info["actor"],
                reason=req.reason,
                new_confidence=req.confidence,
                new_verification_status=req.verification_status,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/memory/{record_id}/versions")
    def get_memory_versions(record_id: str, request: Request):
        rt = get_runtime()
        memory = rt["memory"]
        auth = rt["auth"]
        actor = auth.resolve_optional_actor(request)
        rec = memory.storage.get_record(record_id)
        mem_auth.require_record_read(rec, actor["role"])
        return {"record_id": record_id, "versions": memory.versioning.get_history(record_id)}

    @router.post("/memory/backup")
    def backup_memory(request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        memory = rt["memory"]
        auth.authenticate_request(request, required_capability="memory:backup")
        return memory.export_backup()

    @router.get("/knowledge")
    def get_knowledge_graph(request: Request, entity: Optional[str] = None, query: Optional[str] = None):
        rt = get_runtime()
        memory = rt["memory"]
        auth = rt["auth"]
        actor = auth.resolve_optional_actor(request)
        allowed_ids = mem_auth.visible_record_ids(memory.storage, actor["role"])

        raw_rels = memory.knowledge_graph.query_entity(entity) if entity else memory.storage.list_relations()
        rels = mem_auth.filter_relations(raw_rels, allowed_ids, actor["role"])
        contras = mem_auth.filter_contradictions(memory.storage.list_contradictions(), allowed_ids, actor["role"])
        raw_results = memory.retrieve(query=query, top_k=5) if query else []
        results = mem_auth.filter_records(raw_results, actor["role"])
        return {
            "relations": rels,
            "contradictions": contras,
            "search_results": results,
        }

    @router.post("/knowledge/ingest")
    def ingest_external_knowledge(req: V2HybridIngestRequest, request: Request):
        """GNNM Hybrid Knowledge: Authenticated external document ingestion with provenance & chunking."""
        rt = get_runtime()
        auth = rt["auth"]
        ingestion = rt["knowledge_ingestion"]
        identity = rt["identity"]
        blockchain = rt["blockchain"]

        actor_info = auth.authenticate_request(request, required_capability="memory:write")
        res = ingestion.ingest_text(
            content=req.content,
            source_type=req.source_type,
            title=req.title,
            location=req.location,
            author=req.author or actor_info["actor"],
            organization=req.organization,
            license_name=req.license,
            visibility_policy=req.visibility_policy.upper(),
            verification_status=req.verification_status.upper(),
            owner_node_id=identity.node_id,
            training_approved=req.training_approved,
        )
        if res.get("status") == "INGESTED":
            blockchain.record_event(
                LedgerEventType.KNOWLEDGE_PROPOSED,
                f"ext:{res['source_id']}|chunks={res['chunks_indexed']}",
                identity,
                metadata={"source_id": res["source_id"], "chunks": res["chunks_indexed"]},
            )
        return res

    @router.post("/knowledge/hybrid-search")
    def hybrid_knowledge_search(req: V2HybridSearchRequest, request: Request):
        """GNNM Hybrid Knowledge: Authorization-enforced hybrid retrieval across external & memory layers."""
        rt = get_runtime()
        auth = rt["auth"]
        controller = rt["hybrid_knowledge"]
        identity = rt["identity"]

        actor = auth.resolve_optional_actor(request)
        ctx = KnowledgeAccessContext(
            actor_id=actor["actor"],
            actor_role=actor["role"],
            node_id=identity.node_id,
            include_external=req.include_external,
            include_memory=req.include_memory,
            max_results=req.top_k,
        )
        pkg = controller.query(
            query=req.query,
            access_context=ctx,
            top_k=req.top_k,
            metadata_filters=req.metadata_filters,
        )
        return {
            "retrieval_executed": pkg["retrieval_executed"],
            "reason": pkg["reason"],
            "knowledge_layers_used": pkg["knowledge_layers_used"],
            "conflicts": pkg["conflicts"],
            "results": [r.to_dict() for r in pkg["results"]],
        }

    @router.get("/knowledge/sources")
    def list_knowledge_sources(request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        source_reg = rt["source_registry"]
        auth.authenticate_request(request, required_capability="memory:read")
        sources = source_reg.list_sources()
        return {"count": len(sources), "sources": sources}

    @router.get("/learning/registry")
    def list_gnnm_model_registry(request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        reg = rt["gnnm_registry"]
        auth.authenticate_request(request, required_capability="memory:read")
        return {
            "active_version_id": reg.active_version_id,
            "versions": reg.list_versions(),
        }

    @router.post("/learning/train")
    def trigger_controlled_gnnm_training(req: V2KnowledgeTrainRequest, request: Request):
        """Isolated Knowledge-to-Model training job (NEVER overwrites active production weights)."""
        rt = get_runtime()
        auth = rt["auth"]
        pipeline = rt["training_pipeline"]
        retriever = rt["hybrid_retriever"]
        source_reg = rt["source_registry"]
        audit_logger = rt["audit_logger"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        actor_info = auth.authenticate_request(request, required_capability="learning:approve")
        sources_map = {s["source_id"]: source_reg.get(s["source_id"]) for s in source_reg.list_sources()}
        sources_map = {k: v for k, v in sources_map.items() if v is not None}

        try:
            res = pipeline.run_training_job(
                base_checkpoint_dir=req.base_checkpoint_dir,
                mode=req.mode,
                records=retriever.list_external_records(),
                sft_examples=req.sft_examples,
                sources_by_id=sources_map,
                seed=req.seed,
                steps=req.steps,
                lr=req.lr,
                auto_promote_if_approved=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        audit_logger.record(
            event_type="GNNM_TRAINING_JOB",
            severity="INFO",
            actor=actor_info["actor"],
            action="POST /api/v2/learning/train",
            outcome=res.get("status", "UNKNOWN"),
            details={"run_id": res.get("run_id"), "mode": req.mode},
        )
        blockchain.record_event(
            LedgerEventType.MODEL_VERSION_CHANGED,
            f"train:{res.get('run_id')}|status={res.get('status')}",
            identity,
            metadata={"run_id": res.get("run_id"), "status": res.get("status"), "mode": req.mode},
        )
        return res

    @router.post("/learning/promote/{version_id}")
    def promote_gnnm_model_version(version_id: str, req: V2PromoteModelRequest, request: Request):
        """Controlled operator-approved promotion of an APPROVED GNNM checkpoint."""
        rt = get_runtime()
        auth = rt["auth"]
        reg = rt["gnnm_registry"]
        audit_logger = rt["audit_logger"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        actor_info = auth.authenticate_request(request, required_capability="learning:approve")
        try:
            promoted = reg.promote(version_id, operator_approved=req.operator_approved)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        audit_logger.record(
            event_type="GNNM_MODEL_PROMOTED",
            severity="INFO",
            actor=actor_info["actor"],
            action=f"POST /api/v2/learning/promote/{version_id}",
            outcome="ACTIVE",
            details={"version_id": version_id, "notes": req.notes},
        )
        blockchain.record_event(
            LedgerEventType.MODEL_VERSION_CHANGED,
            f"promote:{version_id}|by={actor_info['actor']}",
            identity,
            metadata={"version_id": version_id, "rollback_reference": promoted.get("rollback_reference")},
        )
        return {"status": "promoted", "active_version": promoted}

    @router.post("/learning/rollback")
    def rollback_gnnm_model_version(request: Request):
        """Reversibly restore the previous ACTIVE GNNM checkpoint."""
        rt = get_runtime()
        auth = rt["auth"]
        reg = rt["gnnm_registry"]
        audit_logger = rt["audit_logger"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        actor_info = auth.authenticate_request(request, required_capability="learning:approve")
        try:
            restored = reg.rollback()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        audit_logger.record(
            event_type="GNNM_MODEL_ROLLBACK",
            severity="WARN",
            actor=actor_info["actor"],
            action="POST /api/v2/learning/rollback",
            outcome="RESTORED",
            details={"restored_version_id": restored["version_id"]},
        )
        blockchain.record_event(
            LedgerEventType.MODEL_VERSION_CHANGED,
            f"rollback:{restored['version_id']}|by={actor_info['actor']}",
            identity,
            metadata={"restored_version_id": restored["version_id"]},
        )
        return {"status": "rolled_back", "active_version": restored}

    @router.post("/knowledge/promote/{record_id}")
    def promote_quarantined_knowledge(record_id: str, req: V2PromoteKnowledgeRequest, request: Request):
        """
        P2 Knowledge Integrity: Promote Quarantined record ONLY after independent source
        corroboration and contradiction checks (never solely from absence of contradictions).
        """
        rt = get_runtime()
        auth = rt["auth"]
        memory = rt["memory"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        actor_info = auth.authenticate_request(request, required_capability="memory:write")
        rec = memory.storage.get_record(record_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"Memory record '{record_id}' not found.")
        if rec.get("visibility_policy") == VisibilityPolicy.PRIVATE.value:
            raise HTTPException(status_code=403, detail="PRIVATE memory cannot be promoted to COLLECTIVE.")

        clean_sources = [s.strip() for s in req.independent_sources if s and len(s.strip()) >= 4]
        if not clean_sources:
            raise HTTPException(
                status_code=400,
                detail="Promotion requires at least one traceable independent source or verifier reference.",
            )

        open_contras = [
            c for c in memory.storage.list_contradictions()
            if c.get("resolution_status") == "OPEN"
            and (c.get("record_a_id") == record_id or c.get("record_b_id") == record_id)
        ]
        if open_contras:
            rec["verification_status"] = VerificationStatus.DISPUTED.value
            memory.storage.save_record(rec)
            blockchain.record_event(
                LedgerEventType.KNOWLEDGE_REJECTED,
                f"{record_id}|open_contradictions={len(open_contras)}",
                identity,
                metadata={"record_id": record_id, "reason": "open_contradictions"},
            )
            raise HTTPException(
                status_code=409,
                detail=f"Cannot promote record '{record_id}': {len(open_contras)} unresolved contradiction(s).",
            )

        discovery = rt["discovery"]
        verified_attestations: List[Dict[str, str]] = []
        seen_verifiers = set()
        distinct_evidence_hashes = set()

        for att in req.node_attestations:
            v_node = att.node_id.strip()
            if v_node in seen_verifiers or identity.is_revoked(v_node):
                continue
            # Verifier must be either the local node or an explicitly TRUSTED peer
            is_trusted_verifier = (
                v_node == identity.node_id
                or discovery.get_trust_state(v_node) == PeerTrustState.TRUSTED
            )
            if not is_trusted_verifier:
                continue
            att_payload = canonical_json_bytes({
                "record_id": record_id,
                "record_hash": rec.get("integrity_hash", ""),
                "evidence_hash": att.evidence_hash.strip(),
                "source_ref": att.source_ref.strip(),
                "verifier_node_id": v_node,
            })
            if not identity.verify_message(att_payload, att.signature.strip(), v_node):
                continue
            seen_verifiers.add(v_node)
            distinct_evidence_hashes.add(att.evidence_hash.strip())
            verified_attestations.append({
                "node_id": v_node,
                "evidence_hash": att.evidence_hash.strip(),
                "source_ref": att.source_ref.strip(),
            })

        has_cryptographic_independence = (
            len(verified_attestations) >= 2 and len(distinct_evidence_hashes) >= 2
        )

        if req.verification_method == "independent_node_verification" and not has_cryptographic_independence:
            raise HTTPException(
                status_code=400,
                detail=(
                    "independent_node_verification requires >= 2 cryptographically verified Ed25519 "
                    "node_attestations from distinct TRUSTED verifiers with distinct evidence_hash values."
                ),
            )

        # Plain text strings can AT MOST produce SOURCE_CORROBORATED, NEVER INDEPENDENTLY_VERIFIED.
        new_status = (
            VerificationStatus.INDEPENDENTLY_VERIFIED.value
            if has_cryptographic_independence
            else VerificationStatus.SOURCE_CORROBORATED.value
        )
        rec["verification_status"] = new_status
        rec["visibility_policy"] = VisibilityPolicy.COLLECTIVE.value
        prov = rec.get("source_provenance") or {}
        if isinstance(prov, dict):
            prov["corroborating_sources"] = clean_sources
            prov["verification_method"] = req.verification_method
            prov["verified_attestations"] = verified_attestations
            rec["source_provenance"] = prov
        memory.storage.save_record(rec)
        memory.versioning.record_version_snapshot(
            rec, modified_by=actor_info["actor"], reason=f"promoted:{req.verification_method}"
        )
        blockchain.record_event(
            LedgerEventType.KNOWLEDGE_ACCEPTED,
            f"{record_id}|status={new_status}|attestations={len(verified_attestations)}|sources={len(clean_sources)}",
            identity,
            metadata={
                "record_id": record_id,
                "epistemic_status": new_status,
                "independent_sources": clean_sources,
                "verified_attestations_count": len(verified_attestations),
                "verification_method": req.verification_method,
            },
        )
        return {
            "status": "promoted",
            "epistemic_status": new_status,
            "verified_attestations": len(verified_attestations),
            "record": rec,
        }

    # ─── 4. /api/v2/agents (Internal Operational Endpoints — Authenticated) ───
    @router.get("/agents")
    def get_agents_status(request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        scheduler = rt["scheduler"]
        auth.authenticate_request(request, required_capability="agent:read")
        return {
            "agents": scheduler.list_agents(),
            "tasks": [t.to_dict() for t in scheduler.tasks.values()],
        }

    @router.post("/agents/run")
    def run_autonomous_agent(req: V2AgentRunRequest, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        scheduler = rt["scheduler"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        # P0: Authenticate agent execution AND always force human_approved=False on submission
        actor_info = auth.authenticate_request(request, required_capability="agent:run")
        task = scheduler.submit_objective(
            objective=req.objective,
            requires_human_approval=req.requires_human_approval,
            human_approved=False,
            max_iterations=req.max_iterations,
            requested_tools=req.requested_tools,
        )
        blockchain.record_event(
            LedgerEventType.TASK_CREATED,
            f"{task.task_id}|state={task.state.value}",
            identity,
            metadata={
                "task_id": task.task_id,
                "actor": actor_info["actor"],
                "requires_human_approval": req.requires_human_approval,
            },
        )
        return task.to_dict()

    @router.post("/agents/tasks/{task_id}/approve")
    def approve_agent_task(task_id: str, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        scheduler = rt["scheduler"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]
        audit_logger = rt["audit_logger"]

        actor_info = auth.authenticate_request(request, required_capability="agent:approve")
        try:
            task = scheduler.approve_task(task_id)
            blockchain.record_event(
                LedgerEventType.RESULT_VERIFIED,
                f"approve:{task_id}|by={actor_info['actor']}",
                identity,
                metadata={"task_id": task_id, "approved_by": actor_info["actor"], "state": task.state.value},
            )
            audit_logger.record(
                event_type="HUMAN_APPROVAL_GRANTED",
                severity="INFO",
                actor=actor_info["actor"],
                action=f"POST /api/v2/agents/tasks/{task_id}/approve",
                outcome="APPROVED",
                details={"task_id": task_id, "state": task.state.value},
            )
            return task.to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ─── 5. /api/v2/network & /api/v2/nodes (P3 Full GIP/2.0 Handlers) ────────
    @router.get("/network")
    def get_network_status():
        rt = get_runtime()
        p2p = rt["p2p"]
        discovery = rt["discovery"]
        identity = rt["identity"]
        return {
            "protocol_version": "GIP/2.0",
            "local_node_id": identity.node_id,
            "transport": "Authenticated-HTTPS (libp2p-ready abstraction)",
            "peers_urls": list(p2p.peers),
            "discovered_nodes": discovery.list_active_peers(),
            "revoked_node_ids": sorted(identity.revoked_node_ids),
        }

    @router.post("/network/gip")
    def receive_gip_endpoint(msg: GIPMessage):
        rt = get_runtime()
        validator = rt["gip_validator"]
        discovery = rt["discovery"]
        identity = rt["identity"]
        audit_logger = rt["audit_logger"]
        synchronizer = rt["synchronizer"]
        cognition = rt["cognition"]
        memory = rt["memory"]
        blockchain = rt["blockchain"]

        # Step 1: Cryptographic & replay validation
        valid, reason = validator.validate(msg)
        if not valid:
            audit_logger.record(
                event_type="GIP_REJECTED",
                severity="HIGH",
                actor=msg.sender_node_id or "unknown",
                action=f"POST /api/v2/network/gip ({msg.message_type})",
                outcome="REJECTED",
                details={"reason": reason},
            )
            raise HTTPException(status_code=403, detail=f"GIP message rejected: {reason}")

        # Step 2: Explicit Peer Trust State Enforcement (UNKNOWN -> PENDING -> AUTHENTICATED -> TRUSTED -> REVOKED)
        # Signature validity proves key ownership, NOT network trust.
        # Operational GIP namespaces ALWAYS require TRUSTED state (even when peer registry is empty!).
        policy = rt["policy"]
        sender_trust = discovery.get_trust_state(msg.sender_node_id)
        if msg.message_type not in (GIPMessageType.GIP_IDENTITY, GIPMessageType.GIP_DISCOVERY, GIPMessageType.GIP_CAPABILITY):
            if msg.sender_node_id != identity.node_id and sender_trust != PeerTrustState.TRUSTED:
                audit_logger.record(
                    event_type="GIP_UNTRUSTED_PEER_REJECTED",
                    severity="HIGH",
                    actor=msg.sender_node_id[:16],
                    action=f"GIP:{msg.message_type}",
                    outcome="REJECTED",
                    details={"trust_state": sender_trust.value},
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Peer signature valid, but peer trust_state='{sender_trust.value}' is not TRUSTED.",
                )

        # Step 3: Full 8-namespace GIP/2.0 operational dispatch
        if msg.message_type in (GIPMessageType.GIP_IDENTITY, GIPMessageType.GIP_DISCOVERY, GIPMessageType.GIP_CAPABILITY):
            raw_ep = str(msg.payload.get("endpoint", "")).strip()
            if not raw_ep:
                raise HTTPException(status_code=400, detail="GIP discovery requires a valid endpoint URL.")
            ok_ep, clean_or_reason = validate_peer_url(raw_ep, allow_local=policy.allow_local_peers)
            if not ok_ep:
                audit_logger.record(
                    event_type="GIP_SSRF_BLOCKED",
                    severity="HIGH",
                    actor=msg.sender_node_id[:16],
                    action=f"GIP:{msg.message_type}",
                    outcome="BLOCKED",
                    details={"endpoint": raw_ep, "reason": clean_or_reason},
                )
                raise HTTPException(status_code=400, detail=f"GIP endpoint blocked by SSRF guard: {clean_or_reason}")

            # Optional signed peer key rotation certificate propagation (Round 3 Finding E)
            rot_cert = msg.payload.get("rotation_certificate")
            rotation_applied = False
            if isinstance(rot_cert, dict):
                if not identity.apply_peer_rotation_certificate(rot_cert):
                    raise HTTPException(
                        status_code=403,
                        detail="Invalid dual-signed key rotation certificate.",
                    )
                rotation_applied = True

            # Discovered peers start in PENDING (never auto-TRUSTED until operator approves)
            desc = discovery.register_peer(
                node_id=msg.sender_node_id,
                endpoint=clean_or_reason,
                node_name=msg.payload.get("node_name", "peer"),
                capabilities=msg.payload.get("capabilities", ["symbolic", "evidence"]),
                trust_state=PeerTrustState.PENDING,
            )
            ack = GIPMessage(
                message_type=GIPMessageType.GIP_CAPABILITY,
                sender_node_id=identity.node_id,
                recipient_node_id=msg.sender_node_id,
                payload={
                    "status": "registered_pending_trust",
                    "trust_state": desc.trust_state.value,
                    "rotation_applied": rotation_applied,
                    "in_reply_to": msg.message_id,
                    "capabilities": ["symbolic", "evidence", "verification", "memory"],
                },
            ).sign(identity)
            return {"accepted": True, "trust_state": desc.trust_state.value, "rotation_applied": rotation_applied, "ack": ack.model_dump()}

        if msg.message_type == GIPMessageType.GIP_TASK:
            objective = str(msg.payload.get("objective", "")).strip()
            task_id = str(msg.payload.get("task_id", msg.message_id))
            if not objective:
                raise HTTPException(status_code=400, detail="GIP-TASK requires non-empty objective.")
            blockchain.record_event(
                LedgerEventType.TASK_ACCEPTED, f"{task_id}|{objective[:64]}", identity,
                metadata={"task_id": task_id, "sender": msg.sender_node_id[:16]},
            )
            local_recs = mem_auth.filter_records(memory.retrieve(query=objective, top_k=4), "peer")
            res = cognition.execute(objective=objective, memory_records=local_recs)
            blockchain.record_event(
                LedgerEventType.EVIDENCE_SUBMITTED, f"{task_id}|conf={res.confidence}", identity,
                metadata={"task_id": task_id, "verification_status": res.verification_status},
            )
            reply = GIPMessage(
                message_type=GIPMessageType.GIP_VERIFY,
                sender_node_id=identity.node_id,
                recipient_node_id=msg.sender_node_id,
                payload={
                    "task_id": task_id,
                    "conclusion": res.conclusion,
                    "confidence": res.confidence,
                    "verification_status": res.verification_status,
                    "symbolic_verified": bool(res.capabilities_used.get("symbolic_used_in_this_call")),
                    "evidence": [e.model_dump() for e in res.evidence],
                },
            ).sign(identity)
            return {"accepted": True, "reply": reply.model_dump()}

        if msg.message_type == GIPMessageType.GIP_KNOWLEDGE:
            content = str(msg.payload.get("content", "")).strip()
            if not content:
                raise HTTPException(status_code=400, detail="GIP-KNOWLEDGE requires content.")
            # P2 Quarantine: incoming peer knowledge enters LOCAL Quarantine (UNVERIFIED)
            rec = memory.store(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                origin_node_id=msg.sender_node_id,
                source_type="unverified_peer",
                author=f"peer:{msg.sender_node_id[:12]}",
                signature=msg.signature,
                confidence=min(float(msg.payload.get("confidence", 0.5)), 0.6),
                verification_status=VerificationStatus.UNVERIFIED,
                visibility_policy=VisibilityPolicy.LOCAL,
            )
            blockchain.record_event(
                LedgerEventType.KNOWLEDGE_PROPOSED, f"{rec['record_id']}|from={msg.sender_node_id[:16]}", identity,
                metadata={"record_id": rec["record_id"], "quarantined": True},
            )
            return {"accepted": True, "quarantined_record_id": rec["record_id"], "status": "QUARANTINED_UNVERIFIED"}

        if msg.message_type == GIPMessageType.GIP_VERIFY:
            task_id = str(msg.payload.get("task_id", msg.message_id))
            # Single-node submission verified by outer Ed25519 envelope — NOT multi-node consensus!
            blockchain.record_event(
                LedgerEventType.EVIDENCE_SUBMITTED,
                f"verify:{task_id}|from={msg.sender_node_id[:16]}",
                identity,
                metadata={"task_id": task_id, "sender": msg.sender_node_id, "single_node_submission": True},
            )
            return {
                "accepted": True,
                "status": "SINGLE_NODE_SUBMISSION_VERIFIED",
                "multi_node_consensus": False,
                "task_id": task_id,
                "sender_node_id": msg.sender_node_id,
            }

        if msg.message_type == GIPMessageType.GIP_CONSENSUS:
            # P1 + Round 3 Consensus Integrity:
            # Cryptographic Ed25519 verification, duplicate-node rejection, revocation check,
            # policy quorum (>= 2), and evidence independence are enforced directly inside
            # QuorumConsensusManager.evaluate_submissions().
            task_id = str(msg.payload.get("task_id", msg.message_id))
            raw_subs = msg.payload.get("submissions", [])
            if not isinstance(raw_subs, list):
                raise HTTPException(status_code=400, detail="GIP-CONSENSUS submissions must be a list.")

            policy_quorum = max(2, int(getattr(policy, "min_quorum_size", 2)))
            eval_res = consensus_mgr.evaluate_submissions(
                task_id=task_id,
                submissions=raw_subs,
                quorum_size=policy_quorum,
                verifier_identity=identity,
            )
            return {"accepted": True, "consensus": eval_res}

        if msg.message_type == GIPMessageType.GIP_SYNC:
            sync_report = synchronizer.ingest_sync_message(msg)
            return {"accepted": True, "sync_report": sync_report}

        ack = GIPMessage(
            message_type=GIPMessageType.GIP_CAPABILITY,
            sender_node_id=identity.node_id,
            recipient_node_id=msg.sender_node_id,
            payload={"status": "accepted", "in_reply_to": msg.message_id},
        ).sign(identity)
        return {"accepted": True, "ack": ack.model_dump()}

    @router.get("/nodes")
    def list_nodes():
        rt = get_runtime()
        discovery = rt["discovery"]
        identity = rt["identity"]
        return {
            "self_node": {
                "node_id": identity.node_id,
                "key_version": identity.key_version,
            },
            "peers": discovery.list_active_peers(),
        }

    @router.post("/nodes/register")
    def register_node(req: V2RegisterPeerRequest, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        p2p = rt["p2p"]
        discovery = rt["discovery"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        auth.authenticate_request(request, required_capability="network:manage")
        try:
            clean_ep = p2p.validate_and_add_peer(req.endpoint)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"SSRF / Peer validation failed: {e}")

        desc = discovery.register_peer(
            node_id=req.node_id,
            endpoint=clean_ep,
            node_name=req.node_name,
            capabilities=req.capabilities,
            trust_state=PeerTrustState.TRUSTED,
        )
        blockchain.record_event(
            LedgerEventType.NODE_REGISTERED,
            f"{req.node_id[:16]}|{clean_ep}",
            identity,
            metadata={"peer_node_id": req.node_id, "endpoint": clean_ep, "trust_state": "TRUSTED"},
        )
        return desc.to_dict()

    @router.post("/nodes/{node_id}/trust")
    def update_peer_trust_state(node_id: str, req: V2PeerTrustRequest, request: Request):
        """P1 Peer Trust Lifecycle Management (Admin/Operator explicit promotion to TRUSTED or REVOKED)."""
        rt = get_runtime()
        auth = rt["auth"]
        discovery = rt["discovery"]
        identity = rt["identity"]
        blockchain = rt["blockchain"]
        auth.authenticate_request(request, required_capability="network:manage")
        try:
            state = PeerTrustState(req.trust_state.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid trust_state: {req.trust_state}")
        try:
            desc = discovery.set_trust_state(node_id, state)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        if state == PeerTrustState.REVOKED:
            identity.revoke_peer(node_id)
        blockchain.record_event(
            LedgerEventType.NODE_REGISTERED,
            f"trust:{node_id[:16]}={state.value}",
            identity,
            metadata={"peer_node_id": node_id, "trust_state": state.value},
        )
        return desc.to_dict()

    # ─── 6. /api/v2/federation (Stage 4 Collective Knowledge Network) ─────────
    @router.get("/federation")
    def list_federation_organizations():
        rt = get_runtime()
        federation = rt["federation"]
        return {
            "count": len(federation.list_organizations()),
            "organizations": federation.list_organizations(),
        }

    @router.post("/federation/organizations")
    def register_federation_organization(profile: OrganizationProfile, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        federation = rt["federation"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]
        policy = rt["policy"]

        auth.authenticate_request(request, required_capability="network:manage")
        ok_ep, clean_ep = validate_peer_url(profile.endpoint, allow_local=policy.allow_local_peers)
        if not ok_ep:
            raise HTTPException(status_code=400, detail=f"Federation endpoint blocked by SSRF guard: {clean_ep}")
        profile.endpoint = clean_ep
        saved = federation.register_organization(profile)
        blockchain.record_event(
            LedgerEventType.NODE_REGISTERED,
            f"org:{saved.org_id}|{saved.org_type}|{saved.node_id[:12]}",
            identity,
            metadata={"org_id": saved.org_id, "org_type": saved.org_type, "node_id": saved.node_id},
        )
        return saved.model_dump()

    # ─── 7. /api/v2/system ────────────────────────────────────────────────────
    @router.get("/system")
    @router.get("/system/health")
    def system_health(request: Request):
        rt = get_runtime()
        engine = rt["engine"]
        memory = rt["memory"]
        blockchain = rt["blockchain"]
        audit_logger = rt["audit_logger"]
        policy = rt["policy"]
        learning = rt["learning"]
        auth = rt["auth"]
        actor = auth.resolve_optional_actor(request)
        is_privileged = actor["role"] in ("admin", "operator")

        audit_summary: Dict[str, Any] = {
            "events_count": len(audit_logger.events),
            "integrity_verified": audit_logger.verify_integrity(),
        }
        if is_privileged:
            audit_summary["recent_events"] = audit_logger.get_recent(limit=10)

        return {
            "status": "healthy" if blockchain.status == "HEALTHY" else blockchain.status,
            "version": "2.0.0",
            "capabilities": engine.capability_status(),
            "storage": memory.storage.health(),
            "ledger": {
                "type": "tamper_evident_local_signed_event_ledger",
                "status": blockchain.status,
                "height": len(blockchain.chain),
                "valid": blockchain.is_chain_valid(),
            },
            "security_audit": audit_summary,
            "governance_policy": policy.to_dict(),
            "learning": {
                "active_strategy": learning.adapter.active_strategy,
                "experiments_count": len(learning.experiments),
            },
        }

    @router.post("/system/rotate-key")
    def rotate_node_key(request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        identity = rt["identity"]
        audit_logger = rt["audit_logger"]
        auth.authenticate_request(request, required_capability="security:rotate_key")
        cert = identity.rotate_key()
        audit_logger.record(
            event_type="KEY_ROTATED",
            severity="INFO",
            actor="admin",
            action="POST /api/v2/system/rotate-key",
            outcome="SUCCESS",
            details=cert,
        )
        return cert

    @router.post("/system/learning/experiment")
    def run_learning_experiment(req: V2LearningExperimentRequest, request: Request):
        rt = get_runtime()
        auth = rt["auth"]
        learning = rt["learning"]
        engine = rt["engine"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        auth.authenticate_request(request, required_capability="learning:approve")
        if req.simulate_failure:
            candidate_fn = lambda _p: "BROKEN_OUTPUT"
        else:
            candidate_fn = lambda p: engine.generate(p)

        exp = learning.propose_and_evaluate_experiment(
            name=req.name,
            level=req.level,
            proposed_changes=req.proposed_changes,
            candidate_fn=candidate_fn,
            policy_approved=True,
        )
        if exp["active_model_affected"]:
            blockchain.record_event(
                LedgerEventType.MODEL_VERSION_CHANGED,
                f"{exp['experiment_id']}|v={exp['post_strategy_version']}",
                identity,
                metadata={"experiment_id": exp["experiment_id"], "version": exp["post_strategy_version"]},
            )
        return exp

    return router
