from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from gajahmada_ai.blockchain.chain import LedgerEventType
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.consensus import QuorumConsensusManager
from gajahmada_ai.network.discovery import PeerTrustState
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


class V2PromoteKnowledgeRequest(BaseModel):
    """P2 Remediation: Promotion requires explicit independent sources & verification method."""
    independent_sources: List[str] = Field(..., min_length=1)
    verification_method: str = Field(default="source_corroboration", min_length=3, max_length=64)
    verifier_notes: str = Field(default="", max_length=1024)


class V2PeerTrustRequest(BaseModel):
    trust_state: str = Field(..., description="PENDING | AUTHENTICATED | TRUSTED | REVOKED")


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

    # ─── 2. /api/v2/reasoning ─────────────────────────────────────────────────
    @router.get("/reasoning")
    def list_reasoning_history():
        rt = get_runtime()
        cognition = rt["cognition"]
        items = list(cognition.history.values())[-25:]
        return {
            "count": len(items),
            "results": [r.model_dump() for r in items],
        }

    @router.post("/reasoning")
    def run_reasoning(req: V2AskRequest, request: Request):
        return intelligence_ask(req, request)

    @router.get("/reasoning/{reasoning_id}")
    def inspect_reasoning(reasoning_id: str):
        rt = get_runtime()
        cognition = rt["cognition"]
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

        new_status = (
            VerificationStatus.INDEPENDENTLY_VERIFIED.value
            if len(clean_sources) >= 2 or req.verification_method == "independent_node_verification"
            else VerificationStatus.SOURCE_CORROBORATED.value
        )
        rec["verification_status"] = new_status
        rec["visibility_policy"] = VisibilityPolicy.COLLECTIVE.value
        prov = rec.get("source_provenance") or {}
        if isinstance(prov, dict):
            prov["corroborating_sources"] = clean_sources
            prov["verification_method"] = req.verification_method
            rec["source_provenance"] = prov
        memory.storage.save_record(rec)
        memory.versioning.record_version_snapshot(
            rec, modified_by=actor_info["actor"], reason=f"promoted:{req.verification_method}"
        )
        blockchain.record_event(
            LedgerEventType.KNOWLEDGE_ACCEPTED,
            f"{record_id}|status={new_status}|sources={len(clean_sources)}",
            identity,
            metadata={
                "record_id": record_id,
                "epistemic_status": new_status,
                "independent_sources": clean_sources,
                "verification_method": req.verification_method,
            },
        )
        return {"status": "promoted", "epistemic_status": new_status, "record": rec}

    # ─── 4. /api/v2/agents ────────────────────────────────────────────────────
    @router.get("/agents")
    def get_agents_status():
        rt = get_runtime()
        scheduler = rt["scheduler"]
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
                    "in_reply_to": msg.message_id,
                    "capabilities": ["symbolic", "evidence", "verification", "memory"],
                },
            ).sign(identity)
            return {"accepted": True, "trust_state": desc.trust_state.value, "ack": ack.model_dump()}

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
            # P1 Consensus Integrity:
            # 1. NEVER trust caller-supplied `signature_verified`.
            # 2. Verify EACH submission's individual Ed25519 signature over its canonical payload.
            # 3. Reject duplicate node_ids (Sybil / replay inflation).
            # 4. Enforce quorum_size strictly from local GovernancePolicy (>= 2), NEVER peer payload.
            task_id = str(msg.payload.get("task_id", msg.message_id))
            raw_subs = msg.payload.get("submissions", [])
            if not isinstance(raw_subs, list):
                raise HTTPException(status_code=400, detail="GIP-CONSENSUS submissions must be a list.")

            verified_subs: List[Dict[str, Any]] = []
            seen_nodes = set()
            rejected_count = 0
            for sub in raw_subs:
                if not isinstance(sub, dict):
                    rejected_count += 1
                    continue
                sub_node_id = str(sub.get("node_id", "")).strip()
                sub_sig = str(sub.get("signature", "")).strip()
                if not sub_node_id or not sub_sig or sub_node_id in seen_nodes:
                    rejected_count += 1
                    continue
                if identity.is_revoked(sub_node_id):
                    rejected_count += 1
                    continue
                canonical_sub = canonical_json_bytes({
                    "task_id": task_id,
                    "node_id": sub_node_id,
                    "conclusion": str(sub.get("conclusion", "")),
                    "confidence": float(sub.get("confidence", 0.5)),
                    "evidence": sub.get("evidence", []),
                })
                if not identity.verify_message(canonical_sub, sub_sig, sub_node_id):
                    rejected_count += 1
                    continue
                seen_nodes.add(sub_node_id)
                verified_subs.append({
                    "node_id": sub_node_id,
                    "signature_verified": True,
                    "conclusion": str(sub.get("conclusion", "")),
                    "confidence": float(sub.get("confidence", 0.5)),
                    "symbolic_verified": bool(sub.get("symbolic_verified", False)),
                    "evidence": sub.get("evidence", []),
                })

            policy_quorum = max(2, int(getattr(policy, "min_quorum_size", 2)))
            eval_res = consensus_mgr.evaluate_submissions(
                task_id=task_id, submissions=verified_subs, quorum_size=policy_quorum
            )
            eval_res["rejected_submissions"] = rejected_count
            eval_res["enforced_quorum_size"] = policy_quorum
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

    # ─── 6. /api/v2/system ────────────────────────────────────────────────────
    @router.get("/system")
    @router.get("/system/health")
    def system_health():
        rt = get_runtime()
        engine = rt["engine"]
        memory = rt["memory"]
        blockchain = rt["blockchain"]
        audit_logger = rt["audit_logger"]
        policy = rt["policy"]
        learning = rt["learning"]
        return {
            "status": "healthy" if blockchain.status == "HEALTHY" else blockchain.status,
            "version": "2.0.0",
            "capabilities": engine.capability_status(),
            "storage": memory.storage.health(),
            "ledger": {
                "status": blockchain.status,
                "height": len(blockchain.chain),
                "valid": blockchain.is_chain_valid(),
            },
            "security_audit": {
                "events_count": len(audit_logger.events),
                "integrity_verified": audit_logger.verify_integrity(),
                "recent_events": audit_logger.get_recent(limit=10),
            },
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
