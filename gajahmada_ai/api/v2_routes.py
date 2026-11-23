from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from gajahmada_ai.blockchain.chain import LedgerEventType
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.consensus import QuorumConsensusManager
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType


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
    def intelligence_ask(req: V2AskRequest):
        rt = get_runtime()
        cognition = rt["cognition"]
        memory = rt["memory"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        retrieved = memory.retrieve(query=req.objective, top_k=4)
        result = cognition.execute(
            objective=req.objective,
            memory_records=retrieved,
            symbolic_facts=req.symbolic_facts,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        # P2 Knowledge Quarantine Policy:
        # Unverified reasoning outputs enter LOCAL Quarantine (UNVERIFIED) — never auto-published to COLLECTIVE.
        clean_ans = result.conclusion.split("\n\n---\n📚")[0].strip()
        qa_entry = f"Pertanyaan: {req.objective.strip()} | Jawaban: {clean_ans}"
        is_generic = (
            "Permintaan telah diproses oleh **GajahMada Hybrid Cognitive Engine**" in result.conclusion
            or "Belum ada data spesifik untuk" in result.conclusion
        )
        can_promote_collective = (
            not is_generic
            and result.verification_status == "VERIFIED"
            and float(result.confidence) >= 0.85
            and not result.contradictions
        )

        rag = rt.get("rag")
        engine = rt.get("engine")
        if rag and engine and can_promote_collective and qa_entry not in rag.doc_store:
            emb = engine.create_embedding(qa_entry)
            if emb:
                rag.ingest(qa_entry, emb)
                rag.flush()

        memory.store(
            content=qa_entry,
            memory_type=MemoryType.EPISODIC if is_generic else MemoryType.SEMANTIC,
            origin_node_id=identity.node_id,
            source_type="cognitive_brain_qa_update",
            author="cognitive_brain",
            confidence=result.confidence,
            verification_status=(
                VerificationStatus.VERIFIED if can_promote_collective else VerificationStatus.UNVERIFIED
            ),
            visibility_policy=(
                VisibilityPolicy.COLLECTIVE if can_promote_collective else VisibilityPolicy.LOCAL
            ),
        )
        memory.store(
            content=f"Q: {req.objective[:160]} -> A ({result.verification_status}, conf={result.confidence})",
            memory_type=MemoryType.WORKING,
            origin_node_id=identity.node_id,
            source_type="cognitive_pipeline",
            confidence=result.confidence,
            verification_status=VerificationStatus(result.verification_status),
        )
        blockchain.record_event(
            LedgerEventType.TASK_COMPLETED,
            f"{result.reasoning_id}|status={result.verification_status}|conf={result.confidence}",
            identity,
            metadata={"reasoning_id": result.reasoning_id, "verification_status": result.verification_status},
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
    def run_reasoning(req: V2AskRequest):
        return intelligence_ask(req)

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
    def promote_quarantined_knowledge(record_id: str, request: Request):
        """P2 Knowledge Integrity: Promote a Quarantined (UNVERIFIED/LOCAL) record to VERIFIED/COLLECTIVE after checks."""
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

        open_contras = [
            c for c in memory.storage.list_contradictions()
            if c.get("resolution_status") == "OPEN"
            and (c.get("record_a_id") == record_id or c.get("record_b_id") == record_id)
        ]
        if open_contras:
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

        rec["verification_status"] = VerificationStatus.VERIFIED.value
        rec["visibility_policy"] = VisibilityPolicy.COLLECTIVE.value
        rec["confidence"] = max(float(rec.get("confidence", 0.75)), 0.85)
        memory.storage.save_record(rec)
        memory.versioning.record_version_snapshot(
            rec, modified_by=actor_info["actor"], reason="independent_verification_promoted"
        )
        blockchain.record_event(
            LedgerEventType.KNOWLEDGE_ACCEPTED,
            f"{record_id}|promoted_by={actor_info['actor']}",
            identity,
            metadata={"record_id": record_id, "actor": actor_info["actor"]},
        )
        return {"status": "promoted", "record": rec}

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
    def run_autonomous_agent(req: V2AgentRunRequest):
        rt = get_runtime()
        scheduler = rt["scheduler"]
        blockchain = rt["blockchain"]
        identity = rt["identity"]

        # P0: Always force human_approved=False on submission; approval must go through authenticated /approve
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
            metadata={"task_id": task.task_id, "requires_human_approval": req.requires_human_approval},
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

        # Step 2: Peer authorization (separate from signature validity!)
        # Handshake types (IDENTITY/DISCOVERY) register peer; operational types require registered or allowed peer
        known_ids = {p["node_id"] for p in discovery.list_active_peers()}
        if msg.message_type not in (GIPMessageType.GIP_IDENTITY, GIPMessageType.GIP_DISCOVERY):
            if validator.allowed_peers is not None and msg.sender_node_id not in validator.allowed_peers:
                raise HTTPException(status_code=403, detail="Peer signature valid, but peer is not authorized.")
            if known_ids and msg.sender_node_id not in known_ids and msg.sender_node_id != identity.node_id:
                raise HTTPException(
                    status_code=403,
                    detail="Peer signature valid, but peer has not completed GIP-IDENTITY/DISCOVERY registration.",
                )

        # Step 3: Full 8-namespace GIP/2.0 operational dispatch
        if msg.message_type in (GIPMessageType.GIP_IDENTITY, GIPMessageType.GIP_DISCOVERY, GIPMessageType.GIP_CAPABILITY):
            discovery.register_peer(
                node_id=msg.sender_node_id,
                endpoint=msg.payload.get("endpoint", "http://127.0.0.1:8000"),
                node_name=msg.payload.get("node_name", "peer"),
                capabilities=msg.payload.get("capabilities", ["symbolic", "evidence"]),
            )
            ack = GIPMessage(
                message_type=GIPMessageType.GIP_CAPABILITY,
                sender_node_id=identity.node_id,
                recipient_node_id=msg.sender_node_id,
                payload={"status": "registered", "in_reply_to": msg.message_id, "capabilities": ["symbolic", "evidence", "verification", "memory"]},
            ).sign(identity)
            return {"accepted": True, "ack": ack.model_dump()}

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
            submission = {
                "node_id": msg.sender_node_id,
                "signature_verified": True,
                "conclusion": msg.payload.get("conclusion", ""),
                "confidence": float(msg.payload.get("confidence", 0.5)),
                "symbolic_verified": bool(msg.payload.get("symbolic_verified", False)),
                "evidence": msg.payload.get("evidence", []),
            }
            eval_res = consensus_mgr.evaluate_submissions(task_id=task_id, submissions=[submission], quorum_size=1)
            blockchain.record_event(
                LedgerEventType.RESULT_VERIFIED, f"{task_id}|status={eval_res['status']}", identity,
                metadata={"task_id": task_id, "consensus_status": eval_res["status"]},
            )
            return {"accepted": True, "verification": eval_res}

        if msg.message_type == GIPMessageType.GIP_CONSENSUS:
            task_id = str(msg.payload.get("task_id", msg.message_id))
            submissions = msg.payload.get("submissions", [])
            quorum_size = int(msg.payload.get("quorum_size", 2))
            eval_res = consensus_mgr.evaluate_submissions(task_id=task_id, submissions=submissions, quorum_size=quorum_size)
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
        )
        blockchain.record_event(
            LedgerEventType.NODE_REGISTERED,
            f"{req.node_id[:16]}|{clean_ep}",
            identity,
            metadata={"peer_node_id": req.node_id, "endpoint": clean_ep},
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
