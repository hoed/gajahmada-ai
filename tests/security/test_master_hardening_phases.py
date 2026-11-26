"""
Master Engineering Hardening Tests (Phases 1-5):
- Phase 1 P0: Direct MemoryManager authorization across PRIVATE/SESSION/AGENT/ORGANIZATION/PUBLIC
  + AccessContext propagation + privacy-safe access-denied audit logging.
- Phase 2 P1: Embedding metadata honesty (heuristic vs learned) + concurrent per-request ContextBudget.
- Phase 4 P2: SignedKnowledgeEnvelope + independent organization counting (same org != independent).
- Phase 5 P2: Privacy-safe LifecycleObservabilityLogger + /api/v2/system/health/subsystems.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import time

from fastapi.testclient import TestClient
import pytest

from gajahmada_ai.api.server import app
from gajahmada_ai.knowledge.controller import GNNMKnowledgeAdapter, HybridKnowledgeController
from gajahmada_ai.knowledge.embedding import (
    FutureGNNMEmbeddingProvider,
    LocalLearnedEmbeddingProvider,
    NativeSemanticEmbeddingProvider,
)
from gajahmada_ai.knowledge.interfaces import HybridKnowledgeConfig, KnowledgeRecord
from gajahmada_ai.knowledge.retriever import HybridKnowledgeRetriever
from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.signed_knowledge import SignedKnowledgeEnvelope, count_independent_operators
from gajahmada_ai.neural.router import ModelRouter
from gajahmada_ai.security.audit import SecurityAuditLogger
from gajahmada_ai.security.context import AccessContext
from gajahmada_ai.security.crypto import NodeIdentity
from gajahmada_ai.security.observability import LifecycleObservabilityLogger


def test_phase1_direct_memory_manager_enforces_cross_user_session_agent_org_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        audit = SecurityAuditLogger(storage_path=os.path.join(tmp, "audit.json"))
        mem = MemoryManager(
            sqlite_path=os.path.join(tmp, "mem.db"),
            backup_dir=os.path.join(tmp, "objects"),
        )
        mem.authorization.audit_logger = audit

        # Store records across all sensitive scopes
        mem.store(
            content="Alpha secret vault code 771122 for Alice.",
            visibility_policy=VisibilityPolicy.PRIVATE,
            owner_actor_id="user:alice",
        )
        mem.store(
            content="Session note 771122 for sess-A.",
            visibility_policy=VisibilityPolicy.SESSION,
            session_id="sess-A",
        )
        mem.store(
            content="Agent telemetry 771122 for agent-1.",
            visibility_policy=VisibilityPolicy.AGENT,
            agent_id="agent-1",
        )
        mem.store(
            content="University research 771122 for org-UGM.",
            visibility_policy=VisibilityPolicy.ORGANIZATION,
            organization_id="org-UGM",
        )
        mem.store(
            content="Public announcement 771122 for everyone.",
            visibility_policy=VisibilityPolicy.PUBLIC,
            verification_status=VerificationStatus.VERIFIED,
        )

        # 1. Missing AccessContext (direct internal call) MUST NOT return PRIVATE/SESSION/AGENT/ORG!
        no_ctx_hits = mem.retrieve("771122", top_k=10)
        assert all(r["visibility_policy"] == "PUBLIC" for r in no_ctx_hits)

        # 2. Bob (operator) cannot see Alice's PRIVATE record, sess-A, agent-1, or org-UGM
        bob_ctx = AccessContext(
            actor_id="user:bob",
            actor_role="operator",
            organization_id="org-ITB",
            agent_id="agent-2",
            session_id="sess-B",
        )
        bob_hits = mem.retrieve("771122", top_k=10, access_context=bob_ctx)
        assert all(r["visibility_policy"] == "PUBLIC" for r in bob_hits)

        # Verify access-denied events were logged WITHOUT leaking private content!
        denied_events = [e for e in audit.events if e["event_type"] == "MEMORY_ACCESS_DENIED"]
        assert len(denied_events) >= 3
        for ev in denied_events:
            assert "771122" not in str(ev)
            assert "vault code" not in str(ev)

        # 3. Matching owner/session/agent/org CAN retrieve their own scoped records
        alice_ctx = AccessContext(
            actor_id="user:alice",
            actor_role="operator",
            organization_id="org-UGM",
            agent_id="agent-1",
            session_id="sess-A",
        )
        alice_hits = mem.retrieve("771122", top_k=10, access_context=alice_ctx)
        vis_seen = {r["visibility_policy"] for r in alice_hits}
        assert {"PRIVATE", "SESSION", "AGENT", "ORGANIZATION", "PUBLIC"}.issubset(vis_seen)


def test_phase2_embedding_honesty_and_concurrent_immutable_context_budget():
    # Heuristic provider honestly reports is_learned_model=False
    sem = NativeSemanticEmbeddingProvider()
    meta = sem.identity_metadata()
    assert meta["is_learned_model"] is False
    assert meta["encoder_kind"] == "heuristic_semantic_projection"
    assert len(meta["embedding_fingerprint"]) == 16

    # LocalLearnedEmbeddingProvider refuses to fake inference without trained weights
    learned = LocalLearnedEmbeddingProvider(weights_path=None)
    assert learned.is_trained_and_loaded is False
    with pytest.raises(RuntimeError):
        learned.embed("test text")

    # FutureGNNMEmbeddingProvider honestly reports weights not loaded
    fut = FutureGNNMEmbeddingProvider(checkpoint_path=None)
    assert fut.identity_metadata()["gnnm_embedding_weights_loaded"] is False

    # Concurrent GNNMKnowledgeAdapter calls with different context budgets must NOT mutate shared state
    ret = HybridKnowledgeRetriever(config=HybridKnowledgeConfig(semantic_embedding=True))
    ret.index_record(
        KnowledgeRecord(
            content="GajahMada Sovereign AI concurrent retrieval verification text " * 20,
            source_id="src-conc",
            visibility_policy="PUBLIC",
            verification_status="VERIFIED",
        )
    )
    ctrl = HybridKnowledgeController(ret)
    orig_max_seq = ctrl.context_builder.max_seq_len
    adapter = GNNMKnowledgeAdapter(ModelRouter(), ctrl)
    ctx = AccessContext(actor_id="op", actor_role="operator").to_knowledge_access_context()

    def worker(budget_tokens: int):
        out = adapter.generate(
            "GajahMada concurrent retrieval",
            ctx,
            options={"max_context_tokens": budget_tokens, "max_tokens": 32},
        )
        return out["token_budget"]["available_context_tokens"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(worker, [64, 128, 256, 64, 128, 256]))

    assert results[0] <= 64 and results[2] <= 256
    assert ctrl.context_builder.max_seq_len == orig_max_seq


def test_phase4_signed_knowledge_envelope_and_organization_independence():
    verifier = NodeIdentity(key_path=None)
    node_org1_a = NodeIdentity(key_path=None)
    node_org1_b = NodeIdentity(key_path=None)
    node_org2 = NodeIdentity(key_path=None)

    # Two DIFFERENT public keys belonging to the SAME organization MUST count as only 1 independent org!
    env1 = SignedKnowledgeEnvelope(
        content="Ocean temperature at Buoy-09 is 27.8C.",
        source_id="src-buoy-09",
        organization_id="org-bmkg",
        evidence_refs=["sha256:ev-a"],
        verification_status="SOURCE_CORROBORATED",
    ).sign(node_org1_a)
    env2 = SignedKnowledgeEnvelope(
        content="Ocean temperature at Buoy-09 is 27.8C.",
        source_id="src-buoy-09-mirror",
        organization_id="org-bmkg",
        evidence_refs=["sha256:ev-b"],
        verification_status="SOURCE_CORROBORATED",
    ).sign(node_org1_b)

    same_org_res = count_independent_operators([env1, env2], verifier)
    assert same_org_res["distinct_nodes"] == 2
    assert same_org_res["independent_organizations"] == 1
    assert same_org_res["independently_corroborated"] is False

    # Adding a second DISTINCT organization achieves true independent corroboration
    env3 = SignedKnowledgeEnvelope(
        content="Ocean temperature at Buoy-09 is 27.8C.",
        source_id="src-noaa-09",
        organization_id="org-noaa",
        evidence_refs=["sha256:ev-c"],
        verification_status="SOURCE_CORROBORATED",
    ).sign(node_org2)
    diff_org_res = count_independent_operators([env1, env3], verifier)
    assert diff_org_res["independent_organizations"] == 2
    assert diff_org_res["independently_corroborated"] is True

    # Tampering content after signing MUST fail verification
    env1.content = "Tampered temperature 99.9C"
    ok, reason = env1.verify(verifier)
    assert ok is False and reason == "content_hash_mismatch"

    # Expired envelope MUST fail verification
    env_exp = SignedKnowledgeEnvelope(
        content="Old reading",
        source_id="src-old",
        organization_id="org-x",
        expires_at=time.time() - 10.0,
    ).sign(node_org2)
    ok_exp, reason_exp = env_exp.verify(verifier)
    assert ok_exp is False and reason_exp == "envelope_expired"


def test_phase5_observability_privacy_and_subsystem_health_endpoint():
    obs = LifecycleObservabilityLogger()
    ev = obs.record(
        event_name="reasoning_complete",
        request_id="req-123",
        actor_id="alice@university.ac.id",
        active_model_version="gnnm-v1",
        retrieval_duration_ms=4.2,
        retrieved_count=3,
        extra={"prompt": "SECRET PROMPT MUST BE SCRUBBED", "safe_metric": 1},
    )
    assert "alice@university.ac.id" not in ev["actor_tag"]
    assert "prompt" not in ev["extra"]
    assert ev["extra"]["safe_metric"] == 1

    client = TestClient(app)
    r = client.get("/api/v2/system/health/subsystems")
    assert r.status_code == 200
    body = r.json()
    valid_states = {"HEALTHY", "DEGRADED", "UNAVAILABLE", "NOT_CONFIGURED"}
    for sub in ("api", "gnnm_runtime", "knowledge_subsystem", "memory_subsystem", "learning_pipeline", "model_registry", "audit_ledger", "distributed_network", "supabase_remote_store"):
        assert sub in body
        assert body[sub]["status"] in valid_states
