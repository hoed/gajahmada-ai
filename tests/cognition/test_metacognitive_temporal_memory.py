"""
Meta-Cognitive Reasoning & Temporal Memory Correction Tests (All 10 Required Cases + Regression).
"""
import os
import tempfile
import time

from fastapi.testclient import TestClient

from gajahmada_ai.api.server import app
from gajahmada_ai.cognition.intent import IntentClassifier
from gajahmada_ai.cognition.perception import PerceptionModule
from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.knowledge.ingestion import KnowledgeIngestionPipeline
from gajahmada_ai.knowledge.retriever import HybridKnowledgeRetriever
from gajahmada_ai.knowledge.source_registry import SourceRegistry
from gajahmada_ai.learning.learning_events import (
    LearningEvent,
    LearningEventTracker,
    LearningEventType,
)
from gajahmada_ai.security.context import AccessContext


def test_intent_classification_distinguishes_meta_cognitive_from_factual():
    perc = PerceptionModule()
    clf = IntentClassifier()

    assert clf.classify(perc.perceive("Apa yang telah kamu pelajari baru-baru ini?")).primary_intent == "META_LEARNING_QUERY"
    assert clf.classify(perc.perceive("Apa yang kamu pelajari hari ini?")).primary_intent == "RECENT_KNOWLEDGE_QUERY"
    assert clf.classify(perc.perceive("Apa yang kamu pelajari dari dokumen terakhir?")).primary_intent == "SOURCE_SPECIFIC_LEARNING_QUERY"
    assert clf.classify(perc.perceive("Apakah kamu sudah mempelajari informasi Palapa ke dalam model GNNM?")).primary_intent == "SELF_KNOWLEDGE_QUERY"
    assert clf.classify(perc.perceive("Apa yang belum kamu ketahui tentang topik ini?")).primary_intent == "KNOWLEDGE_GAP_QUERY"
    assert clf.classify(perc.perceive("How has your knowledge changed?")).primary_intent == "KNOWLEDGE_COMPARISON_QUERY"

    # Critical regression: factual inquiry MUST NOT be hijacked as meta-cognitive!
    factual = clf.classify(perc.perceive("Apa yang kamu ketahui tentang Era Orde Baru?"))
    assert factual.is_meta_cognitive is False
    assert factual.primary_intent == "factual_inquiry"


def test_metacognitive_queries_empty_stored_trained_and_unauthorized():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = LearningEventTracker(storage_path=os.path.join(tmp, "events.db"))
        pipe = CognitiveReasoningPipeline(learning_event_tracker=tracker)
        op_ctx = AccessContext(actor_id="op-1", actor_role="operator")
        anon_ctx = AccessContext(actor_id="anonymous", actor_role="anonymous")

        # Case 7: No recent learning events -> NEVER fabricates history
        r_empty = pipe.execute("Apa yang telah kamu pelajari baru-baru ini?", actor_context=op_ctx)
        assert r_empty.capabilities_used["meta_cognitive_used_in_this_call"] is True
        assert "tidak ditemukan" in r_empty.conclusion.lower()

        # Ingest a real document via KnowledgeIngestionPipeline (auto-emits DOCUMENT_INGESTED, STORED_ONLY)
        ret = HybridKnowledgeRetriever()
        reg = SourceRegistry()
        ing = KnowledgeIngestionPipeline(ret, reg, learning_event_tracker=tracker)
        ing.ingest_text(
            "Program Satelit Palapa diluncurkan pada tahun 1976 untuk menghubungkan telekomunikasi Nusantara.",
            source_type="txt",
            title="Sejarah Satelit Palapa",
            visibility_policy="COLLECTIVE",
            verification_status="VERIFIED",
            extra_metadata={"domain": "telekomunikasi_indonesia"},
        )

        # Case 1 & 8: Recent learning with stored knowledge but NO model training
        r_recent = pipe.execute("Apa yang telah kamu pelajari baru-baru ini?", actor_context=op_ctx)
        assert "Sejarah Satelit Palapa" in r_recent.conclusion
        assert "STORED_KNOWLEDGE" in r_recent.capabilities_used["knowledge_states"]
        assert "INTERNALIZED_KNOWLEDGE" not in r_recent.capabilities_used["knowledge_states"]
        assert "not yet performed" in r_recent.conclusion.lower()

        # Case 2: Today temporal filter
        r_today = pipe.execute("Apa yang kamu pelajari hari ini?", actor_context=op_ctx)
        assert "hari ini" in r_today.interpretation.lower()
        assert "Sejarah Satelit Palapa" in r_today.conclusion

        # Case 3: Source-specific latest document
        r_last = pipe.execute("Apa yang kamu pelajari dari dokumen terakhir?", actor_context=op_ctx)
        assert "Sejarah Satelit Palapa" in r_last.conclusion
        assert len(r_last.evidence) == 1

        # Case 5: Self-knowledge internalization check (stored, NOT yet internalized into GNNM)
        r_palapa_gnnm = pipe.execute(
            "Apakah kamu sudah mempelajari informasi Palapa ke dalam model GNNM?",
            actor_context=op_ctx,
        )
        assert "belum" in r_palapa_gnnm.conclusion.lower()
        assert "STORED_KNOWLEDGE" in r_palapa_gnnm.capabilities_used["knowledge_states"]
        assert "INTERNALIZED_KNOWLEDGE" not in r_palapa_gnnm.capabilities_used["knowledge_states"]

        # Case 6: Knowledge gap query
        r_gap = pipe.execute("Apa yang belum kamu ketahui tentang topik ini?", actor_context=op_ctx)
        assert "keterbatasan" in r_gap.conclusion.lower()

        # Case 10: Unauthorized actor MUST NOT see PRIVATE learning events
        tracker.record_event(
            LearningEvent(
                event_type=LearningEventType.DOCUMENT_INGESTED.value,
                actor_id="user:alice",
                visibility_policy="PRIVATE",
                source_id="src-priv-99",
                source_title="Dokumen Rahasia Proyek Garuda 999",
                knowledge_domain="confidential",
                summary="Kode rahasia internal 999.",
                verification_status="VERIFIED",
                internalization_status="STORED_ONLY",
            )
        )
        r_anon = pipe.execute("Apa yang telah kamu pelajari baru-baru ini?", actor_context=anon_ctx)
        assert "Garuda 999" not in r_anon.conclusion
        assert "rahasia" not in r_anon.conclusion.lower()

        # Case 9: Completed & verified GNNM training event -> reports INTERNALIZED_KNOWLEDGE
        tracker.record_event(
            LearningEvent(
                event_type=LearningEventType.TRAINING_COMPLETED.value,
                actor_id="learning_pipeline",
                visibility_policy="COLLECTIVE",
                source_id="src-palapa",
                source_type="sft",
                source_title="GNNM SFT Satelit Palapa (gnnm-v2)",
                knowledge_domain="telekomunikasi_indonesia",
                summary="Verified GNNM SFT training completed for Palapa corpus.",
                verification_status="VERIFIED",
                model_version_before="gnnm-v1",
                model_version_after="gnnm-v2",
                training_status="COMPLETED_AND_APPROVED",
                internalization_status="INTERNALIZED_IN_GNNM",
            )
        )
        r_trained = pipe.execute(
            "Apakah kamu sudah mempelajari informasi Palapa ke dalam model GNNM?",
            actor_context=op_ctx,
        )
        assert "INTERNALIZED_KNOWLEDGE" in r_trained.capabilities_used["knowledge_states"]
        assert "demonstrated" in r_trained.conclusion.lower()


def test_critical_regression_chat_endpoint_meta_vs_factual():
    """
    Critical Regression Test on live /ai/chat:
    1. 'Apa yang telah kamu pelajari baru-baru ini?' MUST NOT return an arbitrary Orde Baru passage.
    2. 'Apa yang kamu ketahui tentang Era Orde Baru?' MUST continue using factual RAG retrieval.
    """
    client = TestClient(app)

    r_meta = client.post("/ai/chat", json={"prompt": "Apa yang telah kamu pelajari baru-baru ini?"})
    assert r_meta.status_code == 200
    meta_resp = r_meta.json()["response"].lower()
    # Must NOT dump an arbitrary Orde Baru factual history passage as the answer!
    assert "orde baru" not in meta_resp

    r_fact = client.post("/ai/chat", json={"prompt": "Apa yang kamu ketahui tentang Era Orde Baru?"})
    assert r_fact.status_code == 200
    fact_resp = r_fact.json()["response"].lower()
    # Factual inquiry continues using existing RAG / Indonesia corpus
    assert any(kw in fact_resp for kw in ("orde baru", "soeharto", "1966", "pembangunan", "indonesia"))
