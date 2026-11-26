from dataclasses import dataclass
import re
from typing import List, Optional, Tuple

from gajahmada_ai.cognition.perception import PerceivedInput


META_COGNITIVE_INTENTS = {
    "META_LEARNING_QUERY",
    "RECENT_KNOWLEDGE_QUERY",
    "LEARNING_HISTORY_QUERY",
    "SOURCE_SPECIFIC_LEARNING_QUERY",
    "KNOWLEDGE_DOMAIN_QUERY",
    "SELF_KNOWLEDGE_QUERY",
    "KNOWLEDGE_GAP_QUERY",
    "KNOWLEDGE_COMPARISON_QUERY",
}


@dataclass
class ClassifiedIntent:
    primary_intent: str  # mathematical | symbolic_logic | factual_inquiry | autonomous_task | diagnostic | conversational | META_*
    requires_symbolic: bool
    requires_memory: bool
    requires_neural: bool
    requires_agent_execution: bool
    reasoning_domains: List[str]
    is_meta_cognitive: bool = False
    meta_cognitive_intent: Optional[str] = None
    temporal_scope: Optional[str] = None  # today | last_7_days | last_30_days | most_recent | date_range | all


class IntentClassifier:
    """
    Cognitive Stage 2: Intent & Capability Requirement Classifier.
    Distinguishes ordinary factual inquiries from Meta-Cognitive Learning &
    Temporal Knowledge State queries without overriding factual RAG.
    """

    def _detect_temporal_scope(self, ql: str) -> str:
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", ql):
            return "date_range"
        if any(p in ql for p in ("hari ini", "today", "sekarang ini")):
            return "today"
        if any(p in ql for p in ("7 hari", "minggu ini", "seminggu", "last 7 days", "this week", "past week")):
            return "last_7_days"
        if any(p in ql for p in ("30 hari", "bulan ini", "sebulan", "last 30 days", "this month", "past month")):
            return "last_30_days"
        if any(p in ql for p in ("dokumen terakhir", "last document", "terakhir kali", "paling baru", "most recent", "latest")):
            return "most_recent"
        return "recent"

    def _detect_meta_cognitive_intent(self, ql: str) -> Tuple[Optional[str], Optional[str]]:
        # 1. Source-specific learning ("dokumen terakhir", "from the last document", "dari sumber terakhir")
        if any(
            p in ql
            for p in (
                "dokumen terakhir", "sumber terakhir", "file terakhir",
                "last document", "latest document", "from the last", "dari dokumen",
            )
        ) and any(w in ql for w in ("pelajari", "pelajar", "learn", "ingest", "simpan", "ambil", "stored")):
            return "SOURCE_SPECIFIC_LEARNING_QUERY", "most_recent"

        # 2. Knowledge gap queries ("apa yang belum kamu ketahui", "what don't you know")
        if any(
            p in ql
            for p in (
                "belum kamu ketahui", "tidak kamu ketahui", "belum tahu", "keterbatasan pengetahuan",
                "what don't you know", "what do you not know", "knowledge gap", "blind spot", "gaps in your knowledge",
            )
        ):
            return "KNOWLEDGE_GAP_QUERY", "all"

        # 3. Knowledge comparison / evolution ("how has your knowledge changed", "bagaimana pengetahuanmu berubah")
        if any(
            p in ql
            for p in (
                "how has your knowledge changed", "pengetahuanmu berubah", "pengetahuan kamu berubah",
                "perubahan pengetahuan", "perkembangan pengetahuanmu", "compare your knowledge",
                "evolusi pengetahuan", "model evolution",
            )
        ):
            return "KNOWLEDGE_COMPARISON_QUERY", self._detect_temporal_scope(ql)

        # 4. Self-knowledge / GNNM internalization check ("apakah kamu sudah mempelajari X ke dalam model GNNM")
        if any(
            p in ql
            for p in (
                "ke dalam model gnnm", "ke dalam bobot", "into gnnm", "into model weights",
                "internalized", "sudah dilatih", "already trained", "apakah kamu sudah mempelajari",
                "have you internalized", "have you trained", "tersimpan atau dilatih",
            )
        ):
            return "SELF_KNOWLEDGE_QUERY", "all"

        # 5. Learning history queries ("riwayat pembelajaran", "learning history", "log pembelajaran")
        if any(
            p in ql
            for p in (
                "riwayat pembelajaran", "riwayat belajar", "learning history", "history of learning",
                "daftar pembelajaran", "catatan pembelajaran", "learning events",
            )
        ):
            return "LEARNING_HISTORY_QUERY", self._detect_temporal_scope(ql)

        # 6. Recent knowledge / today queries ("apa yang kamu pelajari hari ini", "what did you learn today")
        if any(
            p in ql
            for p in (
                "pelajari hari ini", "belajar hari ini", "pelajari minggu ini", "pelajari 7 hari",
                "learn today", "learned today", "learned this week", "learned in the last",
                "pengetahuan terbaru", "recent knowledge", "baru masuk", "baru di-ingest",
            )
        ):
            return "RECENT_KNOWLEDGE_QUERY", self._detect_temporal_scope(ql)

        # 7. Meta-learning queries ("apa yang telah kamu pelajari baru-baru ini", "what have you learned recently")
        if any(
            p in ql
            for p in (
                "telah kamu pelajari", "baru kamu pelajari", "kamu pelajari baru-baru ini",
                "yang kamu pelajari", "apa yang sudah kamu pelajari", "sudah kamu pelajari",
                "what have you learned", "what did you learn", "recently learned",
            )
        ):
            return "META_LEARNING_QUERY", self._detect_temporal_scope(ql)

        # 8. Broad knowledge-domain self-inventory (ONLY when explicitly asking about own domains/inventory,
        #    NOT ordinary factual queries like "Apa yang kamu ketahui tentang Era Orde Baru?")
        if any(
            p in ql
            for p in (
                "domain pengetahuan apa saja", "topik apa saja yang kamu miliki",
                "what knowledge domains", "list your knowledge domains", "daftar domain pengetahuan",
            )
        ):
            return "KNOWLEDGE_DOMAIN_QUERY", "all"

        return None, None

    def classify(self, perceived: PerceivedInput) -> ClassifiedIntent:
        ql = perceived.normalized_text.lower()
        domains = []

        if perceived.has_math_expression:
            domains.append("mathematics")
        if perceived.has_logical_implication:
            domains.append("symbolic_logic")
        if any(k in ql for k in ("gajah mada", "majapahit", "nusantara", "indonesia", "sriwijaya", "pancasila", "presiden", "ikn")):
            domains.append("nusantara_history_and_state")
        if any(k in ql for k in ("blockchain", "ed25519", "p2p", "arsitektur", "architecture", "gip", "memory", "node", "gnnm")):
            domains.append("sovereign_architecture")

        meta_intent, temporal_scope = self._detect_meta_cognitive_intent(ql)
        if meta_intent is not None:
            if "meta_cognition" not in domains:
                domains.insert(0, "meta_cognition")
            return ClassifiedIntent(
                primary_intent=meta_intent,
                requires_symbolic=False,
                requires_memory=True,
                requires_neural=False,
                requires_agent_execution=False,
                reasoning_domains=domains,
                is_meta_cognitive=True,
                meta_cognitive_intent=meta_intent,
                temporal_scope=temporal_scope,
            )

        if not domains:
            domains.append("general_knowledge")

        is_task = any(w in ql for w in ("investigate", "investigasi", "analyze", "analisis", "plan", "rencanakan", "execute", "jalankan tugas"))
        is_diag = any(w in ql for w in ("status", "health", "diagnose", "diagnostik", "verify", "audit"))

        if perceived.has_math_expression and not perceived.has_logical_implication and len(perceived.keywords) <= 6:
            primary = "mathematical"
        elif perceived.has_logical_implication:
            primary = "symbolic_logic"
        elif is_task:
            primary = "autonomous_task"
        elif is_diag:
            primary = "diagnostic"
        else:
            primary = "factual_inquiry"

        return ClassifiedIntent(
            primary_intent=primary,
            requires_symbolic=perceived.has_math_expression or perceived.has_logical_implication or primary in ("mathematical", "symbolic_logic"),
            requires_memory=True,
            requires_neural=True,
            requires_agent_execution=is_task,
            reasoning_domains=domains,
            is_meta_cognitive=False,
            meta_cognitive_intent=None,
            temporal_scope=None,
        )
