"""
GajahMada AI — Meta-Cognitive Query Orchestrator & Temporal Response Synthesizer.

Handles Meta-Cognitive intents (`META_LEARNING_QUERY`, `RECENT_KNOWLEDGE_QUERY`,
`LEARNING_HISTORY_QUERY`, `SOURCE_SPECIFIC_LEARNING_QUERY`, `KNOWLEDGE_DOMAIN_QUERY`,
`SELF_KNOWLEDGE_QUERY`, `KNOWLEDGE_GAP_QUERY`, `KNOWLEDGE_COMPARISON_QUERY`) using
temporal learning-event retrieval and live system-state inspection instead of semantic
document RAG.

Never fabricates learning events or timestamps, and never claims GNNM weight
internalization unless an actual `TRAINING_COMPLETED` / `MODEL_PROMOTED` event
demonstrates it.
"""
from datetime import datetime, timedelta, timezone
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from gajahmada_ai.learning.learning_events import (
    KnowledgeStateType,
    LearningEvent,
    LearningEventTracker,
    LearningEventType,
)


WIB_TZ = timezone(timedelta(hours=7))


class MetaCognitiveOrchestrator:
    """Orchestrates temporal retrieval, knowledge-state inspection, and honest synthesis."""

    def __init__(
        self,
        event_tracker: Optional[LearningEventTracker] = None,
        model_registry: Optional[Any] = None,
        source_registry: Optional[Any] = None,
    ):
        self.event_tracker = event_tracker
        self.model_registry = model_registry
        self.source_registry = source_registry

    def _resolve_time_window(self, query: str, scope: Optional[str]) -> Tuple[Optional[float], Optional[float], str]:
        now = time.time()
        # Check explicit YYYY-MM-DD date(s)
        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        if dates:
            try:
                start_dt = datetime.strptime(dates[0], "%Y-%m-%d").replace(tzinfo=WIB_TZ)
                end_dt = (
                    datetime.strptime(dates[1], "%Y-%m-%d").replace(tzinfo=WIB_TZ) + timedelta(days=1)
                    if len(dates) >= 2
                    else start_dt + timedelta(days=1)
                )
                return start_dt.timestamp(), end_dt.timestamp(), f"{dates[0]}..{dates[-1]}"
            except ValueError:
                pass

        if scope == "today":
            today_start = datetime.now(WIB_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
            return today_start.timestamp(), now, "hari ini (today)"
        if scope == "last_7_days":
            return now - (7 * 86400), now, "7 hari terakhir (last 7 days)"
        if scope == "last_30_days":
            return now - (30 * 86400), now, "30 hari terakhir (last 30 days)"
        if scope == "most_recent":
            return None, now, "kejadian pembelajaran terbaru (most recent)"
        if scope == "recent":
            return now - (30 * 86400), now, "baru-baru ini (30 hari terakhir)"
        return None, None, "seluruh riwayat tercatat"

    @staticmethod
    def _extract_topic_focus(query: str) -> Optional[str]:
        ql = query.lower()
        for pat in (
            r"informasi\s+([a-z0-9\-\s]{3,32}?)(?:\s+ke\s+dalam|\?|$)",
            r"tentang\s+([a-z0-9\-\s]{3,32}?)(?:\?|$)",
            r"about\s+([a-z0-9\-\s]{3,32}?)(?:\?|$)",
        ):
            m = re.search(pat, ql)
            if m:
                cand = m.group(1).strip()
                if cand not in ("topik ini", "this topic", "indonesia"):
                    return cand.split()[0]
                return cand
        return None

    @staticmethod
    def _is_english(query: str) -> bool:
        ql = query.lower()
        en_markers = ("what have you", "what did you", "how has your", "don't you know", "last document", "have you")
        return any(m in ql for m in en_markers)

    def handle(
        self,
        query: str,
        meta_intent: str,
        temporal_scope: Optional[str] = None,
        access_context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        is_en = self._is_english(query)
        since_ts, until_ts, window_label = self._resolve_time_window(query, temporal_scope)

        if self.event_tracker is None:
            msg = (
                "No learning event tracker is currently configured, so I have no verified recent learning history to report."
                if is_en
                else "Belum ada pelacak riwayat pembelajaran yang aktif, sehingga tidak ditemukan catatan pembelajaran baru-baru ini."
            )
            return {
                "conclusion": msg,
                "events": [],
                "knowledge_states": [],
                "confidence": 0.90,
                "verification_status": "VERIFIED",
                "window_label": window_label,
            }

        # Determine event-type & topic filters by meta-cognitive intent
        ev_types: Optional[List[str]] = None
        limit = 15
        topic_kw: Optional[str] = None

        if meta_intent == "SOURCE_SPECIFIC_LEARNING_QUERY":
            ev_types = [LearningEventType.DOCUMENT_INGESTED.value, LearningEventType.KNOWLEDGE_STORED.value]
            limit = 1
        elif meta_intent == "SELF_KNOWLEDGE_QUERY":
            topic_kw = self._extract_topic_focus(query)
        elif meta_intent == "KNOWLEDGE_GAP_QUERY":
            topic_kw = self._extract_topic_focus(query)
            if topic_kw in ("topik ini", "this topic"):
                topic_kw = None

        events = self.event_tracker.query_temporal(
            access_context=access_context,
            since_timestamp=since_ts,
            until_timestamp=until_ts,
            event_types=ev_types,
            topic_keyword=topic_kw,
            limit=limit,
        )

        # Also inspect actual GNNM training / promotion events across history
        training_events = self.event_tracker.query_temporal(
            access_context=access_context,
            event_types=[
                LearningEventType.TRAINING_COMPLETED.value,
                LearningEventType.MODEL_PROMOTED.value,
            ],
            topic_keyword=topic_kw,
            limit=10,
        )

        # Classify the 4 mandatory knowledge states strictly from observed events
        has_stored = any(
            e.event_type in (LearningEventType.DOCUMENT_INGESTED.value, LearningEventType.KNOWLEDGE_STORED.value)
            for e in events
        )
        has_retrieved = any(e.event_type == LearningEventType.KNOWLEDGE_RETRIEVED.value for e in events)
        has_verified = any(
            e.verification_status in ("VERIFIED", "SOURCE_CORROBORATED", "INDEPENDENTLY_VERIFIED")
            or e.event_type == LearningEventType.KNOWLEDGE_VERIFIED.value
            for e in events
        )
        has_internalized = any(
            e.internalization_status == "INTERNALIZED_IN_GNNM"
            and e.event_type in (LearningEventType.TRAINING_COMPLETED.value, LearningEventType.MODEL_PROMOTED.value)
            for e in (events + training_events)
        )

        active_states: List[str] = []
        if has_retrieved:
            active_states.append(KnowledgeStateType.RETRIEVED_KNOWLEDGE.value)
        if has_stored:
            active_states.append(KnowledgeStateType.STORED_KNOWLEDGE.value)
        if has_verified:
            active_states.append(KnowledgeStateType.VERIFIED_KNOWLEDGE.value)
        if has_internalized:
            active_states.append(KnowledgeStateType.INTERNALIZED_KNOWLEDGE.value)

        # Handle Knowledge Gap query explicitly
        if meta_intent == "KNOWLEDGE_GAP_QUERY":
            unverified_cnt = sum(1 for e in events if e.verification_status in ("UNVERIFIED", "DISPUTED"))
            msg = (
                f"Berdasarkan inspeksi status pengetahuan saya ({window_label}):\n"
                f"- Kejadian pembelajaran terotorisasi yang tercatat: {len(events)}.\n"
                f"- Pengetahuan belum terverifikasi / dipersengketakan: {unverified_cnt}.\n"
                f"- Internalisasi ke bobot model GNNM: {'sudah ada pelatihan terverifikasi' if has_internalized else 'belum didemonstrasikan (belum ada pelatihan GNNM terverifikasi untuk topik ini)'}.\n"
                f"- Keterbatasan: saya tidak mengetahui informasi di luar dokumen yang telah di-ingest/diverifikasi, fakta real-time tanpa sumber, atau data privat di luar otorisasi aktor."
                if not is_en
                else (
                    f"Based on inspecting my current knowledge state ({window_label}):\n"
                    f"- Authorized learning events recorded: {len(events)}.\n"
                    f"- Unverified or disputed entries: {unverified_cnt}.\n"
                    f"- GNNM weight internalization: {'demonstrated via verified training' if has_internalized else 'not yet demonstrated'}.\n"
                    f"- Limitations: I do not possess facts outside ingested/verified sources, real-time unindexed events, or unauthorized private memory."
                )
            )
            return {
                "conclusion": msg,
                "events": [e.to_dict() for e in events],
                "knowledge_states": active_states,
                "confidence": 0.92,
                "verification_status": "VERIFIED",
                "window_label": window_label,
            }

        # No events found in requested temporal window -> NEVER fabricate history!
        if not events:
            topic_note = f" terkait '{topic_kw}'" if topic_kw else ""
            if is_en:
                msg = (
                    f"No authorized learning events{topic_note} were found for the requested window ({window_label}). "
                    "I have not fabricated any learning history, and no GNNM weight internalization is demonstrated for this query."
                )
            else:
                msg = (
                    f"Tidak ditemukan catatan kejadian pembelajaran yang terotorisasi{topic_note} pada rentang {window_label}. "
                    "Belum ada pengetahuan baru yang di-ingest atau dilatih ke bobot model GNNM pada rentang tersebut."
                )
            return {
                "conclusion": msg,
                "events": [],
                "knowledge_states": active_states,
                "confidence": 0.95,
                "verification_status": "VERIFIED",
                "window_label": window_label,
            }

        # Group events by knowledge domain
        by_domain: Dict[str, List[LearningEvent]] = {}
        for ev in events:
            by_domain.setdefault(ev.knowledge_domain or "general_knowledge", []).append(ev)

        ver_statuses = sorted({e.verification_status for e in events if e.verification_status})
        latest_ev = events[0]
        latest_ts_str = datetime.fromtimestamp(latest_ev.timestamp, tz=WIB_TZ).strftime("%Y-%m-%d %H:%M:%S WIB")

        lines: List[str] = []
        if meta_intent == "SOURCE_SPECIFIC_LEARNING_QUERY":
            if is_en:
                lines.append(
                    f"From the most recent ingested document ('{latest_ev.source_title or latest_ev.source_id}', recorded at {latest_ts_str}):"
                )
            else:
                lines.append(
                    f"Dari dokumen terakhir yang masuk ('{latest_ev.source_title or latest_ev.source_id}', tercatat {latest_ts_str}):"
                )
            lines.append(f"- Ringkasan: {latest_ev.summary or latest_ev.source_title}")
            lines.append(f"- Sumber: `{latest_ev.source_id}` ({latest_ev.source_type})")
        elif meta_intent == "SELF_KNOWLEDGE_QUERY":
            topic_lbl = topic_kw or "topik tersebut"
            if has_internalized:
                lines.append(
                    f"Ya, pengetahuan terkait '{topic_lbl}' telah melalui pelatihan GNNM dan terverifikasi masuk ke bobot model."
                )
            else:
                lines.append(
                    f"Pengetahuan terkait '{topic_lbl}' sudah tersimpan sebagai **STORED KNOWLEDGE** di knowledge base, "
                    "tetapi **BELUM** didemonstrasikan terinternalisasi ke dalam bobot model GNNM melalui pelatihan terverifikasi."
                )
        else:
            if is_en:
                lines.append(f"Recent learning & knowledge events ({window_label}, {len(events)} authorized event(s)):")
            else:
                lines.append(f"Ringkasan riwayat pembelajaran & pengetahuan ({window_label}, {len(events)} kejadian terotorisasi):")
            for dom, dom_evs in by_domain.items():
                titles = ", ".join(
                    dict.fromkeys((e.source_title or e.summary[:60] or e.source_id) for e in dom_evs[:3])
                )
                lines.append(f"- Domain `{dom}` ({len(dom_evs)} event): {titles}")

        lines.append("")
        lines.append("Status Pengetahuan (Distingsi Wajib):")
        lines.append(f"- Knowledge ingestion / storage (`STORED_KNOWLEDGE`): {'completed' if has_stored else 'none in window'}")
        lines.append(f"- Source verification (`VERIFIED_KNOWLEDGE`): {', '.join(ver_statuses) if ver_statuses else 'UNVERIFIED'}")
        lines.append(
            f"- GNNM internal training (`INTERNALIZED_KNOWLEDGE`): "
            f"{'demonstrated (verified training completed)' if has_internalized else 'not yet performed / not demonstrated'}"
        )
        lines.append(f"- Event timestamp terakhir: {latest_ts_str}")
        if not has_internalized and has_stored:
            lines.append(
                "\nKesimpulan: informasi tersebut sudah saya terima dan simpan sebagai external/stored knowledge, "
                "tetapi belum dapat dinyatakan telah terinternalisasi ke dalam bobot model GNNM."
            )

        return {
            "conclusion": "\n".join(lines),
            "events": [e.to_dict() for e in events],
            "knowledge_states": active_states,
            "confidence": 0.94,
            "verification_status": "VERIFIED",
            "window_label": window_label,
        }
