from dataclasses import dataclass, field
import re
from typing import Dict, List


@dataclass
class PerceivedInput:
    raw_text: str
    normalized_text: str
    language: str
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    has_math_expression: bool = False
    has_logical_implication: bool = False
    has_opinion_markers: bool = False


class PerceptionModule:
    """
    Layer 3 & Cognitive Stage 1: Input Perception & Normalization.
    Extracts named entities, keywords, numerical expressions, logical connectives,
    and opinion markers without external API calls.
    """

    _ID_WORDS = {"apa", "siapa", "kapan", "dimana", "bagaimana", "mengapa", "jelaskan", "sejarah", "adalah", "yang", "dan", "untuk"}
    _OPINION_MARKERS = {"menurut pendapat", "opini", "sebaiknya", "in my opinion", "i think", "should be", "best", "terbaik"}
    _STOPWORDS = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
        "adalah", "sebagai", "dalam", "atau", "juga", "oleh", "karena", "tentang",
        "apa", "siapa", "kapan", "dimana", "bagaimana", "mengapa", "kenapa", "berapa",
        "jelaskan", "tolong", "buatkan", "sebutkan", "ceritakan", "kamu", "anda", "saya",
        "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and",
        "or", "what", "who", "when", "where", "how", "why", "can", "you", "me",
    }

    def perceive(self, text: str) -> PerceivedInput:
        raw = (text or "").strip()
        normalized = re.sub(r"\s+", " ", raw)
        lower = normalized.lower()
        words = re.findall(r"[a-z0-9]+", lower)

        id_hits = sum(1 for w in words if w in self._ID_WORDS)
        language = "id" if id_hits >= 1 else "en"

        # Extract capitalized named entities (multi-word proper nouns)
        raw_entities = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b", normalized)
        entities = []
        seen_ent = set()
        for ent in raw_entities:
            if ent.lower() not in self._STOPWORDS and len(ent) > 2 and ent not in seen_ent:
                seen_ent.add(ent)
                entities.append(ent)

        keywords = [w for w in words if len(w) > 2 and w not in self._STOPWORDS]
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", normalized)

        has_math = bool(
            re.search(r"(\d+\s*[\+\-\*\/\%\^×÷]\s*\d+|(?:sqrt|log|sin|cos)\s*\()", lower)
        )
        has_logic = bool(
            re.search(r"\b(if\s+.+\s+then|jika\s+.+\s+maka|all\s+\w+\s+are|semua\s+\w+\s+adalah)\b", lower)
        )
        has_opinion = any(m in lower for m in self._OPINION_MARKERS)

        return PerceivedInput(
            raw_text=raw,
            normalized_text=normalized,
            language=language,
            entities=entities,
            keywords=keywords,
            numbers=numbers,
            has_math_expression=has_math,
            has_logical_implication=has_logic,
            has_opinion_markers=has_opinion,
        )
