"""
Question–answer relevance gate (local, deterministic, no external AI dependency).

Problem it solves: topic keyword rules / retrieval can return a passage that is *about the same
topic* but does not answer what was actually asked (e.g. "ibu kota Indonesia" -> a history
summary, "presiden pertama" -> the current president). This module:

1. Extracts the *focus terms* of a question (content words + qualifiers such as ordinals,
   "saat ini", years), ignoring question words and stopwords.
2. Scores how well a candidate answer covers those focus terms (`relevance_score`).
3. Extracts the best-matching sentence/line window from candidate passages
   (`select_best_passage`) so the system can answer from the part of the evidence that
   actually addresses the question.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

_STOPWORDS = {
    # Indonesian
    "apa", "apakah", "siapa", "siapakah", "kapan", "dimana", "di", "mana", "kemana", "dari", "mengapa",
    "kenapa", "bagaimana", "berapa", "yang", "itu", "ini", "adalah", "ialah", "merupakan", "dan", "atau",
    "dengan", "untuk", "pada", "dalam", "ke", "oleh", "saya", "aku", "kamu", "anda", "tolong", "mohon",
    "jelaskan", "sebutkan", "ceritakan", "terangkan", "tentang", "mengenai", "soal", "cara", "kerja",
    "bisa", "dapat", "akan", "sudah", "telah", "ada", "para", "sebuah", "suatu", "itu", "dong", "ya",
    "kah", "lah", "nya", "sih", "gimana", "tahu", "tau", "arti", "makna", "definisi", "pengertian",
    # English
    "what", "who", "whom", "when", "where", "why", "how", "which", "is", "are", "was", "were", "the", "a",
    "an", "of", "in", "on", "to", "for", "and", "or", "with", "about", "tell", "me", "explain", "please",
    "do", "does", "did", "can", "could", "you", "your", "it", "its", "be", "by", "at",
}

_ORDINALS = {
    "pertama": 1, "kedua": 2, "ketiga": 3, "keempat": 4, "kelima": 5, "keenam": 6, "ketujuh": 7,
    "kedelapan": 8, "kesembilan": 9, "kesepuluh": 10, "first": 1, "second": 2, "third": 3,
}
_CURRENT_WORDS = {"sekarang", "terkini", "terbaru", "kini", "current", "now", "latest"}
_CURRENT_ANSWER_HINTS = ("saat ini", "sekarang", "terkini", "kini", "menjabat sejak", "current", "now")
_COMPOUND_TERMS = ("ibu kota", "kota besar", "mata uang", "lagu kebangsaan", "bahasa resmi")

_TOKEN_RE = re.compile(r"[a-z0-9\u00c0-\u024f]+(?:-[a-z0-9]+)*", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _strip_markup(text: str) -> str:
    text = re.sub(r"---\s*\n?📚.*", "", text or "", flags=re.DOTALL)  # drop reference footer
    return re.sub(r"[*_`#>]", "", text)


class FocusTerm:
    __slots__ = ("term", "weight", "kind", "ordinal")

    def __init__(self, term: str, weight: float = 1.0, kind: str = "word", ordinal: Optional[int] = None):
        self.term, self.weight, self.kind, self.ordinal = term, weight, kind, ordinal

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FocusTerm({self.term!r}, w={self.weight}, {self.kind})"


def extract_focus_terms(question: str) -> List[FocusTerm]:
    q = (question or "").lower()
    terms: List[FocusTerm] = []
    consumed: set = set()

    for comp in _COMPOUND_TERMS:
        if comp in q:
            terms.append(FocusTerm(comp, 2.0, "compound"))
            consumed.update(comp.split())

    for m in re.finditer(r"\bke[-\s]?(\d{1,2})\b", q):
        terms.append(FocusTerm(m.group(0), 3.0, "ordinal", int(m.group(1))))
        consumed.add(m.group(0))

    for tok in _tokens(q):
        if tok in consumed or tok in _STOPWORDS or len(tok) < 2:
            continue
        if tok in _ORDINALS:
            terms.append(FocusTerm(tok, 3.0, "ordinal", _ORDINALS[tok]))
        elif tok in _CURRENT_WORDS:
            terms.append(FocusTerm(tok, 3.0, "current"))
        elif re.fullmatch(r"\d{3,4}", tok):
            terms.append(FocusTerm(tok, 2.0, "year"))
        elif re.fullmatch(r"\d+", tok):
            continue
        else:
            terms.append(FocusTerm(tok, 1.0, "word"))

    if re.search(r"\bsaat ini\b", q):
        terms.append(FocusTerm("saat ini", 3.0, "current"))
    # Answer-type expectations: "kapan/when" needs a date/year, "berapa/how many" needs a number.
    if re.search(r"\b(kapan|when)\b|tahun berapa|tanggal berapa", q):
        terms.append(FocusTerm("<date>", 2.0, "date"))
    if re.search(r"\b(berapa|how many|how much)\b", q) and not re.search(r"tahun berapa|tanggal berapa|jam berapa", q):
        terms.append(FocusTerm("<number>", 2.0, "number"))
    return terms


def _term_present(term: FocusTerm, text_lower: str, text_tokens: Sequence[str]) -> bool:
    if term.kind == "ordinal":
        n = term.ordinal
        if term.term in text_lower:
            return True
        return bool(
            re.search(rf"(?:^|\n)\s*[•\-*]?\s*{n}[.)]\s", text_lower)
            or re.search(rf"\bke[-\s]?{n}\b", text_lower)
            or any(w for w, v in _ORDINALS.items() if v == n and w in text_lower)
        )
    if term.kind == "date":
        return bool(re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text_lower)) or any(
            m in text_lower for m in ("januari", "februari", "maret", "april", "mei ", "juni", "juli", "agustus", "september", "oktober", "november", "desember")
        )
    if term.kind == "number":
        return bool(re.search(r"\d", text_lower))
    if term.kind == "current":
        return term.term in text_lower or any(h in text_lower for h in _CURRENT_ANSWER_HINTS)
    if term.term in text_lower:
        return True
    # partial/compound match, e.g. question "blockchain" vs answer token "chain"
    return any(len(t) >= 5 and (t in term.term or term.term in t) and min(len(t), len(term.term)) >= 5 for t in text_tokens)


def relevance_score(question: str, answer: str, context_title: str = "") -> float:
    """Weighted coverage (0..1) of the question's focus terms by `answer`."""
    terms = extract_focus_terms(question)
    if not terms:
        return 1.0  # nothing discriminative to check (e.g. very short / purely conversational)
    body = f"{context_title}\n{answer}"
    text = _strip_markup(body).lower()
    toks = _tokens(text)
    total = sum(t.weight for t in terms)
    hit = sum(t.weight for t in terms if _term_present(t, text, toks))
    return hit / total if total else 1.0


def _split_units(passage: str) -> Tuple[str, List[str]]:
    clean = _strip_markup(passage).strip()
    lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
    # Only a short header line of a multi-line passage may lend context to the lines under it;
    # a single long paragraph must never credit unrelated sentences with its own first sentence.
    title = lines[0] if len(lines) > 1 and len(lines[0]) <= 120 else ""
    units: List[str] = []
    for ln in lines:
        # "Heading: body" -> score the body only; a heading must not make unrelated body text look relevant.
        m = re.match(r"^([^:]{3,100}):\s+(.{40,})$", ln)
        if m:
            ln = m.group(2)
        if len(ln) > 260:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if s.strip())
        else:
            units.append(ln)
    return title, units


def select_best_passage(
    question: str,
    passages: Iterable[str],
    min_score: float = 0.6,
) -> Optional[Tuple[str, float]]:
    """Return (best text window, score) from `passages` that addresses the question, else None."""
    best: Optional[Tuple[str, float]] = None
    for passage in passages:
        if not passage or not passage.strip():
            continue
        title, units = _split_units(passage)
        for i, unit in enumerate(units):
            window = unit
            score = relevance_score(question, unit, context_title=title if title != unit else "")
            if i + 1 < len(units):  # try 2-unit window for answers spanning two sentences
                w2 = f"{unit} {units[i + 1]}"
                s2 = relevance_score(question, w2, context_title=title if title != unit else "")
                if s2 > score + 0.15:
                    window, score = w2, s2
            if score < min_score:
                continue
            shown = window if window == title or not title.endswith(":") else f"{title} {window}"
            shown = _clip(shown.strip())
            better = (
                best is None
                or round(score, 2) > round(best[1], 2)
                or (round(score, 2) == round(best[1], 2) and len(shown) < len(best[0]))
            )
            if better:
                best = (shown, score)
    return best


def _clip(text: str, limit: int = 420) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;: ") + "…"
