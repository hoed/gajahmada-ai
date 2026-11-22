from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class DetectedContradiction:
    contradiction_id: str
    record_a_id: str
    record_b_id: str
    subject: str
    claim_a: str
    claim_b: str
    reason: str
    severity: str = "HIGH"


class ContradictionDetector:
    """
    Deterministic Contradiction Detection Engine (Layer 5C & Layer 4).
    Detects:
    1. Polarity / Negation contradictions on shared subjects (e.g., "X is Y" vs "X is not Y" / "X bukan Y").
    2. Incompatible numeric/year claims on shared entity+predicate attributes (e.g., "founded in 1293" vs "founded in 1400").
    3. Antonym / mutually exclusive state predicates ("online" vs "offline", "valid" vs "invalid", "allowed" vs "forbidden").
    """

    _NEGATION_TOKENS = {"not", "never", "no", "tidak", "bukan", "belum", "tanpa", "false", "invalid"}
    _EXCLUSIVE_PAIRS = [
        ({"online", "active", "aktif"}, {"offline", "inactive", "nonaktif", "mati"}),
        ({"valid", "verified", "sah"}, {"invalid", "unverified", "forged", "palsu"}),
        ({"allowed", "permitted", "diizinkan"}, {"forbidden", "blocked", "dilarang", "ditolak"}),
        ({"true", "benar"}, {"false", "salah"}),
        ({"decentralized", "terdesentralisasi"}, {"centralized", "terpusat"}),
    ]
    _STOPWORDS = {
        "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and",
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada", "adalah",
    }

    def extract_subject_tokens(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return [
            t for t in tokens
            if t not in self._STOPWORDS and t not in self._NEGATION_TOKENS
        ]

    def extract_numbers(self, text: str) -> List[str]:
        return re.findall(r"\b\d+(?:\.\d+)?\b", text)

    def check_pair(
        self,
        id_a: str,
        text_a: str,
        id_b: str,
        text_b: str,
    ) -> Optional[DetectedContradiction]:
        if not text_a or not text_b or text_a.strip() == text_b.strip():
            return None

        subj_a = set(self.extract_subject_tokens(text_a))
        subj_b = set(self.extract_subject_tokens(text_b))
        shared = subj_a.intersection(subj_b)
        if len(shared) < 2:
            return None

        words_a = set(re.findall(r"[a-z0-9]+", text_a.lower()))
        words_b = set(re.findall(r"[a-z0-9]+", text_b.lower()))
        subject_label = " ".join(sorted(shared)[:4])

        # 1. Negation polarity contradiction on highly overlapping statements
        neg_a = bool(words_a.intersection(self._NEGATION_TOKENS))
        neg_b = bool(words_b.intersection(self._NEGATION_TOKENS))
        overlap_ratio = len(shared) / max(1, min(len(subj_a), len(subj_b)))
        if neg_a != neg_b and overlap_ratio >= 0.55:
            return DetectedContradiction(
                contradiction_id=f"contra-{id_a}-{id_b}",
                record_a_id=id_a,
                record_b_id=id_b,
                subject=subject_label,
                claim_a=text_a,
                claim_b=text_b,
                reason=f"Direct negation polarity conflict on shared subject ({subject_label})",
                severity="HIGH",
            )

        # 2. Mutually exclusive antonym predicates
        for pos_set, neg_set in self._EXCLUSIVE_PAIRS:
            if (words_a.intersection(pos_set) and words_b.intersection(neg_set)) or (
                words_a.intersection(neg_set) and words_b.intersection(pos_set)
            ):
                if overlap_ratio >= 0.5:
                    return DetectedContradiction(
                        contradiction_id=f"contra-{id_a}-{id_b}",
                        record_a_id=id_a,
                        record_b_id=id_b,
                        subject=subject_label,
                        claim_a=text_a,
                        claim_b=text_b,
                        reason=f"Mutually exclusive state predicate on subject ({subject_label})",
                        severity="HIGH",
                    )

        # 3. Conflicting numerical/year value on highly identical predicate context
        nums_a = self.extract_numbers(text_a)
        nums_b = self.extract_numbers(text_b)
        if len(nums_a) == 1 and len(nums_b) == 1 and nums_a[0] != nums_b[0] and overlap_ratio >= 0.65:
            return DetectedContradiction(
                contradiction_id=f"contra-{id_a}-{id_b}",
                record_a_id=id_a,
                record_b_id=id_b,
                subject=subject_label,
                claim_a=text_a,
                claim_b=text_b,
                reason=f"Conflicting numerical value ({nums_a[0]} vs {nums_b[0]}) on shared predicate ({subject_label})",
                severity="MEDIUM",
            )

        return None

    def scan_records(self, records: List[Dict[str, str]]) -> List[DetectedContradiction]:
        """Scan a list of {'id': ..., 'content': ...} dicts and return all detected contradictions."""
        contradictions: List[DetectedContradiction] = []
        n = len(records)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.check_pair(
                    str(records[i].get("id", i)),
                    str(records[i].get("content", "")),
                    str(records[j].get("id", j)),
                    str(records[j].get("content", "")),
                )
                if c is not None:
                    contradictions.append(c)
        return contradictions
