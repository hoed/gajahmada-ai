"""
GajahMada AI — Training Knowledge Filter (Knowledge-to-Model Safety Gate).

Strictly filters candidate records before they can enter Continued Pretraining or SFT:
- Rejects UNVERIFIED, MODEL_ASSESSED, DISPUTED, CONTRADICTED, and REJECTED records.
- Rejects prompt-injection & malicious payload patterns.
- Rejects records lacking valid usage permission / license or explicit training approval.
- Rejects low-quality OCR / garbled text and duplicate content.
- Rejects untrusted agent/peer outputs.
Disputed records remain preserved for external retrieval with labels — never deleted.
"""
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from gajahmada_ai.knowledge.interfaces import KnowledgeRecord, KnowledgeSource


APPROVED_VERIFICATION_STATUSES = {
    "VERIFIED",
    "SOURCE_CORROBORATED",
    "INDEPENDENTLY_VERIFIED",
}

FORBIDDEN_LICENSES = {
    "proprietary-no-train",
    "restricted",
    "confidential",
    "all-rights-reserved-no-ai",
    "unspecified",
    "",
}

UNTRUSTED_SOURCE_TYPES = {
    "unverified_peer",
    "untrusted_agent",
    "raw_web_crawl",
    "anonymous_input",
}

_MALICIOUS_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"),
    re.compile(r"(?i)system\s*:\s*override"),
    re.compile(r"(?i)export_private_key|rm\s+-rf\s+/|drop\s+table"),
    re.compile(r"(?i)<\s*script\b"),
]


class TrainingKnowledgeFilter:
    """Deterministic epistemic, license, OCR-quality, and security filter for GNNM training."""

    def __init__(
        self,
        require_verification: bool = True,
        require_license: bool = True,
        min_chars: int = 24,
        max_non_alnum_ratio: float = 0.40,
    ):
        self.require_verification = require_verification
        self.require_license = require_license
        self.min_chars = min_chars
        self.max_non_alnum_ratio = max_non_alnum_ratio

    def _is_low_quality_ocr(self, text: str) -> bool:
        if len(text) < self.min_chars:
            return True
        non_alnum = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()-'\"/%"))
        if (non_alnum / max(1, len(text))) > self.max_non_alnum_ratio:
            return True
        # Excessive repeated single-character artifacts (common OCR corruption)
        if re.search(r"(.)\1{9,}", text):
            return True
        return False

    def evaluate_record(
        self,
        record: KnowledgeRecord,
        source: Optional[KnowledgeSource] = None,
    ) -> Tuple[bool, str]:
        text = (record.content or "").strip()
        if not text:
            return False, "empty_content"

        # 1. Epistemic verification check
        ver = str(record.verification_status or "").upper().strip()
        if self.require_verification and ver not in APPROVED_VERIFICATION_STATUSES:
            return False, f"unapproved_verification_status:{ver}"

        # 2. Untrusted source type check
        if str(record.source_type or "").lower().strip() in UNTRUSTED_SOURCE_TYPES:
            return False, f"untrusted_source_type:{record.source_type}"

        # 3. License & usage permission check
        license_val = (
            source.license if source is not None else str(record.provenance.get("license", ""))
        ).lower().strip()
        if self.require_license and license_val in FORBIDDEN_LICENSES:
            return False, f"missing_or_restricted_license:{license_val or 'none'}"
        if source is not None and source.usage_restrictions and "no_training" in source.usage_restrictions.lower():
            return False, "usage_restriction_no_training"

        # 4. Prompt-injection & malicious content check
        for pat in _MALICIOUS_PATTERNS:
            if pat.search(text):
                return False, "prompt_injection_or_malicious_content"

        # 5. Low-quality OCR check
        if self._is_low_quality_ocr(text):
            return False, "low_quality_or_corrupted_ocr"

        return True, "APPROVED"

    def filter_batch(
        self,
        records: List[KnowledgeRecord],
        sources_by_id: Optional[Dict[str, KnowledgeSource]] = None,
    ) -> Dict[str, Any]:
        sources_map = sources_by_id or {}
        approved: List[KnowledgeRecord] = []
        rejected: List[Dict[str, str]] = []
        seen_hashes = set()

        for rec in records:
            c_hash = hashlib.sha256(rec.content.strip().lower().encode("utf-8")).hexdigest()
            if c_hash in seen_hashes:
                rejected.append({"record_id": rec.record_id, "reason": "duplicate_content"})
                continue

            src = sources_map.get(rec.source_id)
            ok, reason = self.evaluate_record(rec, source=src)
            if not ok:
                rejected.append({"record_id": rec.record_id, "reason": reason})
                continue

            seen_hashes.add(c_hash)
            approved.append(rec)

        return {
            "approved": approved,
            "rejected": rejected,
            "approved_count": len(approved),
            "rejected_count": len(rejected),
        }
