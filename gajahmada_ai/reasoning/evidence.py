from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List

from gajahmada_ai.reasoning.contradictions import ContradictionDetector, DetectedContradiction


@dataclass
class ScoredEvidence:
    evidence_id: str
    content: str
    source: str
    provenance_hash: str
    reliability_score: float
    relevance_score: float
    combined_weight: float
    verification_status: str = "VERIFIED"


@dataclass
class EvidenceEvaluationResult:
    evidence_items: List[ScoredEvidence] = field(default_factory=list)
    contradictions: List[DetectedContradiction] = field(default_factory=list)
    aggregate_confidence: float = 0.0
    aggregate_uncertainty: float = 1.0
    verification_status: str = "UNVERIFIED"
    summary: str = ""


class EvidenceReasoner:
    """
    Evidence & Provenance Reasoning Engine (Layer 5C).
    - Scores evidence relevance and source reliability.
    - Detects contradictions across retrieved evidence records.
    - Computes calibrated confidence and uncertainty scores.
    - Never treats unverified claims or LLM outputs as verified facts automatically.
    """

    SOURCE_RELIABILITY_WEIGHTS: Dict[str, float] = {
        "sovereign_core_corpus": 0.92,
        "cryptographic_ledger": 0.98,
        "symbolic_proof": 1.00,
        "verified_memory": 0.90,
        "collective_quorum_verified": 0.88,
        "operator_ingested": 0.75,
        "unverified_peer": 0.45,
        "llm_hypothesis": 0.35,
        "unknown": 0.50,
    }

    def __init__(self):
        self.contradiction_detector = ContradictionDetector()

    def score_relevance(self, query: str, content: str) -> float:
        q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        if not q_tokens:
            return 0.5
        c_lower = content.lower()
        hits = sum(1 for t in q_tokens if t in c_lower)
        return round(min(1.0, max(0.15, hits / len(q_tokens))), 4)

    def evaluate_evidence(
        self,
        query: str,
        raw_records: List[Dict[str, Any]],
    ) -> EvidenceEvaluationResult:
        if not raw_records:
            return EvidenceEvaluationResult(
                evidence_items=[],
                contradictions=[],
                aggregate_confidence=0.15,
                aggregate_uncertainty=0.85,
                verification_status="UNVERIFIED",
                summary="No supporting evidence records retrieved from memory.",
            )

        scored: List[ScoredEvidence] = []
        for idx, rec in enumerate(raw_records):
            content = str(rec.get("content", "")).strip()
            if not content:
                continue
            ev_id = str(rec.get("id") or f"ev-{idx+1:03d}")
            source = str(rec.get("source") or "sovereign_core_corpus")
            ver_status = str(rec.get("verification_status") or "VERIFIED")
            prov_hash = rec.get("integrity_hash") or hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]

            base_rel = float(rec.get("reliability") or self.SOURCE_RELIABILITY_WEIGHTS.get(source, 0.65))
            if ver_status == "UNVERIFIED":
                base_rel = min(base_rel, 0.55)
            relevance = self.score_relevance(query, content)
            combined = round((base_rel * 0.6) + (relevance * 0.4), 4)

            scored.append(
                ScoredEvidence(
                    evidence_id=ev_id,
                    content=content,
                    source=source,
                    provenance_hash=prov_hash,
                    reliability_score=round(base_rel, 4),
                    relevance_score=relevance,
                    combined_weight=combined,
                    verification_status=ver_status,
                )
            )

        scored.sort(key=lambda x: x.combined_weight, reverse=True)

        # Check for contradictions across retrieved evidence
        contradictions = self.contradiction_detector.scan_records(
            [{"id": e.evidence_id, "content": e.content} for e in scored]
        )

        if not scored:
            return EvidenceEvaluationResult(
                aggregate_confidence=0.15,
                aggregate_uncertainty=0.85,
                verification_status="UNVERIFIED",
                summary="No valid evidence content available.",
            )

        top_weight = scored[0].combined_weight
        corroboration_bonus = min(0.12, (len(scored) - 1) * 0.04)
        contradiction_penalty = min(0.45, len(contradictions) * 0.25)

        confidence = round(max(0.10, min(0.98, top_weight + corroboration_bonus - contradiction_penalty)), 4)
        uncertainty = round(1.0 - confidence, 4)

        if contradictions:
            status = "CONTRADICTED"
        elif any(e.verification_status == "VERIFIED" and e.combined_weight >= 0.60 for e in scored):
            status = "VERIFIED"
        elif scored:
            status = "PARTIALLY_VERIFIED"
        else:
            status = "UNVERIFIED"

        summary = (
            f"Evaluated {len(scored)} evidence record(s); top weight={top_weight:.2f}, "
            f"contradictions={len(contradictions)}, status={status}."
        )
        return EvidenceEvaluationResult(
            evidence_items=scored,
            contradictions=contradictions,
            aggregate_confidence=confidence,
            aggregate_uncertainty=uncertainty,
            verification_status=status,
            summary=summary,
        )
