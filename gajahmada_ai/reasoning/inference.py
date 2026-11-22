from typing import Any, Dict, Optional

from gajahmada_ai.neural.interfaces import GenerationResult
from gajahmada_ai.reasoning.evidence import EvidenceEvaluationResult
from gajahmada_ai.reasoning.symbolic import SymbolicDeductionResult


class HybridInferenceCombiner:
    """
    Fuses results from:
    1. Symbolic Reasoner (deterministic math & logical deduction)
    2. Evidence Reasoner (retrieved knowledge records with provenance & contradiction tracking)
    3. Neural Inference Backend (when a real local model is loaded and active)
    """

    def fuse(
        self,
        objective: str,
        symbolic_res: SymbolicDeductionResult,
        evidence_res: EvidenceEvaluationResult,
        neural_res: Optional[GenerationResult],
        fallback_synthesizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        verified_facts = []
        unverified_claims = []
        inferences = []
        unknowns = []

        if symbolic_res.applicable and symbolic_res.conclusion:
            verified_facts.append(symbolic_res.conclusion)
            for d in symbolic_res.derived_facts:
                inferences.append(f"Symbolic deduction: {d}")

        for ev in evidence_res.evidence_items:
            snippet = ev.content[:280] + ("..." if len(ev.content) > 280 else "")
            if ev.verification_status == "VERIFIED":
                verified_facts.append(f"[{ev.evidence_id} | src:{ev.source}] {snippet}")
            else:
                unverified_claims.append(f"[{ev.evidence_id} | unverified] {snippet}")

        if neural_res and neural_res.neural_used and neural_res.text:
            # Rule: Never treat LLM-generated statements as verified facts automatically!
            if evidence_res.verification_status == "VERIFIED":
                inferences.append(f"Neural synthesis grounded on verified evidence ({neural_res.model_id}): {neural_res.text[:240]}")
            else:
                unverified_claims.append(f"Unverified neural generation ({neural_res.model_id}): {neural_res.text[:240]}")

        if not verified_facts and not symbolic_res.applicable:
            unknowns.append(f"No verified primary evidence or symbolic proof found for: '{objective[:100]}'")

        # Construct primary conclusion
        parts = []
        if symbolic_res.applicable and symbolic_res.conclusion:
            parts.append(symbolic_res.conclusion)

        deterministic_done = symbolic_res.applicable and symbolic_res.math_value is not None
        if deterministic_done:
            pass  # exact symbolic result is the complete answer; no neural elaboration needed
        elif neural_res and neural_res.neural_used and neural_res.text:
            parts.append(neural_res.text)
        elif fallback_synthesizer is not None:
            context_text = "\n".join(e.content for e in evidence_res.evidence_items[:3])
            parts.append(fallback_synthesizer(objective, context=context_text))
        elif evidence_res.evidence_items:
            parts.append(evidence_res.evidence_items[0].content)
        elif not parts:
            parts.append(
                f"Insufficient verified evidence in local memory to establish a verified conclusion for '{objective}'."
            )

        if evidence_res.contradictions:
            c_notes = "; ".join(
                f"Conflict between {c.record_a_id} and {c.record_b_id}: {c.reason}"
                for c in evidence_res.contradictions
            )
            parts.append(f"\n[Contradiction Alert] {c_notes}")

        # Calibrate combined confidence
        if symbolic_res.applicable and not evidence_res.contradictions:
            confidence = max(0.95, evidence_res.aggregate_confidence)
            ver_status = "VERIFIED"
        else:
            confidence = evidence_res.aggregate_confidence
            ver_status = evidence_res.verification_status

        return {
            "conclusion": "\n\n".join(parts),
            "confidence": round(confidence, 4),
            "uncertainty": round(1.0 - confidence, 4),
            "verification_status": ver_status,
            "verified_facts": verified_facts,
            "unverified_claims": unverified_claims,
            "inferences": inferences,
            "unknown_information": unknowns,
        }
