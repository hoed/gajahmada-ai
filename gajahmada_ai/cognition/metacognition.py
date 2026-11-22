from typing import Any, Dict, List

from gajahmada_ai.cognition.models import HypothesisItem
from gajahmada_ai.reasoning.constraints import ConstraintEvaluator
from gajahmada_ai.reasoning.evaluator import ReasoningQualityEvaluator
from gajahmada_ai.reasoning.evidence import EvidenceEvaluationResult
from gajahmada_ai.reasoning.symbolic import SymbolicDeductionResult


class MetacognitiveCritic:
    """
    Cognitive Stage 8 & 9: Metacognitive Critic & Sufficiency Evaluation (Layer 5D).
    - Evaluates reasoning quality and checks formal constraints.
    - Detects missing information and unsupported claims.
    - Updates hypothesis statuses (SUPPORTED / CONTESTED / UNVERIFIED).
    - Triggers additional investigation requirements when evidence is insufficient.
    """

    def __init__(self):
        self.quality_evaluator = ReasoningQualityEvaluator()
        self.constraint_evaluator = ConstraintEvaluator()

    def critique(
        self,
        hypotheses: List[HypothesisItem],
        symbolic_res: SymbolicDeductionResult,
        evidence_res: EvidenceEvaluationResult,
        fused: Dict[str, Any],
        neural_used: bool,
    ) -> Dict[str, Any]:
        ev_ids = [e.evidence_id for e in evidence_res.evidence_items if e.verification_status == "VERIFIED"]

        for h in hypotheses:
            if evidence_res.contradictions:
                h.status = "CONTESTED"
                h.confidence = fused["confidence"]
                h.supporting_evidence_ids = ev_ids
            elif symbolic_res.applicable or ev_ids:
                h.status = "SUPPORTED"
                h.confidence = fused["confidence"]
                h.supporting_evidence_ids = ev_ids or (["symbolic-proof"] if symbolic_res.applicable else [])
            else:
                h.status = "UNVERIFIED"
                h.confidence = min(0.35, fused["confidence"])

        quality = self.quality_evaluator.evaluate(
            fused=fused,
            evidence_count=len(evidence_res.evidence_items),
            contradiction_count=len(evidence_res.contradictions),
            symbolic_used=symbolic_res.applicable,
            neural_used=neural_used,
        )

        constraint_check = self.constraint_evaluator.evaluate({
            "confidence": fused["confidence"],
            "uncertainty": fused["uncertainty"],
            "verification_status": fused["verification_status"],
            "evidence": evidence_res.evidence_items,
            "symbolic_verified": symbolic_res.applicable,
        })

        # Enforce downgrade if constraint violated
        if not constraint_check.satisfied and fused["verification_status"] == "VERIFIED":
            fused["verification_status"] = "UNVERIFIED"
            fused["confidence"] = min(fused["confidence"], 0.45)
            fused["uncertainty"] = round(1.0 - fused["confidence"], 4)

        additional_investigation_needed = not quality["sufficient"]
        execution_reqs: List[str] = []
        if additional_investigation_needed:
            execution_reqs.extend(quality["recommendations"])

        return {
            "sufficient": quality["sufficient"] and constraint_check.satisfied,
            "quality_score": quality["quality_score"],
            "issues": quality["issues"] + constraint_check.violated_constraints,
            "recommendations": quality["recommendations"],
            "execution_requirements": execution_reqs,
            "constraints_satisfied": constraint_check.satisfied,
        }
