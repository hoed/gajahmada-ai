from typing import Any, Dict, List


class ReasoningQualityEvaluator:
    """
    Evaluates reasoning quality, evidence coverage, constraint satisfaction,
    and epistemic calibration.
    """

    def evaluate(
        self,
        fused: Dict[str, Any],
        evidence_count: int,
        contradiction_count: int,
        symbolic_used: bool,
        neural_used: bool,
    ) -> Dict[str, Any]:
        issues: List[str] = []
        recommendations: List[str] = []

        if evidence_count == 0 and not symbolic_used:
            issues.append("Conclusion lacks both retrieved memory evidence and deterministic symbolic proof.")
            recommendations.append("Acquire or ingest verified domain evidence before treating output as factual.")

        if contradiction_count > 0:
            issues.append(f"Detected {contradiction_count} unresolved contradiction(s) in retrieved evidence.")
            recommendations.append("Investigate provenance timestamps and source reliability of conflicting records.")

        if neural_used and evidence_count == 0 and not symbolic_used:
            issues.append("Neural output generated without corroborating memory evidence.")
            recommendations.append("Classify neural claims as UNVERIFIED until corroborated.")

        sufficient = (evidence_count > 0 or symbolic_used) and contradiction_count == 0
        quality_score = round(
            max(
                0.2,
                min(
                    1.0,
                    (0.45 if symbolic_used else 0.0)
                    + (min(0.45, evidence_count * 0.18))
                    + (0.10 if neural_used else 0.10)
                    - (contradiction_count * 0.25),
                ),
            ),
            4,
        )
        return {
            "sufficient": sufficient,
            "quality_score": quality_score,
            "issues": issues,
            "recommendations": recommendations,
        }
