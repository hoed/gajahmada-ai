from typing import Any, Dict, List

from gajahmada_ai.cognition.models import EpistemicBreakdown, HypothesisItem
from gajahmada_ai.cognition.perception import PerceivedInput


class DecisionEngine:
    """
    Cognitive Stage 10: Decision & Epistemic Separation Engine.
    Assembles the final EpistemicBreakdown distinguishing:
    - Verified facts
    - Unverified claims
    - Hypotheses
    - Inferences
    - Opinions
    - Unknown information
    """

    def build_epistemic_breakdown(
        self,
        perceived: PerceivedInput,
        hypotheses: List[HypothesisItem],
        fused: Dict[str, Any],
    ) -> EpistemicBreakdown:
        opinions: List[str] = []
        if perceived.has_opinion_markers:
            opinions.append(
                f"Input contains subjective/normative opinion framing ('{perceived.normalized_text[:100]}'); "
                "normative preference is separated from empirical facts."
            )

        return EpistemicBreakdown(
            verified_facts=list(fused.get("verified_facts", [])),
            unverified_claims=list(fused.get("unverified_claims", [])),
            hypotheses=[f"{h.hypothesis_id} [{h.status}]: {h.statement}" for h in hypotheses],
            inferences=list(fused.get("inferences", [])),
            opinions=opinions,
            unknown_information=list(fused.get("unknown_information", [])),
        )
