from typing import List, Tuple

from gajahmada_ai.cognition.intent import ClassifiedIntent
from gajahmada_ai.cognition.models import HypothesisItem
from gajahmada_ai.cognition.perception import PerceivedInput


class HypothesisGenerator:
    """
    Cognitive Stage 5: Hypothesis & Assumption Generator.
    Generates explicit testable hypotheses and underlying assumptions before
    parallel reasoning evaluation.
    """

    def generate(
        self,
        perceived: PerceivedInput,
        intent: ClassifiedIntent,
        sub_objectives: List[str],
    ) -> Tuple[List[str], List[HypothesisItem]]:
        assumptions = [
            "Local persistent memory and cryptographic ledger state are used as primary ground truth.",
            "Any claim lacking primary provenance or deterministic symbolic proof remains UNVERIFIED.",
        ]
        if perceived.has_math_expression or perceived.has_logical_implication:
            assumptions.append("Formal mathematical and propositional logic rules hold deterministically.")

        hypotheses: List[HypothesisItem] = []
        for idx, sub in enumerate(sub_objectives, start=1):
            hypotheses.append(
                HypothesisItem(
                    hypothesis_id=f"H-{idx:02d}",
                    statement=f"Verified evidence or deterministic proof exists to resolve: '{sub[:140]}'",
                    status="UNVERIFIED",
                    confidence=0.5,
                )
            )
        return assumptions, hypotheses
