import re
from typing import List

from gajahmada_ai.cognition.intent import ClassifiedIntent
from gajahmada_ai.cognition.perception import PerceivedInput


class ProblemDecomposer:
    """
    Cognitive Stage 4: Problem Decomposition.
    Decomposes compound objectives and multi-part questions into ordered,
    independently verifiable sub-objectives.
    """

    def decompose(self, perceived: PerceivedInput, intent: ClassifiedIntent) -> List[str]:
        text = perceived.normalized_text
        # Split on explicit conjunctions or multi-sentence boundaries
        parts = re.split(r"(?:\?\s+|\.\s+|\;\s+|\s+dan\s+jelaskan\s+|\s+and\s+explain\s+)", text)
        sub_objectives = [p.strip() for p in parts if len(p.strip()) > 3]

        if not sub_objectives:
            sub_objectives = [text]

        # For autonomous or analytical intents, ensure structured verification steps
        if intent.primary_intent == "autonomous_task" and len(sub_objectives) == 1:
            base = sub_objectives[0]
            return [
                f"1. Retrieve and verify existing memory evidence for: {base}",
                f"2. Perform symbolic and contradiction analysis on findings for: {base}",
                f"3. Synthesize verified conclusion and uncertainty bounds for: {base}",
            ]
        return sub_objectives[:6]
