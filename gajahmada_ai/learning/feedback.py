import time
from typing import Any, Dict, List
import uuid


class FeedbackCollector:
    """
    Collects verified operator and outcome feedback for reasoning & agent executions.
    Never modifies neural model weights directly during conversation.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_feedback(
        self,
        reasoning_id: str,
        rating: float,
        verified_correct: bool,
        comment: str = "",
        actor: str = "operator",
    ) -> Dict[str, Any]:
        entry = {
            "feedback_id": f"fb-{uuid.uuid4().hex[:10]}",
            "reasoning_id": reasoning_id,
            "rating": max(0.0, min(1.0, float(rating))),
            "verified_correct": bool(verified_correct),
            "comment": comment.strip(),
            "actor": actor,
            "timestamp": round(time.time(), 4),
        }
        self.records.append(entry)
        return entry
