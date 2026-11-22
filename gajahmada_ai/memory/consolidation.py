from typing import Any, Dict, List

from gajahmada_ai.memory.working import WorkingMemory


class MemoryConsolidator:
    """
    Consolidates high-confidence Working Memory items into persistent
    Episodic or Semantic memory tiers.
    """

    def consolidate(self, working: WorkingMemory, min_confidence: float = 0.75) -> List[Dict[str, Any]]:
        candidates = []
        for item in working.recent(limit=working.capacity):
            if float(item.get("confidence", 0.0)) >= min_confidence and not item.get("consolidated"):
                item["consolidated"] = True
                candidates.append(item)
        return candidates
