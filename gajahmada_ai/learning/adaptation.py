import copy
import time
from typing import Any, Dict, List


class StrategyAdapter:
    """
    Reversible Procedural & Policy Strategy Adaptation Manager with Rollback.
    Maintains a versioned stack of active procedural parameters and guarantees
    instant rollback if a regression or operator rejection occurs.
    """

    def __init__(self):
        self.current_version: int = 1
        self.active_strategy: Dict[str, Any] = {
            "version": 1,
            "retrieval_top_k": 3,
            "min_evidence_weight": 0.60,
            "contradiction_penalty": 0.25,
            "active_model_version": "1.0.0",
        }
        self._snapshots: List[Dict[str, Any]] = [copy.deepcopy(self.active_strategy)]

    def apply_adaptation(self, changes: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
        self._snapshots.append(copy.deepcopy(self.active_strategy))
        self.current_version += 1
        self.active_strategy.update(changes)
        self.active_strategy["version"] = self.current_version
        self.active_strategy["updated_at"] = round(time.time(), 4)
        self.active_strategy["reason"] = reason
        return copy.deepcopy(self.active_strategy)

    def rollback(self) -> Dict[str, Any]:
        if len(self._snapshots) > 1:
            self.active_strategy = self._snapshots.pop()
            self.current_version = int(self.active_strategy.get("version", 1))
        return copy.deepcopy(self.active_strategy)
