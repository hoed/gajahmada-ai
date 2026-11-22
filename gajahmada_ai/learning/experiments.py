import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from gajahmada_ai.learning.adaptation import StrategyAdapter
from gajahmada_ai.learning.datasets import RegressionDatasetManager
from gajahmada_ai.learning.evaluator import LearningExperimentEvaluator


class LearningExperimentManager:
    """
    Controlled 3-Level Learning & Experiment Governance Manager (Section 7).
    Requirements enforced:
    - Do NOT automatically modify neural model weights during normal conversations.
    - Every proposed learning change is:
      1. Recorded
      2. Tested
      3. Evaluated against regression suite
      4. Versioned
      5. Reproducible
      6. Approved according to policy
      7. Reversible (with rollback)
    - Failed learning experiments are automatically rejected without affecting the active model/strategy.
    """

    def __init__(self):
        self.datasets = RegressionDatasetManager()
        self.evaluator = LearningExperimentEvaluator(self.datasets)
        self.adapter = StrategyAdapter()
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def propose_and_evaluate_experiment(
        self,
        name: str,
        level: int,  # 1: Knowledge, 2: Procedural Strategy, 3: Local Model Weights
        proposed_changes: Dict[str, Any],
        candidate_fn: Callable[[str], str],
        min_accuracy: float = 1.0,
        policy_approved: bool = True,
    ) -> Dict[str, Any]:
        exp_id = f"exp-{uuid.uuid4().hex[:10]}"
        pre_state = dict(self.adapter.active_strategy)

        eval_result = self.evaluator.evaluate_candidate(candidate_fn, min_accuracy=min_accuracy)

        if not eval_result["passed"]:
            status = "REJECTED_REGRESSION_FAILURE"
            active_after = pre_state
        elif not policy_approved:
            status = "PENDING_POLICY_APPROVAL"
            active_after = pre_state
        else:
            status = "APPROVED_AND_APPLIED"
            active_after = self.adapter.apply_adaptation(proposed_changes, reason=f"Experiment {exp_id}: {name}")

        record = {
            "experiment_id": exp_id,
            "name": name,
            "level": level,
            "proposed_changes": proposed_changes,
            "evaluation": eval_result,
            "policy_approved": policy_approved,
            "status": status,
            "pre_strategy_version": pre_state["version"],
            "post_strategy_version": active_after["version"],
            "active_model_affected": status == "APPROVED_AND_APPLIED",
            "timestamp": round(time.time(), 4),
        }
        self.experiments[exp_id] = record
        return record

    def rollback_last(self) -> Dict[str, Any]:
        restored = self.adapter.rollback()
        return {
            "status": "ROLLED_BACK",
            "active_strategy": restored,
        }
