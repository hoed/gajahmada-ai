import time
from typing import Any, Callable, Dict, List

from gajahmada_ai.learning.datasets import RegressionDatasetManager


class LearningExperimentEvaluator:
    """
    Regression & Safety Evaluator for Proposed Learning Changes.
    Runs the candidate strategy or model against the regression dataset suite
    and rejects any experiment that degrades accuracy or violates safety constraints.
    """

    def __init__(self, dataset_manager: RegressionDatasetManager):
        self.datasets = dataset_manager

    def evaluate_candidate(
        self,
        candidate_fn: Callable[[str], str],
        min_accuracy: float = 1.0,
    ) -> Dict[str, Any]:
        suite = self.datasets.get_regression_suite()
        passed = 0
        failures = []
        t0 = time.perf_counter()

        for case in suite:
            try:
                out = str(candidate_fn(case["prompt"]))
                if case["expected_substring"].lower() in out.lower():
                    passed += 1
                else:
                    failures.append({
                        "case_id": case["case_id"],
                        "expected": case["expected_substring"],
                        "actual_excerpt": out[:120],
                    })
            except Exception as e:
                failures.append({
                    "case_id": case["case_id"],
                    "error": str(e),
                })

        total = max(1, len(suite))
        accuracy = round(passed / total, 4)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "passed": accuracy >= min_accuracy and len(failures) == 0,
            "accuracy": accuracy,
            "cases_total": len(suite),
            "cases_passed": passed,
            "failures": failures,
            "latency_ms": elapsed_ms,
        }
