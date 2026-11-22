from typing import Any, Dict, List


class RegressionDatasetManager:
    """
    Curated Verification & Regression Benchmark Dataset Manager.
    Used to test every proposed learning adaptation or model experiment before activation.
    """

    def __init__(self):
        self.cases: List[Dict[str, Any]] = [
            {
                "case_id": "reg-math-01",
                "prompt": "Berapa hasil 144 / 12 + 88?",
                "expected_substring": "100",
                "category": "symbolic_math",
            },
            {
                "case_id": "reg-logic-01",
                "prompt": "All sovereign_nodes are cryptographic_entities. All cryptographic_entities are verifiable_systems. gajahmada is a sovereign_nodes.",
                "expected_substring": "verifiable_systems",
                "category": "symbolic_logic",
            },
            {
                "case_id": "reg-hist-01",
                "prompt": "Siapa yang mengucapkan Sumpah Palapa pada tahun 1336 M?",
                "expected_substring": "Gajah Mada",
                "category": "historical_evidence",
            },
        ]

    def get_regression_suite(self) -> List[Dict[str, Any]]:
        return list(self.cases)
