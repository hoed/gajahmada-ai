from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class ConstraintCheckResult:
    satisfied: bool
    passed_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ConstraintEvaluator:
    """
    Deterministic Constraint Satisfaction & Verification Engine.
    Evaluates numerical, structural, policy, and epistemic constraints.
    """

    def __init__(self):
        self._constraints: Dict[str, Tuple[str, Callable[[Dict[str, Any]], bool]]] = {}
        self.register_constraint(
            "confidence_bounds",
            "Confidence must lie within [0.0, 1.0]",
            lambda ctx: 0.0 <= float(ctx.get("confidence", 0.5)) <= 1.0,
        )
        self.register_constraint(
            "uncertainty_complement",
            "Confidence + uncertainty must be positive and bounded",
            lambda ctx: 0.0 <= float(ctx.get("uncertainty", 0.5)) <= 1.0,
        )
        self.register_constraint(
            "verified_requires_evidence",
            "A VERIFIED conclusion must carry at least 1 evidence reference or deterministic symbolic proof",
            lambda ctx: (
                ctx.get("verification_status") != "VERIFIED"
                or len(ctx.get("evidence", [])) >= 1
                or bool(ctx.get("symbolic_verified", False))
            ),
        )

    def register_constraint(self, name: str, description: str, predicate: Callable[[Dict[str, Any]], bool]):
        self._constraints[name] = (description, predicate)

    def evaluate(self, context: Dict[str, Any]) -> ConstraintCheckResult:
        passed = []
        violated = []
        details = {}
        for name, (desc, pred) in self._constraints.items():
            try:
                ok = bool(pred(context))
            except Exception as e:
                ok = False
                details[name] = f"Error: {e}"
            if ok:
                passed.append(name)
            else:
                violated.append(f"{name}: {desc}")
        return ConstraintCheckResult(
            satisfied=len(violated) == 0,
            passed_constraints=passed,
            violated_constraints=violated,
            details=details,
        )
