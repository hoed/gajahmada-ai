from dataclasses import dataclass
from typing import List

from gajahmada_ai.cognition.perception import PerceivedInput


@dataclass
class ClassifiedIntent:
    primary_intent: str  # "mathematical", "symbolic_logic", "factual_inquiry", "autonomous_task", "diagnostic", "conversational"
    requires_symbolic: bool
    requires_memory: bool
    requires_neural: bool
    requires_agent_execution: bool
    reasoning_domains: List[str]


class IntentClassifier:
    """
    Cognitive Stage 2: Intent & Capability Requirement Classifier.
    """

    def classify(self, perceived: PerceivedInput) -> ClassifiedIntent:
        ql = perceived.normalized_text.lower()
        domains = []

        if perceived.has_math_expression:
            domains.append("mathematics")
        if perceived.has_logical_implication:
            domains.append("symbolic_logic")
        if any(k in ql for k in ("gajah mada", "majapahit", "nusantara", "indonesia", "sriwijaya", "pancasila", "presiden", "ikn")):
            domains.append("nusantara_history_and_state")
        if any(k in ql for k in ("blockchain", "ed25519", "p2p", "arsitektur", "architecture", "gip", "memory", "node")):
            domains.append("sovereign_architecture")

        if not domains:
            domains.append("general_knowledge")

        is_task = any(w in ql for w in ("investigate", "investigasi", "analyze", "analisis", "plan", "rencanakan", "execute", "jalankan tugas"))
        is_diag = any(w in ql for w in ("status", "health", "diagnose", "diagnostik", "verify", "audit"))

        if perceived.has_math_expression and not perceived.has_logical_implication and len(perceived.keywords) <= 6:
            primary = "mathematical"
        elif perceived.has_logical_implication:
            primary = "symbolic_logic"
        elif is_task:
            primary = "autonomous_task"
        elif is_diag:
            primary = "diagnostic"
        else:
            primary = "factual_inquiry"

        return ClassifiedIntent(
            primary_intent=primary,
            requires_symbolic=perceived.has_math_expression or perceived.has_logical_implication or primary in ("mathematical", "symbolic_logic"),
            requires_memory=True,
            requires_neural=True,
            requires_agent_execution=is_task,
            reasoning_domains=domains,
        )
