import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from gajahmada_ai.cognition.decision import DecisionEngine
from gajahmada_ai.cognition.decomposition import ProblemDecomposer
from gajahmada_ai.cognition.hypotheses import HypothesisGenerator
from gajahmada_ai.cognition.intent import IntentClassifier
from gajahmada_ai.cognition.metacognition import MetacognitiveCritic
from gajahmada_ai.cognition.models import (
    ContradictionItem,
    EvidenceItem,
    ReasoningResult,
    ReasoningStepSummary,
)
from gajahmada_ai.cognition.perception import PerceptionModule
from gajahmada_ai.neural.router import ModelRouter
from gajahmada_ai.neural.interfaces import GenerationResult
from gajahmada_ai.reasoning.evidence import EvidenceReasoner
from gajahmada_ai.reasoning.inference import HybridInferenceCombiner
from gajahmada_ai.reasoning.symbolic import SymbolicReasoner


class CognitiveReasoningPipeline:
    """
    GajahMada 2.0 Native 14-Stage Hybrid Cognitive Reasoning Pipeline:
    INPUT -> PERCEPTION -> INTENT -> MEMORY RETRIEVAL -> DECOMPOSITION ->
    HYPOTHESIS GENERATION -> PARALLEL REASONING (Neural + Symbolic + Evidence) ->
    EVIDENCE FUSION -> METACOGNITIVE CRITIC -> SUFFICIENCY EVALUATION ->
    DECISION -> RESPONSE -> MEMORY UPDATE -> OUTCOME EVALUATION.
    """

    def __init__(
        self,
        model_router: Optional[ModelRouter] = None,
        symbolic_reasoner: Optional[SymbolicReasoner] = None,
        evidence_reasoner: Optional[EvidenceReasoner] = None,
        fallback_synthesizer: Optional[Callable[[str, str], str]] = None,
    ):
        self.perception = PerceptionModule()
        self.intent_classifier = IntentClassifier()
        self.decomposer = ProblemDecomposer()
        self.hypothesis_generator = HypothesisGenerator()
        self.model_router = model_router or ModelRouter()
        self.symbolic_reasoner = symbolic_reasoner or SymbolicReasoner()
        self.evidence_reasoner = evidence_reasoner or EvidenceReasoner()
        self.combiner = HybridInferenceCombiner()
        self.critic = MetacognitiveCritic()
        self.decision_engine = DecisionEngine()
        self.fallback_synthesizer = fallback_synthesizer
        self.history: Dict[str, ReasoningResult] = {}

    def execute(
        self,
        objective: str,
        memory_records: Optional[List[Dict[str, Any]]] = None,
        symbolic_facts: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        use_neural: bool = True,
    ) -> ReasoningResult:
        t0 = time.time()
        reasoning_id = f"rsn-{uuid.uuid4().hex[:12]}"
        steps: List[ReasoningStepSummary] = []

        # 1. PERCEPTION
        perceived = self.perception.perceive(objective)
        steps.append(
            ReasoningStepSummary(
                step_index=1,
                stage="PERCEPTION",
                summary=f"Normalized input (lang={perceived.language}, entities={len(perceived.entities)}, math={perceived.has_math_expression}, logic={perceived.has_logical_implication})",
                outcome="COMPLETED",
            )
        )

        # 2. INTENT CLASSIFICATION
        intent = self.intent_classifier.classify(perceived)
        steps.append(
            ReasoningStepSummary(
                step_index=2,
                stage="INTENT_CLASSIFICATION",
                summary=f"Classified primary intent='{intent.primary_intent}' across domains={intent.reasoning_domains}",
                outcome="COMPLETED",
            )
        )

        # 3. MEMORY RETRIEVAL
        raw_evidence = memory_records or []
        steps.append(
            ReasoningStepSummary(
                step_index=3,
                stage="MEMORY_RETRIEVAL",
                summary=f"Retrieved {len(raw_evidence)} memory record(s) for evidence evaluation",
                outcome="COMPLETED",
            )
        )

        # 4. PROBLEM DECOMPOSITION
        sub_objectives = self.decomposer.decompose(perceived, intent)
        steps.append(
            ReasoningStepSummary(
                step_index=4,
                stage="PROBLEM_DECOMPOSITION",
                summary=f"Decomposed objective into {len(sub_objectives)} sub-objective(s)",
                outcome="COMPLETED",
            )
        )

        # 5. HYPOTHESIS GENERATION
        assumptions, hypotheses = self.hypothesis_generator.generate(perceived, intent, sub_objectives)
        steps.append(
            ReasoningStepSummary(
                step_index=5,
                stage="HYPOTHESIS_GENERATION",
                summary=f"Formulated {len(hypotheses)} testable hypothesis/hypotheses with {len(assumptions)} assumption(s)",
                outcome="COMPLETED",
            )
        )

        # 6A. SYMBOLIC REASONING
        symbolic_res = self.symbolic_reasoner.reason(perceived.normalized_text, facts=symbolic_facts)
        steps.append(
            ReasoningStepSummary(
                step_index=6,
                stage="SYMBOLIC_REASONING",
                summary=(
                    f"Symbolic engine applicable={symbolic_res.applicable} "
                    f"(math={symbolic_res.math_value}, derived_facts={len(symbolic_res.derived_facts)})"
                ),
                outcome="VERIFIED" if symbolic_res.applicable else "NOT_APPLICABLE",
            )
        )

        # 6B. EVIDENCE REASONING
        evidence_res = self.evidence_reasoner.evaluate_evidence(perceived.normalized_text, raw_evidence)
        steps.append(
            ReasoningStepSummary(
                step_index=7,
                stage="EVIDENCE_REASONING",
                summary=evidence_res.summary,
                outcome=evidence_res.verification_status,
            )
        )

        # 6C. NEURAL REASONING (Only when a local neural backend is actually loaded)
        # Skip neural generation when a deterministic/real-time specialist should answer directly
        # (exact math, live weather/temperature, exact Al-Qur'an verse, or live system clock).
        ql_norm = perceived.normalized_text.lower()
        is_exact_math = bool(symbolic_res.applicable and symbolic_res.math_value is not None)
        is_realtime_or_scripture = any(
            kw in ql_norm
            for kw in (
                "suhu", "cuaca", "temperatur", "temperature", "weather", "kelembapan",
                "quran", "qur'an", "alquran", "al-quran", "surat ", "surah ", "ayat ",
                "jam berapa", "tanggal berapa", "hari apa", "waktu sekarang", "jam sekarang", "what time",
            )
        )
        run_neural = use_neural and not is_exact_math and not is_realtime_or_scripture

        context_str = "\n".join(e.content for e in evidence_res.evidence_items[:3])
        neural_res = self.model_router.route_generation(
            prompt=perceived.normalized_text,
            context=context_str,
            task_type=intent.primary_intent,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if run_neural else GenerationResult(
            text="", backend_name="skipped", model_id="-", neural_used=False,
            tokens_generated=0, latency_ms=0.0,
            warning="Neural step skipped: handled by deterministic/live specialist resolver.",
        )
        steps.append(
            ReasoningStepSummary(
                step_index=8,
                stage="NEURAL_REASONING",
                summary=(
                    f"Neural backend '{neural_res.backend_name}' executed (neural_used={neural_res.neural_used})"
                    if neural_res.neural_used
                    else f"Neural inference inactive ({neural_res.warning}); relying on Symbolic + Evidence reasoning"
                ),
                outcome="EXECUTED" if neural_res.neural_used else "UNAVAILABLE_REPORTED",
            )
        )

        # 7. EVIDENCE FUSION
        fused = self.combiner.fuse(
            objective=perceived.normalized_text,
            symbolic_res=symbolic_res,
            evidence_res=evidence_res,
            neural_res=neural_res,
            fallback_synthesizer=self.fallback_synthesizer,
        )
        steps.append(
            ReasoningStepSummary(
                step_index=9,
                stage="EVIDENCE_FUSION",
                summary=f"Fused hybrid streams -> status={fused['verification_status']}, confidence={fused['confidence']}",
                outcome="COMPLETED",
            )
        )

        # 8. METACOGNITIVE CRITIC & SUFFICIENCY EVALUATION
        critique = self.critic.critique(
            hypotheses=hypotheses,
            symbolic_res=symbolic_res,
            evidence_res=evidence_res,
            fused=fused,
            neural_used=neural_res.neural_used,
        )
        steps.append(
            ReasoningStepSummary(
                step_index=10,
                stage="METACOGNITIVE_CRITIC",
                summary=f"Sufficiency={critique['sufficient']}, quality_score={critique['quality_score']}, issues={len(critique['issues'])}",
                outcome="SUFFICIENT" if critique["sufficient"] else "INSUFFICIENT_FLAGGED",
            )
        )

        # 9. DECISION & EPISTEMIC SEPARATION
        epistemic = self.decision_engine.build_epistemic_breakdown(perceived, hypotheses, fused)
        steps.append(
            ReasoningStepSummary(
                step_index=11,
                stage="DECISION",
                summary=(
                    f"Separated verified_facts={len(epistemic.verified_facts)}, "
                    f"unverified_claims={len(epistemic.unverified_claims)}, "
                    f"inferences={len(epistemic.inferences)}, unknowns={len(epistemic.unknown_information)}"
                ),
                outcome="COMPLETED",
            )
        )

        evidence_models = [
            EvidenceItem(
                evidence_id=e.evidence_id,
                source=e.source,
                provenance_hash=e.provenance_hash,
                excerpt=e.content[:320],
                reliability_score=e.reliability_score,
                relevance_score=e.relevance_score,
                verification_status=e.verification_status,
            )
            for e in evidence_res.evidence_items
        ]
        contradiction_models = [
            ContradictionItem(
                contradiction_id=c.contradiction_id,
                record_a_id=c.record_a_id,
                record_b_id=c.record_b_id,
                subject=c.subject,
                claim_a=c.claim_a,
                claim_b=c.claim_b,
                reason=c.reason,
                severity=c.severity,
            )
            for c in evidence_res.contradictions
        ]

        cap_report = self.model_router.capability_report()
        cap_report["neural_used_in_this_call"] = neural_res.neural_used
        cap_report["symbolic_used_in_this_call"] = symbolic_res.applicable
        cap_report["evidence_count_in_this_call"] = len(evidence_models)

        result = ReasoningResult(
            reasoning_id=reasoning_id,
            objective=perceived.normalized_text,
            interpretation=f"Intent: {intent.primary_intent} | Domains: {', '.join(intent.reasoning_domains)} | Sub-objectives: {len(sub_objectives)}",
            assumptions=assumptions,
            hypotheses=hypotheses,
            evidence=evidence_models,
            contradictions=contradiction_models,
            reasoning_steps=steps,
            epistemic_breakdown=epistemic,
            conclusion=fused["conclusion"],
            confidence=fused["confidence"],
            uncertainty=fused["uncertainty"],
            verification_status=fused["verification_status"],
            memory_references=[e.evidence_id for e in evidence_models],
            execution_requirements=critique["execution_requirements"],
            capabilities_used=cap_report,
            timestamp=round(t0, 4),
        )
        self.history[reasoning_id] = result
        return result
