from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HypothesisItem(BaseModel):
    hypothesis_id: str
    statement: str
    status: Literal["SUPPORTED", "REFUTED", "UNVERIFIED", "CONTESTED"] = "UNVERIFIED"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence_ids: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    source: str
    provenance_hash: str
    excerpt: str
    reliability_score: float = Field(default=0.75, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.75, ge=0.0, le=1.0)
    verification_status: str = "VERIFIED"


class ContradictionItem(BaseModel):
    contradiction_id: str
    record_a_id: str
    record_b_id: str
    subject: str
    claim_a: str
    claim_b: str
    reason: str
    severity: str = "HIGH"


class ReasoningStepSummary(BaseModel):
    step_index: int
    stage: str
    summary: str
    outcome: str


class EpistemicBreakdown(BaseModel):
    """
    Explicit epistemic distinction required by GajahMada 2.0:
    Never conflates verified facts with unverified claims, hypotheses, inferences,
    opinions, or unknown information.
    """
    verified_facts: List[str] = Field(default_factory=list)
    unverified_claims: List[str] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    inferences: List[str] = Field(default_factory=list)
    opinions: List[str] = Field(default_factory=list)
    unknown_information: List[str] = Field(default_factory=list)


class ReasoningResult(BaseModel):
    reasoning_id: str
    objective: str
    interpretation: str
    assumptions: List[str] = Field(default_factory=list)
    hypotheses: List[HypothesisItem] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    reasoning_steps: List[ReasoningStepSummary] = Field(default_factory=list)
    epistemic_breakdown: EpistemicBreakdown = Field(default_factory=EpistemicBreakdown)
    conclusion: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_status: Literal["VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED", "CONTRADICTED"] = "UNVERIFIED"
    memory_references: List[str] = Field(default_factory=list)
    execution_requirements: List[str] = Field(default_factory=list)
    capabilities_used: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0
