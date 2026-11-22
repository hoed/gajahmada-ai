from typing import Any, Dict

from gajahmada_ai.agents.base import AgentCapability, AgentTask, BaseAgent
from gajahmada_ai.execution.sandbox import SandboxedExecutor


class VerifierAgent(BaseAgent):
    """
    Autonomous Independent Verifier Agent.
    Responsibilities:
    - Evaluates evidence provenance and reliability weights.
    - Runs contradiction detection across all task findings.
    - Determines whether evidence is sufficient or requires plan revision / self-correction.
    """

    def __init__(self, sandbox: SandboxedExecutor):
        super().__init__(
            AgentCapability(
                agent_id="agent.verifier",
                role="Epistemic & Provenance Verifier",
                allowed_tools={"evidence_verify", "contradiction_check"},
                permissions={"reasoning:verify", "audit:inspect"},
                description="Audits provenance, detects contradictions, and verifies epistemic sufficiency.",
            )
        )
        self.sandbox = sandbox

    def execute_step(self, task: AgentTask, context: Dict[str, Any]) -> Dict[str, Any]:
        res = self.sandbox.execute_tool(
            tool_name="evidence_verify",
            arguments={"query": task.objective, "records": task.findings},
            agent_id=self.capability.agent_id,
            human_approved=task.human_approved,
        )
        ver_data = res.get("result", {}) if res.get("status") == "COMPLETED" else {}
        symbolic_ok = bool(task.analysis.get("symbolic_applicable"))

        confidence = float(ver_data.get("aggregate_confidence", 0.15))
        status = ver_data.get("verification_status", "UNVERIFIED")
        contradictions = ver_data.get("contradictions", [])

        if symbolic_ok and not contradictions:
            confidence = max(0.95, confidence)
            status = "VERIFIED"

        task.confidence = round(confidence, 4)
        task.verification = {
            "verification_status": status,
            "confidence": task.confidence,
            "uncertainty": round(1.0 - task.confidence, 4),
            "contradictions_count": len(contradictions),
            "contradictions": contradictions,
            "requires_self_correction": (len(task.findings) == 0 and not symbolic_ok),
        }

        for sub in task.subtasks:
            if sub.get("assigned_agent") == self.capability.agent_id:
                sub["status"] = "COMPLETED"

        return {
            "agent": self.capability.agent_id,
            "verification": task.verification,
        }
