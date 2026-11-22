from typing import Any, Dict

from gajahmada_ai.agents.base import AgentCapability, AgentTask, BaseAgent
from gajahmada_ai.execution.sandbox import SandboxedExecutor


class AnalystAgent(BaseAgent):
    """
    Autonomous Analyst Agent.
    Responsibilities:
    - Runs deterministic symbolic deduction and mathematical evaluation via sandboxed tools.
    - Synthesizes structured inferences from retrieved findings.
    """

    def __init__(self, sandbox: SandboxedExecutor):
        super().__init__(
            AgentCapability(
                agent_id="agent.analyst",
                role="Symbolic & Quantitative Analyst",
                allowed_tools={"symbolic_math", "symbolic_logic"},
                permissions={"reasoning:symbolic", "reasoning:analyze"},
                description="Applies formal logic rules, mathematical evaluation, and quantitative synthesis.",
            )
        )
        self.sandbox = sandbox

    def execute_step(self, task: AgentTask, context: Dict[str, Any]) -> Dict[str, Any]:
        res = self.sandbox.execute_tool(
            tool_name="symbolic_math",
            arguments={"query": task.objective, "facts": context.get("facts", [])},
            agent_id=self.capability.agent_id,
            human_approved=task.human_approved,
        )
        sym_data = res.get("result", {}) if res.get("status") == "COMPLETED" else {}
        task.analysis = {
            "symbolic_applicable": sym_data.get("applicable", False),
            "math_expression": sym_data.get("math_expression"),
            "math_value": sym_data.get("math_value"),
            "derived_facts": sym_data.get("derived_facts", []),
            "conclusion": sym_data.get("conclusion"),
            "evidence_count_analyzed": len(task.findings),
        }

        for sub in task.subtasks:
            if sub.get("assigned_agent") == self.capability.agent_id:
                sub["status"] = "COMPLETED"

        return {
            "agent": self.capability.agent_id,
            "analysis": task.analysis,
        }
