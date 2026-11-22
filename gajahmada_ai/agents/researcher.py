from typing import Any, Dict, List

from gajahmada_ai.agents.base import AgentCapability, AgentTask, BaseAgent
from gajahmada_ai.execution.sandbox import SandboxedExecutor


class ResearcherAgent(BaseAgent):
    """
    Autonomous Researcher Agent.
    Responsibilities:
    - Executes memory and knowledge graph queries inside the SandboxedExecutor.
    - Collects primary evidence records with provenance hashes.
    - Identifies knowledge gaps when evidence is missing.
    """

    def __init__(self, sandbox: SandboxedExecutor):
        super().__init__(
            AgentCapability(
                agent_id="agent.researcher",
                role="Memory & Evidence Investigator",
                allowed_tools={"memory_search", "knowledge_graph_query"},
                permissions={"memory:read", "knowledge:query"},
                description="Retrieves semantic, episodic, and collective memory records with provenance.",
            )
        )
        self.sandbox = sandbox

    def execute_step(self, task: AgentTask, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.can_use_tool("memory_search"):
            raise PermissionError("ResearcherAgent lacks 'memory_search' permission")

        res = self.sandbox.execute_tool(
            tool_name="memory_search",
            arguments={"query": task.objective, "top_k": 5},
            agent_id=self.capability.agent_id,
            human_approved=task.human_approved,
        )
        records: List[Dict[str, Any]] = res.get("result", []) if res.get("status") == "COMPLETED" else []
        task.findings = records

        for sub in task.subtasks:
            if sub.get("assigned_agent") == self.capability.agent_id:
                sub["status"] = "COMPLETED"
                sub["records_found"] = len(records)

        return {
            "agent": self.capability.agent_id,
            "records_retrieved": len(records),
            "missing_evidence": len(records) == 0,
        }
