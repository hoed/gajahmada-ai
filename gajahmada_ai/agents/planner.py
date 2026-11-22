from typing import Any, Dict, List

from gajahmada_ai.agents.base import AgentCapability, AgentTask, BaseAgent, ExecutionState
from gajahmada_ai.cognition.decomposition import ProblemDecomposer
from gajahmada_ai.cognition.intent import IntentClassifier
from gajahmada_ai.cognition.perception import PerceptionModule


class PlannerAgent(BaseAgent):
    """
    Autonomous Planner Agent.
    Responsibilities:
    - Perceives and classifies the objective.
    - Decomposes the objective into bounded subtasks.
    - Maps each subtask to required specialist agents and sandboxed tools.
    - Transitions task from CREATED -> PLANNED.
    """

    def __init__(self):
        super().__init__(
            AgentCapability(
                agent_id="agent.planner",
                role="Objective Decomposition & Execution Planner",
                allowed_tools={"system_diagnostics"},
                permissions={"task:plan", "task:decompose"},
                description="Decomposes objectives into structured subtasks with tool and capability assignments.",
            )
        )
        self.perception = PerceptionModule()
        self.intent_classifier = IntentClassifier()
        self.decomposer = ProblemDecomposer()

    def execute_step(self, task: AgentTask, context: Dict[str, Any]) -> Dict[str, Any]:
        perceived = self.perception.perceive(task.objective)
        intent = self.intent_classifier.classify(perceived)
        raw_subtasks = self.decomposer.decompose(perceived, intent)

        structured_subtasks: List[Dict[str, Any]] = []
        for idx, sub in enumerate(raw_subtasks[: task.budget.max_subtasks], start=1):
            structured_subtasks.append({
                "subtask_id": f"{task.task_id}-sub-{idx}",
                "index": idx,
                "description": sub,
                "assigned_agent": "agent.researcher",
                "required_tool": "memory_search",
                "status": "PLANNED",
            })

        # Add quantitative/symbolic analysis subtask
        structured_subtasks.append({
            "subtask_id": f"{task.task_id}-sub-analysis",
            "index": len(structured_subtasks) + 1,
            "description": f"Execute symbolic and quantitative analysis on: {task.objective[:100]}",
            "assigned_agent": "agent.analyst",
            "required_tool": "symbolic_math",
            "status": "PLANNED",
        })

        # Add independent verification subtask
        structured_subtasks.append({
            "subtask_id": f"{task.task_id}-sub-verify",
            "index": len(structured_subtasks) + 1,
            "description": f"Verify provenance, constraints, and contradictions for: {task.objective[:100]}",
            "assigned_agent": "agent.verifier",
            "required_tool": "evidence_verify",
            "status": "PLANNED",
        })

        task.subtasks = structured_subtasks
        task.assigned_agents = ["agent.planner", "agent.researcher", "agent.analyst", "agent.verifier"]
        task.transition(
            ExecutionState.PLANNED,
            f"Generated execution plan with {len(structured_subtasks)} subtasks (intent={intent.primary_intent})",
        )
        return {
            "agent": self.capability.agent_id,
            "intent": intent.primary_intent,
            "domains": intent.reasoning_domains,
            "subtasks_planned": len(structured_subtasks),
        }
