import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.agents.analyst import AnalystAgent
from gajahmada_ai.agents.base import AgentTask, ExecutionBudget, ExecutionState
from gajahmada_ai.agents.planner import PlannerAgent
from gajahmada_ai.agents.researcher import ResearcherAgent
from gajahmada_ai.agents.verifier import VerifierAgent
from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.execution.sandbox import SandboxedExecutor
from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus
from gajahmada_ai.reasoning.evidence import EvidenceReasoner
from gajahmada_ai.reasoning.symbolic import SymbolicReasoner
from gajahmada_ai.security.audit import SecurityAuditLogger
from gajahmada_ai.security.policies import GovernancePolicy


class AutonomousTaskScheduler:
    """
    Layer 6 Autonomous Agent Runtime & Task Scheduler.
    Orchestrates the full lifecycle:
    CREATED -> PLANNED -> APPROVED (or WAITING if human approval required) ->
    RUNNING -> (Self-Correction loop if missing evidence) -> COMPLETED / FAILED / CANCELLED.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        cognitive_pipeline: CognitiveReasoningPipeline,
        policy: Optional[GovernancePolicy] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self.memory = memory_manager
        self.cognition = cognitive_pipeline
        self.policy = policy or GovernancePolicy()
        self.audit_logger = audit_logger
        self.sandbox = SandboxedExecutor(policy=self.policy, audit_logger=self.audit_logger)
        self._register_default_sandbox_tools()

        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent(self.sandbox)
        self.analyst = AnalystAgent(self.sandbox)
        self.verifier = VerifierAgent(self.sandbox)
        self.tasks: Dict[str, AgentTask] = {}

    def _register_default_sandbox_tools(self):
        symbolic = SymbolicReasoner()
        evidence_engine = EvidenceReasoner()

        self.sandbox.register_tool(
            "memory_search",
            lambda args: self.memory.retrieve(
                query=str(args.get("query", "")),
                top_k=int(args.get("top_k", 5)),
            ),
        )
        self.sandbox.register_tool(
            "knowledge_graph_query",
            lambda args: self.memory.knowledge_graph.query_entity(str(args.get("entity", ""))),
        )
        self.sandbox.register_tool(
            "symbolic_math",
            lambda args: symbolic.reason(
                query=str(args.get("query", "")),
                facts=args.get("facts", []),
            ).__dict__,
        )
        self.sandbox.register_tool(
            "evidence_verify",
            lambda args: {
                "aggregate_confidence": (res := evidence_engine.evaluate_evidence(args.get("query", ""), args.get("records", []))).aggregate_confidence,
                "verification_status": res.verification_status,
                "contradictions": [c.__dict__ for c in res.contradictions],
                "summary": res.summary,
            },
        )

    def submit_objective(
        self,
        objective: str,
        requires_human_approval: bool = False,
        human_approved: bool = False,
        max_iterations: int = 8,
        requested_tools: Optional[List[str]] = None,
    ) -> AgentTask:
        t0 = time.perf_counter()
        budget = ExecutionBudget(
            max_iterations=min(max_iterations, self.policy.max_agent_iterations),
            max_seconds=self.policy.max_agent_execution_seconds,
        )
        # Check if any requested tool is sensitive
        needs_approval = requires_human_approval or any(
            self.policy.is_action_sensitive(t) for t in (requested_tools or [])
        )

        task = AgentTask(
            objective=objective.strip(),
            requires_human_approval=needs_approval,
            human_approved=human_approved,
            budget=budget,
        )
        task.transition(ExecutionState.CREATED, "Objective received by AutonomousTaskScheduler")
        self.tasks[task.task_id] = task

        # Step 1: Planning (CREATED -> PLANNED)
        task.budget.iterations_used += 1
        self.planner.execute_step(task, {})

        # Step 2: Approval Gate (PLANNED -> WAITING or APPROVED)
        if task.requires_human_approval and not task.human_approved:
            task.transition(
                ExecutionState.WAITING,
                "Waiting for explicit operator approval before execution",
            )
            task.budget.elapsed_seconds = time.perf_counter() - t0
            return task

        task.transition(ExecutionState.APPROVED, "Authorized for sandboxed execution")
        return self._run_approved_task(task, t0, requested_tools=requested_tools)

    def approve_task(self, task_id: str) -> AgentTask:
        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(f"Task '{task_id}' not found")
        if task.state == ExecutionState.CANCELLED:
            raise ValueError("Cannot approve a CANCELLED task")
        task.human_approved = True
        task.transition(ExecutionState.APPROVED, "Human operator granted execution approval")
        return self._run_approved_task(task, time.perf_counter())

    def cancel_task(self, task_id: str) -> AgentTask:
        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(f"Task '{task_id}' not found")
        task.transition(ExecutionState.CANCELLED, "Cancelled by operator")
        return task

    def _run_approved_task(
        self,
        task: AgentTask,
        t0: float,
        requested_tools: Optional[List[str]] = None,
    ) -> AgentTask:
        try:
            task.transition(ExecutionState.RUNNING, "Executing specialist agent pipeline")

            # Enforce sandbox check on any explicitly requested tools
            for tool_name in (requested_tools or []):
                self.sandbox.execute_tool(
                    tool_name=tool_name,
                    arguments={"objective": task.objective},
                    agent_id="agent.scheduler",
                    human_approved=task.human_approved,
                )

            # Researcher step
            if task.budget.is_exhausted():
                raise TimeoutError("Execution budget exhausted before research phase")
            task.budget.iterations_used += 1
            self.researcher.execute_step(task, {})

            # Analyst step
            if task.budget.is_exhausted():
                raise TimeoutError("Execution budget exhausted before analysis phase")
            task.budget.iterations_used += 1
            self.analyst.execute_step(task, {})

            # Verifier step
            if task.budget.is_exhausted():
                raise TimeoutError("Execution budget exhausted before verification phase")
            task.budget.iterations_used += 1
            self.verifier.execute_step(task, {})

            # Self-Correction: if no direct memory findings were retrieved, broaden query by keywords
            if task.verification.get("requires_self_correction") and not task.budget.is_exhausted():
                task.budget.iterations_used += 1
                keywords = self.planner.perception.perceive(task.objective).keywords
                if keywords:
                    broad_query = " ".join(keywords[:3])
                    broad_hits = self.memory.retrieve(query=broad_query, top_k=3)
                    if broad_hits:
                        task.findings = broad_hits
                        task.state_history.append({
                            "state": ExecutionState.RUNNING.value,
                            "timestamp": round(time.time(), 4),
                            "note": f"Self-correction revised search query to '{broad_query}' -> {len(broad_hits)} record(s)",
                        })
                        self.verifier.execute_step(task, {})

            # Final cognitive synthesis report
            reasoning_res = self.cognition.execute(
                objective=task.objective,
                memory_records=task.findings,
            )
            task.final_report = reasoning_res.conclusion
            task.confidence = reasoning_res.confidence

            # Store validated episodic experience (Step 11)
            self.memory.store(
                content=f"Executed autonomous task '{task.task_id}' for objective '{task.objective[:120]}' -> status={reasoning_res.verification_status}, confidence={task.confidence}",
                memory_type=MemoryType.EPISODIC,
                source_type="autonomous_agent_experience",
                author="agent.scheduler",
                confidence=task.confidence,
                verification_status=VerificationStatus(reasoning_res.verification_status),
            )

            task.budget.elapsed_seconds = time.perf_counter() - t0
            task.transition(
                ExecutionState.COMPLETED,
                f"Completed in {task.budget.iterations_used} iteration(s) with confidence={task.confidence}",
            )
        except Exception as e:
            task.error = str(e)
            task.budget.elapsed_seconds = time.perf_counter() - t0
            task.transition(ExecutionState.FAILED, f"Execution failed: {e}")

        return task

    def list_agents(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent_id": a.capability.agent_id,
                "role": a.capability.role,
                "allowed_tools": sorted(a.capability.allowed_tools),
                "permissions": sorted(a.capability.permissions),
                "description": a.capability.description,
            }
            for a in (self.planner, self.researcher, self.analyst, self.verifier)
        ]
