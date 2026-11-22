from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set
import uuid


class ExecutionState(str, Enum):
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ExecutionBudget:
    max_iterations: int = 8
    max_seconds: float = 20.0
    max_subtasks: int = 6
    iterations_used: int = 0
    elapsed_seconds: float = 0.0

    def is_exhausted(self) -> bool:
        return self.iterations_used >= self.max_iterations or self.elapsed_seconds >= self.max_seconds


@dataclass
class AgentCapability:
    agent_id: str
    role: str
    allowed_tools: Set[str]
    permissions: Set[str]
    description: str


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: f"tsk-{uuid.uuid4().hex[:10]}")
    objective: str = ""
    state: ExecutionState = ExecutionState.CREATED
    requires_human_approval: bool = False
    human_approved: bool = False
    assigned_agents: List[str] = field(default_factory=list)
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    final_report: str = ""
    confidence: float = 0.0
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    def transition(self, new_state: ExecutionState, note: str = ""):
        self.state = new_state
        self.updated_at = time.time()
        self.state_history.append({
            "state": new_state.value,
            "timestamp": round(self.updated_at, 4),
            "note": note,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "state": self.state.value,
            "requires_human_approval": self.requires_human_approval,
            "human_approved": self.human_approved,
            "assigned_agents": self.assigned_agents,
            "subtasks": self.subtasks,
            "findings": self.findings,
            "analysis": self.analysis,
            "verification": self.verification,
            "final_report": self.final_report,
            "confidence": self.confidence,
            "budget": {
                "max_iterations": self.budget.max_iterations,
                "iterations_used": self.budget.iterations_used,
                "max_seconds": self.budget.max_seconds,
                "elapsed_seconds": round(self.budget.elapsed_seconds, 4),
            },
            "state_history": self.state_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


class BaseAgent(ABC):
    """
    Base Autonomous Agent with distinct capabilities, tool permissions, and responsibilities.
    Not a prompt template wrapper—each concrete subclass operates on specific cognitive,
    memory, symbolic, or verification primitives.
    """

    def __init__(self, capability: AgentCapability):
        self.capability = capability

    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.capability.allowed_tools

    @abstractmethod
    def execute_step(self, task: AgentTask, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
