import time
from typing import Any, Callable, Dict, Optional

from gajahmada_ai.security.audit import SecurityAuditLogger
from gajahmada_ai.security.policies import GovernancePolicy


class SandboxedExecutor:
    """
    Controlled, Policy-Guarded Sandboxed Execution Engine (Layer 6 & Layer 8).
    - Prohibits unrestricted shell, arbitrary filesystem writes, or external network access.
    - Enforces explicit tool allowlists and human approval gates for sensitive actions.
    - Tracks resource execution time against bounded budgets.
    """

    def __init__(
        self,
        policy: Optional[GovernancePolicy] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self.policy = policy or GovernancePolicy()
        self.audit_logger = audit_logger
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register_tool(self, name: str, handler: Callable[[Dict[str, Any]], Any]):
        self._tools[name] = handler

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_id: str,
        human_approved: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Check if action is sensitive and requires human approval
        if self.policy.is_action_sensitive(tool_name):
            if self.policy.require_human_approval_for_sensitive_tools and not human_approved:
                if self.audit_logger:
                    self.audit_logger.record(
                        event_type="SANDBOX_BLOCKED_APPROVAL_REQUIRED",
                        severity="WARN",
                        actor=agent_id,
                        action=f"execute_tool:{tool_name}",
                        outcome="WAITING_APPROVAL",
                        details={"arguments": str(arguments)[:200]},
                    )
                return {
                    "status": "WAITING_APPROVAL",
                    "tool": tool_name,
                    "error": f"Sensitive tool '{tool_name}' requires explicit human approval.",
                }

        # 2. Enforce strict sandbox tool allowlist
        if not self.policy.is_tool_allowed(tool_name) and not (
            self.policy.is_action_sensitive(tool_name) and human_approved
        ):
            if self.audit_logger:
                self.audit_logger.record(
                    event_type="SANDBOX_VIOLATION",
                    severity="HIGH",
                    actor=agent_id,
                    action=f"execute_tool:{tool_name}",
                    outcome="BLOCKED",
                    details={"reason": "Tool not in sandbox allowlist"},
                )
            raise PermissionError(f"Sandbox violation: Tool '{tool_name}' is not permitted by GovernancePolicy.")

        handler = self._tools.get(tool_name)
        if handler is None:
            raise KeyError(f"Tool '{tool_name}' is not registered in sandbox.")

        result = handler(arguments)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "status": "COMPLETED",
            "tool": tool_name,
            "result": result,
            "latency_ms": elapsed_ms,
        }
