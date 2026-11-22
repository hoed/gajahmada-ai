from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class GovernancePolicy:
    """
    Layer 8 Intelligence Governance Policies:
    - Permission boundaries & tool sandboxing
    - Execution policies & human approval thresholds
    - Evidence & knowledge trust thresholds
    - Peer allowlists and resource ceilings
    """
    require_admin_auth_on_v2: bool = True
    require_human_approval_for_sensitive_tools: bool = True
    min_evidence_for_verified_fact: int = 1
    min_confidence_for_collective_memory: float = 0.70
    max_agent_iterations: int = 10
    max_agent_execution_seconds: float = 30.0
    max_input_length_chars: int = 32768
    allow_local_peers: bool = True
    allow_external_ai_providers: bool = False
    auto_modify_neural_weights: bool = False
    allowed_sandbox_tools: Set[str] = field(
        default_factory=lambda: {
            "symbolic_math",
            "symbolic_logic",
            "memory_search",
            "evidence_verify",
            "contradiction_check",
            "knowledge_graph_query",
            "system_diagnostics",
        }
    )
    sensitive_actions_requiring_approval: Set[str] = field(
        default_factory=lambda: {
            "shell_command",
            "filesystem_write",
            "external_network_fetch",
            "memory_purge",
            "model_weight_update",
            "policy_override",
        }
    )
    trusted_peer_ids: Set[str] = field(default_factory=set)
    revoked_peer_ids: Set[str] = field(default_factory=set)

    def is_action_sensitive(self, action_name: str) -> bool:
        return action_name in self.sensitive_actions_requiring_approval

    def is_tool_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed_sandbox_tools

    def to_dict(self) -> Dict:
        return {
            "require_admin_auth_on_v2": self.require_admin_auth_on_v2,
            "require_human_approval_for_sensitive_tools": self.require_human_approval_for_sensitive_tools,
            "min_evidence_for_verified_fact": self.min_evidence_for_verified_fact,
            "min_confidence_for_collective_memory": self.min_confidence_for_collective_memory,
            "max_agent_iterations": self.max_agent_iterations,
            "max_agent_execution_seconds": self.max_agent_execution_seconds,
            "max_input_length_chars": self.max_input_length_chars,
            "allow_local_peers": self.allow_local_peers,
            "allow_external_ai_providers": self.allow_external_ai_providers,
            "auto_modify_neural_weights": self.auto_modify_neural_weights,
            "allowed_sandbox_tools": sorted(self.allowed_sandbox_tools),
            "sensitive_actions_requiring_approval": sorted(self.sensitive_actions_requiring_approval),
        }
