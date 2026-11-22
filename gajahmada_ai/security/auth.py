from enum import Enum
import hmac
import os
import secrets
from typing import Dict, Optional, Set
from fastapi import HTTPException, Request

from gajahmada_ai.security.audit import SecurityAuditLogger
from gajahmada_ai.security.crypto import NodeIdentity


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    PEER = "peer"
    READER = "reader"


ROLE_CAPABILITIES: Dict[Role, Set[str]] = {
    Role.ADMIN: {
        "system:admin",
        "config:write",
        "logs:clear",
        "memory:read",
        "memory:write",
        "memory:delete",
        "memory:backup",
        "reasoning:execute",
        "agent:run",
        "agent:approve",
        "network:manage",
        "network:gip",
        "learning:propose",
        "learning:approve",
        "security:rotate_key",
    },
    Role.OPERATOR: {
        "memory:read",
        "memory:write",
        "memory:backup",
        "reasoning:execute",
        "agent:run",
        "network:gip",
        "learning:propose",
    },
    Role.PEER: {
        "network:gip",
        "reasoning:execute",
        "memory:read",
    },
    Role.READER: {
        "memory:read",
        "reasoning:execute",
    },
}


class AuthManager:
    """
    Role and capability-based authentication and authorization manager.
    Supports:
    - API Keys (`X-API-Key` header or `Authorization: Bearer <token>`)
    - Cryptographically signed Ed25519 peer headers (`X-Node-Id`, `X-Signature`, `X-Timestamp`)
    """

    def __init__(
        self,
        identity: Optional[NodeIdentity] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self.identity = identity
        self.audit_logger = audit_logger
        self._tokens: Dict[str, Role] = {}
        self.admin_api_key = os.environ.get("GAJAHMADA_ADMIN_KEY", "").strip()
        if not self.admin_api_key:
            self.admin_api_key = "gm-sovereign-admin-" + (
                identity.node_id[:24] if identity and identity.node_id else secrets.token_urlsafe(18)
            )
        self._tokens[self.admin_api_key] = Role.ADMIN

        operator_key = os.environ.get("GAJAHMADA_OPERATOR_KEY", "").strip()
        if operator_key:
            self._tokens[operator_key] = Role.OPERATOR

    def register_token(self, token: str, role: Role):
        if token and len(token) >= 16:
            self._tokens[token] = role

    def authenticate_request(
        self,
        request: Request,
        required_capability: str = "system:admin",
    ) -> Dict[str, str]:
        api_key = request.headers.get("x-api-key", "").strip()
        auth_header = request.headers.get("authorization", "").strip()
        if not api_key and auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

        if api_key:
            for registered_token, role in self._tokens.items():
                if hmac.compare_digest(api_key, registered_token):
                    caps = ROLE_CAPABILITIES.get(role, set())
                    if required_capability in caps:
                        return {"actor": f"role:{role.value}", "role": role.value}
                    self._log_denial(request, f"role:{role.value}", required_capability, "insufficient_capability")
                    raise HTTPException(
                        status_code=403,
                        detail=f"Role '{role.value}' lacks required capability '{required_capability}'.",
                    )

        # Also check Ed25519 signed administrative request from local node identity
        node_id = request.headers.get("x-node-id", "").strip()
        signature = request.headers.get("x-signature", "").strip()
        timestamp = request.headers.get("x-timestamp", "").strip()
        if node_id and signature and timestamp and self.identity:
            msg = f"{request.method}|{request.url.path}|{timestamp}".encode("utf-8")
            if node_id == self.identity.node_id and self.identity.verify_message(msg, signature, node_id):
                return {"actor": f"node:{node_id[:12]}", "role": Role.ADMIN.value}

        self._log_denial(request, "anonymous", required_capability, "unauthenticated")
        raise HTTPException(
            status_code=401,
            detail=f"Authentication required for capability '{required_capability}'. Supply valid X-API-Key or Bearer token.",
        )

    def _log_denial(self, request: Request, actor: str, capability: str, reason: str):
        if self.audit_logger:
            self.audit_logger.record(
                event_type="AUTH_REJECTED",
                severity="WARN",
                actor=actor,
                action=f"{request.method} {request.url.path}",
                outcome="DENIED",
                details={"required_capability": capability, "reason": reason},
            )
