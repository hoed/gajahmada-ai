from collections import deque
from enum import Enum
import hashlib
import hmac
import os
import secrets
import time
from typing import Deque, Dict, Optional, Set, Tuple
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
        "reasoning:read",
        "reasoning:execute",
        "agent:read",
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
        "reasoning:read",
        "reasoning:execute",
        "agent:read",
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
        admin_key_file: Optional[str] = None,
        allow_generated_key: bool = True,
    ):
        self.identity = identity
        self.audit_logger = audit_logger
        self.admin_key_file = admin_key_file
        self.allow_generated_key = allow_generated_key
        self._tokens: Dict[str, Role] = {}
        self._seen_nonces: Set[Tuple[str, str]] = set()
        self._nonce_order: Deque[Tuple[str, str]] = deque(maxlen=4096)
        self.admin_api_key = os.environ.get("GAJAHMADA_ADMIN_KEY", "").strip()
        if not self.admin_api_key and admin_key_file and os.path.exists(admin_key_file):
            try:
                with open(admin_key_file, "r", encoding="utf-8") as f:
                    self.admin_api_key = f.read().strip()
            except Exception:
                self.admin_api_key = ""
        if not self.admin_api_key:
            if not allow_generated_key:
                # Serverless disks are ephemeral. A random key written to /tmp cannot
                # be recovered by the operator and changes on every cold start.
                self.admin_api_key = ""
            else:
                # Cryptographically random 256-bit secret (NEVER derived from public node_id!)
                self.admin_api_key = "gm-admin-" + secrets.token_urlsafe(32)
                if admin_key_file:
                    try:
                        os.makedirs(os.path.dirname(os.path.abspath(admin_key_file)), exist_ok=True)
                        with open(admin_key_file, "w", encoding="utf-8") as f:
                            f.write(self.admin_api_key)
                        os.chmod(admin_key_file, 0o600)
                    except Exception:
                        pass
        if self.admin_api_key:
            self._tokens[self.admin_api_key] = Role.ADMIN

        operator_key = os.environ.get("GAJAHMADA_OPERATOR_KEY", "").strip()
        if operator_key:
            self._tokens[operator_key] = Role.OPERATOR

    def rotate_admin_key(self) -> str:
        """Rotate the admin API key to a fresh cryptographically random secret."""
        old_key = self.admin_api_key
        new_key = "gm-admin-" + secrets.token_urlsafe(32)
        self._tokens.pop(old_key, None)
        self.admin_api_key = new_key
        self._tokens[new_key] = Role.ADMIN
        if self.admin_key_file:
            try:
                with open(self.admin_key_file, "w", encoding="utf-8") as f:
                    f.write(new_key)
                os.chmod(self.admin_key_file, 0o600)
            except Exception:
                pass
        return new_key

    def resolve_optional_actor(self, request: Request) -> Dict[str, str]:
        """Inspect request credentials without raising; returns anonymous/reader if unauthenticated."""
        api_key = request.headers.get("x-api-key", "").strip()
        auth_header = request.headers.get("authorization", "").strip()
        if not api_key and auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()
        if api_key:
            for registered_token, role in self._tokens.items():
                if hmac.compare_digest(api_key, registered_token):
                    return {"actor": f"role:{role.value}", "role": role.value}
        return {"actor": "anonymous", "role": "anonymous"}

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

        # Hardened Ed25519 signed request auth (opt-in via GAJAHMADA_ALLOW_NODE_SIG_AUTH=1):
        # Requires domain-separated envelope (method, path, body_hash, timestamp, nonce, audience),
        # strict timestamp freshness (+/- 60s), replay nonce cache, and NEVER blindly grants ADMIN.
        node_id = request.headers.get("x-node-id", "").strip()
        signature = request.headers.get("x-signature", "").strip()
        timestamp = request.headers.get("x-timestamp", "").strip()
        nonce = request.headers.get("x-nonce", "").strip()
        body_hash = request.headers.get("x-body-sha256", "").strip() or hashlib.sha256(b"").hexdigest()
        audience = request.headers.get("x-auth-audience", "").strip()
        if (
            os.environ.get("GAJAHMADA_ALLOW_NODE_SIG_AUTH") == "1"
            and node_id
            and signature
            and timestamp
            and nonce
            and audience == "gajahmada-operator-auth"
            and self.identity
        ):
            try:
                ts_val = float(timestamp)
            except ValueError:
                self._log_denial(request, f"node:{node_id[:12]}", required_capability, "invalid_timestamp")
                raise HTTPException(status_code=401, detail="Invalid X-Timestamp header.")
            if abs(time.time() - ts_val) > 60.0:
                self._log_denial(request, f"node:{node_id[:12]}", required_capability, "stale_timestamp")
                raise HTTPException(status_code=401, detail="Signed request timestamp expired (>60s skew).")
            nonce_key = (node_id, nonce)
            if nonce_key in self._seen_nonces:
                self._log_denial(request, f"node:{node_id[:12]}", required_capability, "replayed_nonce")
                raise HTTPException(status_code=401, detail="Signed request nonce replay detected.")
            msg = f"gajahmada-auth-v2|{request.method}|{request.url.path}|{body_hash}|{timestamp}|{nonce}|{audience}".encode("utf-8")
            if node_id == self.identity.node_id and self.identity.verify_message(msg, signature, node_id):
                if len(self._nonce_order) == self._nonce_order.maxlen:
                    self._seen_nonces.discard(self._nonce_order.popleft())
                self._nonce_order.append(nonce_key)
                self._seen_nonces.add(nonce_key)
                # Bound node-signature auth to OPERATOR capabilities (never automatic full ADMIN)
                if required_capability in ROLE_CAPABILITIES[Role.OPERATOR]:
                    return {"actor": f"node:{node_id[:12]}", "role": Role.OPERATOR.value}
                self._log_denial(request, f"node:{node_id[:12]}", required_capability, "insufficient_capability")
                raise HTTPException(
                    status_code=403,
                    detail=f"Node signature auth lacks capability '{required_capability}'.",
                )

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
