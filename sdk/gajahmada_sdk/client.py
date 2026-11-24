"""
GajahMada 2.0 Python SDK Client (Stage 5 — Independent AI Ecosystem).
Supports both live HTTP (`httpx`) and in-process ASGI / TestClient transports.
"""
from typing import Any, Dict, List, Optional
import httpx

from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType
from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


class NodeKeyPair:
    """Ergonomic wrapper around Ed25519 NodeIdentity for SDK developers."""

    def __init__(self, key_path: Optional[str] = None):
        self.identity = NodeIdentity(key_path=key_path)

    @property
    def node_id(self) -> str:
        return self.identity.node_id

    def create_attestation(
        self,
        record_id: str,
        record_hash: str,
        evidence_hash: str,
        source_ref: str,
    ) -> Dict[str, str]:
        payload = canonical_json_bytes({
            "record_id": record_id,
            "record_hash": record_hash,
            "evidence_hash": evidence_hash,
            "source_ref": source_ref,
            "verifier_node_id": self.node_id,
        })
        return {
            "node_id": self.node_id,
            "signature": self.identity.sign_message(payload),
            "evidence_hash": evidence_hash,
            "source_ref": source_ref,
        }


def build_signed_gip_message(
    keypair: NodeKeyPair,
    message_type: GIPMessageType,
    payload: Dict[str, Any],
    recipient_node_id: str = "*",
) -> Dict[str, Any]:
    msg = GIPMessage(
        message_type=message_type,
        sender_node_id=keypair.node_id,
        recipient_node_id=recipient_node_id,
        payload=payload,
    ).sign(keypair.identity)
    return msg.model_dump()


class GajahMadaClient:
    """
    Official Python SDK Client for GajahMada 2.0 Sovereign Nodes.
    Pass `http_client` (e.g. FastAPI TestClient or custom httpx.Client) for testing or custom transports.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        http_client: Optional[Any] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _url(self, path: str) -> str:
        # TestClient accepts relative paths directly; httpx.Client with base_url also accepts relative paths
        return path if path.startswith("/") else f"/{path}"

    def health(self) -> Dict[str, Any]:
        r = self._client.get(self._url("/api/v2/system/health"), headers=self._headers())
        r.raise_for_status()
        return r.json()

    def diagnose(self) -> Dict[str, Any]:
        r = self._client.get(self._url("/api/v2/intelligence/diagnose"), headers=self._headers())
        r.raise_for_status()
        return r.json()

    def ask(self, objective: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        r = self._client.post(
            self._url("/api/v2/intelligence/ask"),
            headers=self._headers(),
            json={"objective": objective, "max_tokens": max_tokens, "temperature": temperature},
        )
        r.raise_for_status()
        return r.json()

    def store_memory(
        self,
        content: str,
        memory_type: str = "semantic",
        visibility_policy: str = "LOCAL",
        verification_status: str = "UNVERIFIED",
        confidence: float = 0.8,
    ) -> Dict[str, Any]:
        r = self._client.post(
            self._url("/api/v2/memory"),
            headers=self._headers(),
            json={
                "content": content,
                "memory_type": memory_type,
                "visibility_policy": visibility_policy,
                "verification_status": verification_status,
                "confidence": confidence,
            },
        )
        r.raise_for_status()
        return r.json()

    def promote_knowledge(
        self,
        record_id: str,
        independent_sources: List[str],
        verification_method: str = "source_corroboration",
        node_attestations: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        r = self._client.post(
            self._url(f"/api/v2/knowledge/promote/{record_id}"),
            headers=self._headers(),
            json={
                "independent_sources": independent_sources,
                "verification_method": verification_method,
                "node_attestations": node_attestations or [],
            },
        )
        r.raise_for_status()
        return r.json()

    def send_gip(self, signed_message: Dict[str, Any]) -> Dict[str, Any]:
        r = self._client.post(
            self._url("/api/v2/network/gip"),
            headers=self._headers(),
            json=signed_message,
        )
        r.raise_for_status()
        return r.json()

    def list_federation(self) -> Dict[str, Any]:
        r = self._client.get(self._url("/api/v2/federation"), headers=self._headers())
        r.raise_for_status()
        return r.json()

    def register_organization(
        self,
        org_id: str,
        org_name: str,
        org_type: str,
        node_id: str,
        endpoint: str,
    ) -> Dict[str, Any]:
        r = self._client.post(
            self._url("/api/v2/federation/organizations"),
            headers=self._headers(),
            json={
                "org_id": org_id,
                "org_name": org_name,
                "org_type": org_type,
                "node_id": node_id,
                "endpoint": endpoint,
            },
        )
        r.raise_for_status()
        return r.json()
