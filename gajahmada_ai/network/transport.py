from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import httpx

from gajahmada_ai.network.protocol import GIPMessage
from gajahmada_ai.security.ssrf import validate_peer_url


class TransportBackend(ABC):
    """
    Abstract P2P Transport Interface (Layer 2).
    Designed to support Authenticated HTTPS transport initially and pluggable
    libp2p (QUIC/Noise) transports in future deployments.
    """

    @abstractmethod
    async def send_message(self, peer_endpoint: str, message: GIPMessage) -> Dict[str, Any]:
        pass


class AuthenticatedHTTPTransport(TransportBackend):
    """
    Authenticated HTTP/HTTPS Transport for GIP/2.0 messages.
    Also supports direct in-process routing table registration so multiple
    independently configured node instances can exchange signed GIP messages
    over local ASGI/HTTP or in-memory test transports without mocking cryptography.
    """

    def __init__(self, timeout_sec: float = 5.0, allow_local: bool = True):
        self.timeout = httpx.Timeout(timeout_sec)
        self.allow_local = allow_local
        self._direct_handlers: Dict[str, Callable[[GIPMessage], Dict[str, Any]]] = {}

    def register_direct_peer(self, endpoint: str, handler: Callable[[GIPMessage], Dict[str, Any]]):
        clean = endpoint.strip().rstrip("/")
        self._direct_handlers[clean] = handler

    async def send_message(self, peer_endpoint: str, message: GIPMessage) -> Dict[str, Any]:
        ok, clean = validate_peer_url(peer_endpoint, allow_local=self.allow_local)
        if not ok:
            raise ValueError(f"Transport SSRF guard blocked peer '{peer_endpoint}': {clean}")

        if clean in self._direct_handlers:
            return self._direct_handlers[clean](message)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(
                f"{clean}/api/v2/network/gip",
                json=message.model_dump(),
            )
            res.raise_for_status()
            return res.json()
