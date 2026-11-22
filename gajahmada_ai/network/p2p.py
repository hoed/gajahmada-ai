import asyncio
from typing import Any, Dict, List, Optional, Set
import httpx

from gajahmada_ai.security.ssrf import validate_peer_url

_DEFAULT_TIMEOUT = httpx.Timeout(5.0)
_MAX_PEERS = 64


class P2PNetwork:
    """
    Asynchronous P2P Network Manager with strict SSRF protection.
    Preserves v1 methods (`add_peer`, `remove_peer`, `broadcast`, `sync_peers`)
    while preventing registration of cloud metadata or restricted internal ports.
    """

    def __init__(self, host: str, port: int, allow_local: bool = True):
        self.host = host
        self.port = int(port)
        self.allow_local = allow_local
        self.node_url = f"http://{host}:{self.port}"
        self._self_urls = {
            f"http://{host}:{self.port}",
            f"http://0.0.0.0:{self.port}",
            f"http://127.0.0.1:{self.port}",
            f"http://localhost:{self.port}",
        }
        self.peers: Set[str] = set()

    def add_peer(self, peer_url: str, allow_local: Optional[bool] = None) -> bool:
        use_local = self.allow_local if allow_local is None else allow_local
        ok, clean_or_reason = validate_peer_url(peer_url, allow_local=use_local)
        if not ok:
            return False
        if clean_or_reason in self._self_urls or len(self.peers) >= _MAX_PEERS:
            return False
        self.peers.add(clean_or_reason)
        return True

    def validate_and_add_peer(self, peer_url: str, allow_local: Optional[bool] = None) -> str:
        use_local = self.allow_local if allow_local is None else allow_local
        ok, clean_or_reason = validate_peer_url(peer_url, allow_local=use_local)
        if not ok:
            raise ValueError(clean_or_reason)
        if clean_or_reason in self._self_urls:
            raise ValueError("Cannot register self URL as external peer")
        if len(self.peers) >= _MAX_PEERS:
            raise ValueError("Maximum peer table capacity reached")
        self.peers.add(clean_or_reason)
        return clean_or_reason

    def remove_peer(self, peer_url: str):
        self.peers.discard(peer_url.strip().rstrip("/"))

    async def broadcast(self, endpoint: str, data: Dict[str, Any]):
        """Broadcast signed payload to all peers with timeout."""
        if not self.peers:
            return []
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            tasks = [
                client.post(f"{peer}{endpoint}", json=data)
                for peer in list(self.peers)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

    async def sync_peers(self, bootstrap_peers: List[str]):
        """Discover additional peers from bootstrap nodes."""
        for bp in bootstrap_peers:
            self.add_peer(bp)

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            for peer in list(self.peers):
                try:
                    res = await client.get(f"{peer}/node/peers")
                    if res.status_code == 200:
                        for np_url in res.json().get("peers", []):
                            self.add_peer(np_url)
                except Exception:
                    pass
