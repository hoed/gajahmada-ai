import asyncio
from typing import Any, Dict, List, Set
from urllib.parse import urlparse
import httpx

_DEFAULT_TIMEOUT = httpx.Timeout(5.0)
_MAX_PEERS = 64


class P2PNetwork:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self.node_url = f"http://{host}:{self.port}"
        self._self_urls = {
            f"http://{host}:{self.port}",
            f"http://0.0.0.0:{self.port}",
            f"http://127.0.0.1:{self.port}",
            f"http://localhost:{self.port}",
        }
        self.peers: Set[str] = set()

    def add_peer(self, peer_url: str) -> bool:
        clean = peer_url.strip().rstrip("/")
        parsed = urlparse(clean)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        if clean in self._self_urls or len(self.peers) >= _MAX_PEERS:
            return False
        self.peers.add(clean)
        return True

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
