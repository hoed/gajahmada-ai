"""
HTTP edge helpers: trusted-proxy client IP resolution and bounded long-lived connections.

- `resolve_client_ip` only honours forwarding headers when the deployment is explicitly behind a
  trusted proxy (Vercel auto-detected, or `GAJAHMADA_TRUST_PROXY=1`). Otherwise the socket peer is
  used, so clients cannot spoof `X-Forwarded-For` to evade rate limits.
- `ConnectionLimiter` caps concurrent SSE/streaming connections per IP and globally.
"""
from __future__ import annotations

import ipaddress
import os
import threading
from collections import defaultdict
from typing import Dict, Optional

from fastapi import Request


def proxy_trusted() -> bool:
    flag = os.environ.get("GAJAHMADA_TRUST_PROXY", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return bool(os.environ.get("VERCEL"))


def _valid_ip(value: str) -> Optional[str]:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def resolve_client_ip(request: Request, trust_proxy: Optional[bool] = None) -> str:
    trust = proxy_trusted() if trust_proxy is None else trust_proxy
    if trust:
        real = _valid_ip(request.headers.get("x-real-ip", ""))
        if real:
            return real
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            first = _valid_ip(xff.split(",")[0])
            if first:
                return first
    return request.client.host if request.client else "unknown"


class ConnectionLimiter:
    """Thread-safe concurrent connection cap (per key and global)."""

    def __init__(self, per_key: int = 3, global_max: int = 50):
        self.per_key = per_key
        self.global_max = global_max
        self._counts: Dict[str, int] = defaultdict(int)
        self._total = 0
        self._lock = threading.Lock()

    def acquire(self, key: str) -> bool:
        with self._lock:
            if self._total >= self.global_max or self._counts[key] >= self.per_key:
                return False
            self._counts[key] += 1
            self._total += 1
            return True

    def release(self, key: str) -> None:
        with self._lock:
            if self._counts.get(key, 0) > 0:
                self._counts[key] -= 1
                self._total = max(0, self._total - 1)
                if self._counts[key] == 0:
                    del self._counts[key]

    @property
    def active(self) -> int:
        with self._lock:
            return self._total


def build_sse_limiter() -> ConnectionLimiter:
    return ConnectionLimiter(
        per_key=int(os.environ.get("GAJAHMADA_SSE_PER_IP", "3")),
        global_max=int(os.environ.get("GAJAHMADA_SSE_MAX", "50")),
    )
