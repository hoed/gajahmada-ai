"""Regression tests: trusted-proxy client IP, SSE connection caps, rate-limiter pruning."""
from __future__ import annotations

import time

from starlette.requests import Request

from gajahmada_ai.api.http_utils import ConnectionLimiter, resolve_client_ip
from gajahmada_ai.security.rate_limit import SlidingWindowRateLimiter


def _req(headers: dict, peer: str = "10.0.0.9") -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/", "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (peer, 1234),
    })


def test_forwarded_headers_ignored_when_proxy_not_trusted():
    r = _req({"x-forwarded-for": "6.6.6.6", "x-real-ip": "7.7.7.7"})
    assert resolve_client_ip(r, trust_proxy=False) == "10.0.0.9"


def test_forwarded_headers_used_when_proxy_trusted():
    assert resolve_client_ip(_req({"x-real-ip": "203.0.113.5"}), trust_proxy=True) == "203.0.113.5"
    assert resolve_client_ip(_req({"x-forwarded-for": "198.51.100.2, 10.1.1.1"}), trust_proxy=True) == "198.51.100.2"


def test_invalid_forwarded_value_falls_back_to_peer():
    assert resolve_client_ip(_req({"x-forwarded-for": "not-an-ip"}), trust_proxy=True) == "10.0.0.9"


def test_connection_limiter_per_key_and_global_caps():
    lim = ConnectionLimiter(per_key=2, global_max=3)
    assert lim.acquire("a") and lim.acquire("a")
    assert not lim.acquire("a")
    assert lim.acquire("b")
    assert not lim.acquire("c")  # global cap reached
    lim.release("a")
    assert lim.acquire("c")
    for k in ("a", "b", "c", "c"):
        lim.release(k)
    assert lim.active == 0


def test_rate_limiter_prunes_idle_buckets():
    rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=0.01)
    for i in range(50):
        rl.check(f"ip-{i}")
    time.sleep(0.05)
    for _ in range(1024):
        rl.check("hot")
    assert len(rl._buckets) < 10


def test_sse_stream_returns_429_when_cap_exceeded():
    from fastapi.testclient import TestClient
    from gajahmada_ai.api import server

    server.ensure_initialized()
    client = TestClient(server.app)
    from gajahmada_ai.api.observatory_routes import SSE_LIMITER as limiter
    held = []
    while limiter.acquire("testclient"):
        held.append("testclient")
    try:
        assert client.get("/api/v2/observatory/stream").status_code == 429
    finally:
        for k in held:
            limiter.release(k)
