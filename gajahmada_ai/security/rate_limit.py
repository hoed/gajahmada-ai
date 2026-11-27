from collections import defaultdict, deque
import time
from typing import Deque, Dict, Tuple


class SlidingWindowRateLimiter:
    """
    In-memory sliding-window rate limiter to protect endpoints and peer requests
    against flooding and computational DoS.
    """

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._calls = 0

    def check(self, key: str, cost: int = 1) -> Tuple[bool, int]:
        now = time.time()
        cutoff = now - self.window_seconds
        self._calls += 1
        if self._calls % 1024 == 0:
            # Prune idle buckets so per-IP keys cannot grow memory without bound.
            for k in [k for k, b in self._buckets.items() if not b or b[-1] < cutoff]:
                del self._buckets[k]
        bucket = self._buckets[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) + cost > self.max_requests:
            return False, 0

        for _ in range(cost):
            bucket.append(now)
        remaining = max(0, self.max_requests - len(bucket))
        return True, remaining
