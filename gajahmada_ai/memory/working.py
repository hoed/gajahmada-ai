from collections import deque
from typing import Any, Deque, Dict, List


class WorkingMemory:
    """
    Tier 1: Working Memory.
    Bounded context-window memory holding active reasoning states, short-term
    observations, and current session items.
    """

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._items: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def push(self, item: Dict[str, Any]):
        self._items.append(item)

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        items = list(self._items)
        return items[-limit:]

    def clear(self):
        self._items.clear()

    def count(self) -> int:
        return len(self._items)
