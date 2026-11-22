from typing import Any, Dict, List

from gajahmada_ai.memory.models import MemoryType
from gajahmada_ai.storage.interfaces import StorageBackend


class EpisodicMemory:
    """
    Tier 3: Episodic Memory.
    Stores temporal experiences, executed reasoning sessions, task outcomes,
    and historical events with timestamps.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def list_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.storage.list_records(memory_type=MemoryType.EPISODIC.value, limit=limit)
