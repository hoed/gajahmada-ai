from typing import Any, Dict, List

from gajahmada_ai.memory.models import MemoryType
from gajahmada_ai.storage.interfaces import StorageBackend


class CollectiveMemory:
    """
    Tier 5: Collective Memory.
    Stores peer-synchronized knowledge verified across the distributed GajahMada
    network while strictly preserving the original peer's `origin_node_id` and signature.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def list_collective(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.storage.list_records(memory_type=MemoryType.COLLECTIVE.value, limit=limit)
