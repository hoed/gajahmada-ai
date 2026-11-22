from typing import Any, Dict, List

from gajahmada_ai.memory.models import MemoryType
from gajahmada_ai.storage.interfaces import StorageBackend


class ProceduralMemory:
    """
    Tier 4: Procedural Memory.
    Stores validated execution strategies, workflow templates, and learned
    problem-solving procedures (Level 2 learning updates).
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def list_procedures(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.storage.list_records(memory_type=MemoryType.PROCEDURAL.value, limit=limit)
