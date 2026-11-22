from typing import Any, Dict, List

from gajahmada_ai.memory.models import MemoryType
from gajahmada_ai.storage.interfaces import StorageBackend


class SemanticMemory:
    """
    Tier 2: Semantic Memory.
    Persistent conceptual, historical, scientific, and factual records with
    384-d semantic vectors, provenance, and verification status.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def list_facts(self, limit: int = 200) -> List[Dict[str, Any]]:
        return self.storage.list_records(memory_type=MemoryType.SEMANTIC.value, limit=limit)
