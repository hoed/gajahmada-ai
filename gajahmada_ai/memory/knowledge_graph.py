import re
import time
from typing import Any, Dict, List

from gajahmada_ai.storage.interfaces import StorageBackend


class KnowledgeGraphManager:
    """
    Relational Entity & Knowledge Relationship Graph stored inside the primary
    relational store (PostgreSQL/SQLite) without requiring an external graph DB.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def extract_and_store_relations(self, record_id: str, content: str, entities: List[str], confidence: float = 0.8):
        if len(entities) < 2:
            return
        primary = entities[0]
        for target in entities[1:6]:
            if primary.lower() == target.lower():
                continue
            rel_id = f"rel-{record_id}-{re.sub(r'[^a-z0-9]', '', primary.lower())[:12]}-{re.sub(r'[^a-z0-9]', '', target.lower())[:12]}"
            self.storage.save_relation({
                "relation_id": rel_id,
                "source_entity": primary,
                "predicate": "co_occurs_with",
                "target_entity": target,
                "record_id": record_id,
                "confidence": confidence,
                "created_at": time.time(),
            })

    def query_entity(self, entity: str) -> List[Dict[str, Any]]:
        return self.storage.list_relations(entity=entity)
