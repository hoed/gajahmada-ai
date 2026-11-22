import time
from typing import Any, Dict, List

from gajahmada_ai.storage.interfaces import StorageBackend


class MemoryVersionController:
    """
    Immutable Version Tracking & Audit Trail for Memory Records.
    Rule 3: New evidence may supersede older conclusions.
    Rule 4: Previous versions must remain auditable.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def record_version_snapshot(
        self,
        record_dict: Dict[str, Any],
        modified_by: str = "local_node",
        reason: str = "initial_creation",
    ):
        ver = int(record_dict.get("version", 1))
        entry = {
            "version_id": f"{record_dict['record_id']}-v{ver}",
            "record_id": record_dict["record_id"],
            "version": ver,
            "content": record_dict["content"],
            "confidence": float(record_dict.get("confidence", 0.75)),
            "verification_status": record_dict.get("verification_status", "UNVERIFIED"),
            "integrity_hash": record_dict["integrity_hash"],
            "modified_by": modified_by,
            "reason": reason,
            "timestamp": round(time.time(), 4),
        }
        self.storage.save_version(entry)

    def get_history(self, record_id: str) -> List[Dict[str, Any]]:
        return self.storage.get_versions(record_id)
