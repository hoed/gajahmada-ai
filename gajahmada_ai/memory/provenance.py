import hashlib
from typing import Any, Dict

from gajahmada_ai.memory.models import MemoryRecord, SourceProvenance


class ProvenanceValidator:
    """
    Validates content integrity hashes, origin node attribution, and sharing policies.
    Rule 5: Collective knowledge must preserve its original source.
    Rule 6: Memory synchronization must respect node permissions.
    """

    def verify_record_integrity(self, rec: Dict[str, Any]) -> bool:
        expected_str = (
            f"{rec['record_id']}|{rec['content']}|{rec['memory_type']}|"
            f"{rec['version']}|{rec.get('source_provenance', {}).get('origin_node_id', 'local')}|{rec['verification_status']}"
        )
        expected_hash = hashlib.sha256(expected_str.encode("utf-8")).hexdigest()
        return rec.get("integrity_hash") == expected_hash

    def can_share_externally(self, rec: Dict[str, Any]) -> bool:
        vis = str(rec.get("visibility_policy", "LOCAL")).upper()
        return vis in ("SHARED", "COLLECTIVE")
