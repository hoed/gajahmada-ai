from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

WIB_TZ = timezone(timedelta(hours=7), name="WIB")


class SecurityAuditLogger:
    """
    Immutable, hash-chained security and governance audit event logger.
    Records authentication failures, SSRF blocks, peer revocations, policy violations,
    key rotations, and execution approvals.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.events: List[Dict[str, Any]] = []
        self._last_hash = "0" * 64
        if self.storage_path and os.path.exists(self.storage_path):
            self._load()

    def record(
        self,
        event_type: str,
        severity: str,
        actor: str,
        action: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(WIB_TZ)
        ts = round(now.timestamp(), 4)
        idx = len(self.events) + 1
        details_dict = details or {}
        raw_str = f"{idx}|{ts}|{event_type}|{severity}|{actor}|{action}|{outcome}|{json.dumps(details_dict, sort_keys=True)}|{self._last_hash}"
        event_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

        entry = {
            "event_id": f"sec-{idx:06d}",
            "index": idx,
            "timestamp": ts,
            "iso_time": now.isoformat(),
            "event_type": event_type,
            "severity": severity,
            "actor": actor,
            "action": action,
            "outcome": outcome,
            "details": details_dict,
            "previous_hash": self._last_hash,
            "hash": event_hash,
        }
        self._last_hash = event_hash
        self.events.append(entry)
        self._save()
        return entry

    def verify_integrity(self) -> bool:
        prev = "0" * 64
        for ev in self.events:
            raw_str = (
                f"{ev['index']}|{ev['timestamp']}|{ev['event_type']}|{ev['severity']}|"
                f"{ev['actor']}|{ev['action']}|{ev['outcome']}|"
                f"{json.dumps(ev.get('details', {}), sort_keys=True)}|{prev}"
            )
            expected = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
            if ev.get("previous_hash") != prev or ev.get("hash") != expected:
                return False
            prev = expected
        return True

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.events[-limit:]

    def _save(self):
        if not self.storage_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.events, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _load(self):
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    self.events = loaded
                    if self.events:
                        self._last_hash = self.events[-1].get("hash", "0" * 64)
        except Exception:
            self.events = []
