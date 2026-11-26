"""
GajahMada AI — Versioned GNNM Model Registry with Controlled Promotion & Rollback.

Preserves existing `gajahmada_ai/neural/model_registry.py` (runtime backend router)
while adding persistent checkpoint lifecycle governance:
States: DEVELOPMENT -> TRAINING -> EVALUATING -> APPROVED -> ACTIVE -> ARCHIVED (or REJECTED).
"""
from enum import Enum
import json
import os
import time
from typing import Any, Dict, List, Optional


class ModelPromotionStatus(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class GNNMModelRegistry:
    """Persistent atomic JSON registry for GNNM checkpoint versions, promotion, and rollback."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._versions: Dict[str, Dict[str, Any]] = {}
        self.active_version_id: Optional[str] = None
        if self.storage_path and os.path.exists(self.storage_path):
            self._load()

    def register_version(
        self,
        version_id: str,
        model_name: str,
        architecture: str,
        checkpoint_path: str,
        base_checkpoint: Optional[str] = None,
        dataset_versions: Optional[List[str]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        status: ModelPromotionStatus = ModelPromotionStatus.DEVELOPMENT,
    ) -> Dict[str, Any]:
        entry = {
            "version_id": version_id,
            "model_name": model_name,
            "architecture": architecture,
            "checkpoint_path": checkpoint_path,
            "base_checkpoint": base_checkpoint,
            "dataset_versions": list(dataset_versions or []),
            "training_config": dict(training_config or {}),
            "training_timestamp": round(time.time(), 3),
            "evaluation_metrics": dict(evaluation_metrics or {}),
            "promotion_status": status.value if isinstance(status, ModelPromotionStatus) else str(status),
            "provenance": dict(provenance or {}),
            "rollback_reference": None,
        }
        self._versions[version_id] = entry
        self._save()
        return entry

    def update_status(
        self,
        version_id: str,
        status: ModelPromotionStatus,
        evaluation_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = self._versions.get(version_id)
        if entry is None:
            raise KeyError(f"Model version '{version_id}' not found.")
        entry["promotion_status"] = status.value
        if evaluation_metrics is not None:
            entry["evaluation_metrics"] = dict(evaluation_metrics)
        self._save()
        return entry

    def promote(self, version_id: str, operator_approved: bool = False) -> Dict[str, Any]:
        """Controlled promotion: requires APPROVED state AND explicit operator approval."""
        entry = self._versions.get(version_id)
        if entry is None:
            raise KeyError(f"Model version '{version_id}' not found.")
        if not operator_approved:
            raise PermissionError("Model promotion requires explicit operator approval.")
        if entry["promotion_status"] not in (ModelPromotionStatus.APPROVED.value, ModelPromotionStatus.ACTIVE.value):
            raise ValueError(
                f"Cannot promote version '{version_id}' from status '{entry['promotion_status']}'; must pass evaluation gate (APPROVED) first."
            )

        previous_active = self.active_version_id
        if previous_active and previous_active != version_id and previous_active in self._versions:
            self._versions[previous_active]["promotion_status"] = ModelPromotionStatus.ARCHIVED.value

        entry["rollback_reference"] = previous_active
        entry["promotion_status"] = ModelPromotionStatus.ACTIVE.value
        self.active_version_id = version_id
        self._save()
        return entry

    def rollback(self) -> Dict[str, Any]:
        """Reversibly restore the previous ACTIVE model version."""
        if not self.active_version_id or self.active_version_id not in self._versions:
            raise RuntimeError("No active model version available for rollback.")
        current = self._versions[self.active_version_id]
        target_id = current.get("rollback_reference")
        if not target_id or target_id not in self._versions:
            raise RuntimeError(f"Active version '{self.active_version_id}' has no valid rollback_reference.")

        current["promotion_status"] = ModelPromotionStatus.ARCHIVED.value
        restored = self._versions[target_id]
        restored["promotion_status"] = ModelPromotionStatus.ACTIVE.value
        self.active_version_id = target_id
        self._save()
        return restored

    def get_active(self) -> Optional[Dict[str, Any]]:
        if self.active_version_id:
            return self._versions.get(self.active_version_id)
        return None

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        return self._versions.get(version_id)

    def list_versions(self) -> List[Dict[str, Any]]:
        return list(self._versions.values())

    def _save(self):
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        tmp = f"{self.storage_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"active_version_id": self.active_version_id, "versions": self._versions},
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.storage_path)

    def _load(self):
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.active_version_id = raw.get("active_version_id")
                self._versions = dict(raw.get("versions", {}))
        except Exception:
            self.active_version_id = None
            self._versions = {}
