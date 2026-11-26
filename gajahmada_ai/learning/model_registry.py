"""
GajahMada AI — Versioned GNNM Model Registry with Strict State Transitions,
Checkpoint Integrity Verification, and Atomic Runtime Coupling (Phase 1 & 3).

Allowed Lifecycle Transitions:
  DEVELOPMENT -> TRAINING -> EVALUATING -> APPROVED -> ACTIVE -> ARCHIVED
  (TRAINING / EVALUATING / APPROVED may also transition to REJECTED)
  (ARCHIVED may transition back to ACTIVE via controlled rollback)

Invalid transitions raise `ValueError`.
Promoting to `ACTIVE` requires:
1. `operator_approved=True`
2. Current status `APPROVED`
3. Valid, loadable checkpoint on disk matching recorded SHA-256 checksum
4. When a runtime `ModelRouter` is attached, safe staging load + smoke test + atomic switch
   (if staging load fails, the previous `ACTIVE` model is preserved untouched!).
"""
from enum import Enum
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class ModelPromotionStatus(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    ModelPromotionStatus.DEVELOPMENT.value: {
        ModelPromotionStatus.TRAINING.value,
        ModelPromotionStatus.APPROVED.value,  # Only for pre-verified baseline registration
        ModelPromotionStatus.REJECTED.value,
        ModelPromotionStatus.ARCHIVED.value,
    },
    ModelPromotionStatus.TRAINING.value: {
        ModelPromotionStatus.EVALUATING.value,
        ModelPromotionStatus.REJECTED.value,
    },
    ModelPromotionStatus.EVALUATING.value: {
        ModelPromotionStatus.APPROVED.value,
        ModelPromotionStatus.REJECTED.value,
    },
    ModelPromotionStatus.APPROVED.value: {
        ModelPromotionStatus.ACTIVE.value,
        ModelPromotionStatus.REJECTED.value,
        ModelPromotionStatus.ARCHIVED.value,
    },
    ModelPromotionStatus.ACTIVE.value: {
        ModelPromotionStatus.ARCHIVED.value,
    },
    ModelPromotionStatus.REJECTED.value: {
        ModelPromotionStatus.ARCHIVED.value,
    },
    ModelPromotionStatus.ARCHIVED.value: {
        ModelPromotionStatus.ACTIVE.value,
    },
}


def compute_checkpoint_checksum(checkpoint_dir: str) -> Tuple[bool, str, Dict[str, str]]:
    """
    Compute deterministic SHA-256 checksums for mandatory GNNM checkpoint artifacts
    (`config.json`, `tokenizer.json`, `model.pt`).
    Returns (is_complete, combined_sha256, per_file_hashes).
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return False, "", {}
    mandatory = ("config.json", "tokenizer.json", "model.pt")
    file_hashes: Dict[str, str] = {}
    for fname in mandatory:
        fpath = os.path.join(checkpoint_dir, fname)
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            return False, "", file_hashes
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        file_hashes[fname] = h.hexdigest()

    # Validate config.json & tokenizer.json parse cleanly as JSON
    try:
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(checkpoint_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            tok = json.load(f)
        if not isinstance(cfg, dict) or "d_model" not in cfg or not isinstance(tok, dict):
            return False, "", file_hashes
    except Exception:
        return False, "", file_hashes

    combined = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return True, combined, file_hashes


class GNNMModelRegistry:
    """Persistent atomic JSON registry for GNNM checkpoint versions, runtime switching, and rollback."""

    def __init__(self, storage_path: Optional[str] = None, runtime_router: Optional[Any] = None):
        self.storage_path = storage_path
        self._runtime_router = runtime_router
        self._switch_lock = threading.Lock()
        self._versions: Dict[str, Dict[str, Any]] = {}
        self.active_version_id: Optional[str] = None
        if self.storage_path and os.path.exists(self.storage_path):
            self._load()

    def attach_runtime_router(self, model_router: Any):
        """Couple the ModelRegistry to the live ModelRouter for atomic runtime model switching."""
        self._runtime_router = model_router

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
        checkpoint_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        ckpt_ok, computed_sha, file_hashes = compute_checkpoint_checksum(checkpoint_path)
        entry = {
            "version_id": version_id,
            "model_name": model_name,
            "architecture": architecture,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256 or computed_sha,
            "checkpoint_files_sha256": file_hashes,
            "checkpoint_verified": ckpt_ok,
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
        checkpoint_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._versions.get(version_id)
        if entry is None:
            raise KeyError(f"Model version '{version_id}' not found.")
        current_st = entry["promotion_status"]
        target_st = status.value if isinstance(status, ModelPromotionStatus) else str(status)
        if current_st != target_st:
            allowed = ALLOWED_TRANSITIONS.get(current_st, set())
            if target_st not in allowed:
                raise ValueError(
                    f"Invalid model lifecycle transition for '{version_id}': {current_st} -> {target_st}"
                )
        entry["promotion_status"] = target_st
        if evaluation_metrics is not None:
            entry["evaluation_metrics"] = dict(evaluation_metrics)
        if checkpoint_sha256 is not None:
            entry["checkpoint_sha256"] = checkpoint_sha256
        self._save()
        return entry

    def _stage_and_switch_runtime(self, checkpoint_path: str, version_id: str) -> Tuple[bool, str]:
        """
        Safely stage candidate checkpoint on a fresh PyTorchNativeBackend, run a smoke inference
        check (when PyTorch is installed), and atomically swap into the live ModelRouter.
        If staging fails, the previous runtime backend remains untouched!
        """
        if self._runtime_router is None:
            return True, "no_runtime_router_attached"

        from gajahmada_ai.neural.pytorch_backend import PyTorchNativeBackend

        staged = PyTorchNativeBackend(model_path=checkpoint_path)
        if staged._torch is None:
            # PyTorch unavailable in this process; record metadata without faking a neural load
            return True, "pytorch_unavailable_metadata_only"

        if not staged.load_model(checkpoint_path):
            return False, f"Staged checkpoint failed to load: {staged._load_error}"

        smoke = staged.generate("GajahMada", max_tokens=8, temperature=0.5)
        if not smoke.neural_used:
            return False, f"Staged checkpoint failed inference smoke test: {smoke.warning}"

        staged.metadata.version = version_id
        with self._switch_lock:
            self._runtime_router.native_backend = staged
            self._runtime_router.registry.register_backend("gajahmada-native", staged, set_active=True)
        return True, "switched_atomically"

    def promote(self, version_id: str, operator_approved: bool = False) -> Dict[str, Any]:
        """Controlled promotion: requires APPROVED state, valid checkpoint integrity, & explicit approval."""
        entry = self._versions.get(version_id)
        if entry is None:
            raise KeyError(f"Model version '{version_id}' not found.")
        if not operator_approved:
            raise PermissionError("Model promotion requires explicit operator approval.")
        if entry["promotion_status"] not in (ModelPromotionStatus.APPROVED.value, ModelPromotionStatus.ACTIVE.value):
            raise ValueError(
                f"Cannot promote version '{version_id}' from status '{entry['promotion_status']}'; must pass evaluation gate (APPROVED) first."
            )

        # Verify checkpoint exists on disk and integrity matches recorded SHA-256
        ckpt_ok, live_sha, _ = compute_checkpoint_checksum(entry["checkpoint_path"])
        if not ckpt_ok:
            raise ValueError(
                f"Cannot promote '{version_id}': checkpoint at '{entry['checkpoint_path']}' is missing or invalid."
            )
        recorded_sha = entry.get("checkpoint_sha256", "")
        if recorded_sha and recorded_sha != live_sha:
            raise ValueError(
                f"Cannot promote '{version_id}': checkpoint checksum mismatch (tampered or modified on disk)."
            )
        entry["checkpoint_sha256"] = live_sha
        entry["checkpoint_verified"] = True

        # Stage & atomically switch runtime if attached
        switched, switch_note = self._stage_and_switch_runtime(entry["checkpoint_path"], version_id)
        if not switched:
            raise RuntimeError(f"Runtime staging failed for '{version_id}': {switch_note}")

        previous_active = self.active_version_id
        if previous_active and previous_active != version_id and previous_active in self._versions:
            self._versions[previous_active]["promotion_status"] = ModelPromotionStatus.ARCHIVED.value

        entry["rollback_reference"] = previous_active
        entry["promotion_status"] = ModelPromotionStatus.ACTIVE.value
        entry["runtime_switch_note"] = switch_note
        self.active_version_id = version_id
        self._save()
        return entry

    def rollback(self) -> Dict[str, Any]:
        """Reversibly restore the previous ACTIVE model version and atomically switch runtime."""
        if not self.active_version_id or self.active_version_id not in self._versions:
            raise RuntimeError("No active model version available for rollback.")
        current = self._versions[self.active_version_id]
        target_id = current.get("rollback_reference")
        if not target_id or target_id not in self._versions:
            raise RuntimeError(f"Active version '{self.active_version_id}' has no valid rollback_reference.")

        restored = self._versions[target_id]
        ckpt_ok, live_sha, _ = compute_checkpoint_checksum(restored["checkpoint_path"])
        if not ckpt_ok:
            raise RuntimeError(f"Rollback target '{target_id}' checkpoint is missing or invalid.")

        switched, switch_note = self._stage_and_switch_runtime(restored["checkpoint_path"], target_id)
        if not switched:
            raise RuntimeError(f"Rollback runtime switch failed for '{target_id}': {switch_note}")

        current["promotion_status"] = ModelPromotionStatus.ARCHIVED.value
        restored["promotion_status"] = ModelPromotionStatus.ACTIVE.value
        restored["checkpoint_sha256"] = live_sha
        restored["runtime_switch_note"] = switch_note
        self.active_version_id = target_id
        self._save()
        return restored

    def runtime_status(self) -> Dict[str, Any]:
        active = self.get_active()
        return {
            "active_version_id": self.active_version_id,
            "active_checkpoint_path": active.get("checkpoint_path") if active else None,
            "active_checkpoint_sha256": active.get("checkpoint_sha256") if active else None,
            "runtime_router_coupled": self._runtime_router is not None,
            "total_versions": len(self._versions),
        }

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
