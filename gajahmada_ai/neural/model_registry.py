import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.neural.interfaces import InferenceBackend, ModelCapabilityMetadata


class ModelRegistry:
    """
    Registry of local neural models and inference backends with version tracking
    and capability metadata. External cloud providers are disabled by default.
    """

    def __init__(self):
        self._backends: Dict[str, InferenceBackend] = {}
        self._metadata: Dict[str, ModelCapabilityMetadata] = {}
        self._version_history: List[Dict[str, Any]] = []
        self.active_model_id: Optional[str] = None

    def register_backend(
        self,
        name: str,
        backend: InferenceBackend,
        metadata: Optional[ModelCapabilityMetadata] = None,
        set_active: bool = False,
    ):
        meta = metadata or getattr(backend, "metadata", None)
        if meta and meta.is_external_provider:
            raise PermissionError(
                "External cloud AI providers are disabled by default per GajahMada Sovereign Policy."
            )
        self._backends[name] = backend
        if meta:
            self._metadata[name] = meta
        if set_active or self.active_model_id is None:
            self.active_model_id = name

        self._version_history.append({
            "backend_name": name,
            "model_id": meta.model_id if meta else name,
            "version": meta.version if meta else "1.0.0",
            "timestamp": round(time.time(), 3),
            "action": "REGISTERED",
        })

    def get_backend(self, name: Optional[str] = None) -> Optional[InferenceBackend]:
        target = name or self.active_model_id
        if target and target in self._backends:
            return self._backends[target]
        return None

    def set_active_backend(self, name: str) -> bool:
        if name not in self._backends:
            return False
        old = self.active_model_id
        self.active_model_id = name
        meta = self._metadata.get(name)
        self._version_history.append({
            "backend_name": name,
            "previous_backend": old,
            "model_id": meta.model_id if meta else name,
            "version": meta.version if meta else "1.0.0",
            "timestamp": round(time.time(), 3),
            "action": "ACTIVATED",
        })
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for name, backend in self._backends.items():
            h = backend.health()
            meta = self._metadata.get(name)
            out.append({
                "name": name,
                "model_id": meta.model_id if meta else name,
                "backend_type": meta.backend_type if meta else h.get("backend"),
                "version": meta.version if meta else "1.0.0",
                "is_active_route": name == self.active_model_id,
                "neural_inference_active": h.get("neural_inference_active", False),
                "health": h,
            })
        return out

    def get_version_history(self) -> List[Dict[str, Any]]:
        return list(self._version_history)
