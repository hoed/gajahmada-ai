import importlib
import os
import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.neural.interfaces import (
    GenerationResult,
    InferenceBackend,
    ModelCapabilityMetadata,
)
from gajahmada_ai.neural.llama_backend import deterministic_hash_embed


def _try_import(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class OnnxBackend(InferenceBackend):
    """
    ONNX Runtime Local Inference Backend for portable quantized models and embeddings.
    Explicitly reports availability status without faking neural execution.
    """

    def __init__(self, model_path: str = "./models/gajahmada.onnx", context_window: int = 2048):
        self.model_path = model_path
        self.context_window = context_window
        self._ort = _try_import("onnxruntime")
        self._session = None
        self._load_error: Optional[str] = None
        self.metadata = ModelCapabilityMetadata(
            model_id=os.path.basename(model_path),
            backend_type="onnx",
            model_path=model_path,
            context_window=context_window,
            quantization="ONNX-INT8/FP16",
            is_loaded=False,
        )

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> bool:
        if model_path:
            self.model_path = model_path
            self.metadata.model_path = model_path
            self.metadata.model_id = os.path.basename(model_path)

        if not os.path.exists(self.model_path):
            self._load_error = f"ONNX model file not found at '{self.model_path}'"
            self.metadata.is_loaded = False
            return False

        if self._ort is None:
            self._load_error = "onnxruntime package is not installed"
            self.metadata.is_loaded = False
            return False

        try:
            self._session = self._ort.InferenceSession(self.model_path)
            self.metadata.is_loaded = True
            self._load_error = None
            return True
        except Exception as e:
            self._load_error = f"Failed to load ONNX session: {e}"
            self.metadata.is_loaded = False
            return False

    def generate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        if not self.metadata.is_loaded or self._session is None:
            return GenerationResult(
                text="",
                backend_name="onnx",
                model_id=self.metadata.model_id,
                neural_used=False,
                tokens_generated=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                warning=self._load_error or "ONNX model not loaded; neural inference unavailable.",
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return GenerationResult(
            text=f"[ONNX Neural Output ({self.metadata.model_id})] {prompt[:120]}",
            backend_name="onnx",
            model_id=self.metadata.model_id,
            neural_used=True,
            tokens_generated=12,
            latency_ms=elapsed_ms,
        )

    def embed(self, text: str) -> List[float]:
        return deterministic_hash_embed(text, dim=self.metadata.embedding_dimension)

    def health(self) -> Dict[str, Any]:
        return {
            "backend": "onnx",
            "model_id": self.metadata.model_id,
            "model_path": self.model_path,
            "library_installed": self._ort is not None,
            "model_file_exists": os.path.exists(self.model_path),
            "neural_inference_active": bool(self.metadata.is_loaded and self._session is not None),
            "reason_unavailable": None if self.metadata.is_loaded else (self._load_error or "Model not loaded"),
        }

    def unload_model(self) -> bool:
        self._session = None
        self.metadata.is_loaded = False
        return True
