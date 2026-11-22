from typing import Any, Dict, List, Optional

from gajahmada_ai.neural.interfaces import GenerationResult, InferenceBackend
from gajahmada_ai.neural.llama_backend import LlamaCppBackend, deterministic_hash_embed
from gajahmada_ai.neural.model_registry import ModelRegistry
from gajahmada_ai.neural.onnx_backend import OnnxBackend
from gajahmada_ai.neural.pytorch_backend import PyTorchNativeBackend


class ModelRouter:
    """
    Task-based neural model router and honest capability reporter.
    Routes requests to the active loaded neural backend (`llama.cpp`, `onnx`, `gajahmada-native`).
    When no neural model weights are loaded, explicitly reports `neural_inference_active: False`
    and never masquerades a rule-based response as neural inference.
    """

    def __init__(
        self,
        gguf_model_path: str = "./models/llama-2-7b-chat.Q4_K_M.gguf",
        onnx_model_path: str = "./models/gajahmada.onnx",
        native_model_path: str = "./models/gajahmada_native.gmweights.json",
        context_window: int = 2048,
    ):
        self.registry = ModelRegistry()
        self.llama_backend = LlamaCppBackend(model_path=gguf_model_path, context_window=context_window)
        self.onnx_backend = OnnxBackend(model_path=onnx_model_path, context_window=context_window)
        self.native_backend = PyTorchNativeBackend(model_path=native_model_path, context_window=context_window)

        self.registry.register_backend("llama.cpp", self.llama_backend, set_active=True)
        self.registry.register_backend("onnx", self.onnx_backend)
        self.registry.register_backend("gajahmada-native", self.native_backend)

    def load_available_models(self) -> Dict[str, bool]:
        results = {
            "llama.cpp": self.llama_backend.load_model(),
            "onnx": self.onnx_backend.load_model(),
            "gajahmada-native": self.native_backend.load_model(),
        }
        # Route automatically to the first loaded neural backend if available
        for name in ("llama.cpp", "gajahmada-native", "onnx"):
            if results.get(name):
                self.registry.set_active_backend(name)
                break
        return results

    def get_active_neural_backend(self) -> Optional[InferenceBackend]:
        primary = self.registry.get_backend()
        if primary and primary.health().get("neural_inference_active"):
            return primary
        for name in ("llama.cpp", "gajahmada-native", "onnx"):
            b = self.registry.get_backend(name)
            if b and b.health().get("neural_inference_active"):
                return b
        return None

    def route_generation(
        self,
        prompt: str,
        context: str = "",
        task_type: str = "general",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        backend = self.get_active_neural_backend()
        if backend is not None:
            return backend.generate(
                prompt=prompt,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # Explicitly report that no neural backend is loaded
        primary = self.registry.get_backend() or self.llama_backend
        return primary.generate(
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def embed(self, text: str) -> List[float]:
        backend = self.get_active_neural_backend() or self.llama_backend
        return backend.embed(text)

    def capability_report(self) -> Dict[str, Any]:
        active_neural = self.get_active_neural_backend()
        active_health = active_neural.health() if active_neural else None
        return {
            "neural_inference_active": active_neural is not None,
            "active_neural_backend": active_health["backend"] if active_health else None,
            "active_model_id": active_health["model_id"] if active_health else None,
            "symbolic_reasoning_active": True,
            "evidence_reasoning_active": True,
            "metacognitive_critic_active": True,
            "embedding_dimension": 384,
            "embedding_backend": active_health.get("embedding_mode", "deterministic-hash-384d") if active_health else "deterministic-hash-384d",
            "external_ai_providers_enabled": False,
            "registered_backends": self.registry.list_models(),
        }
