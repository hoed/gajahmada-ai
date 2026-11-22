"""GajahMada 2.0 Provider-Independent Neural Inference Package."""
from gajahmada_ai.neural.interfaces import (
    GenerationResult,
    InferenceBackend,
    ModelCapabilityMetadata,
)
from gajahmada_ai.neural.llama_backend import LlamaCppBackend
from gajahmada_ai.neural.model_registry import ModelRegistry
from gajahmada_ai.neural.onnx_backend import OnnxBackend
from gajahmada_ai.neural.pytorch_backend import PyTorchNativeBackend
from gajahmada_ai.neural.router import ModelRouter

__all__ = [
    "InferenceBackend",
    "ModelCapabilityMetadata",
    "GenerationResult",
    "LlamaCppBackend",
    "OnnxBackend",
    "PyTorchNativeBackend",
    "ModelRegistry",
    "ModelRouter",
]
