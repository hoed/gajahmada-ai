from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelCapabilityMetadata:
    model_id: str
    backend_type: str  # "llama.cpp", "onnx", "pytorch", "gajahmada-native"
    model_path: str
    context_window: int = 2048
    embedding_dimension: int = 384
    supports_generation: bool = True
    supports_embedding: bool = True
    quantization: str = "unknown"
    version: str = "1.0.0"
    supported_tasks: List[str] = field(default_factory=lambda: ["general", "reasoning", "embedding"])
    min_ram_mb: int = 512
    is_loaded: bool = False
    is_external_provider: bool = False


@dataclass
class GenerationResult:
    text: str
    backend_name: str
    model_id: str
    neural_used: bool
    tokens_generated: int
    latency_ms: float
    warning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):
    """
    Provider-independent inference backend interface.
    CRITICAL RULE:
    Never pretend neural inference is operational when a neural model is not loaded.
    """

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """Load model weights into memory. Returns True only if real neural weights are loaded."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Run neural generation. If model is not loaded, explicitly set neural_used=False."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate a normalized vector embedding."""
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return explicit operational health and whether real neural inference is active."""
        pass

    @abstractmethod
    def unload_model(self) -> bool:
        """Release model weights from memory."""
        pass
