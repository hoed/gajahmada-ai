import importlib
import json
import math
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


class PyTorchNativeBackend(InferenceBackend):
    """
    PyTorch & GajahMada-Native Neural Research Backend.
    Supports loading either PyTorch checkpoints (`.pt` / `.pth`) or GajahMada
    NumPy/JSON neural weight matrices (`.gmweights.json`) for local neural
    representation and classification research without external cloud dependencies.
    """

    def __init__(self, model_path: str = "./models/gajahmada_native.gmweights.json", context_window: int = 2048):
        self.model_path = model_path
        self.context_window = context_window
        self._torch = _try_import("torch")
        self._np = _try_import("numpy")
        self._weights: Optional[Dict[str, Any]] = None
        self._gnnm_engine: Optional[Any] = None
        self._load_error: Optional[str] = None
        self.metadata = ModelCapabilityMetadata(
            model_id=os.path.basename(model_path),
            backend_type="gajahmada-native",
            model_path=model_path,
            context_window=context_window,
            quantization="FP32-Native",
            is_loaded=False,
        )

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> bool:
        if model_path:
            self.model_path = model_path
            self.metadata.model_path = model_path
            self.metadata.model_id = os.path.basename(model_path)

        # Auto-discover trained GNNM checkpoint directory (./models/gnnm-*) only when using default placeholder path
        if (
            not os.path.exists(self.model_path)
            and os.path.normpath(self.model_path) == os.path.normpath("./models/gajahmada_native.gmweights.json")
            and os.path.isdir("./models")
        ):
            for entry in sorted(os.listdir("./models")):
                cand = os.path.join("./models", entry)
                if entry.startswith("gnnm-") and os.path.isdir(cand) and os.path.exists(os.path.join(cand, "model.pt")):
                    self.model_path = cand
                    self.metadata.model_path = cand
                    self.metadata.model_id = entry
                    break

        if not os.path.exists(self.model_path):
            self._load_error = f"Native neural weight file not found at '{self.model_path}'"
            self.metadata.is_loaded = False
            return False

        try:
            # Support loading a trained GNNM checkpoint directory (config.json + tokenizer.json + model.pt)
            if os.path.isdir(self.model_path) and os.path.exists(os.path.join(self.model_path, "model.pt")):
                if self._torch is None:
                    self._load_error = "PyTorch package is not installed for GNNM checkpoint loading"
                    self.metadata.is_loaded = False
                    return False
                from gajahmada_ai.gnnm.inference import GNNMInferenceEngine

                self._gnnm_engine = GNNMInferenceEngine(self.model_path)
                self._weights = {"type": "gnnm_transformer", "name": self._gnnm_engine.cfg.name}
                self.metadata.model_id = self._gnnm_engine.cfg.name
                self.metadata.context_window = self._gnnm_engine.cfg.max_seq_len
                self.metadata.quantization = "FP32-GNNM-Native"
                self.metadata.is_loaded = True
                self._load_error = None
                return True
            if self.model_path.endswith(".json"):
                with open(self.model_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "weights" not in data:
                    raise ValueError("Missing 'weights' matrix in native model file")
                self._weights = data
                self.metadata.version = data.get("version", "1.0.0")
                self.metadata.is_loaded = True
                self._load_error = None
                return True
            elif self._torch is not None:
                self._weights = {"torch_state": self._torch.load(self.model_path, map_location="cpu", weights_only=True)}
                self.metadata.is_loaded = True
                self._load_error = None
                return True
            else:
                self._load_error = "PyTorch package is not installed for .pt/.pth checkpoint loading"
                self.metadata.is_loaded = False
                return False
        except Exception as e:
            self._load_error = f"Failed to load native neural weights: {e}"
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
        if not self.metadata.is_loaded or self._weights is None:
            return GenerationResult(
                text="",
                backend_name="gajahmada-native",
                model_id=self.metadata.model_id,
                neural_used=False,
                tokens_generated=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                warning=self._load_error or "Native neural weights not loaded.",
            )

        if self._gnnm_engine is not None:
            # Token-budget-aware context construction respecting GNNM max_seq_len
            # (replaces legacy fixed 240-char truncation without altering checkpoint format)
            max_seq = int(self._gnnm_engine.cfg.max_seq_len)
            gen_budget = min(max(16, int(max_tokens)), max(16, max_seq // 3), 96)
            tok = self._gnnm_engine.tokenizer
            prompt_ids = tok.encode(prompt, add_bos=False, add_eos=False)
            if context:
                ctx_budget = max(0, max_seq - gen_budget - len(prompt_ids) - 8)
                ctx_ids = tok.encode(context, add_bos=False, add_eos=False)
                if len(ctx_ids) > ctx_budget:
                    ctx_ids = ctx_ids[:ctx_budget]
                bounded_ctx = tok.decode(ctx_ids, skip_special_tokens=True).strip() if ctx_ids else ""
                full_prompt = f"{bounded_ctx}\n{prompt}" if bounded_ctx else prompt
            else:
                full_prompt = prompt

            text = self._gnnm_engine.generate(
                full_prompt,
                max_new_tokens=gen_budget,
                temperature=min(max(temperature, 0.2), 1.0),
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return GenerationResult(
                text=text,
                backend_name="gajahmada-native",
                model_id=self.metadata.model_id,
                neural_used=True,
                tokens_generated=len(text.split()),
                latency_ms=elapsed_ms,
                metadata={
                    "architecture": "GNNM-DecoderOnly-Transformer",
                    "version": self.metadata.version,
                    "max_seq_len": max_seq,
                    "gen_token_budget": gen_budget,
                },
            )

        # Forward pass over 384-d input embedding using loaded projection weights
        emb = self.embed(prompt)
        bias = float(self._weights.get("bias", 0.0))
        w_vec = self._weights.get("weights", [0.01] * len(emb))
        activation = math.tanh(sum(e * w for e, w in zip(emb, w_vec)) + bias)
        templates = self._weights.get("responses", {})
        text = templates.get(
            "default",
            f"[GajahMada Native Neural Model v{self.metadata.version}] Activation={activation:.4f} | Objective: {prompt[:160]}",
        )
        if context:
            text += f"\nGrounded Context: {context[:300]}"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return GenerationResult(
            text=text,
            backend_name="gajahmada-native",
            model_id=self.metadata.model_id,
            neural_used=True,
            tokens_generated=len(text.split()),
            latency_ms=elapsed_ms,
            metadata={"activation": round(activation, 5), "version": self.metadata.version},
        )

    def embed(self, text: str) -> List[float]:
        base = deterministic_hash_embed(text, dim=self.metadata.embedding_dimension)
        if self.metadata.is_loaded and self._weights and "embed_scale" in self._weights:
            scale = float(self._weights["embed_scale"])
            normed = [v * scale for v in base]
            n = math.sqrt(sum(x * x for x in normed)) or 1.0
            return [round(x / n, 6) for x in normed]
        return base

    def health(self) -> Dict[str, Any]:
        return {
            "backend": "gajahmada-native",
            "model_id": self.metadata.model_id,
            "model_path": self.model_path,
            "pytorch_installed": self._torch is not None,
            "numpy_installed": self._np is not None,
            "model_file_exists": os.path.exists(self.model_path),
            "neural_inference_active": bool(self.metadata.is_loaded and self._weights is not None),
            "version": self.metadata.version,
            "reason_unavailable": None if self.metadata.is_loaded else (self._load_error or "Model not loaded"),
        }

    def unload_model(self) -> bool:
        self._weights = None
        self.metadata.is_loaded = False
        return True
