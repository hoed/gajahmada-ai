from dataclasses import asdict, dataclass
import json
from typing import Any, Dict


@dataclass
class GNNMConfig:
    """
    Modular, scalable configuration for GajahMada Native Neural Model (GNNM)
    Decoder-Only Transformer.

    Architecture baseline:
      Pre-Norm RMSNorm + Rotary Positional Embeddings (RoPE) +
      Grouped/Multi-Head Causal Self-Attention + SwiGLU Feed-Forward.

    Note: A configuration defines an architecture specification, NOT trained weights.
    """

    name: str = "GNNM-01-25M"
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 6
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    language_profile: str = "multilingual-balanced-id-en"
    intelligence_mode: str = "hybrid-cognitive"
    # Extended modular architecture fields (backward-compatible defaults)
    family: str = "BUMBLECORE"
    specialization: str = "GENERAL"
    architecture_type: str = "dense"  # "dense" | "moe"
    positional_encoding: str = "rope"
    attention_type: str = "gqa"  # "gqa" | "mha"
    normalization: str = "rmsnorm"
    activation: str = "swiglu"
    precision: str = "bf16"  # "fp32" | "bf16" | "fp16"
    gradient_checkpointing: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    lifecycle_status: str = "ARCHITECTURE_DESIGNED"

    def __post_init__(self):
        if self.vocab_size < 16:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be >= 16")
        if self.d_model < 16 or self.n_layers < 1 or self.n_heads < 1 or self.n_kv_heads < 1:
            raise ValueError("Model dimensions (d_model, n_layers, n_heads, n_kv_heads) must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.architecture_type not in ("dense", "moe"):
            raise ValueError(f"Unsupported architecture_type '{self.architecture_type}' (expected 'dense' or 'moe')")
        if self.architecture_type == "moe":
            if self.num_experts < 2:
                raise ValueError("MoE architecture requires num_experts >= 2")
            if not (1 <= self.num_experts_per_tok <= self.num_experts):
                raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        else:
            self.num_experts = 1
            self.num_experts_per_tok = 1
        self.attention_type = "mha" if self.n_kv_heads == self.n_heads else "gqa"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def parameter_breakdown(self) -> Dict[str, int]:
        """Exact analytical parameter breakdown (separate accounting for Dense vs. MoE)."""
        embed_params = self.vocab_size * self.d_model
        lm_head_params = 0 if self.tie_embeddings else embed_params

        q_proj = self.d_model * self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        k_proj = self.d_model * kv_dim
        v_proj = self.d_model * kv_dim
        o_proj = self.d_model * self.d_model
        attn_per_layer = q_proj + k_proj + v_proj + o_proj

        # SwiGLU uses 3 linear projections: w_gate, w_up, w_down
        single_ffn = 3 * self.d_model * self.d_ff
        if self.architecture_type == "moe":
            router_per_layer = self.d_model * self.num_experts
            ffn_per_layer_total = router_per_layer + (self.num_experts * single_ffn)
            ffn_per_layer_active = router_per_layer + (self.num_experts_per_tok * single_ffn)
        else:
            router_per_layer = 0
            ffn_per_layer_total = single_ffn
            ffn_per_layer_active = single_ffn

        norms_per_layer = 2 * self.d_model
        final_norm = self.d_model

        total_attn = self.n_layers * attn_per_layer
        total_ffn = self.n_layers * ffn_per_layer_total
        active_ffn = self.n_layers * ffn_per_layer_active
        total_norms = (self.n_layers * norms_per_layer) + final_norm

        total_params = embed_params + lm_head_params + total_attn + total_ffn + total_norms
        active_params = embed_params + lm_head_params + total_attn + active_ffn + total_norms
        return {
            "embedding_params": embed_params,
            "lm_head_params": lm_head_params,
            "attention_params": total_attn,
            "ffn_params_total": total_ffn,
            "ffn_params_active": active_ffn,
            "router_params": self.n_layers * router_per_layer,
            "norm_params": total_norms,
            "total_parameters": total_params,
            "active_parameters": active_params,
        }

    def estimate_parameters(self) -> int:
        """Exact analytical total parameter count for this GNNM Transformer configuration."""
        return self.parameter_breakdown()["total_parameters"]

    def estimate_active_parameters(self) -> int:
        """Analytical active parameter count per forward pass (equals total for dense models)."""
        return self.parameter_breakdown()["active_parameters"]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        bd = self.parameter_breakdown()
        d["estimated_parameters"] = bd["total_parameters"]
        d["active_parameters"] = bd["active_parameters"]
        d["parameter_breakdown"] = bd
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GNNMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for computed in ("estimated_parameters", "active_parameters", "parameter_breakdown"):
            data.pop(computed, None)
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ─── GNNM MODEL FAMILY PRESETS ────────────────────────────────────────────────
# Preserves legacy GNNM-01..04 aliases AND formalizes BUMBLECORE, OPTICORE, and TITANFORGE.
# IMPORTANT: Presets are architecture configurations, NOT claims of trained model weights.

GNNM_PRESETS: Dict[str, GNNMConfig] = {
    # Legacy / Research Smoke-Test Presets
    "GNNM-01-Tiny": GNNMConfig(
        name="GNNM-01-Tiny", family="BUMBLECORE", vocab_size=1024, d_model=128, n_layers=4,
        n_heads=4, n_kv_heads=4, d_ff=384, max_seq_len=256, precision="fp32",
        lifecycle_status="CONFIGURATION_VALIDATED",
    ),
    "GNNM-01-12M": GNNMConfig(
        name="GNNM-01-12M", family="BUMBLECORE", vocab_size=4096, d_model=384, n_layers=6,
        n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=512, precision="fp32",
        lifecycle_status="CONFIGURATION_VALIDATED",
    ),
    "GNNM-01-25M": GNNMConfig(
        name="GNNM-01-25M", family="BUMBLECORE", vocab_size=8192, d_model=384, n_layers=8,
        n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=1024, precision="bf16",
        lifecycle_status="CONFIGURATION_VALIDATED",
    ),
    "GNNM-02-100M": GNNMConfig(
        name="GNNM-02-100M", family="BUMBLECORE", vocab_size=32000, d_model=768, n_layers=12,
        n_heads=12, n_kv_heads=4, d_ff=2048, max_seq_len=2048, precision="bf16",
        lifecycle_status="ARCHITECTURE_DESIGNED",
    ),
    "GNNM-03-300M": GNNMConfig(
        name="GNNM-03-300M", family="BUMBLECORE", vocab_size=32000, d_model=1024, n_layers=20,
        n_heads=16, n_kv_heads=4, d_ff=2816, max_seq_len=4096, precision="bf16",
        lifecycle_status="ARCHITECTURE_DESIGNED",
    ),
    "GNNM-04-1B": GNNMConfig(
        name="GNNM-04-1B", family="OPTICORE", vocab_size=48000, d_model=2048, n_layers=22,
        n_heads=16, n_kv_heads=4, d_ff=5632, max_seq_len=4096, precision="bf16",
        lifecycle_status="ARCHITECTURE_DESIGNED",
    ),

    # ── BUMBLECORE (Lightweight Family: 12M, 100M, 300M) ──
    "BUMBLECORE-12M": GNNMConfig(
        name="BUMBLECORE-12M", family="BUMBLECORE", vocab_size=4096, d_model=384, n_layers=6,
        n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=512, precision="fp32",
        lifecycle_status="CONFIGURATION_VALIDATED",
    ),
    "BUMBLECORE-100M": GNNMConfig(
        name="BUMBLECORE-100M", family="BUMBLECORE", vocab_size=32000, d_model=768, n_layers=12,
        n_heads=12, n_kv_heads=4, d_ff=2048, max_seq_len=2048, precision="bf16",
        lifecycle_status="ARCHITECTURE_DESIGNED",
    ),
    "BUMBLECORE-300M": GNNMConfig(
        name="BUMBLECORE-300M", family="BUMBLECORE", vocab_size=32000, d_model=1024, n_layers=20,
        n_heads=16, n_kv_heads=4, d_ff=2816, max_seq_len=4096, precision="bf16",
        lifecycle_status="ARCHITECTURE_DESIGNED",
    ),

    # ── OPTICORE (General-Purpose Family: 1B, 3B, 7B) ──
    "OPTICORE-1B": GNNMConfig(
        name="OPTICORE-1B", family="OPTICORE", vocab_size=48000, d_model=2048, n_layers=22,
        n_heads=16, n_kv_heads=4, d_ff=5632, max_seq_len=4096, precision="bf16",
        gradient_checkpointing=True, lifecycle_status="ARCHITECTURE_DESIGNED",
    ),
    "OPTICORE-3B": GNNMConfig(
        name="OPTICORE-3B", family="OPTICORE", vocab_size=64000, d_model=3072, n_layers=28,
        n_heads=24, n_kv_heads=8, d_ff=8192, max_seq_len=8192, rope_theta=500000.0,
        tie_embeddings=False, precision="bf16", gradient_checkpointing=True,
        lifecycle_status="PLANNED",
    ),
    "OPTICORE-7B": GNNMConfig(
        name="OPTICORE-7B", family="OPTICORE", vocab_size=64000, d_model=4096, n_layers=32,
        n_heads=32, n_kv_heads=8, d_ff=11008, max_seq_len=8192, rope_theta=500000.0,
        tie_embeddings=False, precision="bf16", gradient_checkpointing=True,
        lifecycle_status="PLANNED",
    ),

    # ── TITANFORGE (Large-Scale Native Family: 30B, 70B, 120B+) ──
    "TITANFORGE-30B": GNNMConfig(
        name="TITANFORGE-30B", family="TITANFORGE", vocab_size=128000, d_model=6144, n_layers=56,
        n_heads=48, n_kv_heads=8, d_ff=18432, max_seq_len=8192, rope_theta=500000.0,
        tie_embeddings=False, precision="bf16", gradient_checkpointing=True,
        lifecycle_status="PLANNED",
    ),
    # Exact ~70.55B Dense Transformer Baseline (80 layers, d_model=8192, GQA 64:8, SwiGLU 28672)
    "TITANFORGE-70B": GNNMConfig(
        name="TITANFORGE-70B", family="TITANFORGE", vocab_size=128000, d_model=8192, n_layers=80,
        n_heads=64, n_kv_heads=8, d_ff=28672, max_seq_len=8192, rope_theta=500000.0,
        tie_embeddings=False, precision="bf16", gradient_checkpointing=True,
        lifecycle_status="ARCHITECTURE_DESIGNED",
    ),
    # Experimental MoE variant (separate architecture & parameter accounting: ~72B total / ~15.8B active)
    "TITANFORGE-70B-MoE-Exp": GNNMConfig(
        name="TITANFORGE-70B-MoE-Exp", family="TITANFORGE", architecture_type="moe",
        vocab_size=128000, d_model=5120, n_layers=48, n_heads=40, n_kv_heads=8,
        d_ff=12288, max_seq_len=8192, rope_theta=500000.0, tie_embeddings=False,
        precision="bf16", gradient_checkpointing=True, num_experts=8, num_experts_per_tok=2,
        lifecycle_status="PLANNED",
    ),
    "TITANFORGE-120B": GNNMConfig(
        name="TITANFORGE-120B", family="TITANFORGE", vocab_size=128000, d_model=10240, n_layers=88,
        n_heads=80, n_kv_heads=8, d_ff=35840, max_seq_len=8192, rope_theta=500000.0,
        tie_embeddings=False, precision="bf16", gradient_checkpointing=True,
        lifecycle_status="PLANNED",
    ),
}
