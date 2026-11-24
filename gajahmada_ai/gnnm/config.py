from dataclasses import asdict, dataclass
import json
from typing import Any, Dict


@dataclass
class GNNMConfig:
    """
    Configuration for GajahMada Native Neural Model (GNNM) Decoder-Only Transformer.
    Architecture: Pre-Norm RMSNorm + Rotary Positional Embeddings (RoPE) +
                  Grouped/Multi-Head Causal Attention + SwiGLU Feed-Forward.
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

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_parameters(self) -> int:
        """Exact analytical parameter count for this GNNM Transformer configuration."""
        embed_params = self.vocab_size * self.d_model
        lm_head_params = 0 if self.tie_embeddings else embed_params

        q_proj = self.d_model * self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        k_proj = self.d_model * kv_dim
        v_proj = self.d_model * kv_dim
        o_proj = self.d_model * self.d_model
        attn_params = q_proj + k_proj + v_proj + o_proj

        # SwiGLU uses 3 linear projections: w_gate, w_up, w_down
        ffn_params = 3 * self.d_model * self.d_ff
        norms_per_layer = 2 * self.d_model

        per_layer = attn_params + ffn_params + norms_per_layer
        final_norm = self.d_model
        return embed_params + lm_head_params + (self.n_layers * per_layer) + final_norm

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["estimated_parameters"] = self.estimate_parameters()
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GNNMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("estimated_parameters", None)
        return cls(**data)


GNNM_PRESETS: Dict[str, GNNMConfig] = {
    # Ultra-fast CPU smoke-test / unit-test preset (~1.0M params)
    "GNNM-01-Tiny": GNNMConfig(
        name="GNNM-01-Tiny", vocab_size=1024, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4, d_ff=384, max_seq_len=256
    ),
    # Stage 1 10M-50M Entry Proof-of-Concept (~12.4M params, CPU-trainable checkpoint proof)
    "GNNM-01-12M": GNNMConfig(
        name="GNNM-01-12M", vocab_size=4096, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=512
    ),
    # Stage 1 Proof-of-Concept (~18M params)
    "GNNM-01-25M": GNNMConfig(
        name="GNNM-01-25M", vocab_size=8192, d_model=384, n_layers=8, n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=1024
    ),
    # Stage 2 Basic Multilingual (~105M params)
    "GNNM-02-100M": GNNMConfig(
        name="GNNM-02-100M", vocab_size=32000, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, d_ff=2048, max_seq_len=2048
    ),
    # Stage 3 Understanding & Generation (~320M params)
    "GNNM-03-300M": GNNMConfig(
        name="GNNM-03-300M", vocab_size=32000, d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4, d_ff=2816, max_seq_len=4096
    ),
    # Stage 4 Independent Sovereign LM (~1.1B params)
    "GNNM-04-1B": GNNMConfig(
        name="GNNM-04-1B", vocab_size=48000, d_model=2048, n_layers=22, n_heads=16, n_kv_heads=4, d_ff=5632, max_seq_len=4096
    ),
}
