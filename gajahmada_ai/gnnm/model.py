"""
GajahMada Native Neural Model (GNNM) — Custom Decoder-Only Transformer in PyTorch.
Built from scratch with:
- RMSNorm (Pre-Normalization)
- Rotary Positional Embeddings (RoPE)
- Grouped / Multi-Head Causal Self-Attention
- SwiGLU Feed-Forward Network
- Autoregressive next-token generation (temperature + top-k sampling)
"""

import importlib
from typing import Any, Optional, Tuple
from gajahmada_ai.gnnm.config import GNNMConfig

try:
    torch: Any = importlib.import_module("torch")
    nn: Any = importlib.import_module("torch.nn")
    F: Any = importlib.import_module("torch.nn.functional")
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


if HAS_TORCH:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            norm = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(norm + self.eps) * self.weight


    def build_rope_cache(
        seq_len: int, head_dim: int, theta: float = 10000.0, device=None
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(seq_len, device=device).float()
        angles = torch.outer(positions, freqs)
        return torch.cos(angles), torch.sin(angles)


    def apply_rope(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
        # x: [B, n_heads, T, head_dim]
        t = x.size(2)
        c = cos[:t].unsqueeze(0).unsqueeze(0)
        s = sin[:t].unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * c - x_odd * s
        out_odd = x_even * s + x_odd * c
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: GNNMConfig):
            super().__init__()
            self.n_heads = cfg.n_heads
            self.n_kv_heads = cfg.n_kv_heads
            self.head_dim = cfg.head_dim
            self.n_rep = cfg.n_heads // cfg.n_kv_heads

            self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
            self.w_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
            self.w_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
            self.w_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
            self.dropout = cfg.dropout

        def forward(
            self, x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor"
        ) -> "torch.Tensor":
            bsz, seq_len, _ = x.shape
            q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.w_k(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.w_v(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
            return self.w_o(attn_out)


    class SwiGLUFeedForward(nn.Module):
        def __init__(self, cfg: GNNMConfig):
            super().__init__()
            self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


    class GNNMTransformerBlock(nn.Module):
        def __init__(self, cfg: GNNMConfig):
            super().__init__()
            self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.attn = CausalSelfAttention(cfg)
            self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.ffn = SwiGLUFeedForward(cfg)

        def forward(self, x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
            x = x + self.attn(self.attn_norm(x), cos, sin)
            x = x + self.ffn(self.ffn_norm(x))
            return x


    class GNNMTransformer(nn.Module):
        """GajahMada Native Neural Model (GNNM) Decoder-Only Language Model."""

        def __init__(self, cfg: GNNMConfig):
            super().__init__()
            self.cfg = cfg
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.layers = nn.ModuleList([GNNMTransformerBlock(cfg) for _ in range(cfg.n_layers)])
            self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.token_emb.weight

            cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self._init_weights()

        def _init_weights(self) -> None:
            """LLaMA/GPT-style weight initialization so initial CE loss starts near ln(vocab_size)."""
            resid_std = 0.02 / ((2 * self.cfg.n_layers) ** 0.5)
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    std = resid_std if name.endswith(("w_o", "w_down")) else 0.02
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(
            self,
            input_ids: "torch.Tensor",
            targets: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
            x = self.token_emb(input_ids)
            for layer in self.layers:
                x = layer(x, self.rope_cos, self.rope_sin)
            logits = self.lm_head(self.final_norm(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
            return logits, loss

        @torch.no_grad()
        def generate(
            self,
            input_ids: "torch.Tensor",
            max_new_tokens: int = 64,
            temperature: float = 0.7,
            top_k: int = 40,
            repetition_penalty: float = 1.15,
            eos_id: Optional[int] = None,
        ) -> "torch.Tensor":
            self.eval()
            out = input_ids
            for _ in range(max_new_tokens):
                ctx = out[:, -self.cfg.max_seq_len :]
                logits, _ = self.forward(ctx)
                next_logits = logits[:, -1, :]
                if repetition_penalty > 1.0:
                    for b in range(out.size(0)):
                        recent = set(out[b, -32:].tolist())
                        for tok_id in recent:
                            if next_logits[b, tok_id] > 0:
                                next_logits[b, tok_id] /= repetition_penalty
                            else:
                                next_logits[b, tok_id] *= repetition_penalty
                next_logits = next_logits / max(temperature, 1e-4)
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                out = torch.cat([out, next_tok], dim=1)
                if eos_id is not None and int(next_tok[0, 0].item()) == eos_id:
                    break
            return out
else:
    class GNNMTransformer:  # type: ignore
        def __init__(self, cfg: GNNMConfig):
            raise ImportError(
                "PyTorch (`torch`) is required to instantiate GNNMTransformer. "
                "Install it with: pip install torch"
            )
