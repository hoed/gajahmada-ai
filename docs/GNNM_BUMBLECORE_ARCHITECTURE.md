# GajahMada GNNM BumbleCore 300M — Architecture Specification (Release 2.5)

- **Full Name:** GajahMada GNNM BumbleCore 300M (`GNNM-BumbleCore-300M`)
- **Implementation:** `gajahmada_ai/gnnm/bumblecore/architecture.py` (`BumbleCoreConfig`, `BumbleCoreModel`) extending native `gajahmada_ai/gnnm/model.py` (`GNNMTransformer`).
- **Architecture:** Dense Decoder-Only Autoregressive Transformer (`RMSNorm` + `RoPE` + `Grouped Query Attention 16:4` + `SwiGLU` + `Gradient Checkpointing` + `Tied Embeddings`).

## Exact Parameter Accounting (Verified Against Instantiated PyTorch Model)
| Component | Formula | Exact Parameters |
| :--- | :--- | :---: |
| **Token Embeddings (`vocab=32000, d=1024`)** | `32000 × 1024` | `32,768,000` |
| **Output Projection (`lm_head`)** | Tied with `token_emb` | `0` |
| **Attention (24 layers, Q=16, KV=4, head_dim=64)** | `24 × (1024² + 2×1024×256 + 1024²)` | `62,914,560` |
| **SwiGLU FFN (24 layers, `d_ff=2816`)** | `24 × (3 × 1024 × 2816)` | `207,618,048` |
| **RMSNorm (48 layer norms + final norm)** | `(24 × 2 + 1) × 1024` | `50,176` |
| **Total Trainable Parameters** | **Verified `actual == analytical` (`diff = 0`)** | **`303,350,784` (`303.35M`)** |

Configurations outside `[285M, 315M]` are automatically rejected unless `allow_target_override=True` is passed explicitly. MoE is explicitly disallowed in BumbleCore v1.
