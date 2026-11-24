"""
GajahMada Native Neural Model (GNNM) — Sovereign From-Scratch Language Model Stack.
Locked Architectural Direction:
1. Architecture: Custom Decoder-Only Transformer (RoPE + RMSNorm + SwiGLU + Causal Attention)
2. Identity: Balanced Multilingual (Indonesian + International languages)
3. Priority: Hybrid Intelligence (compact native neural LM paired with GajahMada's
   14-stage Cognitive, Symbolic, Evidence, and Collective Verification Engine)
"""

from gajahmada_ai.gnnm.config import GNNMConfig, GNNM_PRESETS
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.gnnm.dataset import GNNMCorpusBuilder

__all__ = [
    "GNNMConfig",
    "GNNM_PRESETS",
    "GNNMBPETokenizer",
    "GNNMCorpusBuilder",
]
