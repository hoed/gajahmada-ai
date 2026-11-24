"""
GajahMada Native Neural Model (GNNM) — Native Inference Engine (`GNNMInferenceEngine`).
Loads a self-trained GNNM checkpoint directory (`config.json`, `tokenizer.json`, `model.pt`)
and performs autoregressive text generation on CPU or CUDA with zero external model dependencies.
"""

import os
from typing import Optional

from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.model import GNNMTransformer, HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer


class GNNMInferenceEngine:
    def __init__(self, checkpoint_dir: str, device: Optional[str] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch (`torch`) is required to run GNNMInferenceEngine.")
        self.checkpoint_dir = checkpoint_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = GNNMConfig.load(os.path.join(checkpoint_dir, "config.json"))
        self.tokenizer = GNNMBPETokenizer.load(os.path.join(checkpoint_dir, "tokenizer.json"))
        self.model = GNNMTransformer(self.cfg).to(self.device)
        state = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.eos_id,
        )
        generated_ids = out[0].tolist()[len(ids):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
