"""
GajahMada Native Neural Model (GNNM) — Custom Pretraining Loop (`GNNMTrainer`).
Supports CPU and NVIDIA CUDA with AdamW, cosine learning-rate decay, gradient
clipping, and reproducible checkpoint serialization (`config.json`, `tokenizer.json`, `model.pt`).
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.model import GNNMTransformer, HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer


class GNNMTrainer:
    """From-scratch causal language model trainer for GNNM-01..GNNM-05."""

    def __init__(
        self,
        cfg: GNNMConfig,
        tokenizer: GNNMBPETokenizer,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch (`torch`) is required to run GNNMTrainer. Install with: pip install torch")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GNNMTransformer(cfg).to(self.device)
        self.lr = lr
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
        )
        self.loss_history: List[float] = []

    def train_steps(
        self,
        token_pairs: List[Tuple[List[int], List[int]]],
        steps: int = 100,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        if not token_pairs:
            raise ValueError("token_pairs is empty; build corpus and pack sequences first.")
        self.model.train()
        t0 = time.perf_counter()
        n_pairs = len(token_pairs)

        for step in range(steps):
            # Cosine LR schedule with minimum 10% floor
            progress = step / max(1, steps - 1)
            lr_now = self.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_now

            batch_x, batch_y = [], []
            for b in range(batch_size):
                x, y = token_pairs[(step * batch_size + b) % n_pairs]
                batch_x.append(x)
                batch_y.append(y)

            x_t = torch.tensor(batch_x, dtype=torch.long, device=self.device)
            y_t = torch.tensor(batch_y, dtype=torch.long, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(x_t, targets=y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.loss_history.append(round(float(loss.item()), 5))

        return {
            "model_name": self.cfg.name,
            "device": self.device,
            "steps": steps,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "initial_loss": self.loss_history[0],
            "final_loss": self.loss_history[-1],
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }

    def save_checkpoint(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self.cfg.save(os.path.join(output_dir, "config.json"))
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        weights_path = os.path.join(output_dir, "model.pt")
        torch.save(self.model.state_dict(), weights_path)
        return output_dir
