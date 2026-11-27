"""
GajahMada Native Neural Model (GNNM) — Extended Pretraining & SFT Engine (`GNNMTrainer`).

Preserves 100% backward compatibility with existing GNNM callers while adding Phase 3
production training capabilities for BumbleCore 300M:
- Gradient accumulation (`grad_accum_steps`)
- Learning-rate warmup (`warmup_steps`) + cosine decay
- Mixed precision (`bf16` / `fp16` via `torch.amp.autocast` when supported)
- Gradient norm, tokens/sec, samples/sec, & memory telemetry logging
- Deterministic seeding
- Atomic resumable checkpoints (model + optimizer + global_step + RNG state)
- Emergency checkpoint on interruption without corrupting the last valid checkpoint
- Strict validation (NEVER returns fake `val_loss=0.0, perplexity=1.0` on empty val sets)
"""

from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.model import GNNMTransformer, HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class GNNMTrainer:
    """From-scratch causal language model & SFT trainer for GNNM / BumbleCore 300M."""

    def __init__(
        self,
        cfg: GNNMConfig,
        tokenizer: GNNMBPETokenizer,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
        seed: Optional[int] = 42,
        precision: Optional[str] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch (`torch`) is required to run GNNMTrainer. Install with: pip install torch")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.model = GNNMTransformer(cfg).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.precision = (precision or cfg.precision or "fp32").lower()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
        )
        self.global_step: int = 0
        self.tokens_processed: int = 0
        self.loss_history: List[float] = []
        self.step_logs: List[Dict[str, Any]] = []

    def _compute_lr(self, step: int, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.1) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return self.lr * float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return self.lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _amp_context(self):
        if self.device.startswith("cuda") and self.precision in ("bf16", "fp16"):
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def train_steps(
        self,
        token_pairs: List[Tuple[List[int], List[int]]],
        steps: int = 100,
        batch_size: int = 8,
        grad_accum_steps: int = 1,
        warmup_steps: int = 0,
        val_pairs: Optional[List[Tuple[List[int], List[int]]]] = None,
        eval_interval: int = 0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 0,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if not token_pairs:
            raise ValueError("token_pairs is empty; build corpus and pack sequences first.")
        self.model.train()
        t0 = time.perf_counter()
        n_pairs = len(token_pairs)
        grad_accum = max(1, int(grad_accum_steps))
        start_step = self.global_step
        total_target_steps = start_step + steps
        step_losses: List[float] = []
        last_val: Optional[Dict[str, Any]] = None
        last_ckpt_path: Optional[str] = None

        try:
            for local_step in range(steps):
                step_t0 = time.perf_counter()
                cur_step = self.global_step
                lr_now = self._compute_lr(cur_step, total_target_steps, warmup_steps=warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_now

                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                step_tokens = 0

                for micro in range(grad_accum):
                    batch_x, batch_y = [], []
                    base_idx = ((cur_step * grad_accum) + micro) * batch_size
                    for b in range(batch_size):
                        x, y = token_pairs[(base_idx + b) % n_pairs]
                        batch_x.append(x)
                        batch_y.append(y)
                        step_tokens += len(x)

                    x_t = torch.tensor(batch_x, dtype=torch.long, device=self.device)
                    y_t = torch.tensor(batch_y, dtype=torch.long, device=self.device)

                    with self._amp_context():
                        _, loss = self.model(x_t, targets=y_t)
                        scaled_loss = loss / grad_accum
                    scaled_loss.backward()
                    accum_loss += float(loss.item()) / grad_accum

                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
                if not math.isfinite(accum_loss) or not math.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite training state at step {cur_step}: loss={accum_loss}, grad_norm={grad_norm}")

                self.optimizer.step()
                self.global_step += 1
                self.tokens_processed += step_tokens
                step_dur = max(1e-6, time.perf_counter() - step_t0)

                loss_val = round(accum_loss, 5)
                self.loss_history.append(loss_val)
                step_losses.append(loss_val)

                gpu_mem_mb = 0.0
                peak_mem_mb = 0.0
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    gpu_mem_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                    peak_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

                if val_pairs is not None and eval_interval > 0 and (self.global_step % eval_interval == 0):
                    last_val = self.evaluate_loss(val_pairs)
                    self.model.train()

                if checkpoint_dir and checkpoint_interval > 0 and (self.global_step % checkpoint_interval == 0):
                    last_ckpt_path = self.save_resumable_checkpoint(checkpoint_dir, metrics={"step": self.global_step, "loss": loss_val})

                log_entry = {
                    "global_step": self.global_step,
                    "train_loss": loss_val,
                    "val_loss": last_val.get("val_loss") if last_val else None,
                    "lr": round(lr_now, 8),
                    "grad_norm": round(grad_norm, 5),
                    "tokens_processed": self.tokens_processed,
                    "tokens_per_sec": round(step_tokens / step_dur, 1),
                    "samples_per_sec": round((batch_size * grad_accum) / step_dur, 2),
                    "gpu_memory_mb": gpu_mem_mb,
                    "peak_memory_mb": peak_mem_mb,
                    "checkpoint_id": last_ckpt_path,
                }
                self.step_logs.append(log_entry)
                if on_step is not None:
                    on_step(log_entry)
        except KeyboardInterrupt:
            if checkpoint_dir:
                self.save_resumable_checkpoint(os.path.join(checkpoint_dir, "emergency_interrupt"), metrics={"interrupted_at_step": self.global_step})
            raise

        elapsed = round(time.perf_counter() - t0, 3)
        return {
            "model_name": self.cfg.name,
            "device": self.device,
            "precision": self.precision,
            "steps": steps,
            "global_step": self.global_step,
            "grad_accum_steps": grad_accum,
            "tokens_processed": self.tokens_processed,
            "tokens_per_sec": round(self.tokens_processed / max(1e-6, elapsed), 1),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "initial_loss": step_losses[0],
            "final_loss": step_losses[-1],
            "last_validation": last_val,
            "elapsed_sec": elapsed,
        }

    def evaluate_loss(
        self,
        val_pairs: List[Tuple[List[int], List[int]]],
        max_batches: int = 16,
    ) -> Dict[str, Any]:
        """
        Compute validation cross-entropy loss and perplexity without updating weights.
        Never fabricates `val_loss=0.0, perplexity=1.0` when validation data is empty/invalid.
        """
        if not val_pairs:
            return {
                "status": "UNAVAILABLE",
                "val_loss": None,
                "perplexity": None,
                "samples_evaluated": 0,
                "error": "Validation dataset is empty; cannot compute authentic validation metrics.",
            }
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_pairs[:max_batches]:
                if all(t == -100 for t in y):
                    continue
                x_t = torch.tensor([x], dtype=torch.long, device=self.device)
                y_t = torch.tensor([y], dtype=torch.long, device=self.device)
                _, loss = self.model(x_t, targets=y_t)
                if loss is not None and math.isfinite(float(loss.item())):
                    total_loss += float(loss.item())
                    count += 1
        if count == 0:
            return {
                "status": "INVALID_VALIDATION_DATA",
                "val_loss": None,
                "perplexity": None,
                "samples_evaluated": 0,
                "error": "No finite non-masked validation samples available.",
            }
        avg_loss = total_loss / count
        ppl = math.exp(min(avg_loss, 20.0))
        return {
            "status": "COMPUTED",
            "val_loss": round(avg_loss, 5),
            "perplexity": round(ppl, 3),
            "samples_evaluated": count,
        }

    def train_sft(
        self,
        sft_pairs: List[Tuple[List[int], List[int]]],
        steps: int = 40,
        batch_size: int = 4,
        sft_lr: Optional[float] = None,
        grad_accum_steps: int = 1,
        warmup_steps: int = 0,
    ) -> Dict[str, Any]:
        """Supervised Fine-Tuning (SFT) stage with lower LR and prompt-masked CE loss (`ignore_index=-100`)."""
        orig_lr = self.lr
        self.lr = sft_lr if sft_lr is not None else max(self.lr * 0.25, 1e-4)
        try:
            res = self.train_steps(
                sft_pairs,
                steps=steps,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                warmup_steps=warmup_steps,
            )
            res["stage"] = "sft_instruction_tuning"
            return res
        finally:
            self.lr = orig_lr

    def save_checkpoint(self, output_dir: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Atomic weight-only inference checkpoint save (backward compatible)."""
        os.makedirs(output_dir, exist_ok=True)
        self.cfg.save(os.path.join(output_dir, "config.json"))
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        weights_path = os.path.join(output_dir, "model.pt")
        tmp_weights = weights_path + ".tmp"
        torch.save(self.model.state_dict(), tmp_weights)
        os.replace(tmp_weights, weights_path)
        meta = dict(metrics or {})
        meta["model_sha256"] = _sha256_file(weights_path)
        meta["global_step"] = self.global_step
        meta["tokens_processed"] = self.tokens_processed
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return output_dir

    def save_resumable_checkpoint(self, output_dir: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a full resumable training checkpoint atomically (weights + optimizer +
        global_step + RNG state + loss_history) without corrupting previous files.
        """
        self.save_checkpoint(output_dir, metrics=metrics)
        state_path = os.path.join(output_dir, "trainer_state.pt")
        tmp_state = state_path + ".tmp"
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_processed": self.tokens_processed,
            "loss_history": self.loss_history,
            "seed": self.seed,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(state, tmp_state)
        os.replace(tmp_state, state_path)
        return output_dir

    def resume_from_checkpoint(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Restore model weights, optimizer state, global step, and RNG state from checkpoint."""
        weights_path = os.path.join(checkpoint_dir, "model.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        restored_optimizer = False
        if os.path.isfile(trainer_state_path):
            t_state = torch.load(trainer_state_path, map_location=self.device, weights_only=False)
            if "optimizer_state_dict" in t_state:
                self.optimizer.load_state_dict(t_state["optimizer_state_dict"])
                restored_optimizer = True
            self.global_step = int(t_state.get("global_step", 0))
            self.tokens_processed = int(t_state.get("tokens_processed", 0))
            self.loss_history = list(t_state.get("loss_history", []))
            if "python_rng_state" in t_state:
                random.setstate(t_state["python_rng_state"])
            if "torch_rng_state" in t_state:
                torch.set_rng_state(t_state["torch_rng_state"].cpu())

        return {
            "resumed_from": checkpoint_dir,
            "global_step": self.global_step,
            "tokens_processed": self.tokens_processed,
            "optimizer_restored": restored_optimizer,
        }
