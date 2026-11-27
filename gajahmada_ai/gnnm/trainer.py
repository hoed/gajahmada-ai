"""
GajahMada Native Neural Model (GNNM) — Hardened Pretraining & SFT Engine (`GNNMTrainer`, Release 2.6).

Preserves 100% backward compatibility with existing callers while enforcing Release 2.6 integrity:
- True Token Accounting (Phase 2):
  Tracks `raw_input_tokens`, `valid_training_tokens` (non-padding), `loss_bearing_target_tokens`
  (`y != -100`), `tokens_consumed` (primary metric: non-padding tokens consumed by optimizer),
  `validation_tokens` (strictly excluded from training counters), `target_tokens`,
  `token_budget_remaining`, and `token_budget_reached`. Stops cleanly at `target_tokens`.
- SFT Learning Rate Correctness (Phase 4):
  Respects configured `sft_lr` / `self.lr` directly with ZERO hidden `max(..., 1e-4)` override.
- Exact Training Resume & Reproducibility (Phase 5):
  Saves & restores model, optimizer, scheduler step/horizon, `dataset_cursor`, token counters,
  CPU/CUDA/Python/NumPy RNG states, FP16 GradScaler, and provenance hashes (`config_sha256`,
  `dataset_manifest_sha256`, `tokenizer_sha256`), rejecting mismatched dataset/tokenizer on resume.
"""

from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
import platform
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


def _try_numpy():
    try:
        return importlib.import_module("numpy")
    except Exception:
        return None


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
        min_lr_ratio: float = 0.1,
        provenance_hashes: Optional[Dict[str, str]] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch (`torch`) is required to run GNNMTrainer. Install with: pip install torch")
        if cfg.vocab_size != tokenizer.vocab_size:
            # Allow legacy callers where vocab_size >= tokenizer.vocab_size, but record exact mismatch
            if tokenizer.vocab_size > cfg.vocab_size:
                raise ValueError(
                    f"Tokenizer vocab_size ({tokenizer.vocab_size}) exceeds model cfg.vocab_size ({cfg.vocab_size})."
                )
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np_mod = _try_numpy()
            if np_mod is not None:
                np_mod.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.model = GNNMTransformer(cfg).to(self.device)
        self.lr = float(lr)
        self.min_lr_ratio = float(min_lr_ratio)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.precision = (precision or cfg.precision or "fp32").lower()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.weight_decay
        )
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=(self.device.startswith("cuda") and self.precision == "fp16"))
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else None
        )

        # Authoritative training & token accounting state
        self.global_step: int = 0
        self.dataset_cursor: int = 0
        self.scheduler_total_steps: int = 0
        self.raw_input_tokens: int = 0
        self.valid_training_tokens: int = 0
        self.loss_bearing_target_tokens: int = 0
        self.tokens_consumed: int = 0  # Primary metric: non-padding tokens consumed by optimizer
        self.validation_tokens: int = 0  # Strictly separate from training counters
        self.target_tokens: Optional[int] = None
        self.token_budget_reached: bool = False
        self.provenance_hashes: Dict[str, str] = dict(provenance_hashes or {})
        self.loss_history: List[float] = []
        self.step_logs: List[Dict[str, Any]] = []

    @property
    def tokens_processed(self) -> int:
        """Backward-compatible alias for primary consumed non-padding training tokens."""
        return self.tokens_consumed

    def token_accounting_snapshot(self) -> Dict[str, Any]:
        remaining = (
            max(0, self.target_tokens - self.tokens_consumed)
            if self.target_tokens is not None
            else None
        )
        return {
            "primary_metric": "tokens_consumed (non-padding training tokens consumed by optimizer)",
            "raw_input_tokens": self.raw_input_tokens,
            "valid_training_tokens": self.valid_training_tokens,
            "loss_bearing_target_tokens": self.loss_bearing_target_tokens,
            "tokens_consumed": self.tokens_consumed,
            "validation_tokens": self.validation_tokens,
            "target_tokens": self.target_tokens,
            "token_budget_remaining": remaining,
            "token_budget_reached": self.token_budget_reached,
        }

    def _compute_lr(self, step: int, total_steps: int, warmup_steps: int = 0) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return self.lr * float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

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
        target_tokens: Optional[int] = None,
        total_schedule_steps: Optional[int] = None,
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
        pad_id = self.tokenizer.pad_id

        if target_tokens is not None:
            self.target_tokens = int(target_tokens)
        if total_schedule_steps is not None:
            self.scheduler_total_steps = int(total_schedule_steps)
        elif self.scheduler_total_steps < self.global_step + steps:
            self.scheduler_total_steps = self.global_step + steps

        step_losses: List[float] = []
        steps_executed = 0
        last_val: Optional[Dict[str, Any]] = None
        last_ckpt_path: Optional[str] = None
        last_lr_used = self.lr

        try:
            for _ in range(steps):
                if self.target_tokens is not None and self.tokens_consumed >= self.target_tokens:
                    self.token_budget_reached = True
                    break

                step_t0 = time.perf_counter()
                cur_step = self.global_step
                lr_now = self._compute_lr(cur_step, self.scheduler_total_steps, warmup_steps=warmup_steps)
                last_lr_used = lr_now
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_now

                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                step_raw = 0
                step_valid = 0
                step_loss_bearing = 0
                micro_executed = 0

                for _micro in range(grad_accum):
                    if self.target_tokens is not None and (self.tokens_consumed + step_valid) >= self.target_tokens:
                        self.token_budget_reached = True
                        break

                    batch_x, batch_y = [], []
                    for _b in range(batch_size):
                        if self.target_tokens is not None and (self.tokens_consumed + step_valid) >= self.target_tokens:
                            self.token_budget_reached = True
                            break
                        x, y = token_pairs[self.dataset_cursor % n_pairs]
                        self.dataset_cursor += 1
                        batch_x.append(x)
                        batch_y.append(y)
                        step_raw += len(x)
                        step_valid += sum(1 for tok in x if tok != pad_id)
                        step_loss_bearing += sum(1 for tok in y if tok != -100)

                    if not batch_x:
                        break

                    x_t = torch.tensor(batch_x, dtype=torch.long, device=self.device)
                    y_t = torch.tensor(batch_y, dtype=torch.long, device=self.device)

                    with self._amp_context():
                        _, loss = self.model(x_t, targets=y_t)
                        scaled_loss = loss / grad_accum
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    accum_loss += float(loss.item())
                    micro_executed += 1

                if micro_executed == 0:
                    break

                # Normalize loss by actual micro-batches executed (handles partial final batch)
                if micro_executed != grad_accum:
                    scale_fix = float(grad_accum) / float(micro_executed)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale_fix)
                avg_step_loss = accum_loss / micro_executed

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
                if not math.isfinite(avg_step_loss) or not math.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite training state at step {cur_step}: loss={avg_step_loss}, grad_norm={grad_norm}")

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.global_step += 1
                steps_executed += 1
                self.raw_input_tokens += step_raw
                self.valid_training_tokens += step_valid
                self.loss_bearing_target_tokens += step_loss_bearing
                self.tokens_consumed += step_valid
                if self.target_tokens is not None and self.tokens_consumed >= self.target_tokens:
                    self.token_budget_reached = True

                step_dur = max(1e-6, time.perf_counter() - step_t0)
                loss_val = round(avg_step_loss, 5)
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
                    "lr": round(lr_now, 10),
                    "grad_norm": round(grad_norm, 5),
                    "tokens_consumed": self.tokens_consumed,
                    "tokens_per_sec": round(step_valid / step_dur, 1),
                    "samples_per_sec": round((batch_size * micro_executed) / step_dur, 2),
                    "gpu_memory_mb": gpu_mem_mb,
                    "peak_memory_mb": peak_mem_mb,
                    "checkpoint_id": last_ckpt_path,
                }
                self.step_logs.append(log_entry)
                if on_step is not None:
                    on_step(log_entry)
        except KeyboardInterrupt:
            if checkpoint_dir:
                self.save_resumable_checkpoint(
                    os.path.join(checkpoint_dir, "emergency_interrupt"),
                    metrics={"interrupted_at_step": self.global_step},
                )
            raise

        elapsed = round(time.perf_counter() - t0, 3)
        return {
            "model_name": self.cfg.name,
            "device": self.device,
            "precision": self.precision,
            "steps": steps_executed,
            "global_step": self.global_step,
            "grad_accum_steps": grad_accum,
            "configured_lr": self.lr,
            "last_optimizer_lr": round(last_lr_used, 10),
            "tokens_processed": self.tokens_consumed,
            "token_accounting": self.token_accounting_snapshot(),
            "tokens_per_sec": round(self.tokens_consumed / max(1e-6, elapsed), 1),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "initial_loss": step_losses[0] if step_losses else None,
            "final_loss": step_losses[-1] if step_losses else None,
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
        Validation tokens are tracked separately and NEVER added to training token counters.
        """
        if not val_pairs:
            return {
                "status": "UNAVAILABLE",
                "val_loss": None,
                "perplexity": None,
                "samples_evaluated": 0,
                "validation_tokens": 0,
                "error": "Validation dataset is empty; cannot compute authentic validation metrics.",
            }
        self.model.eval()
        total_loss = 0.0
        count = 0
        val_toks = 0
        pad_id = self.tokenizer.pad_id
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
                    val_toks += sum(1 for tok in x if tok != pad_id)
        if count == 0:
            return {
                "status": "INVALID_VALIDATION_DATA",
                "val_loss": None,
                "perplexity": None,
                "samples_evaluated": 0,
                "validation_tokens": 0,
                "error": "No finite non-masked validation samples available.",
            }
        self.validation_tokens += val_toks
        avg_loss = total_loss / count
        ppl = math.exp(min(avg_loss, 20.0))
        return {
            "status": "COMPUTED",
            "val_loss": round(avg_loss, 5),
            "perplexity": round(ppl, 3),
            "samples_evaluated": count,
            "validation_tokens": val_toks,
        }

    def train_sft(
        self,
        sft_pairs: List[Tuple[List[int], List[int]]],
        steps: int = 40,
        batch_size: int = 4,
        sft_lr: Optional[float] = None,
        grad_accum_steps: int = 1,
        warmup_steps: int = 0,
        target_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Supervised Fine-Tuning (SFT) stage with prompt-masked CE loss (`ignore_index=-100`).
        Respects `sft_lr` if explicitly provided, otherwise uses `self.lr` directly —
        NEVER silently clamps `5e-5` upward to `1e-4`.
        """
        orig_lr = self.lr
        effective_lr = float(sft_lr) if sft_lr is not None else float(self.lr)
        self.lr = effective_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = effective_lr
        try:
            res = self.train_steps(
                sft_pairs,
                steps=steps,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                warmup_steps=warmup_steps,
                target_tokens=target_tokens,
            )
            res["stage"] = "sft_instruction_tuning"
            res["configured_sft_lr"] = effective_lr
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
        meta["tokens_processed"] = self.tokens_consumed
        meta["token_accounting"] = self.token_accounting_snapshot()
        meta["provenance_hashes"] = dict(self.provenance_hashes)
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return output_dir

    def save_resumable_checkpoint(self, output_dir: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a full resumable training checkpoint atomically (weights + optimizer +
        scheduler horizon + dataset cursor + token counters + CPU/CUDA/Python/NumPy RNG +
        FP16 GradScaler + provenance hashes).
        """
        self.save_checkpoint(output_dir, metrics=metrics)
        state_path = os.path.join(output_dir, "trainer_state.pt")
        tmp_state = state_path + ".tmp"
        np_mod = _try_numpy()
        state: Dict[str, Any] = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "dataset_cursor": self.dataset_cursor,
            "scheduler_total_steps": self.scheduler_total_steps,
            "raw_input_tokens": self.raw_input_tokens,
            "valid_training_tokens": self.valid_training_tokens,
            "loss_bearing_target_tokens": self.loss_bearing_target_tokens,
            "tokens_consumed": self.tokens_consumed,
            "validation_tokens": self.validation_tokens,
            "target_tokens": self.target_tokens,
            "token_budget_reached": self.token_budget_reached,
            "provenance_hashes": dict(self.provenance_hashes),
            "loss_history": self.loss_history,
            "seed": self.seed,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np_mod.random.get_state() if np_mod is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if (self.scaler is not None and self.scaler.is_enabled()) else None,
        }
        torch.save(state, tmp_state)
        os.replace(tmp_state, state_path)
        return output_dir

    def resume_from_checkpoint(
        self,
        checkpoint_dir: str,
        expected_dataset_hash: Optional[str] = None,
        expected_tokenizer_hash: Optional[str] = None,
        allow_resume_mismatch: bool = False,
    ) -> Dict[str, Any]:
        """
        Restore model, optimizer, scheduler, dataset cursor, token counters, and RNG states.
        Detects dataset/tokenizer hash mismatches and stops unless `allow_resume_mismatch=True`.
        """
        weights_path = os.path.join(checkpoint_dir, "model.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")

        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        t_state = (
            torch.load(trainer_state_path, map_location=self.device, weights_only=False)
            if os.path.isfile(trainer_state_path)
            else {}
        )
        saved_hashes = t_state.get("provenance_hashes") or {}
        mismatches: List[str] = []
        if expected_dataset_hash and saved_hashes.get("dataset_manifest_sha256") and expected_dataset_hash != saved_hashes["dataset_manifest_sha256"]:
            mismatches.append("DATASET_MANIFEST_HASH_MISMATCH")
        if expected_tokenizer_hash and saved_hashes.get("tokenizer_sha256") and expected_tokenizer_hash != saved_hashes["tokenizer_sha256"]:
            mismatches.append("TOKENIZER_HASH_MISMATCH")
        if mismatches and not allow_resume_mismatch:
            raise ValueError(
                f"Refusing silent resume due to provenance mismatch: {mismatches}. "
                "Pass allow_resume_mismatch=True explicitly to override."
            )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        restored_optimizer = False
        restored_rng = []
        if t_state:
            if "optimizer_state_dict" in t_state:
                self.optimizer.load_state_dict(t_state["optimizer_state_dict"])
                restored_optimizer = True
            self.global_step = int(t_state.get("global_step", 0))
            self.dataset_cursor = int(t_state.get("dataset_cursor", 0))
            self.scheduler_total_steps = int(t_state.get("scheduler_total_steps", 0))
            self.raw_input_tokens = int(t_state.get("raw_input_tokens", 0))
            self.valid_training_tokens = int(t_state.get("valid_training_tokens", 0))
            self.loss_bearing_target_tokens = int(t_state.get("loss_bearing_target_tokens", 0))
            self.tokens_consumed = int(t_state.get("tokens_consumed", t_state.get("tokens_processed", 0)))
            self.validation_tokens = int(t_state.get("validation_tokens", 0))
            if t_state.get("target_tokens") is not None:
                self.target_tokens = int(t_state["target_tokens"])
            self.token_budget_reached = bool(t_state.get("token_budget_reached", False))
            self.loss_history = list(t_state.get("loss_history", []))
            if t_state.get("python_rng_state") is not None:
                random.setstate(t_state["python_rng_state"])
                restored_rng.append("python")
            if t_state.get("torch_rng_state") is not None:
                torch.set_rng_state(t_state["torch_rng_state"].cpu())
                restored_rng.append("torch_cpu")
            if t_state.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(t_state["cuda_rng_state_all"])
                    restored_rng.append("torch_cuda")
                except Exception:
                    pass
            np_mod = _try_numpy()
            if t_state.get("numpy_rng_state") is not None and np_mod is not None:
                np_mod.random.set_state(t_state["numpy_rng_state"])
                restored_rng.append("numpy")
            if t_state.get("scaler_state_dict") is not None and self.scaler is not None and self.scaler.is_enabled():
                self.scaler.load_state_dict(t_state["scaler_state_dict"])

        return {
            "resumed_from": checkpoint_dir,
            "resume_mode": "EXACT_STATE_RESTORED" if (restored_optimizer and not mismatches) else "OVERRIDE_OR_WEIGHTS_ONLY",
            "global_step": self.global_step,
            "dataset_cursor": self.dataset_cursor,
            "tokens_consumed": self.tokens_consumed,
            "optimizer_restored": restored_optimizer,
            "rng_restored": restored_rng,
            "mismatches": mismatches,
        }
