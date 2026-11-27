"""
GajahMada GNNM BumbleCore 300M — Resource Preflight & Hardware Readiness Checker (Phase 4).

Probes real hardware (PyTorch, CUDA, GPU name/VRAM, BF16/FP16 support, system RAM,
free disk, dataset availability, checkpoint destination) and compares against
analytical training estimates. Stops cleanly before full training if resources are
insufficient — NEVER silently shrinks the model architecture.
"""
from __future__ import annotations

import importlib
import os
import shutil
from typing import Any, Dict, List, Optional

import psutil

from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.resource_estimator import estimate_model_resources


def run_bumblecore_preflight(
    cfg: GNNMConfig,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 16,
    target_tokens: Optional[int] = None,
    dataset_path: Optional[str] = None,
    checkpoint_dir: str = "./models/bumblecore-300m",
    measured_tokens_per_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute hardware, storage, dataset, and analytical memory preflight."""
    is_vercel = os.environ.get("VERCEL") == "1"
    has_torch = False
    cuda_available = False
    gpu_name = None
    gpu_vram_gb = 0.0
    bf16_supported = False
    fp16_supported = False

    if not is_vercel:
        try:
            torch = importlib.import_module("torch")
            has_torch = True
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                bf16_supported = bool(torch.cuda.is_bf16_supported())
                fp16_supported = True
            else:
                bf16_supported = True  # PyTorch CPU supports bfloat16 tensors
        except Exception:
            pass

    vm = psutil.virtual_memory()
    sys_ram_total_gb = round(vm.total / (1024 ** 3), 2)
    sys_ram_avail_gb = round(vm.available / (1024 ** 3), 2)
    disk_free_gb = round(shutil.disk_usage(os.getcwd()).free / (1024 ** 3), 2)

    est = estimate_model_resources(
        cfg,
        training_tokens=target_tokens,
        micro_batch_size=micro_batch_size,
    )
    # Temporary attention score buffer estimate (Flash/SDPA tiled vs naive O(T^2))
    sdpa_temp_mb = round((micro_batch_size * cfg.n_heads * cfg.max_seq_len * cfg.head_dim * 2) / (1024 ** 2), 2)
    approx_total_vram_gb = round(est["training_memory_gb"]["total_sharded_memory_gb"] + (sdpa_temp_mb / 1024.0), 3)
    req_sys_ram_gb = round(max(4.0, est["checkpoint_storage_gb"]["full_resumable_checkpoint_gb"] * 1.5), 2)
    req_disk_gb = round(est["checkpoint_storage_gb"]["full_resumable_checkpoint_gb"] * 3.0 + 2.0, 2)

    # Duration ONLY calculated when measured throughput is actually supplied
    approx_duration_hours = (
        round((target_tokens or est["compute_and_cluster"]["chinchilla_optimal_tokens"]) / (measured_tokens_per_sec * 3600.0), 2)
        if measured_tokens_per_sec and measured_tokens_per_sec > 0
        else None
    )

    blockers: List[str] = []
    if is_vercel:
        blockers.append("VERCEL_FREE_FORBIDDEN: Model training is strictly disabled on Vercel Free.")
    if not has_torch:
        blockers.append("PYTORCH_MISSING: Install PyTorch before launching training.")
    if cfg.estimate_parameters() >= 100_000_000 and not cuda_available:
        blockers.append(
            f"GPU_REQUIRED_FOR_FULL_TRAINING: {cfg.name} ({cfg.estimate_parameters():,} params) requires CUDA GPU (>= {approx_total_vram_gb} GB VRAM); current host is CPU-only."
        )
    if cuda_available and gpu_vram_gb < approx_total_vram_gb:
        blockers.append(f"INSUFFICIENT_VRAM: GPU has {gpu_vram_gb} GB VRAM, estimated requirement is {approx_total_vram_gb} GB.")
    if disk_free_gb < req_disk_gb:
        blockers.append(f"INSUFFICIENT_DISK: Free disk {disk_free_gb} GB < estimated checkpoint storage {req_disk_gb} GB.")
    if dataset_path and not os.path.exists(dataset_path):
        blockers.append(f"DATASET_NOT_FOUND: {dataset_path}")

    return {
        "preflight_passed": len(blockers) == 0,
        "status": "READY" if len(blockers) == 0 else "BLOCKED_INSUFFICIENT_RESOURCES",
        "model_name": cfg.name,
        "exact_parameters": cfg.estimate_parameters(),
        "hardware_measured": {
            "is_vercel": is_vercel,
            "pytorch_available": has_torch,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram_gb,
            "bf16_supported": bf16_supported,
            "fp16_supported": fp16_supported,
            "system_ram_total_gb": sys_ram_total_gb,
            "system_ram_available_gb": sys_ram_avail_gb,
            "free_disk_gb": disk_free_gb,
        },
        "analytical_estimate": {
            "estimate_kind": "THEORETICAL_ANALYTICAL_ESTIMATE",
            "parameter_memory_gb": est["training_memory_gb"]["weights_bf16_gb"],
            "gradient_memory_gb": est["training_memory_gb"]["gradients_bf16_gb"],
            "optimizer_state_memory_gb": est["training_memory_gb"]["optimizer_adamw_fp32_gb"],
            "activation_memory_gb": est["training_memory_gb"]["activations_selected_gb"],
            "temp_attention_memory_mb": sdpa_temp_mb,
            "approx_total_vram_gb": approx_total_vram_gb,
            "recommended_system_ram_gb": req_sys_ram_gb,
            "recommended_free_disk_gb": req_disk_gb,
            "checkpoint_weights_gb": est["checkpoint_storage_gb"]["inference_weights_bf16_gb"],
            "checkpoint_resumable_gb": est["checkpoint_storage_gb"]["full_resumable_checkpoint_gb"],
            "effective_batch_tokens": micro_batch_size * grad_accum_steps * cfg.max_seq_len,
            "measured_throughput_duration_hours": approx_duration_hours,
        },
        "checkpoint_destination": checkpoint_dir,
        "blockers": blockers,
    }
