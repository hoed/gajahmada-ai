"""
GajahMada GNNM Analytical Resource Estimator (Phase 6).

Pure-Python analytical estimator requiring ZERO GPUs and ZERO PyTorch instantiation.
Computes transparent, documented theoretical estimates for any GNNMConfig — including
TITANFORGE 70B — across FP32, BF16, FP16, INT8, and INT4 precisions.

All outputs explicitly distinguish THEORETICAL_ESTIMATE from measured runtime metrics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from gajahmada_ai.gnnm.config import GNNMConfig

BYTES_PER_PARAM = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.55,  # 0.5 bytes/param + ~10% quantization group scales/zeros overhead
}

# Documented reference hardware specifications (theoretical peak BF16 dense TFLOPS & VRAM)
REFERENCE_GPUS: Dict[str, Dict[str, Any]] = {
    "H100-80GB-SXM": {"vram_gb": 80.0, "bf16_tflops": 989.0, "interconnect": "900 GB/s NVLink 4 + 400 Gbps InfiniBand"},
    "A100-80GB-SXM": {"vram_gb": 80.0, "bf16_tflops": 312.0, "interconnect": "600 GB/s NVLink 3 + 200 Gbps InfiniBand"},
    "L40S-48GB": {"vram_gb": 48.0, "bf16_tflops": 362.0, "interconnect": "PCIe Gen4 x16 (limited multi-node bandwidth)"},
    "RTX-4090-24GB": {"vram_gb": 24.0, "bf16_tflops": 165.0, "interconnect": "PCIe Gen4 x16 (consumer, no NVLink)"},
}


def _gb(num_bytes: float) -> float:
    return round(num_bytes / (1024 ** 3), 3)


def estimate_model_resources(
    cfg: GNNMConfig,
    training_tokens: Optional[int] = None,
    micro_batch_size: int = 1,
    seq_len: Optional[int] = None,
    mfu: float = 0.40,
    target_gpu: str = "H100-80GB-SXM",
) -> Dict[str, Any]:
    """
    Analytical memory, storage, FLOPs, GPU count, and inference resource estimator.

    Assumptions (explicitly returned in payload):
      - Mixed-precision AdamW training (BF16/FP16 weights & grads + FP32 master weights +
        FP32 momentum & variance states) = 16 bytes/param (2 + 2 + 4 + 4 + 4).
      - Full FP32 AdamW training = 16 bytes/param (4 weights + 4 grads + 8 optimizer).
      - Dense training FLOPs ≈ 6 * N_active * D_tokens (Kaplan / Hoffmann scaling law).
      - Default Chinchilla-optimal token horizon = 20 * N_active if training_tokens omitted.
      - Model FLOPs Utilization (MFU) defaults to 0.40 (40%, realistic FSDP/TP cluster range 35-45%).
    """
    bd = cfg.parameter_breakdown()
    n_total = bd["total_parameters"]
    n_active = bd["active_parameters"]
    eff_seq = seq_len or cfg.max_seq_len
    chinchilla_tokens = int(20 * n_active)
    tokens_to_train = int(training_tokens) if training_tokens else chinchilla_tokens

    # 1. Precision-specific weight memory & inference memory (including KV cache for 1 request)
    kv_elements_per_token = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim
    inference_by_precision: Dict[str, Dict[str, Any]] = {}
    for prec, bpp in BYTES_PER_PARAM.items():
        w_bytes = n_total * bpp
        kv_bytes_per_elem = 2.0 if prec in ("fp32", "bf16", "fp16") else 1.0
        kv_cache_bytes = kv_elements_per_token * eff_seq * kv_bytes_per_elem
        # ~15% runtime buffer overhead for activations & CUDA allocator
        total_inf_bytes = (w_bytes + kv_cache_bytes) * 1.15
        inference_by_precision[prec] = {
            "weights_gb": _gb(w_bytes),
            "kv_cache_gb_single_seq": _gb(kv_cache_bytes),
            "total_vram_gb_est": _gb(total_inf_bytes),
            "fits_single_80gb_gpu": _gb(total_inf_bytes) <= 74.0,
            "fits_single_24gb_gpu": _gb(total_inf_bytes) <= 21.0,
        }

    # 2. Training memory breakdown (Mixed Precision BF16 + FP32 AdamW states)
    weight_mem_bytes = n_total * 2.0
    grad_mem_bytes = n_total * 2.0
    # FP32 master weights (4B) + AdamW m (4B) + AdamW v (4B) = 12 bytes/param
    optimizer_mem_bytes = n_total * 12.0

    # Activation memory estimate per micro-batch (Transformer standard formula)
    # Without checkpointing: ~n_layers * micro_batch * seq * d_model * 34 bytes (BF16)
    # With gradient checkpointing: stores layer inputs (~2 bytes * d_model * seq * n_layers) + 1 active layer
    act_no_ckpt_bytes = cfg.n_layers * micro_batch_size * eff_seq * cfg.d_model * 34.0
    act_with_ckpt_bytes = (
        (cfg.n_layers * micro_batch_size * eff_seq * cfg.d_model * 2.0)
        + (micro_batch_size * eff_seq * cfg.d_model * 34.0)
    )
    active_act_bytes = act_with_ckpt_bytes if cfg.gradient_checkpointing else act_no_ckpt_bytes

    total_training_mem_bytes = (
        weight_mem_bytes + grad_mem_bytes + optimizer_mem_bytes + active_act_bytes
    )

    # Checkpoint storage: weights (BF16) for inference-only vs full resumable training state (weights+optimizer)
    ckpt_weights_only_gb = _gb(n_total * 2.0)
    ckpt_full_training_gb = _gb(n_total * 14.0)  # BF16 weights + FP32 master + AdamW m/v

    # 3. Compute FLOPs & GPU Cluster Sizing
    training_flops = 6.0 * n_active * tokens_to_train
    gpu_spec = REFERENCE_GPUS.get(target_gpu, REFERENCE_GPUS["H100-80GB-SXM"])
    usable_vram_gb = gpu_spec["vram_gb"] * 0.82  # Leave 18% headroom for NCCL buffers & fragmentation
    min_gpus_for_memory = max(1, int(math.ceil(_gb(total_training_mem_bytes) / usable_vram_gb)))
    # Round up to practical power-of-2 / 8-GPU node multiple for large models
    if min_gpus_for_memory > 8:
        recommended_gpus = int(math.ceil(min_gpus_for_memory / 8) * 8)
    else:
        recommended_gpus = min_gpus_for_memory

    effective_tflops_per_gpu = gpu_spec["bf16_tflops"] * max(0.05, min(0.85, mfu))
    gpu_seconds_single = training_flops / (effective_tflops_per_gpu * 1e12)
    est_days_at_min_cluster = round(gpu_seconds_single / (recommended_gpus * 86400.0), 2)
    est_days_at_64_gpus = round(gpu_seconds_single / (64 * 86400.0), 2)
    est_days_at_256_gpus = round(gpu_seconds_single / (256 * 86400.0), 2)

    # 4. Vercel Free compatibility assessment
    vercel_compatible_inference = (
        _gb(n_total * 2.0) <= 0.15 and cfg.estimate_parameters() <= 50_000_000
    )
    vercel_notes = (
        "Small enough for theoretical CPU evaluation locally, but PyTorch exceeds Vercel Free 50MB lambda limit."
        if vercel_compatible_inference
        else "UNAVAILABLE ON VERCEL FREE: Model memory and compute exceed serverless limits; requires dedicated external GPU infrastructure."
    )

    return {
        "estimate_kind": "THEORETICAL_ANALYTICAL_ESTIMATE",
        "measured_benchmark": False,
        "model_name": cfg.name,
        "family": cfg.family,
        "architecture_type": cfg.architecture_type,
        "parameters": {
            "total": n_total,
            "total_billions": round(n_total / 1e9, 4),
            "active": n_active,
            "active_billions": round(n_active / 1e9, 4),
            "breakdown": bd,
        },
        "training_memory_gb": {
            "weights_bf16_gb": _gb(weight_mem_bytes),
            "gradients_bf16_gb": _gb(grad_mem_bytes),
            "optimizer_adamw_fp32_gb": _gb(optimizer_mem_bytes),
            "activations_no_ckpt_gb": _gb(act_no_ckpt_bytes),
            "activations_with_ckpt_gb": _gb(act_with_ckpt_bytes),
            "activations_selected_gb": _gb(active_act_bytes),
            "total_sharded_memory_gb": _gb(total_training_mem_bytes),
        },
        "checkpoint_storage_gb": {
            "inference_weights_bf16_gb": ckpt_weights_only_gb,
            "full_resumable_checkpoint_gb": ckpt_full_training_gb,
        },
        "compute_and_cluster": {
            "training_tokens": tokens_to_train,
            "chinchilla_optimal_tokens": chinchilla_tokens,
            "estimated_training_flops": training_flops,
            "estimated_training_zettaflops": round(training_flops / 1e21, 4),
            "target_gpu": target_gpu,
            "gpu_vram_gb": gpu_spec["vram_gb"],
            "min_gpus_for_memory_fsdp": min_gpus_for_memory,
            "recommended_min_gpus": recommended_gpus,
            "interconnect_requirement": gpu_spec["interconnect"],
            "assumed_mfu": mfu,
            "approx_duration_days_min_cluster": est_days_at_min_cluster,
            "approx_duration_days_64_gpus": est_days_at_64_gpus,
            "approx_duration_days_256_gpus": est_days_at_256_gpus,
        },
        "inference_memory_by_precision": inference_by_precision,
        "vercel_free_compatibility": {
            "training_allowed": False,
            "native_inference_compatible": False,
            "status": "UNAVAILABLE_ON_VERCEL_FREE",
            "reason": vercel_notes,
        },
        "assumptions": [
            "Analytical calculation only (zero GPUs required; not a measured training run).",
            "Mixed-precision AdamW assumes BF16 weights/grads + FP32 master weights + FP32 m/v (16 bytes/param).",
            "Training FLOPs follow 6 * N_active * D_tokens; actual wall-clock time depends on interconnect and MFU.",
            "INT4 estimate includes ~10% quantization group-scale metadata overhead (0.55 bytes/param).",
        ],
    }
