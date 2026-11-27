"""
GajahMada GNNM Distributed Training & Data Pipeline Preparation (Phase 5.2 & 5.3).

Cleanly separated from Vercel serverless runtime execution.
Provides configuration schemas, validation, and readiness checks for:
- PyTorch Distributed (DDP, FSDP, Tensor Parallelism, Pipeline Parallelism)
- Gradient Accumulation, Mixed Precision, Distributed Checkpointing, Training Recovery
- Modular Data Pipeline Specification (ingestion -> cleaning -> dedup -> lang-id ->
  quality filter -> tokenization -> dataset versioning)

NON-NEGOTIABLE:
- NEVER executes large-scale training inside Vercel (`VERCEL=1`).
- NEVER simulates successful multi-GPU or multi-host distributed training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import os
from typing import Any, Dict, List

from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.resource_estimator import estimate_model_resources


@dataclass
class DataPipelineSpec:
    """Modular pretraining & SFT data pipeline configuration."""
    dataset_version: str = "gm-corpus-v1.0-planned"
    stages: List[str] = field(default_factory=lambda: [
        "dataset_ingestion",
        "text_cleaning_normalization",
        "exact_and_near_deduplication",
        "language_identification_id_en",
        "quality_and_license_filtering",
        "byte_level_bpe_tokenization",
        "sha256_dataset_versioning",
    ])
    languages: List[str] = field(default_factory=lambda: ["id", "en"])
    require_verified_license: bool = True
    dedup_method: str = "sha256_exact + minhash_lsh_planned"
    sft_separated_from_pretrain: bool = True


@dataclass
class DistributedTrainingConfig:
    """Configuration for external multi-GPU / multi-node GNNM training."""
    model_name: str = "TITANFORGE-70B"
    strategy: str = "FSDP_HYBRID_SHARD"  # "DDP" | "FSDP" | "FSDP_HYBRID_SHARD" | "TP_PP_3D"
    tensor_parallel_size: int = 8
    pipeline_parallel_size: int = 2
    data_parallel_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    distributed_checkpointing: bool = True
    checkpoint_interval_steps: int = 500
    resume_from_checkpoint: bool = True
    telemetry_enabled: bool = True
    target_gpu: str = "H100-80GB-SXM"

    @property
    def total_world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size


def inspect_distributed_runtime_readiness(
    cfg: GNNMConfig,
    dist_cfg: DistributedTrainingConfig,
) -> Dict[str, Any]:
    """
    Honest inspection of whether the current machine can launch the requested
    distributed training plan. Never fakes multi-GPU availability.
    """
    is_vercel = os.environ.get("VERCEL") == "1"
    has_torch = False
    cuda_available = False
    gpu_count = 0
    nccl_available = False

    if not is_vercel:
        try:
            torch = importlib.import_module("torch")
            has_torch = True
            cuda_available = bool(torch.cuda.is_available())
            gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
            nccl_available = bool(hasattr(torch.distributed, "is_nccl_available") and torch.distributed.is_nccl_available())
        except Exception:
            pass

    estimate = estimate_model_resources(cfg, target_gpu=dist_cfg.target_gpu)
    required_gpus = max(dist_cfg.total_world_size, estimate["compute_and_cluster"]["min_gpus_for_memory_fsdp"])

    blockers: List[str] = []
    if is_vercel:
        blockers.append("VERCEL_RUNTIME_FORBIDDEN: Large-scale training is strictly disabled on Vercel Free.")
    if not has_torch:
        blockers.append("PYTORCH_UNAVAILABLE: PyTorch is not installed in the active Python environment.")
    if not cuda_available or gpu_count == 0:
        blockers.append("NO_GPU_DETECTED: Zero CUDA GPUs detected in current runtime.")
    elif gpu_count < required_gpus:
        blockers.append(f"INSUFFICIENT_GPUS: Detected {gpu_count} GPU(s), plan requires at least {required_gpus}.")
    if cfg.n_heads % dist_cfg.tensor_parallel_size != 0:
        blockers.append(f"INVALID_TP: n_heads ({cfg.n_heads}) must be divisible by tensor_parallel_size ({dist_cfg.tensor_parallel_size}).")
    if cfg.n_kv_heads % dist_cfg.tensor_parallel_size != 0 and dist_cfg.tensor_parallel_size > 1:
        blockers.append(f"INVALID_TP_KV: n_kv_heads ({cfg.n_kv_heads}) must be divisible by tensor_parallel_size ({dist_cfg.tensor_parallel_size}).")

    can_execute = len(blockers) == 0
    return {
        "execution_ready": can_execute,
        "status": "READY_EXTERNAL_CLUSTER" if can_execute else ("UNAVAILABLE_ON_VERCEL_FREE" if is_vercel else "NOT_CONFIGURED"),
        "distributed_tests_executed": False,
        "runtime_environment": {
            "is_vercel": is_vercel,
            "pytorch_installed": has_torch,
            "cuda_available": cuda_available,
            "detected_gpu_count": gpu_count,
            "nccl_available": nccl_available,
        },
        "distributed_config": {
            **asdict(dist_cfg),
            "total_world_size": dist_cfg.total_world_size,
        },
        "data_pipeline": asdict(DataPipelineSpec()),
        "resource_estimate_summary": {
            "total_parameters_b": estimate["parameters"]["total_billions"],
            "total_sharded_memory_gb": estimate["training_memory_gb"]["total_sharded_memory_gb"],
            "min_gpus_for_memory": estimate["compute_and_cluster"]["min_gpus_for_memory_fsdp"],
        },
        "blockers": blockers,
    }
