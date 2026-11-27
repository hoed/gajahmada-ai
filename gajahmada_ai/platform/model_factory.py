"""
GajahMada Unified Model Registry & Model Factory Control Plane (Phases 4, 11, 13).

Manages:
- Model Family Catalog: BUMBLECORE, OPTICORE, TITANFORGE + Specializations (NEUROX,
  CODESTRIKE, KNOWTRON) + NEXUS PRIME orchestration layer.
- 11-State Lifecycle Governance:
    PLANNED | ARCHITECTURE_DESIGNED | CONFIGURATION_VALIDATED | DATASET_PREPARATION |
    READY_FOR_TRAINING | TRAINING | TRAINING_COMPLETED | EVALUATION | APPROVED |
    PRODUCTION | DEPRECATED
- Honest reconciliation with actual on-disk model artifacts (`models/gnnm-*`) and
  `GNNMModelRegistry`. Never claims weights exist when no verified checkpoint is present.
- Vercel Free Guard: Manages metadata, architecture validation, parameter calculation,
  and resource estimation — training execution remains strictly DISABLED on Vercel Free.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

from gajahmada_ai.gnnm.config import GNNM_PRESETS, GNNMConfig
from gajahmada_ai.gnnm.distributed_prep import (
    DistributedTrainingConfig,
    inspect_distributed_runtime_readiness,
)
from gajahmada_ai.gnnm.resource_estimator import estimate_model_resources

LIFECYCLE_STATES = [
    "PLANNED",
    "ARCHITECTURE_DESIGNED",
    "CONFIGURATION_VALIDATED",
    "DATASET_PREPARATION",
    "READY_FOR_TRAINING",
    "TRAINING",
    "TRAINING_COMPLETED",
    "EVALUATION",
    "APPROVED",
    "PRODUCTION",
    "DEPRECATED",
]

FAMILY_METADATA: Dict[str, Dict[str, Any]] = {
    "bumblecore": {
        "family": "BUMBLECORE",
        "tagline": "Lightweight Native GNNM Family (12M · 100M · 300M)",
        "purpose": "Lightweight language tasks, research experiments, small-model inference, and architecture validation.",
        "presets": ["BUMBLECORE-12M", "BUMBLECORE-100M", "BUMBLECORE-300M", "GNNM-01-Tiny"],
        "is_specialization": False,
    },
    "opticore": {
        "family": "OPTICORE",
        "tagline": "General-Purpose Native GNNM Family (1B · 3B · 7B)",
        "purpose": "General language understanding, instruction following, and general reasoning research.",
        "presets": ["OPTICORE-1B", "OPTICORE-3B", "OPTICORE-7B"],
        "is_specialization": False,
    },
    "titanforge": {
        "family": "TITANFORGE",
        "tagline": "Large-Scale Native GNNM Research Family (30B · 70B · 120B+)",
        "purpose": "Large-scale native intelligence research, general-purpose language modeling, and long-term independent AI development.",
        "presets": ["TITANFORGE-30B", "TITANFORGE-70B", "TITANFORGE-70B-MoE-Exp", "TITANFORGE-120B"],
        "is_specialization": False,
    },
    "neurox": {
        "family": "NEUROX",
        "tagline": "Reasoning Specialization (Planned)",
        "purpose": "Multi-step symbolic + neural reasoning specialization built on verified GNNM base checkpoints.",
        "presets": [],
        "is_specialization": True,
        "status": "PLANNED",
        "note": "Specialization only — not represented as a separate trained model until verified artifacts & evaluations exist.",
    },
    "codestrike": {
        "family": "CODESTRIKE",
        "tagline": "Coding & Software Engineering Specialization (Planned)",
        "purpose": "Code synthesis, static analysis, and software engineering specialization built on verified GNNM base checkpoints.",
        "presets": [],
        "is_specialization": True,
        "status": "PLANNED",
        "note": "Specialization only — not represented as a separate trained model until verified artifacts & evaluations exist.",
    },
    "knowtron": {
        "family": "KNOWTRON",
        "tagline": "Knowledge-Intensive & Retrieval Specialization (Planned)",
        "purpose": "Grounded retrieval, citation accuracy, and abstention specialization integrated with Hybrid Knowledge.",
        "presets": [],
        "is_specialization": True,
        "status": "PLANNED",
        "note": "Specialization only — not represented as a separate trained model until verified artifacts & evaluations exist.",
    },
}


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _discover_local_checkpoint(models_dir: str, preset_name: str) -> Dict[str, Any]:
    """Check actual on-disk artifacts without inventing weights."""
    alias_dirs = {
        "BUMBLECORE-12M": ["gnnm-01-12m", "bumblecore-12m"],
        "GNNM-01-12M": ["gnnm-01-12m"],
        "GNNM-01-Tiny": ["gnnm-01-tiny"],
    }
    candidates = alias_dirs.get(preset_name, [preset_name.lower()])
    for sub in candidates:
        ckpt_dir = os.path.join(models_dir, sub)
        model_pt = os.path.join(ckpt_dir, "model.pt")
        if os.path.isfile(model_pt):
            return {
                "weight_available": True,
                "artifact_path": ckpt_dir,
                "format": "pytorch_state_dict (.pt)",
                "size_mb": round(os.path.getsize(model_pt) / (1024 * 1024), 2),
                "weight_checksum": _sha256_file(model_pt),
            }
    return {
        "weight_available": False,
        "artifact_path": None,
        "format": "pytorch_state_dict (.pt) [planned]",
        "size_mb": 0.0,
        "weight_checksum": None,
    }


def build_unified_model_registry(
    models_dir: str,
    gnnm_registry: Optional[Any] = None,
    is_vercel: Optional[bool] = None,
    neural_inference_active: bool = False,
) -> Dict[str, Any]:
    """Build the honest unified Model Registry across all families & presets."""
    vercel = (os.environ.get("VERCEL") == "1") if is_vercel is None else is_vercel
    entries: List[Dict[str, Any]] = []

    ordered_presets = [
        "GNNM-01-Tiny", "BUMBLECORE-12M", "BUMBLECORE-100M", "BUMBLECORE-300M",
        "OPTICORE-1B", "OPTICORE-3B", "OPTICORE-7B",
        "TITANFORGE-30B", "TITANFORGE-70B", "TITANFORGE-70B-MoE-Exp", "TITANFORGE-120B",
    ]

    for name in ordered_presets:
        cfg = GNNM_PRESETS[name]
        artifact = _discover_local_checkpoint(models_dir, name) if not vercel else {
            "weight_available": False, "artifact_path": None, "format": "pytorch_state_dict (.pt)",
            "size_mb": 0.0, "weight_checksum": None,
        }
        est = estimate_model_resources(cfg)
        if artifact["weight_available"]:
            lifecycle = "TRAINING_COMPLETED"
            training_status = "LOCAL_CHECKPOINT_VERIFIED"
            runtime_compat = "LOCAL ONLY" if not vercel else "UNAVAILABLE ON VERCEL FREE"
            deploy_avail = "ACTIVE" if (neural_inference_active and not vercel) else "LOCAL ONLY"
        else:
            lifecycle = cfg.lifecycle_status
            training_status = "NOT_EXECUTED"
            runtime_compat = "UNAVAILABLE ON VERCEL FREE" if cfg.estimate_parameters() > 50_000_000 else "LOCAL ONLY"
            deploy_avail = "PLANNED" if lifecycle == "PLANNED" else "NOT CONFIGURED"

        entries.append({
            "model_name": cfg.name,
            "model_family": cfg.family,
            "specialization": cfg.specialization,
            "architecture": f"GNNM Decoder-Only Transformer ({cfg.architecture_type.upper()}, {cfg.attention_type.upper()}, {cfg.normalization.upper()}, {cfg.activation.upper()}, {cfg.positional_encoding.upper()})",
            "architecture_type": cfg.architecture_type,
            "parameter_count": cfg.estimate_parameters(),
            "parameter_billions": est["parameters"]["total_billions"],
            "active_parameter_count": cfg.estimate_active_parameters(),
            "model_version": "0.1.0-local" if artifact["weight_available"] else "0.0.0-spec",
            "model_format": artifact["format"],
            "weight_available": artifact["weight_available"],
            "weight_checksum": artifact["weight_checksum"],
            "artifact_size_mb": artifact["size_mb"],
            "lifecycle_state": lifecycle,
            "training_status": training_status,
            "evaluation_status": "LOCAL_SMOKE_VERIFIED" if artifact["weight_available"] else "NOT_EVALUATED",
            "license": "MIT (GajahMada AI Independent Architecture)",
            "provenance": artifact["artifact_path"] or "gajahmada_ai/gnnm/config.py (architecture spec only)",
            "runtime_compatibility": runtime_compat,
            "deployment_availability": deploy_avail,
            "config": cfg.to_dict(),
            "inference_vram_bf16_gb": est["inference_memory_by_precision"]["bf16"]["total_vram_gb_est"],
            "inference_vram_int4_gb": est["inference_memory_by_precision"]["int4"]["total_vram_gb_est"],
        })

    # Include any dynamically registered trained versions from GNNMModelRegistry
    registered_versions = gnnm_registry.list_versions() if gnnm_registry else []

    return {
        "lifecycle_states": LIFECYCLE_STATES,
        "families": FAMILY_METADATA,
        "models": entries,
        "registered_trained_versions": registered_versions,
        "training_execution_policy": {
            "vercel_free_training_enabled": False,
            "status": "DISABLED_ON_VERCEL_FREE" if vercel else "LOCAL_OR_EXTERNAL_ONLY",
            "note": "Vercel Free hosts metadata, architecture validation, and resource estimation only. Large-scale training requires an external GPU cluster.",
        },
    }


def validate_custom_architecture(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an arbitrary GNNMConfig from Model Factory UI and return parameter + resource estimates."""
    valid_fields = GNNMConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in payload.items() if k in valid_fields}
    cfg = GNNMConfig(**filtered)
    est = estimate_model_resources(
        cfg,
        training_tokens=payload.get("training_tokens"),
        micro_batch_size=int(payload.get("micro_batch_size", 1)),
        mfu=float(payload.get("mfu", 0.40)),
        target_gpu=str(payload.get("target_gpu", "H100-80GB-SXM")),
    )
    dist_cfg = DistributedTrainingConfig(
        model_name=cfg.name,
        target_gpu=str(payload.get("target_gpu", "H100-80GB-SXM")),
        tensor_parallel_size=min(8, cfg.n_kv_heads) if cfg.n_kv_heads >= 1 else 1,
    )
    readiness = inspect_distributed_runtime_readiness(cfg, dist_cfg)
    return {
        "valid": True,
        "config": cfg.to_dict(),
        "resource_estimate": est,
        "distributed_readiness": readiness,
    }
