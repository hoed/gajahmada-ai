"""
GajahMada Platform Control Plane API Routes (`/api/v2/platform/*`).

Vercel-Free compatible, request-response only (zero background workers, zero fake data):
- GET  /api/v2/platform/status
- GET  /api/v2/platform/models
- GET  /api/v2/platform/models/{family}
- POST /api/v2/platform/factory/validate
- GET  /api/v2/platform/factory/estimator
- GET  /api/v2/platform/factory/distributed-plan
- GET  /api/v2/platform/providers
- GET  /api/v2/platform/nexus
- GET  /api/v2/platform/benchmarks
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_300m_config
from gajahmada_ai.gnnm.bumblecore.data_pipeline import validate_gnnm_tokenizer
from gajahmada_ai.gnnm.bumblecore.evaluation import RELEASE_GATES
from gajahmada_ai.gnnm.bumblecore.gate_evidence import evaluate_evidence_based_gates
from gajahmada_ai.gnnm.bumblecore.preflight import run_bumblecore_preflight
from gajahmada_ai.gnnm.config import GNNM_PRESETS
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.gnnm.distributed_prep import (
    DistributedTrainingConfig,
    inspect_distributed_runtime_readiness,
)
from gajahmada_ai.gnnm.resource_estimator import REFERENCE_GPUS, estimate_model_resources
from gajahmada_ai.knowledge.providers import RetrievalProviderRegistry
from gajahmada_ai.platform.model_factory import (
    FAMILY_METADATA,
    build_unified_model_registry,
    validate_custom_architecture,
)
from gajahmada_ai.platform.nexus import NexusPrimeOrchestrator


def create_platform_router(get_runtime: Callable[[], Dict[str, Any]], repo_root: str) -> APIRouter:
    router = APIRouter(prefix="/api/v2/platform", tags=["GajahMada Platform Control Plane"])
    models_dir = os.path.join(repo_root, "models")

    def _build_providers(rt: Dict[str, Any]) -> RetrievalProviderRegistry:
        return RetrievalProviderRegistry(
            rag_store=rt.get("rag"),
            hybrid_retriever=rt.get("hybrid_retriever"),
        )

    def _build_nexus(rt: Dict[str, Any]) -> NexusPrimeOrchestrator:
        engine = rt.get("engine")
        return NexusPrimeOrchestrator(
            model_router=getattr(engine, "router", None) if engine else None,
            provider_registry=_build_providers(rt),
            scheduler=rt.get("scheduler"),
        )

    @router.get("/status")
    def platform_status():
        rt = get_runtime()
        is_vercel = os.environ.get("VERCEL") == "1"
        engine = rt.get("engine")
        cap = engine.capability_status() if engine else {}
        neural_active = bool(cap.get("neural_inference_active", False)) and not is_vercel
        supabase_ok = bool(rt.get("supabase") and rt["supabase"].configured)
        providers = _build_providers(rt).report()

        return {
            "platform": "GajahMada AI — Sovereign Independent Intelligence",
            "runtime_environment": "VERCEL_FREE_SERVERLESS" if is_vercel else "LOCAL_OR_SERVER",
            "is_vercel": is_vercel,
            "native_model_inference": {
                "active": neural_active,
                "status_banner": (
                    f"ACTIVE ({cap.get('active_model_id')})"
                    if neural_active
                    else "NATIVE MODEL INFERENCE: UNAVAILABLE IN CURRENT RUNTIME"
                ),
                "active_backend": cap.get("active_neural_backend") if neural_active else None,
                "active_model_id": cap.get("active_model_id") if neural_active else None,
            },
            "storage": {
                "persistent_database_configured": supabase_ok or (not is_vercel),
                "mode": (
                    "SUPABASE_PGVECTOR_CONFIGURED"
                    if supabase_ok
                    else ("EPHEMERAL_VERCEL_TMP_ONLY" if is_vercel else "LOCAL_SQLITE_WAL")
                ),
                "browser_storage_supported": ["localStorage", "IndexedDB"],
                "warning": (
                    "Persistent server storage is not configured on Vercel Free; filesystem writes in /tmp do not persist across serverless cold starts. Configure optional Supabase or use browser IndexedDB/localStorage."
                    if (is_vercel and not supabase_ok)
                    else None
                ),
            },
            "retrieval_providers": providers,
            "training_execution": {
                "enabled_in_runtime": False if is_vercel else True,
                "status": "UNAVAILABLE ON VERCEL FREE" if is_vercel else "LOCAL ONLY",
            },
        }

    @router.get("/models")
    def list_unified_models():
        rt = get_runtime()
        engine = rt.get("engine")
        cap = engine.capability_status() if engine else {}
        return build_unified_model_registry(
            models_dir=models_dir,
            gnnm_registry=rt.get("gnnm_registry"),
            neural_inference_active=bool(cap.get("neural_inference_active", False)),
        )

    @router.get("/models/{family}")
    def get_model_family(family: str):
        key = family.strip().lower()
        if key not in FAMILY_METADATA:
            raise HTTPException(status_code=404, detail=f"Unknown model family '{family}'.")
        rt = get_runtime()
        engine = rt.get("engine")
        cap = engine.capability_status() if engine else {}
        reg = build_unified_model_registry(
            models_dir=models_dir,
            gnnm_registry=rt.get("gnnm_registry"),
            neural_inference_active=bool(cap.get("neural_inference_active", False)),
        )
        fam = FAMILY_METADATA[key]
        fam_models = [m for m in reg["models"] if m["model_family"].lower() == key]
        return {
            "family_info": fam,
            "models": fam_models,
            "training_execution_policy": reg["training_execution_policy"],
        }

    @router.post("/factory/validate")
    def factory_validate_architecture(payload: Dict[str, Any]):
        try:
            return validate_custom_architecture(payload)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @router.get("/factory/estimator")
    def factory_estimator(
        preset: str = Query(default="TITANFORGE-70B"),
        target_gpu: str = Query(default="H100-80GB-SXM"),
        training_tokens: Optional[int] = Query(default=None, ge=1),
        mfu: float = Query(default=0.40, ge=0.05, le=0.85),
    ):
        if preset not in GNNM_PRESETS:
            raise HTTPException(status_code=404, detail=f"Unknown preset '{preset}'. Available: {list(GNNM_PRESETS.keys())}")
        cfg = GNNM_PRESETS[preset]
        return {
            "available_presets": list(GNNM_PRESETS.keys()),
            "available_gpus": list(REFERENCE_GPUS.keys()),
            "estimate": estimate_model_resources(cfg, training_tokens=training_tokens, mfu=mfu, target_gpu=target_gpu),
        }

    @router.get("/factory/distributed-plan")
    def factory_distributed_plan(
        preset: str = Query(default="TITANFORGE-70B"),
        target_gpu: str = Query(default="H100-80GB-SXM"),
    ):
        if preset not in GNNM_PRESETS:
            raise HTTPException(status_code=404, detail=f"Unknown preset '{preset}'.")
        cfg = GNNM_PRESETS[preset]
        tp = min(8, cfg.n_kv_heads) if cfg.n_kv_heads >= 1 else 1
        dist_cfg = DistributedTrainingConfig(model_name=cfg.name, target_gpu=target_gpu, tensor_parallel_size=tp)
        return inspect_distributed_runtime_readiness(cfg, dist_cfg)

    @router.get("/providers")
    def platform_providers():
        rt = get_runtime()
        return _build_providers(rt).report()

    @router.get("/nexus")
    def platform_nexus():
        rt = get_runtime()
        return _build_nexus(rt).status_report()

    @router.get("/bumblecore")
    def platform_bumblecore():
        cfg = build_bumblecore_300m_config()
        preflight = run_bumblecore_preflight(cfg=cfg, target_tokens=3_276_800_000)
        tok_audit = validate_gnnm_tokenizer(GNNMBPETokenizer())
        gate_report = evaluate_evidence_based_gates(models_dir)
        pre_ev = gate_report["evidence"]["pretraining"]
        sft_ev = gate_report["evidence"]["sft"]
        eval_ev = gate_report["evidence"]["evaluation"]
        return {
            "release": "2.6",
            "lifecycle_stage": gate_report["lifecycle_stage"],
            "pretraining_status": gate_report["pretraining_status"],
            "status_taxonomy": [
                "CONFIGURED",
                "READY",
                "RUNNING",
                "SMOKE_TESTED",
                "TRAINED",
                "EVALUATED",
                "RELEASE_CANDIDATE",
                "PUBLISHED",
            ],
            "actual_telemetry": {
                "actual_pretraining_tokens": int(pre_ev.get("actual_tokens") or 0),
                "actual_pretraining_steps": int(pre_ev.get("actual_steps") or 0),
                "actual_sft_steps": int(sft_ev.get("actual_steps") or 0),
                "checkpoint_trust_verified": bool(pre_ev.get("passed", False)),
                "checkpoint_sha256": sft_ev.get("checkpoint_sha256") or pre_ev.get("checkpoint_sha256"),
                "evaluation_passed": bool(eval_ev.get("passed", False)),
                "evaluation_score": eval_ev.get("overall_score"),
                "evaluation_dataset_sha256": eval_ev.get("benchmark_dataset_sha256"),
                "blocked_at_gate": gate_report["release_gates"].get("blocked_at_gate"),
            },
            "theoretical_token_budget": {
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 16,
                "sequence_length": 2048,
                "tokens_per_optimizer_step": 65_536,
                "max_steps": 50_000,
                "total_tokens": 3_276_800_000,
            },
            "model_metadata": cfg.metadata(),
            "tokenizer_audit": tok_audit,
            "preflight": preflight,
            "release_gates": gate_report["release_gates"],
            "gate_evidence": gate_report["evidence"],
            "release_gate_names": RELEASE_GATES,
            "epistemic_separation": {
                "document_retrieval_is_training": False,
                "memory_storage_is_weight_modification": False,
                "neural_learning_requires_verified_checkpoint_delta": True,
            },
        }

    @router.get("/benchmarks")
    def platform_benchmarks():
        report_path = os.path.join(repo_root, "docs", "BENCHMARK_REPORT.json")
        if not os.path.isfile(report_path):
            return {
                "status": "NO DATA AVAILABLE",
                "report": None,
                "note": "No verified benchmark report file found at docs/BENCHMARK_REPORT.json.",
            }
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "status": "AVAILABLE",
                "source": "docs/BENCHMARK_REPORT.json",
                "report": data,
            }
        except Exception as exc:
            return {"status": "ERROR", "report": None, "note": str(exc)}

    return router
