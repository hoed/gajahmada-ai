"""
GajahMada AI — Hardened Isolated Knowledge-to-Model Training Pipeline (Phase 1 P0).

Strict Training Integrity Guarantees:
- NEVER enters `EVALUATING` or `APPROVED` when PyTorch training was NOT actually executed.
- Explicitly distinguishes:
  - `TRAINING_NOT_EXECUTED` (e.g., PyTorch unavailable or no valid approved training pairs)
  - `TRAINING_FAILED` (runtime exception during training)
  - `CHECKPOINT_MISSING` (base or candidate checkpoint artifacts missing)
  - `CHECKPOINT_INVALID` (corrupted JSON/weights, architecture mismatch, or failed inference smoke test)
  - `EVALUATION_FAILED` / `REJECTED` (failed ModelEvaluationGate)
  - `APPROVED` (passed real training, checkpoint verification, real inference, and EvaluationGate)
- Records SHA-256 checkpoint integrity, parent vs. candidate weight hashes, dataset fingerprint,
  config fingerprint, training duration, and loss history.
- NEVER overwrites the active production checkpoint without passing evaluation AND explicit approval.
"""
import hashlib
import json
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from gajahmada_ai.knowledge.interfaces import KnowledgeRecord, KnowledgeSource
from gajahmada_ai.learning.dataset_builder import KnowledgeDatasetBuilder
from gajahmada_ai.learning.evaluation_gate import ModelEvaluationGate
from gajahmada_ai.learning.model_registry import (
    GNNMModelRegistry,
    ModelPromotionStatus,
    compute_checkpoint_checksum,
)


class KnowledgeToModelPipeline:
    """Controlled, reproducible, verifiable pipeline for internalizing verified knowledge into GNNM."""

    def __init__(
        self,
        model_registry: GNNMModelRegistry,
        dataset_builder: Optional[KnowledgeDatasetBuilder] = None,
        evaluation_gate: Optional[ModelEvaluationGate] = None,
        workspace_dir: str = "./knowledge_store/training_runs",
    ):
        self.model_registry = model_registry
        self.dataset_builder = dataset_builder or KnowledgeDatasetBuilder()
        self.evaluation_gate = evaluation_gate or ModelEvaluationGate()
        self.workspace_dir = workspace_dir

    def run_training_job(
        self,
        base_checkpoint_dir: str,
        mode: str,  # "continued_pretraining" | "sft"
        records: Optional[List[KnowledgeRecord]] = None,
        sft_examples: Optional[List[Dict[str, Any]]] = None,
        sources_by_id: Optional[Dict[str, KnowledgeSource]] = None,
        seed: int = 42,
        steps: int = 4,
        lr: float = 1e-4,
        candidate_eval_fn: Optional[Callable[[str, str], str]] = None,
        baseline_eval_fn: Optional[Callable[[str, str], str]] = None,
        baseline_perplexity: Optional[float] = None,
        auto_promote_if_approved: bool = False,
        operator_approved: bool = False,
    ) -> Dict[str, Any]:
        if mode not in ("continued_pretraining", "sft"):
            raise ValueError("mode must be 'continued_pretraining' or 'sft'")

        run_id = f"gnnm-cand-{uuid.uuid4().hex[:8]}"
        run_dir = os.path.join(self.workspace_dir, run_id)
        candidate_ckpt_dir = os.path.join(run_dir, "checkpoint")
        dataset_dir = os.path.join(run_dir, "dataset")
        os.makedirs(run_dir, exist_ok=True)
        t_job_start = time.perf_counter()

        # 1. Verify base checkpoint exists and is intact BEFORE training
        base_ok, base_sha, base_files = compute_checkpoint_checksum(base_checkpoint_dir)
        if not base_ok:
            return {
                "run_id": run_id,
                "status": "CHECKPOINT_MISSING",
                "error": f"Base checkpoint at '{base_checkpoint_dir}' is missing or incomplete.",
                "active_version_id": self.model_registry.active_version_id,
            }

        # 2. Build filtered, provenance-tracked dataset
        if mode == "continued_pretraining":
            ds_pkg = self.dataset_builder.build_continued_pretraining_dataset(
                records=records or [],
                sources_by_id=sources_by_id,
                output_dir=dataset_dir,
            )
        else:
            rec_map = {r.record_id: r for r in (records or [])}
            ds_pkg = self.dataset_builder.build_sft_dataset(
                examples=sft_examples or [],
                records_by_id=rec_map,
                sources_by_id=sources_by_id,
                output_dir=dataset_dir,
            )

        manifest = ds_pkg["manifest"]
        if manifest["item_count"] == 0:
            return {
                "run_id": run_id,
                "status": "TRAINING_NOT_EXECUTED",
                "reason": "no_approved_training_data",
                "dataset_manifest": manifest,
                "rejected": ds_pkg["rejected"],
                "active_version_id": self.model_registry.active_version_id,
            }

        # 3. Check PyTorch availability BEFORE registering or pretending to train
        from gajahmada_ai.gnnm.model import HAS_TORCH

        if not HAS_TORCH:
            return {
                "run_id": run_id,
                "status": "TRAINING_NOT_EXECUTED",
                "reason": "pytorch_not_installed",
                "dataset_manifest": manifest,
                "active_version_id": self.model_registry.active_version_id,
            }

        # 4. Clone base checkpoint into isolated candidate workspace
        shutil.copytree(base_checkpoint_dir, candidate_ckpt_dir, dirs_exist_ok=True)

        train_cfg = {"mode": mode, "seed": seed, "steps": steps, "lr": lr}
        cfg_fingerprint = hashlib.sha256(
            json.dumps(train_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.model_registry.register_version(
            version_id=run_id,
            model_name=os.path.basename(os.path.normpath(base_checkpoint_dir)) or "GNNM",
            architecture="GNNM-DecoderOnly-Transformer",
            checkpoint_path=candidate_ckpt_dir,
            base_checkpoint=base_checkpoint_dir,
            dataset_versions=[manifest["dataset_id"]],
            training_config={**train_cfg, "config_fingerprint": cfg_fingerprint},
            provenance={
                "source_ids": manifest["source_ids"],
                "dataset_sha256": manifest["sha256"],
                "parent_checkpoint_sha256": base_sha,
            },
            status=ModelPromotionStatus.TRAINING,
            checkpoint_sha256=base_sha,
        )

        # 5. Execute REAL PyTorch GNNMTrainer update
        train_metrics: Dict[str, Any] = {
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "config_fingerprint": cfg_fingerprint,
            "dataset_sha256": manifest["sha256"],
            "torch_executed": False,
        }
        cand_ppl: Optional[float] = None
        try:
            from gajahmada_ai.gnnm.config import GNNMConfig
            from gajahmada_ai.gnnm.dataset import GNNMCorpusBuilder
            from gajahmada_ai.gnnm.inference import GNNMInferenceEngine
            from gajahmada_ai.gnnm.model import torch
            from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
            from gajahmada_ai.gnnm.trainer import GNNMTrainer

            torch.manual_seed(seed)
            cfg = GNNMConfig.load(os.path.join(candidate_ckpt_dir, "config.json"))
            tok = GNNMBPETokenizer.load(os.path.join(candidate_ckpt_dir, "tokenizer.json"))
            trainer = GNNMTrainer(cfg, tok, lr=lr)
            trainer.model.load_state_dict(
                torch.load(os.path.join(candidate_ckpt_dir, "model.pt"), map_location=trainer.device, weights_only=True)
            )
            cb = GNNMCorpusBuilder()
            if mode == "continued_pretraining":
                for it in ds_pkg["items"]:
                    cb.add_document(it["text"], source=it.get("source_id", "approved"))
                pairs = cb.pack_token_blocks(tok, seq_len=min(64, cfg.max_seq_len))
                if not pairs:
                    self.model_registry.update_status(run_id, ModelPromotionStatus.REJECTED)
                    return {
                        "run_id": run_id,
                        "status": "TRAINING_NOT_EXECUTED",
                        "reason": "insufficient_tokens_for_sequence_packing",
                        "active_version_id": self.model_registry.active_version_id,
                    }
                tm = trainer.train_steps(pairs, steps=steps, batch_size=min(2, len(pairs)))
                val_m = trainer.evaluate_loss(pairs[:4])
                cand_ppl = val_m["perplexity"]
                train_metrics.update({
                    **tm,
                    **val_m,
                    "loss_history": list(trainer.loss_history),
                    "torch_executed": True,
                })
            else:
                sft_pairs = cb.pack_instruction_pairs(tok, ds_pkg["items"], seq_len=min(64, cfg.max_seq_len))
                if not sft_pairs:
                    self.model_registry.update_status(run_id, ModelPromotionStatus.REJECTED)
                    return {
                        "run_id": run_id,
                        "status": "TRAINING_NOT_EXECUTED",
                        "reason": "no_valid_sft_pairs",
                        "active_version_id": self.model_registry.active_version_id,
                    }
                tm = trainer.train_sft(sft_pairs, steps=steps, batch_size=min(2, len(sft_pairs)), sft_lr=lr)
                val_m = trainer.evaluate_loss(sft_pairs[:4])
                cand_ppl = val_m["perplexity"]
                train_metrics.update({
                    **tm,
                    **val_m,
                    "loss_history": list(trainer.loss_history),
                    "torch_executed": True,
                })

            train_metrics["training_duration_sec"] = round(time.perf_counter() - t_job_start, 3)
            trainer.save_checkpoint(candidate_ckpt_dir, metrics=train_metrics)
        except Exception as exc:
            self.model_registry.update_status(
                run_id,
                ModelPromotionStatus.REJECTED,
                evaluation_metrics={"error": str(exc), "status": "TRAINING_FAILED"},
            )
            return {
                "run_id": run_id,
                "status": "TRAINING_FAILED",
                "error": str(exc),
                "active_version_id": self.model_registry.active_version_id,
            }

        # 6. Strict Candidate Checkpoint Integrity & Real Inference Verification
        cand_ok, cand_sha, cand_files = compute_checkpoint_checksum(candidate_ckpt_dir)
        if not cand_ok:
            self.model_registry.update_status(run_id, ModelPromotionStatus.REJECTED)
            return {
                "run_id": run_id,
                "status": "CHECKPOINT_INVALID",
                "error": "Candidate checkpoint missing or failed structural verification after training.",
                "active_version_id": self.model_registry.active_version_id,
            }

        weights_changed = cand_files.get("model.pt") != base_files.get("model.pt")
        train_metrics["candidate_checkpoint_sha256"] = cand_sha
        train_metrics["parent_checkpoint_sha256"] = base_sha
        train_metrics["weights_changed"] = weights_changed

        try:
            loaded_engine = GNNMInferenceEngine(candidate_ckpt_dir)
            smoke_text = loaded_engine.generate("GajahMada", max_new_tokens=8, temperature=0.5)
            train_metrics["inference_smoke_verified"] = True
            train_metrics["inference_smoke_sample_len"] = len(smoke_text)
        except Exception as exc:
            self.model_registry.update_status(
                run_id,
                ModelPromotionStatus.REJECTED,
                evaluation_metrics={"error": str(exc), "status": "CHECKPOINT_INVALID"},
            )
            return {
                "run_id": run_id,
                "status": "CHECKPOINT_INVALID",
                "error": f"Candidate checkpoint failed to load or execute inference: {exc}",
                "active_version_id": self.model_registry.active_version_id,
            }

        # 7. Enter EVALUATING ONLY after real training + valid checkpoint + real inference passed!
        self.model_registry.update_status(
            run_id,
            ModelPromotionStatus.EVALUATING,
            checkpoint_sha256=cand_sha,
        )

        # Evaluate using the REAL loaded candidate engine unless an explicit evaluator callable is supplied
        if candidate_eval_fn is None:
            def candidate_eval_fn(prompt: str, ctx: str) -> str:
                full = f"{ctx}\n{prompt}" if ctx else prompt
                return loaded_engine.generate(full[: loaded_engine.cfg.max_seq_len], max_new_tokens=32, temperature=0.3)

        eval_report = self.evaluation_gate.evaluate_candidate(
            candidate_fn=candidate_eval_fn,
            baseline_fn=baseline_eval_fn,
            candidate_perplexity=cand_ppl,
            baseline_perplexity=baseline_perplexity,
            model_version=run_id,
            training_texts=[it.get("text", it.get("instruction", "")) for it in ds_pkg["items"]],
        )

        # 8. Transition status strictly based on measured evaluation results
        if eval_report["passed"]:
            self.model_registry.update_status(
                run_id,
                ModelPromotionStatus.APPROVED,
                evaluation_metrics={"training": train_metrics, "evaluation": eval_report},
                checkpoint_sha256=cand_sha,
            )
            final_status = ModelPromotionStatus.APPROVED.value
            if auto_promote_if_approved and operator_approved:
                self.model_registry.promote(run_id, operator_approved=True)
                final_status = ModelPromotionStatus.ACTIVE.value
        else:
            self.model_registry.update_status(
                run_id,
                ModelPromotionStatus.REJECTED,
                evaluation_metrics={"training": train_metrics, "evaluation": eval_report},
                checkpoint_sha256=cand_sha,
            )
            final_status = "EVALUATION_FAILED"

        return {
            "run_id": run_id,
            "status": final_status,
            "candidate_checkpoint_dir": candidate_ckpt_dir,
            "checkpoint_sha256": cand_sha,
            "weights_changed": weights_changed,
            "dataset_manifest": manifest,
            "training_metrics": train_metrics,
            "evaluation_report": eval_report,
            "active_version_id": self.model_registry.active_version_id,
            "completed_at": round(time.time(), 3),
        }
