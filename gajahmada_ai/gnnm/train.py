"""
GajahMada GNNM BumbleCore — Controlled Pretraining & SFT CLI (`python -m gajahmada_ai.gnnm.train`).

Usage:
  # 1. Preflight check ONLY (never starts training)
  .venv312/bin/python -m gajahmada_ai.gnnm.train --config configs/bumblecore_300m.yaml --stage pretraining --preflight

  # 2. Small CPU smoke validation (proves end-to-end training, resume, validation, & checkpointing)
  .venv312/bin/python -m gajahmada_ai.gnnm.train --config configs/bumblecore_300m.yaml --stage pretraining --smoke --steps 4

  # 3. Full pretraining / SFT (requires passing preflight on suitable GPU hardware)
  .venv312/bin/python -m gajahmada_ai.gnnm.train --config configs/bumblecore_300m.yaml --stage pretraining --resume auto
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional

import yaml

from gajahmada_ai.gnnm.bumblecore.architecture import (
    BumbleCoreConfig,
    build_bumblecore_300m_config,
    build_bumblecore_smoke_config,
)
from gajahmada_ai.gnnm.bumblecore.data_pipeline import (
    BumbleCoreDataPipeline,
    pack_multiturn_sft_pairs,
)
from gajahmada_ai.gnnm.bumblecore.preflight import run_bumblecore_preflight
from gajahmada_ai.gnnm.dataset import GNNMCorpusBuilder
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.gnnm.trainer import GNNMTrainer


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config_from_yaml(raw: Dict[str, Any], smoke: bool = False) -> BumbleCoreConfig:
    if smoke:
        return build_bumblecore_smoke_config(vocab_size=512, max_seq_len=64)
    m = dict(raw.get("model", {}))
    valid = BumbleCoreConfig.__dataclass_fields__.keys()
    return build_bumblecore_300m_config(**{k: v for k, v in m.items() if k in valid})


def _build_seed_multiturn_sft() -> List[List[Dict[str, str]]]:
    sys_msg = "Kamu adalah GajahMada BumbleCore, model bahasa asli Indonesia. Jawab akurat, jujur, dan ringkas."
    return [
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "Siapa Mahapatih Gajah Mada?"},
            {"role": "assistant", "content": "Gajah Mada adalah Mahapatih Majapahit yang mengucapkan Sumpah Palapa pada tahun 1336."},
            {"role": "user", "content": "Apa tujuan sumpah itu?"},
            {"role": "assistant", "content": "Tujuannya adalah menyatukan seluruh wilayah Nusantara di bawah Majapahit."},
        ],
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "What is RMSNorm in a Transformer?"},
            {"role": "assistant", "content": "RMSNorm normalizes activations using their root mean square without mean centering."},
        ],
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "Jika kamu tidak tahu jawabannya, apa yang harus dilakukan?"},
            {"role": "assistant", "content": "Saya harus menyatakan dengan jujur bahwa informasi belum tersedia dan tidak mengarang jawaban."},
        ],
    ]


def execute_training_run(
    config_path: str,
    stage: str = "pretraining",
    preflight_only: bool = False,
    resume: str = "none",
    smoke: bool = False,
    dataset_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    raw_cfg = load_yaml_config(config_path)
    cfg = build_config_from_yaml(raw_cfg, smoke=smoke)
    tr_cfg = raw_cfg.get("sft" if stage == "sft" else "training", {})
    out_dir = output_dir or (
        os.path.join(raw_cfg.get("output", {}).get("checkpoint_dir", "./models/bumblecore-300m"), stage)
        if not smoke
        else os.path.join("./models/bumblecore-smoke", stage)
    )

    preflight = run_bumblecore_preflight(
        cfg=cfg,
        micro_batch_size=int(tr_cfg.get("micro_batch_size", 2)),
        grad_accum_steps=int(tr_cfg.get("gradient_accumulation_steps", 1 if smoke else 16)),
        target_tokens=raw_cfg.get("training", {}).get("target_tokens"),
        dataset_path=dataset_path,
        checkpoint_dir=out_dir,
    )
    if preflight_only:
        return {"mode": "preflight", "stage": stage, "preflight": preflight}

    if not preflight["preflight_passed"]:
        return {
            "mode": "blocked",
            "stage": stage,
            "status": "PREFLIGHT_BLOCKED",
            "preflight": preflight,
            "error": "Preflight failed; training refused to start without required hardware/resources.",
        }

    run_id = f"bc-{stage}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. Build reproducible dataset & manifest
    pipe = BumbleCoreDataPipeline(
        dataset_id=raw_cfg.get("dataset", {}).get("dataset_id", "gajahmada-bumblecore-id-en-v1"),
        dataset_version=raw_cfg.get("dataset", {}).get("dataset_version", "1.0.0"),
        train_ratio=0.80 if smoke else float(raw_cfg.get("dataset", {}).get("train_ratio", 0.90)),
        val_ratio=0.10 if smoke else float(raw_cfg.get("dataset", {}).get("val_ratio", 0.05)),
        test_ratio=0.10 if smoke else float(raw_cfg.get("dataset", {}).get("test_ratio", 0.05)),
    )
    if dataset_path and os.path.isfile(dataset_path):
        for _ in pipe.stream_jsonl(dataset_path):
            pass
    else:
        seed_builder = GNNMCorpusBuilder()
        seed_builder.load_seed_multilingual_corpus()
        for doc in seed_builder.documents:
            pipe.ingest_record(
                text=doc["text"],
                source_name=doc["source"],
                license_name="MIT-GajahMada-Seed",
                language=doc["lang"],
            )

    manifest = pipe.manifest()

    # 2. Train or load tokenizer
    tok_path = os.path.join(out_dir, "tokenizer.json")
    if resume != "none" and os.path.isfile(tok_path):
        tokenizer = GNNMBPETokenizer.load(tok_path)
    else:
        tokenizer = GNNMBPETokenizer()
        tokenizer.train([r["text"] for r in pipe.records], target_vocab_size=cfg.vocab_size)
        cfg.vocab_size = tokenizer.vocab_size

    # 3. Pack pairs for stage
    seq_len = min(cfg.max_seq_len, 64 if smoke else cfg.max_seq_len)
    if stage == "sft":
        sft_convs = _build_seed_multiturn_sft()
        train_pairs = pack_multiturn_sft_pairs(tokenizer, sft_convs, seq_len=seq_len)
        val_pairs = train_pairs[-1:] if len(train_pairs) >= 2 else train_pairs
    else:
        train_docs = pipe.get_split("train") or pipe.records
        val_docs = pipe.get_split("validation") or pipe.records[-2:]
        cb_tr = GNNMCorpusBuilder()
        cb_tr.documents = [{"text": r["text"]} for r in train_docs]
        cb_va = GNNMCorpusBuilder()
        cb_va.documents = [{"text": r["text"]} for r in val_docs]
        train_pairs = cb_tr.pack_token_blocks(tokenizer, seq_len=seq_len)
        val_pairs = cb_va.pack_token_blocks(tokenizer, seq_len=seq_len)
        if not val_pairs and len(train_pairs) >= 2:
            val_pairs = train_pairs[-2:]

    # 4. Initialize trainer & optional resume
    trainer = GNNMTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        lr=float(tr_cfg.get("lr", 3e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.01)),
        grad_clip=float(tr_cfg.get("grad_clip", 1.0)),
        seed=int(raw_cfg.get("training", {}).get("seed", 42)),
        precision=cfg.precision,
    )

    resumed_info = None
    if resume == "auto" and os.path.isfile(os.path.join(out_dir, "model.pt")):
        resumed_info = trainer.resume_from_checkpoint(out_dir)
    elif resume not in ("none", "auto") and os.path.isdir(resume):
        resumed_info = trainer.resume_from_checkpoint(resume)

    steps = steps_override if steps_override is not None else (4 if smoke else int(tr_cfg.get("max_steps", 100)))
    batch_size = 2 if smoke else int(tr_cfg.get("micro_batch_size", 2))
    grad_accum = 1 if smoke else int(tr_cfg.get("gradient_accumulation_steps", 16))
    warmup = 1 if smoke else int(tr_cfg.get("warmup_steps", 500))

    if stage == "sft":
        train_metrics = trainer.train_sft(
            train_pairs, steps=steps, batch_size=batch_size, grad_accum_steps=grad_accum, warmup_steps=warmup
        )
    else:
        train_metrics = trainer.train_steps(
            train_pairs,
            steps=steps,
            batch_size=batch_size,
            grad_accum_steps=grad_accum,
            warmup_steps=warmup,
            val_pairs=val_pairs,
            eval_interval=max(1, steps // 2),
        )

    val_metrics = trainer.evaluate_loss(val_pairs)
    os.makedirs(out_dir, exist_ok=True)
    pipe.save_manifest(os.path.join(out_dir, "dataset_manifest.json"))
    trainer.save_resumable_checkpoint(out_dir, metrics={"train": train_metrics, "validation": val_metrics})

    ended_at = datetime.now(timezone.utc).isoformat()
    run_meta = {
        "run_id": run_id,
        "stage": stage,
        "smoke_mode": smoke,
        "git_commit": _git_commit(),
        "config_sha256": _sha256_json(cfg.to_dict()),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_sha256": _sha256_json({"merges": len(tokenizer.merges), "vocab": tokenizer.vocab_size}),
        "seed": trainer.seed,
        "python_version": platform.python_version(),
        "hardware": preflight["hardware_measured"],
        "started_at": started_at,
        "ended_at": ended_at,
        "resumed_from": resumed_info,
        "actual_steps": train_metrics["steps"],
        "global_step": train_metrics["global_step"],
        "actual_tokens_processed": train_metrics["tokens_processed"],
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "checkpoint_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    return run_meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GajahMada GNNM BumbleCore Training CLI")
    parser.add_argument("--config", default="configs/bumblecore_300m.yaml", help="Path to YAML config")
    parser.add_argument("--stage", choices=["pretraining", "sft"], default="pretraining")
    parser.add_argument("--preflight", action="store_true", help="Run hardware/resource preflight only")
    parser.add_argument("--resume", default="none", help="'none', 'auto', or checkpoint dir path")
    parser.add_argument("--smoke", action="store_true", help="Run small CPU smoke validation")
    parser.add_argument("--dataset", default=None, help="Optional JSONL dataset path")
    parser.add_argument("--output-dir", default=None, help="Checkpoint output directory")
    parser.add_argument("--steps", type=int, default=None, help="Override training step count")
    args = parser.parse_args(argv)

    result = execute_training_run(
        config_path=args.config,
        stage=args.stage,
        preflight_only=args.preflight,
        resume=args.resume,
        smoke=args.smoke,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        steps_override=args.steps,
    )
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") == "PREFLIGHT_BLOCKED":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
