"""
GajahMada GNNM BumbleCore — Hardened Pretraining & SFT CLI (Release 2.6).

Enforces:
- Strict dataset governance: seed corpus allowed ONLY with `--smoke`; full pretraining and
  full SFT strictly reject missing/unapproved external datasets & manifests.
- Tokenizer lock BEFORE model instantiation (verifies vocab size & 303M tolerance).
- True token accounting & stopping at `target_tokens`.
- Exact configured SFT learning rate (no hidden upward override).
- Verifiable run metadata & provenance hashes (`config_sha256`, `dataset_manifest_sha256`, `tokenizer_sha256`).
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
from gajahmada_ai.gnnm.bumblecore.dataset_governance import (
    sha256_file,
    validate_dataset_and_manifest,
    validate_sft_dataset,
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


def build_config_from_yaml(raw: Dict[str, Any], smoke: bool = False, vocab_override: Optional[int] = None) -> BumbleCoreConfig:
    if smoke:
        return build_bumblecore_smoke_config(vocab_size=vocab_override or 512, max_seq_len=64)
    m = dict(raw.get("model", {}))
    if vocab_override is not None:
        m["vocab_size"] = vocab_override
    valid = BumbleCoreConfig.__dataclass_fields__.keys()
    return build_bumblecore_300m_config(**{k: v for k, v in m.items() if k in valid})


def _build_seed_multiturn_sft() -> List[List[Dict[str, str]]]:
    """Seed SFT conversations permitted ONLY for `--smoke` validation."""
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


def _load_jsonl_texts(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("text"):
                texts.append(str(obj["text"]))
    return texts


def _load_sft_conversations(path: str) -> List[List[Dict[str, str]]]:
    convs: List[List[Dict[str, str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
                convs.append(obj["messages"])
    return convs


def execute_training_run(
    config_path: str,
    stage: str = "pretraining",
    preflight_only: bool = False,
    resume: str = "none",
    smoke: bool = False,
    dataset_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    steps_override: Optional[int] = None,
    target_tokens_override: Optional[int] = None,
    allow_resume_mismatch: bool = False,
) -> Dict[str, Any]:
    raw_cfg = load_yaml_config(config_path)
    cfg = build_config_from_yaml(raw_cfg, smoke=smoke)
    tr_cfg = raw_cfg.get("sft" if stage == "sft" else "training", {})
    out_dir = output_dir or (
        os.path.join(raw_cfg.get("output", {}).get("checkpoint_dir", "./models/bumblecore-300m"), stage)
        if not smoke
        else os.path.join("./models/bumblecore-smoke", stage)
    )

    # ── 1. Strict Dataset Governance (NO seed fallback for full pretraining or full SFT) ──
    dataset_validation: Optional[Dict[str, Any]] = None
    if not smoke:
        if not dataset_path:
            return {
                "mode": "blocked",
                "stage": stage,
                "status": "DATASET_GOVERNANCE_FAILED",
                "error": (
                    f"Full {stage} strictly requires an approved external dataset directory with dataset_manifest.json "
                    "(--dataset <dir>). Seed corpus is permitted ONLY with --smoke."
                ),
            }
        if stage == "sft":
            min_sft = int(raw_cfg.get("sft", {}).get("min_sft_examples", 100))
            dataset_validation = validate_sft_dataset(dataset_path, min_examples=min_sft, require_approved=True)
        else:
            dataset_validation = validate_dataset_and_manifest(dataset_path, require_approved=True, min_documents=1)
        if not dataset_validation["valid"]:
            return {
                "mode": "blocked",
                "stage": stage,
                "status": "DATASET_GOVERNANCE_FAILED",
                "dataset_errors": dataset_validation["errors"],
                "error": "External dataset manifest/license/integrity validation failed; training refused.",
            }

    preflight = run_bumblecore_preflight(
        cfg=cfg,
        micro_batch_size=int(tr_cfg.get("micro_batch_size", 2)),
        grad_accum_steps=int(tr_cfg.get("gradient_accumulation_steps", 1 if smoke else 16)),
        target_tokens=target_tokens_override or raw_cfg.get("training", {}).get("target_tokens"),
        dataset_path=dataset_path,
        checkpoint_dir=out_dir,
    )
    if preflight_only:
        return {"mode": "preflight", "stage": stage, "preflight": preflight, "dataset_validation": dataset_validation}

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

    # ── 2. Load dataset splits (external validated OR smoke seed) ──
    if smoke:
        pipe = BumbleCoreDataPipeline(
            dataset_id="smoke-seed-validation",
            dataset_version="smoke-1.0",
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
        )
        seed_builder = GNNMCorpusBuilder()
        seed_builder.load_seed_multilingual_corpus()
        for doc in seed_builder.documents:
            pipe.ingest_record(text=doc["text"], source_name=doc["source"], license_name="MIT", language=doc["lang"])
        manifest = pipe.manifest()
        dataset_manifest_sha256 = manifest["manifest_sha256"]
        train_texts = [r["text"] for r in (pipe.get_split("train") or pipe.records)]
        val_texts = [r["text"] for r in (pipe.get_split("validation") or pipe.records[-2:])]
        sft_convs = _build_seed_multiturn_sft()
        sft_val_convs = sft_convs[-1:]
    else:
        assert dataset_validation is not None and dataset_path is not None
        manifest = dataset_validation["manifest"]
        dataset_manifest_sha256 = sha256_file(os.path.join(dataset_path, "dataset_manifest.json"))
        tr_file = os.path.join(dataset_path, manifest.get("train_split", "train.jsonl"))
        va_file = os.path.join(dataset_path, manifest.get("validation_split", "validation.jsonl"))
        if stage == "sft":
            sft_convs = _load_sft_conversations(tr_file)
            sft_val_convs = _load_sft_conversations(va_file) if os.path.isfile(va_file) else sft_convs[-2:]
            train_texts = []
            val_texts = []
        else:
            train_texts = _load_jsonl_texts(tr_file)
            val_texts = _load_jsonl_texts(va_file) if os.path.isfile(va_file) else []
            sft_convs = []
            sft_val_convs = []

    # ── 3. Tokenizer Lock BEFORE Model Instantiation (Phase 3) ──
    resolved_tok_path = tokenizer_path or (os.path.join(out_dir, "tokenizer.json") if resume != "none" else None)
    if resolved_tok_path and os.path.isfile(resolved_tok_path):
        tokenizer = GNNMBPETokenizer.load(resolved_tok_path)
    else:
        tokenizer = GNNMBPETokenizer()
        if train_texts:
            tokenizer.train(train_texts, target_vocab_size=cfg.vocab_size)

    # Rebuild & re-validate config against locked tokenizer vocab_size BEFORE model instantiation
    cfg = build_config_from_yaml(raw_cfg, smoke=smoke, vocab_override=tokenizer.vocab_size)
    tokenizer_sha256 = _sha256_json({"format": "gnnm-bpe-v1", "vocab_size": tokenizer.vocab_size, "merges": tokenizer.merges})
    config_sha256 = _sha256_json(cfg.to_dict())

    # ── 4. Pack Sequences ──
    seq_len = min(cfg.max_seq_len, 64 if smoke else cfg.max_seq_len)
    if stage == "sft":
        train_pairs = pack_multiturn_sft_pairs(tokenizer, sft_convs, seq_len=seq_len)
        val_pairs = pack_multiturn_sft_pairs(tokenizer, sft_val_convs, seq_len=seq_len)
    else:
        cb_tr = GNNMCorpusBuilder()
        cb_tr.documents = [{"text": t} for t in train_texts]
        cb_va = GNNMCorpusBuilder()
        cb_va.documents = [{"text": t} for t in val_texts]
        train_pairs = cb_tr.pack_token_blocks(tokenizer, seq_len=seq_len)
        val_pairs = cb_va.pack_token_blocks(tokenizer, seq_len=seq_len)
        if smoke and not val_pairs and len(train_pairs) >= 2:
            val_pairs = train_pairs[-2:]

    # ── 5. Initialize Trainer with Exact Configured LR & Provenance Hashes ──
    configured_lr = float(tr_cfg.get("lr", 5e-5 if stage == "sft" else 3e-4))
    trainer = GNNMTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        lr=configured_lr,
        min_lr_ratio=float(tr_cfg.get("min_lr_ratio", 0.1)),
        weight_decay=float(raw_cfg.get("training", {}).get("weight_decay", 0.01)),
        grad_clip=float(raw_cfg.get("training", {}).get("grad_clip", 1.0)),
        seed=int(raw_cfg.get("training", {}).get("seed", 42)),
        precision=cfg.precision,
        provenance_hashes={
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_sha256": tokenizer_sha256,
        },
    )

    resumed_info = None
    if resume == "auto" and os.path.isfile(os.path.join(out_dir, "model.pt")):
        resumed_info = trainer.resume_from_checkpoint(
            out_dir,
            expected_dataset_hash=dataset_manifest_sha256,
            expected_tokenizer_hash=tokenizer_sha256,
            allow_resume_mismatch=allow_resume_mismatch,
        )
    elif resume not in ("none", "auto") and os.path.isdir(resume):
        resumed_info = trainer.resume_from_checkpoint(
            resume,
            expected_dataset_hash=dataset_manifest_sha256,
            expected_tokenizer_hash=tokenizer_sha256,
            allow_resume_mismatch=allow_resume_mismatch,
        )

    steps = steps_override if steps_override is not None else (4 if smoke else int(tr_cfg.get("max_steps", 100)))
    batch_size = 2 if smoke else int(tr_cfg.get("micro_batch_size", 2))
    grad_accum = 1 if smoke else int(tr_cfg.get("gradient_accumulation_steps", 16))
    warmup = 1 if smoke else int(tr_cfg.get("warmup_steps", 500))
    target_toks = target_tokens_override if target_tokens_override is not None else (
        None if smoke else raw_cfg.get("training", {}).get("target_tokens")
    )

    if stage == "sft":
        train_metrics = trainer.train_sft(
            train_pairs,
            steps=steps,
            batch_size=batch_size,
            sft_lr=configured_lr,
            grad_accum_steps=grad_accum,
            warmup_steps=warmup,
            target_tokens=target_toks,
        )
    else:
        train_metrics = trainer.train_steps(
            train_pairs,
            steps=steps,
            batch_size=batch_size,
            grad_accum_steps=grad_accum,
            warmup_steps=warmup,
            target_tokens=target_toks,
            val_pairs=val_pairs,
            eval_interval=max(1, steps // 2),
        )

    val_metrics = trainer.evaluate_loss(val_pairs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    trainer.save_resumable_checkpoint(out_dir, metrics={"train": train_metrics, "validation": val_metrics})
    ckpt_sha256 = sha256_file(os.path.join(out_dir, "model.pt"))

    ended_at = datetime.now(timezone.utc).isoformat()
    run_meta = {
        "release": "bumblecore-2.6",
        "run_id": run_id,
        "stage": stage,
        "smoke_mode": smoke,
        "git_commit": _git_commit(),
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "checkpoint_sha256": ckpt_sha256,
        "exact_trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "seed": trainer.seed,
        "python_version": platform.python_version(),
        "hardware": preflight["hardware_measured"],
        "started_at": started_at,
        "ended_at": ended_at,
        "resumed_from": resumed_info,
        "configured_lr": configured_lr,
        "actual_steps": train_metrics["steps"],
        "global_step": train_metrics["global_step"],
        "actual_tokens_processed": train_metrics["tokens_processed"],
        "token_accounting": train_metrics["token_accounting"],
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "checkpoint_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    return run_meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GajahMada GNNM BumbleCore Training CLI (Release 2.6)")
    parser.add_argument("--config", default="configs/bumblecore_300m.yaml", help="Path to YAML config")
    parser.add_argument("--stage", choices=["pretraining", "sft"], default="pretraining")
    parser.add_argument("--preflight", action="store_true", help="Run hardware/resource preflight only")
    parser.add_argument("--resume", default="none", help="'none', 'auto', or checkpoint dir path")
    parser.add_argument("--smoke", action="store_true", help="Run small CPU smoke validation (seed data allowed ONLY here)")
    parser.add_argument("--dataset", default=None, help="Approved external dataset directory with dataset_manifest.json")
    parser.add_argument("--tokenizer", default=None, help="Frozen tokenizer.json path")
    parser.add_argument("--output-dir", default=None, help="Checkpoint output directory")
    parser.add_argument("--steps", type=int, default=None, help="Override training step count")
    parser.add_argument("--target-tokens", type=int, default=None, help="Override target consumed token budget")
    parser.add_argument("--allow-resume-mismatch", action="store_true", help="Explicitly override dataset/tokenizer hash mismatch on resume")
    args = parser.parse_args(argv)

    result = execute_training_run(
        config_path=args.config,
        stage=args.stage,
        preflight_only=args.preflight,
        resume=args.resume,
        smoke=args.smoke,
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        steps_override=args.steps,
        target_tokens_override=args.target_tokens,
        allow_resume_mismatch=args.allow_resume_mismatch,
    )
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") in ("PREFLIGHT_BLOCKED", "DATASET_GOVERNANCE_FAILED"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
