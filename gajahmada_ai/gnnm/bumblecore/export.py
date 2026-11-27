"""
GajahMada GNNM BumbleCore — Checkpoint Export, Parity Verification & Hugging Face Prep (Phases 8 & 9).

- Keeps lightweight public inference artifacts separate from resumable `trainer_state.pt`.
- Supports `safetensors` export when installed + native `model.pt` (`custom_gnnm_native`).
- Never falsely labels custom GNNM checkpoints as Hugging Face `transformers`-compatible.
- Implements strict export/import parity verification (tensor shapes + deterministic logits).
- Generates Hugging Face Model Card (`README.md`) & optional dry-run/auth-gated upload helper.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from typing import Any, Dict, List, Optional

from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.model import GNNMTransformer, HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_safetensors() -> bool:
    try:
        importlib.import_module("safetensors.torch")
        return True
    except Exception:
        return False


def generate_hf_model_card(
    cfg: GNNMConfig,
    repo_id: str = "GajahMada/GNNM-BumbleCore-300M",
    training_metadata: Optional[Dict[str, Any]] = None,
    benchmark_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an honest Hugging Face README.md model card without fabricated scores."""
    params = cfg.estimate_parameters()
    eval_summary = (
        f"Overall local held-out score: `{benchmark_report.get('overall_score')}` "
        f"({benchmark_report.get('actual_sample_count')} samples, `{benchmark_report.get('benchmark_version')}`)."
        if benchmark_report
        else "No production benchmark results attached yet."
    )
    train_summary = (
        f"Steps: `{training_metadata.get('actual_steps')}`, Tokens processed: `{training_metadata.get('actual_tokens_processed')}`."
        if training_metadata
        else "Architecture/checkpoint metadata only."
    )
    return f"""---
license: mit
language:
  - id
  - en
library_name: gajahmada-gnnm
tags:
  - gajahmada
  - gnnm
  - bumblecore
  - custom-transformer
  - indonesian
---

# {cfg.name} (`{repo_id}`)

**Full Name:** GajahMada GNNM BumbleCore 300M  
**Architecture:** Custom Native GNNM Decoder-Only Transformer (`RMSNorm + RoPE + GQA + SwiGLU`, **NOT** standard Hugging Face `transformers` AutoModel format — load with `gajahmada_ai.gnnm.inference.GNNMInferenceEngine`).

## Model Specifications
- **Exact Trainable Parameters:** `{params:,}` (`{params / 1e6:.2f}M`)
- **Layers / Hidden / Heads (Q:KV) / FFN:** `{cfg.n_layers}` / `{cfg.d_model}` / `{cfg.n_heads}:{cfg.n_kv_heads}` / `{cfg.d_ff}`
- **Vocabulary Size / Context Length:** `{cfg.vocab_size:,}` (`gnnm-bpe-v1` byte-level BPE) / `{cfg.max_seq_len}`

## Training & Evaluation Status
- **Training Provenance:** {train_summary}
- **Evaluation Status:** {eval_summary}

## Limitations
- A 300M-parameter model is a lightweight foundation milestone, not a frontier general-intelligence model.
- Verify factual claims against GajahMada's Hybrid Knowledge Architecture.

## Loading Locally
```python
from gajahmada_ai.gnnm.inference import GNNMInferenceEngine
engine = GNNMInferenceEngine("./models/bumblecore-300m-export")
print(engine.generate("Apa itu Sumpah Palapa?", max_new_tokens=64))
```
"""


def export_inference_artifact(
    source_checkpoint_dir: str,
    export_dir: str,
    repo_id: str = "GajahMada/GNNM-BumbleCore-300M",
    training_metadata: Optional[Dict[str, Any]] = None,
    benchmark_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export lightweight public inference artifact (excludes trainer_state.pt / optimizer)."""
    if not HAS_TORCH:
        raise ImportError("PyTorch (`torch`) is required to export GNNM checkpoints.")

    cfg = GNNMConfig.load(os.path.join(source_checkpoint_dir, "config.json"))
    tok = GNNMBPETokenizer.load(os.path.join(source_checkpoint_dir, "tokenizer.json"))
    state_dict = torch.load(os.path.join(source_checkpoint_dir, "model.pt"), map_location="cpu", weights_only=True)

    os.makedirs(export_dir, exist_ok=True)
    cfg.save(os.path.join(export_dir, "config.json"))
    tok.save(os.path.join(export_dir, "tokenizer.json"))

    gen_cfg = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_k": 40,
        "repetition_penalty": 1.15,
        "bos_token_id": tok.bos_id,
        "eos_token_id": tok.eos_id,
        "pad_token_id": tok.pad_id,
    }
    with open(os.path.join(export_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)

    # Always write native model.pt; also write model.safetensors when safetensors is installed
    pt_path = os.path.join(export_dir, "model.pt")
    torch.save(state_dict, pt_path)
    exported_files = ["config.json", "tokenizer.json", "generation_config.json", "model.pt"]

    safetensors_used = False
    if _has_safetensors():
        st_mod = importlib.import_module("safetensors.torch")
        # Clone tied tensors so safetensors does not reject shared storage
        clean_sd = {k: v.clone().contiguous() for k, v in state_dict.items()}
        st_path = os.path.join(export_dir, "model.safetensors")
        st_mod.save_file(clean_sd, st_path)
        exported_files.append("model.safetensors")
        safetensors_used = True

    card_path = os.path.join(export_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(generate_hf_model_card(cfg, repo_id=repo_id, training_metadata=training_metadata, benchmark_report=benchmark_report))
    exported_files.append("README.md")

    hashes = {fn: _sha256_file(os.path.join(export_dir, fn)) for fn in exported_files}
    manifest = {
        "format": "custom_gnnm_native",
        "transformers_auto_compatible": False,
        "safetensors_exported": safetensors_used,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model_name": cfg.name,
        "exact_parameters": cfg.estimate_parameters(),
        "repo_id": repo_id,
        "excludes_optimizer_state": True,
        "files_sha256": hashes,
    }
    with open(os.path.join(export_dir, "artifact_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_export_import_parity(source_dir: str, export_dir: str) -> Dict[str, Any]:
    """Reload source & exported checkpoints and verify exact tensor shapes & deterministic outputs."""
    if not HAS_TORCH:
        raise ImportError("PyTorch (`torch`) is required to verify export/import parity.")

    cfg_a = GNNMConfig.load(os.path.join(source_dir, "config.json"))
    cfg_b = GNNMConfig.load(os.path.join(export_dir, "config.json"))
    tok_a = GNNMBPETokenizer.load(os.path.join(source_dir, "tokenizer.json"))
    tok_b = GNNMBPETokenizer.load(os.path.join(export_dir, "tokenizer.json"))

    sd_a = torch.load(os.path.join(source_dir, "model.pt"), map_location="cpu", weights_only=True)
    sd_b = torch.load(os.path.join(export_dir, "model.pt"), map_location="cpu", weights_only=True)

    shapes_match = set(sd_a.keys()) == set(sd_b.keys()) and all(
        tuple(sd_a[k].shape) == tuple(sd_b[k].shape) for k in sd_a
    )

    m_a = GNNMTransformer(cfg_a)
    m_b = GNNMTransformer(cfg_b)
    m_a.load_state_dict(sd_a)
    m_b.load_state_dict(sd_b)
    m_a.eval()
    m_b.eval()

    sample_ids = tok_a.encode("GajahMada BumbleCore", add_bos=True)[: cfg_a.max_seq_len]
    tok_match = sample_ids == tok_b.encode("GajahMada BumbleCore", add_bos=True)[: cfg_b.max_seq_len]
    x = torch.tensor([sample_ids], dtype=torch.long)
    with torch.no_grad():
        out_a, _ = m_a(x)
        out_b, _ = m_b(x)
    outputs_match = bool(torch.allclose(out_a, out_b, atol=1e-6))

    return {
        "parity_verified": shapes_match and tok_match and outputs_match and (cfg_a.to_dict() == cfg_b.to_dict()),
        "shapes_match": shapes_match,
        "tokenizer_match": tok_match,
        "config_match": cfg_a.to_dict() == cfg_b.to_dict(),
        "deterministic_outputs_match": outputs_match,
    }


def prepare_hf_upload(
    export_dir: str,
    repo_id: str = "GajahMada/GNNM-BumbleCore-300M",
    private: bool = True,
    dry_run: bool = True,
    confirm_upload: bool = False,
) -> Dict[str, Any]:
    """
    Optional Hugging Face upload preparation.
    NEVER uploads automatically — requires explicit confirm_upload=True, dry_run=False, and HF_TOKEN.
    """
    forbidden_patterns = ("trainer_state.pt", ".env", "node_key.pem", "admin_api.key")
    files = sorted(os.listdir(export_dir)) if os.path.isdir(export_dir) else []
    blocked = [f for f in files if any(p in f for p in forbidden_patterns)]
    if blocked:
        raise ValueError(f"Refusing upload: forbidden private/training files present in export_dir: {blocked}")

    if dry_run or not confirm_upload:
        return {
            "status": "DRY_RUN_ONLY",
            "uploaded": False,
            "repo_id": repo_id,
            "private": private,
            "files_ready": files,
            "note": "Explicit user authorization (dry_run=False, confirm_upload=True, HF_TOKEN) required to upload.",
        }

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return {"status": "MISSING_HF_TOKEN", "uploaded": False, "repo_id": repo_id}

    hf_hub = importlib.import_module("huggingface_hub")
    api = hf_hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=export_dir, repo_id=repo_id, repo_type="model")
    return {"status": "UPLOADED", "uploaded": True, "repo_id": repo_id, "private": private, "files": files}
