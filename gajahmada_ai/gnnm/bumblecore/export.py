"""
GajahMada GNNM BumbleCore — Checkpoint Export, Parity Verification & Hugging Face Prep (Batch 8 Hardened).

- Keeps lightweight public inference artifacts separate from resumable `trainer_state.pt`.
- Supports `safetensors` export when installed + native `model.pt` (`custom_gnnm_native`).
- Never falsely labels custom GNNM checkpoints as Hugging Face `transformers`-compatible.
- Never labels untrained, random-init, or smoke checkpoints as pretrained production models.
- Validates config/tokenizer/generation_config consistency, tensor finiteness, Model Card schema,
  and strict export/import parity (tensor names, shapes, dtypes, numerical values, deterministic logits).
- Enforces recursive safety scan & allowlist manifest; NEVER uploads automatically.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import re
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


def classify_training_status(training_metadata: Optional[Dict[str, Any]]) -> str:
    """Classify checkpoint training status honestly — never claims pretrained without proof."""
    if not training_metadata or not isinstance(training_metadata, dict):
        return "UNTRAINED_OR_RANDOM_INIT"
    if training_metadata.get("smoke_mode") is True:
        return "SMOKE_TEST_ONLY"
    steps = int(training_metadata.get("actual_steps") or 0)
    tokens = int(training_metadata.get("actual_tokens_processed") or 0)
    if steps <= 0 or tokens <= 0:
        return "UNTRAINED_OR_RANDOM_INIT"
    return "TRAINED_WITH_PROVENANCE"


def generate_hf_model_card(
    cfg: GNNMConfig,
    repo_id: str = "GajahMada/GNNM-BumbleCore-300M",
    training_metadata: Optional[Dict[str, Any]] = None,
    benchmark_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an honest Hugging Face README.md model card without fabricated scores or false pretrained claims."""
    params = cfg.estimate_parameters()
    training_status = classify_training_status(training_metadata)

    if training_status == "TRAINED_WITH_PROVENANCE" and training_metadata:
        train_summary = (
            f"Status: `{training_status}` | Steps: `{training_metadata.get('actual_steps')}`, "
            f"Tokens processed: `{training_metadata.get('actual_tokens_processed')}`."
        )
    elif training_status == "SMOKE_TEST_ONLY" and training_metadata:
        train_summary = (
            f"Status: `{training_status}` (NOT a production pretrained model) | "
            f"Smoke steps: `{training_metadata.get('actual_steps')}`, "
            f"Smoke tokens: `{training_metadata.get('actual_tokens_processed')}`."
        )
    else:
        train_summary = (
            f"Status: `{training_status}` — architecture/untrained weights only; "
            "NOT pretrained on production data."
        )

    eval_summary = (
        f"Overall local held-out score: `{benchmark_report.get('overall_score')}` "
        f"({benchmark_report.get('actual_sample_count')} samples, `{benchmark_report.get('benchmark_version')}`)."
        if benchmark_report
        else "No production benchmark results attached yet."
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
- **Training Status:** `{training_status}`
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


REQUIRED_MODEL_CARD_SECTIONS = (
    "## Model Specifications",
    "## Training & Evaluation Status",
    "## Limitations",
    "## Loading Locally",
)


def validate_hf_model_card(
    readme_text: str,
    expected_license: str = "mit",
    expected_training_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate Hugging Face Model Card frontmatter, license, required sections, and honest status labeling."""
    issues: List[str] = []
    if not readme_text or not readme_text.strip():
        return {"valid": False, "issues": ["MODEL_CARD_EMPTY"]}

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", readme_text, flags=re.DOTALL)
    if not fm_match:
        issues.append("MODEL_CARD_FRONTMATTER_MISSING")
        frontmatter = ""
    else:
        frontmatter = fm_match.group(1)

    if frontmatter:
        lic_match = re.search(r"^\s*license\s*:\s*([^\n#]+)", frontmatter, flags=re.MULTILINE)
        if not lic_match:
            issues.append("MODEL_CARD_LICENSE_MISSING")
        else:
            found_lic = lic_match.group(1).strip().strip("'\"").lower()
            if expected_license and found_lic != expected_license.lower():
                issues.append(f"MODEL_CARD_LICENSE_MISMATCH: expected={expected_license}, got={found_lic}")
        if "library_name: gajahmada-gnnm" not in frontmatter:
            issues.append("MODEL_CARD_LIBRARY_NAME_INVALID")

    for section in REQUIRED_MODEL_CARD_SECTIONS:
        if section not in readme_text:
            issues.append(f"MODEL_CARD_SECTION_MISSING: {section}")

    if "AutoModelForCausalLM" in readme_text and "NOT" not in readme_text:
        issues.append("MODEL_CARD_FALSE_TRANSFORMERS_COMPATIBILITY_CLAIM")

    if expected_training_status and expected_training_status != "TRAINED_WITH_PROVENANCE":
        if "TRAINED_WITH_PROVENANCE" in readme_text:
            issues.append("MODEL_CARD_FALSE_PRETRAINED_CLAIM")

    return {"valid": len(issues) == 0, "issues": issues}


def export_inference_artifact(
    source_checkpoint_dir: str,
    export_dir: str,
    repo_id: str = "GajahMada/GNNM-BumbleCore-300M",
    training_metadata: Optional[Dict[str, Any]] = None,
    benchmark_report: Optional[Dict[str, Any]] = None,
    require_trained: bool = False,
) -> Dict[str, Any]:
    """Export lightweight public inference artifact (excludes trainer_state.pt / optimizer)."""
    if not HAS_TORCH:
        raise ImportError("PyTorch (`torch`) is required to export GNNM checkpoints.")

    cfg_path = os.path.join(source_checkpoint_dir, "config.json")
    tok_path = os.path.join(source_checkpoint_dir, "tokenizer.json")
    src_pt_path = os.path.join(source_checkpoint_dir, "model.pt")

    cfg = GNNMConfig.load(cfg_path)
    tok = GNNMBPETokenizer.load(tok_path)

    # Enforce config <-> tokenizer consistency before export
    if int(cfg.vocab_size) != int(tok.vocab_size):
        raise ValueError(
            f"CONFIG_TOKENIZER_VOCAB_MISMATCH: config.vocab_size={cfg.vocab_size} != tokenizer.vocab_size={tok.vocab_size}"
        )

    # Auto-load run_metadata.json if present and caller did not pass explicit metadata
    if training_metadata is None:
        meta_path = os.path.join(source_checkpoint_dir, "run_metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                training_metadata = json.load(f)

    training_status = classify_training_status(training_metadata)
    if require_trained and training_status != "TRAINED_WITH_PROVENANCE":
        raise ValueError(
            f"REFUSING_EXPORT_UNTRAINED_OR_SMOKE_CHECKPOINT: status={training_status}"
        )

    state_dict = torch.load(src_pt_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("INVALID_SOURCE_STATE_DICT: expected non-empty tensor dict.")

    # Reject non-finite weights before export
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            raise ValueError(f"NON_TENSOR_STATE_DICT_ENTRY: {k}")
        if v.is_floating_point() and not bool(torch.isfinite(v).all()):
            raise ValueError(f"NON_FINITE_TENSOR_IN_EXPORT_SOURCE: {k}")

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
    card_text = generate_hf_model_card(
        cfg,
        repo_id=repo_id,
        training_metadata=training_metadata,
        benchmark_report=benchmark_report,
    )
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_text)
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
        "training_status": training_status,
        "source_checkpoint_sha256": _sha256_file(src_pt_path),
        "config_tokenizer_consistent": True,
        "excludes_optimizer_state": True,
        "files_sha256": hashes,
    }
    with open(os.path.join(export_dir, "artifact_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_export_import_parity(source_dir: str, export_dir: str) -> Dict[str, Any]:
    """Reload source & exported checkpoints and verify exact tensor names, shapes, dtypes, values & deterministic outputs."""
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
    dtypes_match = set(sd_a.keys()) == set(sd_b.keys()) and all(
        sd_a[k].dtype == sd_b[k].dtype for k in sd_a
    )
    values_match = shapes_match and all(
        bool(torch.allclose(sd_a[k].float(), sd_b[k].float(), atol=1e-6)) for k in sd_a
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
    max_logit_diff = float((out_a - out_b).abs().max().item())
    outputs_finite = bool(torch.isfinite(out_a).all() and torch.isfinite(out_b).all())
    outputs_match = outputs_finite and bool(torch.allclose(out_a, out_b, atol=1e-6))

    # Also verify Safetensors reload parity if model.safetensors was exported and library is available
    safetensors_checked = False
    safetensors_parity = None
    st_path = os.path.join(export_dir, "model.safetensors")
    if os.path.isfile(st_path) and _has_safetensors():
        safetensors_checked = True
        st_mod = importlib.import_module("safetensors.torch")
        sd_st = st_mod.load_file(st_path, device="cpu")
        st_shapes = set(sd_a.keys()) == set(sd_st.keys()) and all(
            tuple(sd_a[k].shape) == tuple(sd_st[k].shape)
            and sd_a[k].dtype == sd_st[k].dtype
            and torch.allclose(sd_a[k].float(), sd_st[k].float(), atol=1e-6)
            for k in sd_a
        )
        m_st = GNNMTransformer(cfg_b)
        m_st.load_state_dict(sd_st)
        m_st.eval()
        with torch.no_grad():
            out_st, _ = m_st(x)
        safetensors_parity = (
            st_shapes
            and bool(torch.isfinite(out_st).all())
            and bool(torch.allclose(out_a, out_st, atol=1e-6))
        )

    all_ok = (
        shapes_match
        and dtypes_match
        and values_match
        and tok_match
        and outputs_match
        and (cfg_a.to_dict() == cfg_b.to_dict())
    )
    if safetensors_checked:
        all_ok = all_ok and bool(safetensors_parity)

    return {
        "parity_verified": all_ok,
        "shapes_match": shapes_match,
        "dtypes_match": dtypes_match,
        "values_match": values_match,
        "tokenizer_match": tok_match,
        "config_match": cfg_a.to_dict() == cfg_b.to_dict(),
        "deterministic_outputs_match": outputs_match,
        "max_logit_diff": max_logit_diff,
        "safetensors_checked": safetensors_checked,
        "safetensors_parity": safetensors_parity,
    }


RELEASE_ARTIFACT_ALLOWLIST = {
    "config.json",
    "tokenizer.json",
    "generation_config.json",
    "model.pt",
    "model.safetensors",
    "README.md",
    "artifact_manifest.json",
    "LICENSE",
}

FORBIDDEN_UPLOAD_PATTERNS = (
    "trainer_state.pt",
    ".env",
    "node_key.pem",
    "admin_api.key",
    ".pem",
    ".key",
    ".db",
    ".sqlite",
    ".log",
    "fingerprints.db",
)


def scan_export_directory_recursive(export_dir: str) -> Dict[str, Any]:
    """
    Recursively inspect every file and symlink under `export_dir` against an explicit
    release allowlist and forbidden-file policy. Returns an auditable upload manifest.
    """
    if not os.path.isdir(export_dir):
        return {"safe": False, "violations": [f"Export directory not found: {export_dir}"], "manifest": []}

    violations: List[str] = []
    manifest: List[Dict[str, Any]] = []

    for root, dirs, files in os.walk(export_dir, followlinks=False):
        # Reject unexpected symlinks on both directories and files
        for d in dirs:
            full_d = os.path.join(root, d)
            if os.path.islink(full_d):
                violations.append(f"SYMLINK_FORBIDDEN: {os.path.relpath(full_d, export_dir)}")
        for fn in sorted(files):
            full_p = os.path.join(root, fn)
            rel_p = os.path.relpath(full_p, export_dir).replace("\\", "/")
            if os.path.islink(full_p):
                violations.append(f"SYMLINK_FORBIDDEN: {rel_p}")
                continue
            lower = rel_p.lower()
            if any(pat in lower for pat in FORBIDDEN_UPLOAD_PATTERNS):
                violations.append(f"FORBIDDEN_SENSITIVE_FILE: {rel_p}")
                continue
            if rel_p not in RELEASE_ARTIFACT_ALLOWLIST:
                violations.append(f"UNAPPROVED_ARTIFACT_NOT_IN_ALLOWLIST: {rel_p}")
                continue
            manifest.append({
                "relative_path": rel_p,
                "size_bytes": os.path.getsize(full_p),
                "sha256": _sha256_file(full_p),
                "artifact_type": os.path.splitext(fn)[1].lstrip(".") or "text",
                "included": True,
            })

    return {
        "safe": len(violations) == 0 and len(manifest) > 0,
        "violations": violations,
        "manifest": manifest,
    }


def validate_export_bundle(export_dir: str, expected_license: str = "mit") -> Dict[str, Any]:
    """
    Comprehensive pre-upload bundle validation:
    - Recursive safety scan & allowlist enforcement
    - artifact_manifest.json live SHA-256 verification
    - config.json <-> tokenizer.json <-> generation_config.json consistency
    - README.md Model Card schema, license, and honest status check
    """
    scan = scan_export_directory_recursive(export_dir)
    issues: List[str] = []
    if not scan["safe"]:
        issues.extend(scan["violations"])

    manifest: Optional[Dict[str, Any]] = None
    manifest_path = os.path.join(export_dir, "artifact_manifest.json")
    if not os.path.isfile(manifest_path):
        issues.append("ARTIFACT_MANIFEST_MISSING")
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            issues.append(f"ARTIFACT_MANIFEST_UNREADABLE: {exc}")

    if manifest is not None:
        if manifest.get("excludes_optimizer_state") is not True:
            issues.append("OPTIMIZER_STATE_EXCLUSION_UNVERIFIED")
        files_sha = manifest.get("files_sha256") or {}
        for rel_fn, expected_sha in files_sha.items():
            full_p = os.path.join(export_dir, rel_fn)
            if not os.path.isfile(full_p):
                issues.append(f"MANIFEST_FILE_MISSING: {rel_fn}")
            elif _sha256_file(full_p) != expected_sha:
                issues.append(f"MANIFEST_FILE_HASH_MISMATCH: {rel_fn}")

    # Config / tokenizer / generation_config consistency
    cfg_path = os.path.join(export_dir, "config.json")
    tok_path = os.path.join(export_dir, "tokenizer.json")
    gen_path = os.path.join(export_dir, "generation_config.json")
    if os.path.isfile(cfg_path) and os.path.isfile(tok_path):
        try:
            cfg = GNNMConfig.load(cfg_path)
            tok = GNNMBPETokenizer.load(tok_path)
            if int(cfg.vocab_size) != int(tok.vocab_size):
                issues.append(
                    f"CONFIG_TOKENIZER_VOCAB_MISMATCH: config={cfg.vocab_size} tokenizer={tok.vocab_size}"
                )
            if os.path.isfile(gen_path):
                with open(gen_path, "r", encoding="utf-8") as f:
                    gcfg = json.load(f)
                if (
                    gcfg.get("bos_token_id") != tok.bos_id
                    or gcfg.get("eos_token_id") != tok.eos_id
                    or gcfg.get("pad_token_id") != tok.pad_id
                ):
                    issues.append("GENERATION_CONFIG_SPECIAL_TOKENS_MISMATCH")
        except Exception as exc:
            issues.append(f"CONFIG_OR_TOKENIZER_INVALID: {exc}")
    else:
        issues.append("CONFIG_OR_TOKENIZER_MISSING")

    # Model card validation
    card_path = os.path.join(export_dir, "README.md")
    if not os.path.isfile(card_path):
        issues.append("MODEL_CARD_MISSING")
    else:
        with open(card_path, "r", encoding="utf-8") as f:
            card_res = validate_hf_model_card(
                f.read(),
                expected_license=expected_license,
                expected_training_status=(manifest or {}).get("training_status"),
            )
        if not card_res["valid"]:
            issues.extend(card_res["issues"])

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "training_status": (manifest or {}).get("training_status", "UNKNOWN"),
        "scan": scan,
    }


def prepare_hf_upload(
    export_dir: str,
    repo_id: str = "GajahMada/GNNM-BumbleCore-300M",
    private: bool = True,
    dry_run: bool = True,
    confirm_upload: bool = False,
    allow_untrained: bool = False,
) -> Dict[str, Any]:
    """
    Optional Hugging Face upload preparation with recursive safety scan & allowlist.
    NEVER uploads automatically — requires explicit confirm_upload=True, dry_run=False, and HF_TOKEN.
    Live upload additionally requires full bundle validation and trained provenance unless allow_untrained=True.
    """
    scan = scan_export_directory_recursive(export_dir)
    if not scan["safe"]:
        raise ValueError(f"Refusing upload due to recursive safety / forbidden artifact violations: {scan['violations']}")

    files = [m["relative_path"] for m in scan["manifest"]]
    if dry_run or not confirm_upload:
        return {
            "status": "DRY_RUN_ONLY",
            "uploaded": False,
            "repo_id": repo_id,
            "private": private,
            "files_ready": files,
            "upload_manifest": scan["manifest"],
            "note": "Explicit user authorization (dry_run=False, confirm_upload=True, HF_TOKEN) required to upload.",
        }

    bundle = validate_export_bundle(export_dir)
    if not bundle["valid"]:
        raise ValueError(f"Refusing live upload due to bundle validation issues: {bundle['issues']}")
    if not allow_untrained and bundle["training_status"] != "TRAINED_WITH_PROVENANCE":
        raise ValueError(
            f"Refusing live upload of untrained/smoke model (training_status={bundle['training_status']})."
        )

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return {"status": "MISSING_HF_TOKEN", "uploaded": False, "repo_id": repo_id, "upload_manifest": scan["manifest"]}

    hf_hub = importlib.import_module("huggingface_hub")
    api = hf_hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=export_dir, repo_id=repo_id, repo_type="model")
    return {"status": "UPLOADED", "uploaded": True, "repo_id": repo_id, "private": private, "files": files, "upload_manifest": scan["manifest"]}
