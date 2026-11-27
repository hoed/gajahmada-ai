"""
GajahMada GNNM BumbleCore 300M — Checkpoint Trust Layer (Release 2.6, Phase 1).

Eliminates file-hash-only forgery by enforcing three independent layers of verification:
1. Structural Tensor Verification:
   - Rejects empty files, plain-text/corrupt files, and non-PyTorch binaries.
   - Loads `state_dict` safely (`weights_only=True`).
   - Validates every expected tensor key, exact shape, floating-point dtype, missing/unexpected
     keys, absence of NaN/Inf, and exact trainable parameter count against `config.json`.
   - Loads into `GNNMTransformer` (for small/smoke models or when `instantiate_check=True`).
2. Source-File Provenance Cross-Verification:
   - Requires complete run provenance (`run_id`, `git_commit`, `config_sha256`,
     `dataset_manifest_sha256`, `tokenizer_sha256`, `checkpoint_sha256`, `stage`,
     `actual_steps`, `actual_tokens_processed`, `validation_metrics`, `hardware`,
     `started_at`, `ended_at`).
   - Recomputes live hashes against actual on-disk `model.pt`, `config.json`,
     `tokenizer.json`, and `dataset_manifest.json`.
3. Ed25519 Anti-Forgery Signature (Reusing `gajahmada_ai.security.crypto.NodeIdentity`):
   - Signs and verifies training manifests using existing Ed25519 infrastructure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from gajahmada_ai.gnnm.bumblecore.dataset_governance import APPROVED_LICENSES, sha256_file
from gajahmada_ai.gnnm.config import GNNMConfig
from gajahmada_ai.gnnm.model import GNNMTransformer, HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.security.crypto import NodeIdentity

REQUIRED_PROVENANCE_FIELDS = (
    "run_id",
    "git_commit",
    "config_sha256",
    "dataset_manifest_sha256",
    "tokenizer_sha256",
    "checkpoint_sha256",
    "stage",
    "actual_steps",
    "actual_tokens_processed",
    "validation_metrics",
    "hardware",
    "started_at",
    "ended_at",
)

# PyTorch checkpoints saved with modern torch.save are ZIP archives ('PK\x03\x04')
# or legacy pickle protocol magic (\x80\x02..\x80\x05). Plain text files fail immediately.
_PYTORCH_MAGIC_PREFIXES = (b"PK\x03\x04", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")


def canonical_json_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def compute_tokenizer_sha256(tok: GNNMBPETokenizer) -> str:
    return canonical_json_sha256({
        "format": "gnnm-bpe-v1",
        "vocab_size": tok.vocab_size,
        "merges": tok.merges,
    })


def expected_gnnm_tensor_shapes(cfg: GNNMConfig) -> Dict[str, Tuple[int, ...]]:
    """Derive exact expected `state_dict` tensor names and shapes directly from GNNMConfig."""
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    shapes: Dict[str, Tuple[int, ...]] = {
        "token_emb.weight": (cfg.vocab_size, cfg.d_model),
        "final_norm.weight": (cfg.d_model,),
        "lm_head.weight": (cfg.vocab_size, cfg.d_model),
    }
    for i in range(cfg.n_layers):
        p = f"layers.{i}"
        shapes[f"{p}.attn_norm.weight"] = (cfg.d_model,)
        shapes[f"{p}.attn.w_q.weight"] = (cfg.n_heads * cfg.head_dim, cfg.d_model)
        shapes[f"{p}.attn.w_k.weight"] = (kv_dim, cfg.d_model)
        shapes[f"{p}.attn.w_v.weight"] = (kv_dim, cfg.d_model)
        shapes[f"{p}.attn.w_o.weight"] = (cfg.d_model, cfg.n_heads * cfg.head_dim)
        shapes[f"{p}.ffn_norm.weight"] = (cfg.d_model,)
        shapes[f"{p}.ffn.w_gate.weight"] = (cfg.d_ff, cfg.d_model)
        shapes[f"{p}.ffn.w_up.weight"] = (cfg.d_ff, cfg.d_model)
        shapes[f"{p}.ffn.w_down.weight"] = (cfg.d_model, cfg.d_ff)
    return shapes


def _manifest_signing_payload(meta: Dict[str, Any]) -> bytes:
    """Canonical signing payload excluding mutable signature fields."""
    clean = {k: v for k, v in meta.items() if k not in ("manifest_signature", "signer_node_id")}
    return json.dumps(clean, sort_keys=True, default=str).encode("utf-8")


def sign_training_manifest(meta: Dict[str, Any], identity: NodeIdentity) -> Dict[str, Any]:
    """Sign a training run manifest using the repository's existing Ed25519 NodeIdentity."""
    signed = dict(meta)
    payload = _manifest_signing_payload(signed)
    signed["signer_node_id"] = identity.node_id
    signed["manifest_signature"] = identity.sign_message(payload)
    return signed


def verify_training_manifest_signature(meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify Ed25519 signature on `run_metadata.json` when present using NodeIdentity."""
    sig = meta.get("manifest_signature")
    signer = meta.get("signer_node_id")
    if not sig and not signer:
        return True, "UNSIGNED"
    if not sig or not signer:
        return False, "INCOMPLETE_SIGNATURE_FIELDS"
    verifier = NodeIdentity(key_path=None)
    ok = verifier.verify_message(_manifest_signing_payload(meta), str(sig), str(signer))
    return (True, "ED25519_VERIFIED") if ok else (False, "INVALID_MANIFEST_SIGNATURE")


def verify_checkpoint_structure(
    checkpoint_dir: str,
    instantiate_check: bool = False,
) -> Dict[str, Any]:
    """
    Validate actual PyTorch checkpoint file structure, tensor names, shapes, dtypes,
    finite values (no NaN/Inf), missing/unexpected keys, and exact parameter count.
    """
    reasons: List[str] = []
    model_pt = os.path.join(checkpoint_dir, "model.pt")
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")

    if not os.path.isfile(model_pt):
        return {"valid": False, "reasons": ["CHECKPOINT_WEIGHTS_MISSING"], "actual_parameters": 0}
    if os.path.getsize(model_pt) == 0:
        return {"valid": False, "reasons": ["EMPTY_CHECKPOINT_FILE"], "actual_parameters": 0}

    with open(model_pt, "rb") as f:
        magic = f.read(8)
    if not any(magic.startswith(p) for p in _PYTORCH_MAGIC_PREFIXES):
        return {
            "valid": False,
            "reasons": ["INVALID_PYTORCH_CHECKPOINT: file header is not a valid PyTorch archive (possible fake/text file)."],
            "actual_parameters": 0,
        }

    if not os.path.isfile(cfg_path):
        return {"valid": False, "reasons": ["CONFIG_JSON_MISSING"], "actual_parameters": 0}

    try:
        cfg = GNNMConfig.load(cfg_path)
    except Exception as exc:
        return {"valid": False, "reasons": [f"INVALID_CONFIG_JSON: {exc}"], "actual_parameters": 0}

    tok_vocab = None
    if os.path.isfile(tok_path):
        try:
            tok = GNNMBPETokenizer.load(tok_path)
            tok_vocab = tok.vocab_size
            if tok_vocab != cfg.vocab_size:
                reasons.append(f"TOKENIZER_VOCAB_MISMATCH: tokenizer={tok_vocab} config={cfg.vocab_size}")
        except Exception as exc:
            reasons.append(f"INVALID_TOKENIZER_JSON: {exc}")

    if not HAS_TORCH:
        return {
            "valid": False,
            "reasons": reasons + ["PYTORCH_UNAVAILABLE_FOR_TENSOR_VERIFICATION"],
            "actual_parameters": 0,
            "config_parameters": cfg.estimate_parameters(),
        }

    try:
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {
            "valid": False,
            "reasons": [f"INVALID_PYTORCH_CHECKPOINT: torch.load(weights_only=True) failed ({exc})"],
            "actual_parameters": 0,
        }

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return {"valid": False, "reasons": ["EMPTY_STATE_DICT"], "actual_parameters": 0}

    expected = expected_gnnm_tensor_shapes(cfg)
    actual_keys = set(state_dict.keys())
    expected_keys = set(expected.keys())

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        reasons.append(f"MISSING_TENSOR_KEYS: {missing[:5]} (total {len(missing)})")
    if unexpected:
        reasons.append(f"UNEXPECTED_TENSOR_KEYS: {unexpected[:5]} (total {len(unexpected)})")

    allowed_dtypes = {torch.float32, torch.bfloat16, torch.float16}
    dtypes_seen = set()
    actual_param_count = 0

    for k, exp_shape in expected.items():
        if k not in state_dict:
            continue
        t = state_dict[k]
        if not isinstance(t, torch.Tensor):
            reasons.append(f"NON_TENSOR_ENTRY: {k}")
            continue
        if tuple(t.shape) != exp_shape:
            reasons.append(f"TENSOR_SHAPE_MISMATCH({k}): expected {exp_shape}, got {tuple(t.shape)}")
        if t.dtype not in allowed_dtypes:
            reasons.append(f"INVALID_TENSOR_DTYPE({k}): {t.dtype}")
        else:
            dtypes_seen.add(str(t.dtype))
        if not torch.isfinite(t).all():
            reasons.append(f"NON_FINITE_TENSOR_VALUES({k}): contains NaN or Inf")
        # Account for tied embeddings without double counting lm_head.weight
        if not (cfg.tie_embeddings and k == "lm_head.weight"):
            actual_param_count += int(t.numel())

    expected_params = cfg.estimate_parameters()
    if actual_param_count != expected_params:
        reasons.append(f"PARAMETER_COUNT_MISMATCH: actual={actual_param_count:,} expected={expected_params:,}")

    # Optional full model load check (auto-enabled for compact models <= 50M params)
    if not reasons and (instantiate_check or expected_params <= 50_000_000):
        try:
            model = GNNMTransformer(cfg)
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            reasons.append(f"MODEL_LOAD_FAILED: {exc}")

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "actual_parameters": actual_param_count,
        "config_parameters": expected_params,
        "dtypes": sorted(dtypes_seen),
        "tensor_count": len(state_dict),
    }


def verify_trusted_checkpoint(
    checkpoint_dir: str,
    expected_stage: str,
    expected_min_params: int = 285_000_000,
    allow_smoke: bool = False,
    require_signature: bool = False,
) -> Dict[str, Any]:
    """
    Full Checkpoint Trust Layer verification combining:
    1. Structural PyTorch tensor & architecture verification
    2. Live source-file provenance hash verification (`model.pt`, `config.json`, `tokenizer.json`, `dataset_manifest.json`)
    3. Anti-forgery checks (stage, smoke_mode, steps, tokens, val metrics, license, Ed25519 signature)
    """
    reasons: List[str] = []
    model_pt = os.path.join(checkpoint_dir, "model.pt")
    meta_path = os.path.join(checkpoint_dir, "run_metadata.json")
    manifest_path = os.path.join(checkpoint_dir, "dataset_manifest.json")
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")

    for req_name, req_path in (
        ("CHECKPOINT_WEIGHTS_MISSING", model_pt),
        ("RUN_METADATA_MISSING", meta_path),
        ("DATASET_MANIFEST_MISSING", manifest_path),
        ("CONFIG_JSON_MISSING", cfg_path),
        ("TOKENIZER_JSON_MISSING", tok_path),
    ):
        if not os.path.isfile(req_path):
            reasons.append(req_name)

    if reasons:
        return {
            "release": "bumblecore-2.6",
            "gate": expected_stage,
            "status": "NOT_VERIFIED",
            "passed": False,
            "reasons": reasons,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # Layer 1: Structural tensor verification
    struct = verify_checkpoint_structure(checkpoint_dir)
    if not struct["valid"]:
        reasons.extend(struct["reasons"])

    # Layer 2: Metadata & live source-file provenance verification
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        return {"release": "bumblecore-2.6", "gate": expected_stage, "status": "BLOCKED", "passed": False, "reasons": [f"INVALID_RUN_METADATA: {exc}"]}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return {"release": "bumblecore-2.6", "gate": expected_stage, "status": "BLOCKED", "passed": False, "reasons": [f"INVALID_DATASET_MANIFEST: {exc}"]}

    missing_prov = [fld for fld in REQUIRED_PROVENANCE_FIELDS if meta.get(fld) in (None, "")]
    if missing_prov:
        reasons.append(f"MISSING_RUN_PROVENANCE_FIELDS: {missing_prov}")

    if not allow_smoke and meta.get("smoke_mode") is not False:
        reasons.append("SMOKE_CHECKPOINT_REJECTED: smoke_mode must be False for production gates.")
    if meta.get("stage") != expected_stage:
        reasons.append(f"WRONG_STAGE: expected '{expected_stage}', found '{meta.get('stage')}'.")

    # Live SHA-256 verification against actual on-disk source files
    actual_ckpt_hash = sha256_file(model_pt)
    if meta.get("checkpoint_sha256") != actual_ckpt_hash:
        reasons.append("CHECKPOINT_HASH_MISMATCH: live model.pt SHA-256 does not match run_metadata.json.")

    with open(cfg_path, "r", encoding="utf-8") as cf:
        raw_cfg_dict = json.load(cf)
    actual_cfg_hash = canonical_json_sha256(raw_cfg_dict)
    cfg_obj = GNNMConfig.load(cfg_path)
    if meta.get("config_sha256") not in (actual_cfg_hash, canonical_json_sha256(cfg_obj.to_dict())):
        reasons.append("CONFIG_HASH_MISMATCH: live config.json hash does not match run_metadata.json.")

    tok_obj = GNNMBPETokenizer.load(tok_path)
    actual_tok_hash = compute_tokenizer_sha256(tok_obj)
    if meta.get("tokenizer_sha256") != actual_tok_hash:
        reasons.append("TOKENIZER_HASH_MISMATCH: live tokenizer.json hash does not match run_metadata.json.")

    actual_manifest_file_hash = sha256_file(manifest_path)
    declared_ds_hash = meta.get("dataset_manifest_sha256")
    if declared_ds_hash not in (actual_manifest_file_hash, manifest.get("manifest_sha256"), manifest.get("document_hash")):
        reasons.append("DATASET_MANIFEST_HASH_MISMATCH: dataset_manifest.json does not match run_metadata.json.")

    # Actual tensor parameter count vs target & metadata
    actual_params = struct.get("actual_parameters", 0)
    if actual_params < expected_min_params:
        reasons.append(f"PARAMETER_COUNT_BELOW_TARGET: actual={actual_params:,} < required={expected_min_params:,}")
    if meta.get("exact_trainable_parameters") is not None and actual_params > 0 and int(meta["exact_trainable_parameters"]) != actual_params:
        reasons.append(f"METADATA_PARAMETER_MISMATCH: metadata={meta['exact_trainable_parameters']:,} actual={actual_params:,}")

    if int(meta.get("actual_steps", 0)) <= 0:
        reasons.append("ZERO_OPTIMIZER_STEPS")
    if int(meta.get("actual_tokens_processed", 0)) <= 0:
        reasons.append("ZERO_CONSUMED_TOKENS")

    val_m = meta.get("validation_metrics") or {}
    if val_m.get("status") != "COMPUTED" or val_m.get("val_loss") is None:
        reasons.append("VALIDATION_METRICS_MISSING_OR_UNCOMPUTED")

    if not allow_smoke:
        lic = str(manifest.get("license") or "")
        if lic not in APPROVED_LICENSES or str(manifest.get("approval_status", "")).lower() != "approved":
            reasons.append("DATASET_NOT_APPROVED_OR_UNLICENSED")

    # Layer 3: Ed25519 signature verification (when signed or required)
    sig_ok, sig_status = verify_training_manifest_signature(meta)
    if not sig_ok:
        reasons.append(sig_status)
    elif require_signature and sig_status != "ED25519_VERIFIED":
        reasons.append("SIGNATURE_REQUIRED_BUT_UNSIGNED")

    passed = len(reasons) == 0
    return {
        "release": "bumblecore-2.6",
        "gate": expected_stage,
        "status": "PASSED" if passed else "BLOCKED",
        "passed": passed,
        "reasons": reasons,
        "run_id": meta.get("run_id"),
        "git_commit": meta.get("git_commit"),
        "checkpoint_hash": actual_ckpt_hash,
        "config_hash": actual_cfg_hash,
        "tokenizer_hash": actual_tok_hash,
        "dataset_manifest_hash": declared_ds_hash,
        "actual_parameters": actual_params,
        "actual_steps": meta.get("actual_steps", 0),
        "actual_tokens": meta.get("actual_tokens_processed", 0),
        "smoke_mode": meta.get("smoke_mode"),
        "validation_loss": val_m.get("val_loss"),
        "signature_status": sig_status,
        "structure": struct,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
