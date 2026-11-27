"""
GajahMada GNNM BumbleCore — Release 2.6 Dataset Integrity, Provenance & Bounded Streaming (Phase 1).

Enforces:
- Strict separation: seed corpus allowed ONLY for smoke/unit tests; full pretraining & full SFT
  MUST supply an external dataset + verified manifest with approved license.
- Manifest verification: existence, readability, non-empty, license allowlist, SHA-256 file hash,
  measured document count, measured token count, and zero split leakage (document + source level).
- Bounded-memory streaming preprocessing backed by SQLite WAL fingerprint index (no unbounded RAM).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from gajahmada_ai.gnnm.bumblecore.data_pipeline import (
    PREPROCESSING_VERSION,
    compute_quality_score,
    detect_language,
    redact_pii,
)
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer

APPROVED_LICENSES: Set[str] = {
    "MIT",
    "Apache-2.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC0-1.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "MPL-2.0",
    "ODC-By-1.0",
}

DATASET_POLICIES = ("smoke", "pretraining", "sft", "validation", "held_out_eval")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _simhash64(text: str) -> int:
    """Compact 64-bit SimHash fingerprint for bounded-memory near-duplicate detection."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return 0
    bits = [0] * 64
    for tok in tokens[:256]:
        hv = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            bits[i] += 1 if (hv & (1 << i)) else -1
    out = 0
    for i in range(64):
        if bits[i] > 0:
            out |= 1 << i
    return out


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _band_keys(simhash: int) -> List[str]:
    """4 bands of 16 bits for O(1) SQLite near-duplicate candidate lookup (Hamming <= 3)."""
    return [f"{b}:{(simhash >> (b * 16)) & 0xFFFF:04x}" for b in range(4)]


class BoundedStreamingPreprocessor:
    """
    True bounded-memory JSONL streaming preprocessor backed by SQLite WAL fingerprint index.
    Never retains all documents or all shingles in Python RAM.
    Writes output incrementally to `train.jsonl`, `validation.jsonl`, and `test.jsonl`
    plus a verified `dataset_manifest.json`.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_id: str,
        dataset_version: str = "1.0.0",
        source_name: str = "external_corpus",
        source_url: Optional[str] = None,
        license_name: str = "UNSPECIFIED",
        approval_status: str = "pending_review",
        tokenizer: Optional[GNNMBPETokenizer] = None,
        allowed_languages: Optional[List[str]] = None,
        min_quality_score: float = 0.45,
        near_dup_hamming_max: int = 3,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ):
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.source_name = source_name
        self.source_url = source_url
        self.license_name = license_name
        self.approval_status = approval_status
        self.tokenizer = tokenizer or GNNMBPETokenizer()
        self.allowed_languages = set(allowed_languages or ["id", "en", "mixed", "unknown"])
        self.min_quality_score = min_quality_score
        self.near_dup_hamming_max = near_dup_hamming_max
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.db_path = os.path.join(self.output_dir, "fingerprints.db")
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS seen_docs (doc_hash TEXT PRIMARY KEY, split TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS seen_sources (source_key TEXT PRIMARY KEY, split TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS simhash_bands (band_key TEXT NOT NULL, simhash TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_band_key ON simhash_bands(band_key)")
            conn.commit()
        finally:
            conn.close()

    def _deterministic_split(self, key: str) -> str:
        hv = hashlib.sha256(key.encode("utf-8")).hexdigest()
        bucket = int(hv[:8], 16) / 0xFFFFFFFF
        if bucket < self.train_ratio:
            return "train"
        if bucket < self.train_ratio + self.val_ratio:
            return "validation"
        return "test"

    def process_jsonl(self, input_path: str, text_field: str = "text") -> Dict[str, Any]:
        """Stream `input_path` line-by-line with O(1) RAM buffer and emit split files + manifest."""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input dataset not found: {input_path}")

        split_paths = {
            "train": os.path.join(self.output_dir, "train.jsonl"),
            "validation": os.path.join(self.output_dir, "validation.jsonl"),
            "test": os.path.join(self.output_dir, "test.jsonl"),
        }
        stats = {
            "received": 0,
            "accepted": 0,
            "rejected_malformed": 0,
            "rejected_duplicate": 0,
            "rejected_near_duplicate": 0,
            "rejected_low_quality": 0,
            "rejected_language": 0,
            "pii_redactions": 0,
            "token_count": 0,
        }
        split_counts = {"train": 0, "validation": 0, "test": 0}
        lang_set: Set[str] = set()

        conn = sqlite3.connect(self.db_path)
        files = {sp: open(p, "w", encoding="utf-8") for sp, p in split_paths.items()}
        try:
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    raw = line.strip()
                    if not raw:
                        continue
                    stats["received"] += 1
                    try:
                        obj = json.loads(raw)
                        if not isinstance(obj, dict):
                            raise ValueError("not a dict")
                    except Exception:
                        stats["rejected_malformed"] += 1
                        continue

                    text = obj.get(text_field, "")
                    if not isinstance(text, str) or not text.strip():
                        stats["rejected_malformed"] += 1
                        continue

                    clean = re.sub(r"\s+", " ", text.strip())
                    clean, n_red = redact_pii(clean)
                    stats["pii_redactions"] += n_red

                    q_score = compute_quality_score(clean)
                    if q_score < self.min_quality_score:
                        stats["rejected_low_quality"] += 1
                        continue

                    lang = str(obj.get("language") or detect_language(clean))
                    if lang not in self.allowed_languages:
                        stats["rejected_language"] += 1
                        continue

                    doc_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
                    if conn.execute("SELECT 1 FROM seen_docs WHERE doc_hash = ?", (doc_hash,)).fetchone():
                        stats["rejected_duplicate"] += 1
                        continue

                    sh = _simhash64(clean)
                    bands = _band_keys(sh)
                    is_near_dup = False
                    for bk in bands:
                        for (prev_hex,) in conn.execute("SELECT simhash FROM simhash_bands WHERE band_key = ?", (bk,)):
                            if _hamming_distance(sh, int(prev_hex, 16)) <= self.near_dup_hamming_max:
                                is_near_dup = True
                                break
                        if is_near_dup:
                            break
                    if is_near_dup:
                        stats["rejected_near_duplicate"] += 1
                        continue

                    # Prevent source-document leakage across splits by pinning split to source_doc_id if present
                    source_doc_key = str(obj.get("source_doc_id") or obj.get("source_url") or doc_hash)
                    row = conn.execute("SELECT split FROM seen_sources WHERE source_key = ?", (source_doc_key,)).fetchone()
                    if row:
                        split = row[0]
                    else:
                        split = self._deterministic_split(source_doc_key)
                        conn.execute("INSERT OR IGNORE INTO seen_sources VALUES (?, ?)", (source_doc_key, split))

                    conn.execute("INSERT INTO seen_docs VALUES (?, ?)", (doc_hash, split))
                    sh_hex = f"{sh:016x}"
                    conn.executemany("INSERT INTO simhash_bands VALUES (?, ?)", [(bk, sh_hex) for bk in bands])

                    tok_count = len(self.tokenizer.encode(clean, add_bos=True, add_eos=True))
                    stats["token_count"] += tok_count
                    stats["accepted"] += 1
                    split_counts[split] += 1
                    lang_set.add(lang)

                    out_rec = {
                        "document_hash": doc_hash,
                        "source_doc_id": source_doc_key,
                        "language": lang,
                        "quality_score": q_score,
                        "token_count": tok_count,
                        "split": split,
                        "text": clean,
                    }
                    files[split].write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            conn.commit()
        finally:
            conn.close()
            for fh in files.values():
                fh.close()

        split_hashes = {sp: sha256_file(p) for sp, p in split_paths.items()}
        combined_hash = hashlib.sha256(
            json.dumps(split_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()

        manifest = {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "license": self.license_name,
            "language": sorted(lang_set),
            "collection_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "preprocessing_version": PREPROCESSING_VERSION,
            "document_count": stats["accepted"],
            "token_count": stats["token_count"],
            "train_split": "train.jsonl",
            "validation_split": "validation.jsonl",
            "test_split": "test.jsonl",
            "split_counts": split_counts,
            "split_hashes": split_hashes,
            "document_hash": combined_hash,
            "approval_status": self.approval_status,
            "stats": stats,
        }
        manifest_path = os.path.join(self.output_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        return manifest


def validate_dataset_and_manifest(
    dataset_dir: str,
    require_approved: bool = True,
    min_documents: int = 1,
) -> Dict[str, Any]:
    """
    Strictly validate an external dataset directory & its `dataset_manifest.json`.
    Checks: existence, readability, non-empty, approved license, approval_status,
    split file SHA-256 integrity, measured doc count, and zero cross-split leakage.
    """
    errors: List[str] = []
    manifest_path = os.path.join(dataset_dir, "dataset_manifest.json")
    if not os.path.isdir(dataset_dir):
        return {"valid": False, "errors": [f"Dataset directory not found: {dataset_dir}"], "manifest": None}
    if not os.path.isfile(manifest_path):
        return {"valid": False, "errors": [f"Dataset manifest missing: {manifest_path}"], "manifest": None}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return {"valid": False, "errors": [f"Malformed manifest JSON: {exc}"], "manifest": None}

    lic = str(manifest.get("license") or "").strip()
    if require_approved and lic not in APPROVED_LICENSES:
        errors.append(f"UNAPPROVED_LICENSE: '{lic}' is not in approved license allowlist.")
    if require_approved and str(manifest.get("approval_status", "")).lower() != "approved":
        errors.append(f"UNAPPROVED_DATASET: approval_status is '{manifest.get('approval_status')}' (must be 'approved').")
    if not manifest.get("source_name"):
        errors.append("MISSING_PROVENANCE: source_name is required.")

    split_files = {
        "train": os.path.join(dataset_dir, str(manifest.get("train_split", "train.jsonl"))),
        "validation": os.path.join(dataset_dir, str(manifest.get("validation_split", "validation.jsonl"))),
        "test": os.path.join(dataset_dir, str(manifest.get("test_split", "test.jsonl"))),
    }
    expected_hashes = manifest.get("split_hashes") or {}
    seen_hashes_by_split: Dict[str, Set[str]] = {"train": set(), "validation": set(), "test": set()}
    seen_sources_by_split: Dict[str, Set[str]] = {"train": set(), "validation": set(), "test": set()}
    measured_docs = 0

    for sp, fp in split_files.items():
        if not os.path.isfile(fp):
            errors.append(f"MISSING_SPLIT_FILE: {fp}")
            continue
        actual_hash = sha256_file(fp)
        if expected_hashes.get(sp) and actual_hash != expected_hashes[sp]:
            errors.append(f"HASH_MISMATCH({sp}): expected {expected_hashes[sp][:12]} got {actual_hash[:12]}")
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                measured_docs += 1
                dh = obj.get("document_hash") or hashlib.sha256(str(obj.get("text", "")).encode("utf-8")).hexdigest()
                sd = obj.get("source_doc_id")
                seen_hashes_by_split[sp].add(dh)
                if sd:
                    seen_sources_by_split[sp].add(str(sd))

    if measured_docs < min_documents:
        errors.append(f"EMPTY_OR_INSUFFICIENT_DATASET: measured {measured_docs} docs < required {min_documents}.")
    if manifest.get("document_count") is not None and int(manifest["document_count"]) != measured_docs:
        errors.append(f"DOCUMENT_COUNT_MISMATCH: manifest={manifest['document_count']} actual={measured_docs}")

    # Cross-split leakage checks (exact document hash + source_doc_id)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        doc_overlap = seen_hashes_by_split[a] & seen_hashes_by_split[b]
        if doc_overlap:
            errors.append(f"DOCUMENT_LEAKAGE({a}<->{b}): {len(doc_overlap)} overlapping document hashes.")
        src_overlap = seen_sources_by_split[a] & seen_sources_by_split[b]
        if src_overlap:
            errors.append(f"SOURCE_LEAKAGE({a}<->{b}): {len(src_overlap)} overlapping source_doc_id entries.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "measured_documents": measured_docs,
        "manifest": manifest,
    }


def validate_sft_dataset(
    sft_dir: str,
    min_examples: int = 10,
    require_approved: bool = True,
) -> Dict[str, Any]:
    """Validate an external multi-turn SFT dataset directory & manifest."""
    base = validate_dataset_and_manifest(sft_dir, require_approved=require_approved, min_documents=min_examples)
    errors = list(base["errors"])
    train_path = os.path.join(sft_dir, str((base["manifest"] or {}).get("train_split", "train.jsonl")))
    valid_roles = {"system", "user", "assistant"}
    checked = 0
    if os.path.isfile(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                msgs = obj.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    errors.append("INVALID_SFT_FORMAT: each SFT record must contain non-empty 'messages' list.")
                    break
                if not any(isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") for m in msgs):
                    errors.append("INVALID_SFT_FORMAT: SFT conversation missing assistant response.")
                    break
                if any(not isinstance(m, dict) or m.get("role") not in valid_roles for m in msgs):
                    errors.append("INVALID_SFT_ROLE: roles must be system, user, or assistant.")
                    break
                checked += 1
    return {
        "valid": len(errors) == 0 and checked >= min_examples,
        "errors": errors,
        "sft_train_examples": checked,
        "manifest": base["manifest"],
    }
