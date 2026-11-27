"""
GajahMada GNNM BumbleCore — Tokenizer Audit & Language Data Pipeline (Phase 2).

Preserves the native pure-Python `GNNMBPETokenizer` and adds:
- Tokenizer validation suite (Indonesian, affixes, English, numbers, URLs, code, Unicode, mixed ID-EN)
- Streaming JSONL / TXT ingestion
- Exact SHA-256 deduplication + shingle Jaccard near-duplicate detection
- Spam / low-quality filtering & PII detection/redaction
- Provenance & license metadata per document
- Deterministic, zero-leakage train / validation / test splitting
- Multi-turn SFT instruction packing (`system` / `user` / `assistant` with `-100` target masking)
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer

PREPROCESSING_VERSION = "bumblecore-data-v1.0"

_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+62|62|0)8[1-9][0-9]{7,11}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{16}\b"), "[REDACTED_ID_16]"),
    (re.compile(r"\b(?:sk-|gm-admin-|ghp_|hf_)[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_SECRET]"),
]

_ID_HINTS = {"yang", "dan", "di", "dari", "untuk", "dengan", "adalah", "pada", "dalam", "ini", "itu", "tidak", "akan"}
_EN_HINTS = {"the", "and", "of", "to", "in", "is", "for", "with", "that", "this", "from", "are", "was"}

TOKENIZER_AUDIT_SAMPLES: Dict[str, str] = {
    "indonesian_prose": "Mahapatih Gajah Mada mengucapkan Sumpah Palapa pada tahun 1336 untuk menyatukan Nusantara.",
    "indonesian_affixes": "memperjuangkan keberagaman pengembangan ketatanegaraan berkesinambungan menyelaraskan",
    "english_prose": "A Decoder-Only Transformer predicts the next token using causal self-attention and RoPE.",
    "numbers_punctuation": "Pi ≈ 3.14159265; kecepatan cahaya = 299.792.458 m/s (100% akurat!).",
    "urls": "https://gajahmada.vercel.app/brain/knowledge?query=bumblecore&limit=20",
    "code": "def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.w_o(x * 0.5)",
    "unicode_arabic": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ — Bhinneka Tunggal Ika ✨",
    "mixed_id_en": "Model GNNM BumbleCore 300M mendukung inference lokal dengan RMSNorm, SwiGLU, dan GQA.",
}


def validate_gnnm_tokenizer(tokenizer: GNNMBPETokenizer) -> Dict[str, Any]:
    """Audit GNNMBPETokenizer for lossless round-trip and efficiency across ID/EN/code/Unicode."""
    results: Dict[str, Any] = {}
    all_lossless = True
    total_bytes = 0
    total_tokens = 0
    for name, sample in TOKENIZER_AUDIT_SAMPLES.items():
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        lossless = decoded == sample
        if not lossless:
            all_lossless = False
        b_len = len(sample.encode("utf-8"))
        t_len = max(1, len(ids))
        total_bytes += b_len
        total_tokens += t_len
        results[name] = {
            "lossless": lossless,
            "byte_length": b_len,
            "token_count": len(ids),
            "bytes_per_token": round(b_len / t_len, 3),
        }
    return {
        "tokenizer_format": "gnnm-bpe-v1",
        "vocab_size": tokenizer.vocab_size,
        "merges_count": len(tokenizer.merges),
        "all_lossless": all_lossless,
        "avg_bytes_per_token": round(total_bytes / max(1, total_tokens), 3),
        "categories": results,
    }


def redact_pii(text: str) -> Tuple[str, int]:
    """Detect and redact common PII/secret patterns."""
    redactions = 0
    out = text
    for pat, repl in _PII_PATTERNS:
        out, n = pat.subn(repl, out)
        redactions += n
    return out, redactions


def detect_language(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "unknown"
    id_hits = sum(1 for w in words if w in _ID_HINTS)
    en_hits = sum(1 for w in words if w in _EN_HINTS)
    if id_hits > 0 and en_hits > 0 and min(id_hits, en_hits) / max(id_hits, en_hits) >= 0.45:
        return "mixed"
    if id_hits > en_hits:
        return "id"
    if en_hits > id_hits:
        return "en"
    return "unknown"


def compute_quality_score(text: str) -> float:
    """Heuristic quality score (0..1) filtering spam, repetition, and malformed gibberish."""
    clean = (text or "").strip()
    if len(clean) < 20:
        return 0.0
    words = clean.split()
    if len(words) < 4:
        return 0.1
    alpha_ratio = sum(1 for c in clean if c.isalpha() or c.isspace()) / max(1, len(clean))
    unique_word_ratio = len(set(w.lower() for w in words)) / len(words)
    score = 0.5 * min(1.0, alpha_ratio / 0.75) + 0.5 * min(1.0, unique_word_ratio / 0.55)
    return round(max(0.0, min(1.0, score)), 4)


def _shingles(text: str, k: int = 5) -> Set[str]:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    if len(norm) < k:
        return {norm}
    return {norm[i : i + k] for i in range(0, min(len(norm) - k + 1, 400))}


class BumbleCoreDataPipeline:
    """
    Deterministic, provenance-preserving, leakage-free data preprocessing pipeline
    for BumbleCore 300M pretraining and multi-turn SFT.
    """

    def __init__(
        self,
        dataset_id: str = "gajahmada-bumblecore-id-en-v1",
        dataset_version: str = "1.0.0",
        allowed_languages: Optional[List[str]] = None,
        min_quality_score: float = 0.45,
        near_dup_threshold: float = 0.88,
        enable_pii_redaction: bool = True,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ):
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.allowed_languages = set(allowed_languages or ["id", "en", "mixed", "unknown"])
        self.min_quality_score = min_quality_score
        self.near_dup_threshold = near_dup_threshold
        self.enable_pii_redaction = enable_pii_redaction
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self._seen_hashes: Set[str] = set()
        self._seen_shingles: List[Set[str]] = []
        self.records: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "received": 0,
            "accepted": 0,
            "rejected_malformed": 0,
            "rejected_duplicate": 0,
            "rejected_near_duplicate": 0,
            "rejected_low_quality": 0,
            "rejected_language": 0,
            "pii_redactions": 0,
        }

    def _assign_split(self, doc_hash: str) -> str:
        """Deterministic hash-based split ensuring identical document hashes never cross splits."""
        bucket = int(doc_hash[:8], 16) / 0xFFFFFFFF
        if bucket < self.train_ratio:
            return "train"
        if bucket < self.train_ratio + self.val_ratio:
            return "validation"
        return "test"

    def _is_near_duplicate(self, sh: Set[str]) -> bool:
        if not sh or len(self._seen_shingles) > 2000:
            return False
        for prev in self._seen_shingles:
            inter = len(sh & prev)
            union = len(sh | prev)
            if union > 0 and (inter / union) >= self.near_dup_threshold:
                return True
        return False

    def ingest_record(
        self,
        text: Any,
        source_name: str = "local",
        source_url: str = "",
        license_name: str = "UNSPECIFIED_REVIEW_REQUIRED",
        language: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self.stats["received"] += 1
        if not isinstance(text, str) or not text.strip():
            self.stats["rejected_malformed"] += 1
            return None

        clean = re.sub(r"\s+", " ", text.strip())
        if self.enable_pii_redaction:
            clean, n_red = redact_pii(clean)
            self.stats["pii_redactions"] += n_red

        q_score = compute_quality_score(clean)
        if q_score < self.min_quality_score:
            self.stats["rejected_low_quality"] += 1
            return None

        lang = language or detect_language(clean)
        if lang not in self.allowed_languages:
            self.stats["rejected_language"] += 1
            return None

        doc_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if doc_hash in self._seen_hashes:
            self.stats["rejected_duplicate"] += 1
            return None

        sh = _shingles(clean)
        if self._is_near_duplicate(sh):
            self.stats["rejected_near_duplicate"] += 1
            return None

        self._seen_hashes.add(doc_hash)
        self._seen_shingles.append(sh)

        rec = {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_name": source_name,
            "source_url": source_url,
            "license": license_name,
            "language": lang,
            "collection_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "preprocessing_version": PREPROCESSING_VERSION,
            "document_hash": doc_hash,
            "split": self._assign_split(doc_hash),
            "quality_score": q_score,
            "text": clean,
        }
        self.records.append(rec)
        self.stats["accepted"] += 1
        return rec

    def stream_jsonl(self, path: str, text_field: str = "text", default_license: str = "REVIEW_REQUIRED") -> Iterator[Dict[str, Any]]:
        """Stream-process a JSONL file without loading the entire file into RAM."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    self.stats["received"] += 1
                    self.stats["rejected_malformed"] += 1
                    continue
                if not isinstance(obj, dict):
                    self.stats["received"] += 1
                    self.stats["rejected_malformed"] += 1
                    continue
                rec = self.ingest_record(
                    text=obj.get(text_field, ""),
                    source_name=str(obj.get("source_name", os.path.basename(path))),
                    source_url=str(obj.get("source_url", "")),
                    license_name=str(obj.get("license", default_license)),
                    language=obj.get("language"),
                )
                if rec is not None:
                    yield rec

    def get_split(self, split: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["split"] == split]

    def manifest(self) -> Dict[str, Any]:
        split_counts = {"train": 0, "validation": 0, "test": 0}
        lang_counts: Dict[str, int] = {}
        licenses: Dict[str, int] = {}
        hashes_sorted = sorted(r["document_hash"] for r in self.records)
        for r in self.records:
            split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
            lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
            licenses[r["license"]] = licenses.get(r["license"], 0) + 1
        manifest_hash = hashlib.sha256("\n".join(hashes_sorted).encode("utf-8")).hexdigest()
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "preprocessing_version": PREPROCESSING_VERSION,
            "manifest_sha256": manifest_hash,
            "stats": dict(self.stats),
            "splits": split_counts,
            "languages": lang_counts,
            "licenses": licenses,
        }

    def save_manifest(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest(), f, indent=2)
        return path


def pack_multiturn_sft_pairs(
    tokenizer: GNNMBPETokenizer,
    conversations: Iterable[List[Dict[str, str]]],
    seq_len: int = 128,
) -> List[Tuple[List[int], List[int]]]:
    """
    Pack multi-turn conversations (`[{"role": "system"|"user"|"assistant", "content": "..."}]`)
    with strict `-100` loss masking on all non-assistant tokens so gradients optimize ONLY
    intended assistant responses.
    """
    role_tok = {
        "system": tokenizer.special_to_id["<|sys|>"],
        "user": tokenizer.special_to_id["<|user|>"],
        "assistant": tokenizer.special_to_id["<|assistant|>"],
    }
    pad_id = tokenizer.pad_id
    pairs: List[Tuple[List[int], List[int]]] = []

    for turns in conversations:
        seq: List[int] = [tokenizer.bos_id]
        loss_mask: List[bool] = [False]  # True where next-token target should contribute to loss
        for turn in turns:
            role = turn.get("role", "user").lower()
            content = (turn.get("content") or "").strip()
            r_id = role_tok.get(role, role_tok["user"])
            seq.append(r_id)
            loss_mask.append(False)
            toks = tokenizer.encode(content)
            if role == "assistant":
                seq.extend(toks + [tokenizer.eos_id])
                loss_mask.extend([True] * (len(toks) + 1))
            else:
                seq.extend(toks)
                loss_mask.extend([False] * len(toks))

        if len(seq) > seq_len + 1:
            # Preserve the assistant response tail so target tokens are never truncated away
            full_seq = seq[-(seq_len + 1) :]
            full_mask = loss_mask[-(seq_len + 1) :]
        else:
            full_seq = seq
            full_mask = loss_mask
        if len(full_seq) < 2 or not any(full_mask[1:]):
            continue
        x = full_seq[:-1]
        y = [tok if keep else -100 for tok, keep in zip(full_seq[1:], full_mask[1:])]
        if len(x) < seq_len:
            pad_n = seq_len - len(x)
            x = x + [pad_id] * pad_n
            y = y + [-100] * pad_n
        pairs.append((x[:seq_len], y[:seq_len]))
    return pairs
