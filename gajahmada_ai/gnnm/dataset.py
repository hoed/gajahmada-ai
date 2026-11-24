"""
Multilingual Balanced Corpus Cleaning, SHA-256 Deduplication & Sequence Packing Pipeline
for GajahMada Native Neural Model (GNNM).
"""

import hashlib
import re
from typing import Any, Dict, List, Tuple

from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.knowledge.indonesia_corpus import get_indonesia_knowledge_corpus
from gajahmada_ai.knowledge.open_knowledge import OFFLINE_ENCYCLOPEDIA


class GNNMCorpusBuilder:
    """Builds clean, deduplicated, provenance-tracked multilingual training corpora."""

    def __init__(self):
        self._seen_hashes = set()
        self.documents: List[Dict[str, Any]] = []

    @staticmethod
    def clean_text(text: str) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        return clean

    def add_document(self, text: str, lang: str = "id", source: str = "local") -> bool:
        clean = self.clean_text(text)
        if len(clean) < 16:
            return False
        doc_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if doc_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(doc_hash)
        self.documents.append({
            "hash": doc_hash,
            "lang": lang,
            "source": source,
            "text": clean,
        })
        return True

    def load_seed_multilingual_corpus(self) -> int:
        """Seed balanced Indonesian + English foundational knowledge for GNNM-01 PoC."""
        added = 0
        for doc in get_indonesia_knowledge_corpus():
            if self.add_document(doc, lang="id", source="gajahmada_indonesia_corpus"):
                added += 1
        for item in OFFLINE_ENCYCLOPEDIA:
            if self.add_document(f"{item['title']}: {item['extract']}", lang="id", source="gajahmada_encyclopedia"):
                added += 1
        english_seeds = [
            "GajahMada is a sovereign, independent, hybrid distributed intelligence architecture combining native neural language modeling with symbolic and evidence-based reasoning.",
            "A Decoder-Only Transformer predicts the next token autoregressively using causal self-attention, rotary positional embeddings (RoPE), RMSNorm, and SwiGLU feed-forward blocks.",
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy in chlorophyll-bearing plants.",
            "Ed25519 (RFC 8032) provides fast, deterministic public-key digital signatures for cryptographic node identity and tamper-evident audit ledgers.",
            "In distributed collective intelligence, independent verification and source provenance outweigh unverified majority consensus.",
            "Byte-level BPE tokenization encodes any UTF-8 text without unknown token loss across Indonesian, English, Arabic, and programming languages.",
        ]
        for en_doc in english_seeds:
            if self.add_document(en_doc, lang="en", source="gajahmada_multilingual_seed"):
                added += 1
        return added

    def texts(self) -> List[str]:
        return [d["text"] for d in self.documents]

    def pack_token_blocks(
        self,
        tokenizer: GNNMBPETokenizer,
        seq_len: int = 128,
    ) -> List[Tuple[List[int], List[int]]]:
        """Pack tokenized documents into fixed-length (input_ids, target_ids) causal LM training pairs."""
        stream: List[int] = []
        for doc in self.documents:
            stream.extend(tokenizer.encode(doc["text"], add_bos=True, add_eos=True))

        pairs: List[Tuple[List[int], List[int]]] = []
        step = max(1, seq_len)
        for i in range(0, max(0, len(stream) - seq_len), step):
            x = stream[i : i + seq_len]
            y = stream[i + 1 : i + 1 + seq_len]
            if len(x) == seq_len and len(y) == seq_len:
                pairs.append((x, y))
        return pairs
