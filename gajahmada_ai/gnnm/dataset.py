"""
Multilingual Balanced Corpus Cleaning, SHA-256 Deduplication & Sequence Packing Pipeline
for GajahMada Native Neural Model (GNNM).
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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

    def load_from_jsonl(self, path: str, text_field: str = "text", lang: str = "multi", max_docs: int = 100000) -> int:
        """Stream-load and deduplicate documents from a `.jsonl` corpus file."""
        if not os.path.exists(path):
            return 0
        added = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if added >= max_docs:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    txt = obj.get(text_field, "") if isinstance(obj, dict) else str(obj)
                    doc_lang = obj.get("lang", lang) if isinstance(obj, dict) else lang
                    if self.add_document(txt, lang=doc_lang, source=os.path.basename(path)):
                        added += 1
                except Exception:
                    continue
        return added

    def load_from_directory(self, dir_path: str, max_docs: int = 100000) -> int:
        """Load `.txt` and `.jsonl` files from a corpus directory with SHA-256 deduplication."""
        if not os.path.isdir(dir_path):
            return 0
        added = 0
        for root, _, files in os.walk(dir_path):
            for fn in sorted(files):
                if added >= max_docs:
                    break
                full = os.path.join(root, fn)
                if fn.endswith(".jsonl"):
                    added += self.load_from_jsonl(full, max_docs=max_docs - added)
                elif fn.endswith(".txt"):
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            for para in f.read().split("\n\n"):
                                if added >= max_docs:
                                    break
                                if self.add_document(para, lang="multi", source=fn):
                                    added += 1
                    except Exception:
                        continue
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

    @staticmethod
    def build_seed_instruction_examples() -> List[Dict[str, str]]:
        """Balanced Indonesian + English concise instruction-tuning (SFT) seed examples."""
        sys_msg = "Kamu adalah GajahMada, AI berdaulat. Jawab ringkas dan akurat."
        return [
            {"system": sys_msg, "user": "Siapa kamu?", "assistant": "Saya GajahMada, mesin kecerdasan hibrida berdaulat dan terdistribusi."},
            {"system": sys_msg, "user": "Apa isi Sumpah Palapa?", "assistant": "Sumpah Palapa (1336 M) adalah ikrar Mahapatih Gajah Mada untuk menyatukan seluruh Nusantara."},
            {"system": sys_msg, "user": "Apa itu fotosintesis?", "assistant": "Fotosintesis adalah proses tumbuhan mengubah CO2 dan air menjadi glukosa dan oksigen dengan cahaya matahari."},
            {"system": sys_msg, "user": "Who are you?", "assistant": "I am GajahMada, a sovereign hybrid distributed intelligence node."},
            {"system": sys_msg, "user": "What is Ed25519?", "assistant": "Ed25519 (RFC 8032) is a fast, deterministic public-key digital signature algorithm."},
            {"system": sys_msg, "user": "What is RoPE in a Transformer?", "assistant": "RoPE (Rotary Positional Embedding) encodes token positions by rotating query and key vectors."},
        ]

    @staticmethod
    def pack_instruction_pairs(
        tokenizer: GNNMBPETokenizer,
        examples: List[Dict[str, str]],
        seq_len: int = 64,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Build Supervised Fine-Tuning (SFT) (input_ids, target_ids) pairs using GNNM native
        chat tokens (<|sys|>, <|user|>, <|assistant|>, <|eos|>) with `-100` loss masking
        on system/user prompt tokens so gradients only optimize assistant answers.
        """
        s_sys = tokenizer.special_to_id["<|sys|>"]
        s_usr = tokenizer.special_to_id["<|user|>"]
        s_ast = tokenizer.special_to_id["<|assistant|>"]
        pad_id = tokenizer.pad_id

        pairs: List[Tuple[List[int], List[int]]] = []
        for ex in examples:
            prompt_ids = (
                [tokenizer.bos_id, s_sys]
                + tokenizer.encode(ex.get("system", ""))
                + [s_usr]
                + tokenizer.encode(ex.get("user", ""))
                + [s_ast]
            )
            answer_ids = tokenizer.encode(ex.get("assistant", "")) + [tokenizer.eos_id]
            # Guarantee assistant answer tokens fit inside seq_len (never mask 100% of targets)
            max_prompt = max(4, seq_len - min(len(answer_ids), max(8, seq_len // 2)))
            if len(prompt_ids) > max_prompt:
                prompt_ids = prompt_ids[: max_prompt - 1] + [s_ast]
            full = (prompt_ids + answer_ids)[: seq_len + 1]
            if len(full) < 2:
                continue

            x = full[:-1]
            y = full[1:]
            # Mask prompt tokens so CE loss ignores everything before the first assistant token
            mask_len = max(0, len(prompt_ids) - 1)
            y_masked = [(-100 if idx < mask_len else tok) for idx, tok in enumerate(y)]

            if len(x) < seq_len:
                pad_n = seq_len - len(x)
                x = x + [pad_id] * pad_n
                y_masked = y_masked + [-100] * pad_n
            pairs.append((x[:seq_len], y_masked[:seq_len]))
        return pairs
