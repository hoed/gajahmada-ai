"""
GajahMada Native Byte-Level BPE Tokenizer (`GNNMBPETokenizer`).
Implemented from scratch in pure Python with zero external tokenizer dependencies.
Byte-level base vocabulary (256 bytes) guarantees lossless encoding/decoding across
Bahasa Indonesia, regional Nusantara scripts, English, Arabic, math symbols, and code.
"""

from collections import Counter
import json
import os
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS: List[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|sys|>",
    "<|user|>",
    "<|assistant|>",
    "<|sep|>",
]


class GNNMBPETokenizer:
    """Byte-Level Byte-Pair Encoding (BPE) tokenizer built natively for GNNM."""

    def __init__(self):
        # IDs 0..len(SPECIAL_TOKENS)-1 are reserved special tokens
        # IDs offset..offset+255 are raw UTF-8 bytes 0..255
        self.special_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_special: Dict[int, str] = {idx: tok for tok, idx in self.special_to_id.items()}
        self.byte_offset: int = len(SPECIAL_TOKENS)
        self.merges: List[Tuple[int, int]] = []
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.vocab_bytes: Dict[int, bytes] = {
            self.byte_offset + b: bytes([b]) for b in range(256)
        }

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    @staticmethod
    def _merge_sequence(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out: List[int] = []
        i = 0
        n = len(seq)
        a, b = pair
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def train(self, texts: List[str], target_vocab_size: int = 1024, min_frequency: int = 2) -> Dict[str, int]:
        """Train BPE merge rules from scratch on a multilingual corpus."""
        base_size = self.byte_offset + 256
        num_merges = max(0, target_vocab_size - base_size)

        sequences: List[List[int]] = [
            [self.byte_offset + b for b in t.encode("utf-8")]
            for t in texts
            if t and t.strip()
        ]
        self.merges = []
        self.merge_ranks = {}
        self.vocab_bytes = {self.byte_offset + b: bytes([b]) for b in range(256)}

        for step in range(num_merges):
            pair_counts: Counter = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1
            if not pair_counts:
                break
            best_pair, freq = pair_counts.most_common(1)[0]
            if freq < min_frequency:
                break
            new_id = base_size + step
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = new_id
            self.vocab_bytes[new_id] = self.vocab_bytes[best_pair[0]] + self.vocab_bytes[best_pair[1]]
            sequences = [self._merge_sequence(seq, best_pair, new_id) for seq in sequences]

        return {
            "vocab_size": self.vocab_size,
            "merges_learned": len(self.merges),
            "documents_trained": len(sequences),
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [self.byte_offset + b for b in text.encode("utf-8")]
        if self.merge_ranks:
            while len(ids) >= 2:
                best_pair: Optional[Tuple[int, int]] = None
                best_id = 10**9
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    merged_id = self.merge_ranks.get(pair)
                    if merged_id is not None and merged_id < best_id:
                        best_id = merged_id
                        best_pair = pair
                if best_pair is None:
                    break
                ids = self._merge_sequence(ids, best_pair, best_id)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        buf = bytearray()
        for idx in ids:
            if idx in self.id_to_special:
                if not skip_special_tokens:
                    buf.extend(self.id_to_special[idx].encode("utf-8"))
                continue
            chunk = self.vocab_bytes.get(idx)
            if chunk is not None:
                buf.extend(chunk)
        return buf.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "format": "gnnm-bpe-v1",
            "special_tokens": SPECIAL_TOKENS,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GNNMBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls()
        base_size = tok.byte_offset + 256
        for step, pair_list in enumerate(payload.get("merges", [])):
            a, b = int(pair_list[0]), int(pair_list[1])
            new_id = base_size + step
            tok.merges.append((a, b))
            tok.merge_ranks[(a, b)] = new_id
            tok.vocab_bytes[new_id] = tok.vocab_bytes[a] + tok.vocab_bytes[b]
        return tok
