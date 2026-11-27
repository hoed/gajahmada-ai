"""
GajahMada Native Byte-Level BPE Tokenizer (`GNNMBPETokenizer`).
Implemented from scratch in pure Python with zero external tokenizer dependencies.
Byte-level base vocabulary (256 bytes) guarantees lossless encoding/decoding across
Bahasa Indonesia, regional Nusantara scripts, English, Arabic, math symbols, and code.

Uses word-frequency incremental BPE with a max-heap (`heapq`) so training up to
32,000 merges completes in seconds in pure Python instead of hours.
"""

from collections import Counter, defaultdict
import heapq
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


SPECIAL_TOKENS: List[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|sys|>",
    "<|user|>",
    "<|assistant|>",
    "<|sep|>",
]

_CHUNK_RE = re.compile(r"\s*\S+|\s+")


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
        self._encode_cache: Dict[str, List[int]] = {}

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
        """Train BPE merge rules from scratch on a multilingual corpus using fast incremental word-frequency BPE."""
        base_size = self.byte_offset + 256
        num_merges = max(0, target_vocab_size - base_size)

        # Count unique chunk frequencies (lossless pre-tokenization: "".join(chunks) == text)
        word_freq: Counter = Counter()
        doc_count = 0
        for t in texts:
            if not t or not t.strip():
                continue
            doc_count += 1
            for chunk in _CHUNK_RE.findall(t):
                if len(chunk) <= 128:
                    word_freq[chunk.encode("utf-8")] += 1
                else:
                    raw_b = chunk.encode("utf-8")
                    for pos in range(0, len(raw_b), 128):
                        word_freq[raw_b[pos : pos + 128]] += 1

        words: List[List[int]] = []
        freqs: List[int] = []
        for wb, cnt in word_freq.items():
            if len(wb) >= 2:
                words.append([self.byte_offset + b for b in wb])
                freqs.append(cnt)

        self.merges = []
        self.merge_ranks = {}
        self.vocab_bytes = {self.byte_offset + b: bytes([b]) for b in range(256)}
        self._encode_cache = {}

        # Build initial pair counts and inverted index: pair -> set of word indices
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for wid, seq in enumerate(words):
            f = freqs[wid]
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] += f
                pair_to_words[p].add(wid)

        # Max-heap storing (-count, pair) with lazy invalidation
        heap: List[Tuple[int, Tuple[int, int]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items() if cnt >= min_frequency
        ]
        heapq.heapify(heap)

        for step in range(num_merges):
            best_pair: Optional[Tuple[int, int]] = None
            while heap:
                neg_cnt, cand = heapq.heappop(heap)
                cur_cnt = pair_counts.get(cand, 0)
                if cur_cnt < min_frequency:
                    continue
                if -neg_cnt != cur_cnt:
                    # Stale heap entry; re-push current count
                    heapq.heappush(heap, (-cur_cnt, cand))
                    continue
                best_pair = cand
                break

            if best_pair is None:
                break

            new_id = base_size + step
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = new_id
            self.vocab_bytes[new_id] = self.vocab_bytes[best_pair[0]] + self.vocab_bytes[best_pair[1]]

            affected_wids = list(pair_to_words.pop(best_pair, ()))
            pair_counts[best_pair] = 0
            modified_pairs: Set[Tuple[int, int]] = set()

            for wid in affected_wids:
                old_seq = words[wid]
                f = freqs[wid]
                # Remove old pair counts for this word
                for i in range(len(old_seq) - 1):
                    op = (old_seq[i], old_seq[i + 1])
                    if op != best_pair:
                        pair_counts[op] -= f
                        modified_pairs.add(op)
                # Merge this word
                new_seq = self._merge_sequence(old_seq, best_pair, new_id)
                words[wid] = new_seq
                # Add new pair counts for this word
                for i in range(len(new_seq) - 1):
                    np = (new_seq[i], new_seq[i + 1])
                    pair_counts[np] += f
                    pair_to_words[np].add(wid)
                    modified_pairs.add(np)

            for mp in modified_pairs:
                c = pair_counts.get(mp, 0)
                if c >= min_frequency:
                    heapq.heappush(heap, (-c, mp))

        return {
            "vocab_size": self.vocab_size,
            "merges_learned": len(self.merges),
            "documents_trained": doc_count,
        }

    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._encode_cache.get(chunk)
        if cached is not None:
            return cached
        ids: List[int] = [self.byte_offset + b for b in chunk.encode("utf-8")]
        if self.merge_ranks and len(ids) >= 2:
            merge_ranks = self.merge_ranks
            while len(ids) >= 2:
                best_pair: Optional[Tuple[int, int]] = None
                best_id = 10**9
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    merged_id = merge_ranks.get(pair)
                    if merged_id is not None and merged_id < best_id:
                        best_id = merged_id
                        best_pair = pair
                if best_pair is None:
                    break
                ids = self._merge_sequence(ids, best_pair, best_id)
        if len(self._encode_cache) < 65536 and len(chunk) <= 64:
            self._encode_cache[chunk] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        if text:
            if not self.merge_ranks:
                ids.extend(self.byte_offset + b for b in text.encode("utf-8"))
            else:
                for chunk in _CHUNK_RE.findall(text):
                    ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(self.eos_id)
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
