"""
GajahMada GNNM BumbleCore 300M — True Streaming Training Engine (Release 2.6, Phase 3).

Provides bounded-memory, lazy-tokenizing, deterministic streaming loaders for large
pretraining and multi-turn SFT JSONL datasets:
- `StreamingPretrainDataset`: line-by-line JSONL reader -> lazy tokenization with BOS/EOS ->
  streaming fixed-length sequence packing -> bounded deterministic shuffle buffer ->
  cursor-based resume & live RAM/throughput telemetry.
- `StreamingSFTDataset`: line-by-line JSONL conversation reader -> role/format validation ->
  assistant-only `-100` loss masking -> bounded shuffle buffer -> cursor resume.

NEVER loads the full corpus into RAM.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import psutil

from gajahmada_ai.gnnm.bumblecore.data_pipeline import pack_multiturn_sft_pairs
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer

_VALID_SFT_ROLES = {"system", "user", "assistant"}


def _current_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


class StreamingPretrainDataset:
    """
    Bounded-memory streaming pretraining sequence generator.
    Yields `(input_ids, target_ids)` causal LM blocks of exact length `seq_len`.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: GNNMBPETokenizer,
        seq_len: int = 2048,
        text_field: str = "text",
        shuffle_buffer_size: int = 64,
        seed: int = 42,
        skip_sequences: int = 0,
        cycle: bool = True,
    ):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Streaming pretrain dataset not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.text_field = text_field
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.skip_sequences = max(0, int(skip_sequences))
        self.cycle = bool(cycle)

        self.initial_ram_mb: float = _current_rss_mb()
        self.peak_ram_mb: float = self.initial_ram_mb
        self.records_read: int = 0
        self.records_accepted: int = 0
        self.sequences_yielded: int = 0
        self.tokens_yielded: int = 0
        self.data_loading_sec: float = 0.0

    def _raw_sequence_stream(self) -> Iterator[Tuple[List[int], List[int]]]:
        """Stream JSONL -> lazy tokenize -> pack into `seq_len` blocks with bounded token buffer."""
        token_buf: List[int] = []
        block_need = self.seq_len + 1
        while True:
            yielded_in_epoch = False
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    t0 = time.perf_counter()
                    raw = line.strip()
                    if not raw:
                        continue
                    self.records_read += 1
                    try:
                        obj = json.loads(raw)
                        txt = obj.get(self.text_field, "") if isinstance(obj, dict) else ""
                    except Exception:
                        continue
                    if not isinstance(txt, str) or not txt.strip():
                        continue
                    self.records_accepted += 1
                    toks = self.tokenizer.encode(txt.strip(), add_bos=True, add_eos=True)
                    token_buf.extend(toks)
                    self.data_loading_sec += time.perf_counter() - t0

                    while len(token_buf) >= block_need:
                        chunk = token_buf[:block_need]
                        del token_buf[: self.seq_len]
                        yielded_in_epoch = True
                        yield (chunk[:-1], chunk[1:])
            if not self.cycle or not yielded_in_epoch:
                # Flush final partial block padded cleanly if any tokens remain and not cycling
                if not self.cycle and len(token_buf) >= 2:
                    pad_id = self.tokenizer.pad_id
                    x = token_buf[:-1][: self.seq_len]
                    y = token_buf[1:][: self.seq_len]
                    if len(x) < self.seq_len:
                        pad_n = self.seq_len - len(x)
                        x = x + [pad_id] * pad_n
                        y = y + [-100] * pad_n
                    yield (x, y)
                break

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        rng = random.Random(self.seed)
        buf: List[Tuple[List[int], List[int]]] = []
        skipped = 0
        for pair in self._raw_sequence_stream():
            if self.shuffle_buffer_size <= 1:
                out = pair
            else:
                buf.append(pair)
                if len(buf) < self.shuffle_buffer_size:
                    continue
                idx = rng.randrange(len(buf))
                out = buf.pop(idx)
            if skipped < self.skip_sequences:
                skipped += 1
                continue
            self.sequences_yielded += 1
            self.tokens_yielded += sum(1 for t in out[0] if t != self.tokenizer.pad_id)
            if self.sequences_yielded % 32 == 0:
                self.peak_ram_mb = max(self.peak_ram_mb, _current_rss_mb())
            yield out

        while buf:
            idx = rng.randrange(len(buf))
            out = buf.pop(idx)
            if skipped < self.skip_sequences:
                skipped += 1
                continue
            self.sequences_yielded += 1
            self.tokens_yielded += sum(1 for t in out[0] if t != self.tokenizer.pad_id)
            yield out
        self.peak_ram_mb = max(self.peak_ram_mb, _current_rss_mb())

    def telemetry(self) -> Dict[str, Any]:
        return {
            "loader": "StreamingPretrainDataset",
            "initial_ram_mb": self.initial_ram_mb,
            "peak_ram_mb": self.peak_ram_mb,
            "ram_delta_mb": round(max(0.0, self.peak_ram_mb - self.initial_ram_mb), 2),
            "records_read": self.records_read,
            "records_accepted": self.records_accepted,
            "sequences_yielded": self.sequences_yielded,
            "tokens_yielded": self.tokens_yielded,
            "data_loading_sec": round(self.data_loading_sec, 4),
            "tokens_per_sec": round(self.tokens_yielded / max(1e-6, self.data_loading_sec), 1),
        }


class StreamingSFTDataset:
    """
    Bounded-memory streaming multi-turn SFT JSONL dataset.
    Validates message roles and applies strict assistant-only `-100` loss masking.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: GNNMBPETokenizer,
        seq_len: int = 2048,
        shuffle_buffer_size: int = 32,
        seed: int = 42,
        skip_sequences: int = 0,
        cycle: bool = True,
    ):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Streaming SFT dataset not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.skip_sequences = max(0, int(skip_sequences))
        self.cycle = bool(cycle)

        self.initial_ram_mb: float = _current_rss_mb()
        self.peak_ram_mb: float = self.initial_ram_mb
        self.records_read: int = 0
        self.records_accepted: int = 0
        self.sequences_yielded: int = 0
        self.tokens_yielded: int = 0
        self.data_loading_sec: float = 0.0

    def _raw_sft_stream(self) -> Iterator[Tuple[List[int], List[int]]]:
        while True:
            yielded_in_epoch = False
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    t0 = time.perf_counter()
                    raw = line.strip()
                    if not raw:
                        continue
                    self.records_read += 1
                    try:
                        obj = json.loads(raw)
                        msgs = obj.get("messages") if isinstance(obj, dict) else None
                    except Exception:
                        continue
                    if not isinstance(msgs, list) or not msgs:
                        continue
                    if any(not isinstance(m, dict) or m.get("role", "").lower() not in _VALID_SFT_ROLES for m in msgs):
                        continue
                    packed = pack_multiturn_sft_pairs(self.tokenizer, [msgs], seq_len=self.seq_len)
                    self.data_loading_sec += time.perf_counter() - t0
                    if packed:
                        self.records_accepted += 1
                        yielded_in_epoch = True
                        yield packed[0]
            if not self.cycle or not yielded_in_epoch:
                break

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        rng = random.Random(self.seed)
        buf: List[Tuple[List[int], List[int]]] = []
        skipped = 0
        for pair in self._raw_sft_stream():
            if self.shuffle_buffer_size <= 1:
                out = pair
            else:
                buf.append(pair)
                if len(buf) < self.shuffle_buffer_size:
                    continue
                idx = rng.randrange(len(buf))
                out = buf.pop(idx)
            if skipped < self.skip_sequences:
                skipped += 1
                continue
            self.sequences_yielded += 1
            self.tokens_yielded += sum(1 for t in out[0] if t != self.tokenizer.pad_id)
            if self.sequences_yielded % 32 == 0:
                self.peak_ram_mb = max(self.peak_ram_mb, _current_rss_mb())
            yield out
        while buf:
            idx = rng.randrange(len(buf))
            out = buf.pop(idx)
            if skipped < self.skip_sequences:
                skipped += 1
                continue
            self.sequences_yielded += 1
            self.tokens_yielded += sum(1 for t in out[0] if t != self.tokenizer.pad_id)
            yield out
        self.peak_ram_mb = max(self.peak_ram_mb, _current_rss_mb())

    def telemetry(self) -> Dict[str, Any]:
        return {
            "loader": "StreamingSFTDataset",
            "initial_ram_mb": self.initial_ram_mb,
            "peak_ram_mb": self.peak_ram_mb,
            "ram_delta_mb": round(max(0.0, self.peak_ram_mb - self.initial_ram_mb), 2),
            "records_read": self.records_read,
            "records_accepted": self.records_accepted,
            "sequences_yielded": self.sequences_yielded,
            "tokens_yielded": self.tokens_yielded,
            "data_loading_sec": round(self.data_loading_sec, 4),
            "tokens_per_sec": round(self.tokens_yielded / max(1e-6, self.data_loading_sec), 1),
        }
