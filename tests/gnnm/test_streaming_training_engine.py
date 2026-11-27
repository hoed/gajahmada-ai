"""
Release 2.6 Batch 3 — True Streaming Training Engine Acceptance Tests (Phase 3E).

Verifies:
1. Synthetic large JSONL dataset (2,000 docs) streamed with bounded RAM (no full-corpus list in memory)
2. Deterministic sequence ordering under identical seed
3. Exact cursor resume (`skip_sequences=K` matches continuous stream slice `[K:]`)
4. Split isolation (`train.jsonl` vs `validation.jsonl` never mix)
5. Streaming SFT preserves assistant-only `-100` masking & rejects malformed roles
6. End-to-end streaming training into `GNNMTrainer` (PyTorch `.venv312`) reports `data_telemetry`
"""
from __future__ import annotations

import json
try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_smoke_config
from gajahmada_ai.gnnm.bumblecore.streaming_loader import (
    StreamingPretrainDataset,
    StreamingSFTDataset,
)
from gajahmada_ai.gnnm.model import HAS_TORCH
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.gnnm.trainer import GNNMTrainer


def _skipif_no_torch(fn):
    return pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")(fn) if pytest else fn


def test_streaming_pretrain_bounded_ram_determinism_cursor_and_split_isolation(tmp_path):
    tok = GNNMBPETokenizer()
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "validation.jsonl"

    # Write 2,000 synthetic training documents + 50 distinct validation documents
    with open(train_p, "w", encoding="utf-8") as f:
        for i in range(2000):
            f.write(json.dumps({"text": f"TRAIN_DOC_{i:04d} arsitektur GajahMada GNNM BumbleCore 300M."}) + "\n")
    with open(val_p, "w", encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"text": f"VAL_ONLY_DOC_{i:04d} evaluasi terpisah tanpa kebocoran."}) + "\n")

    # 1 & 2. Deterministic ordering + bounded RAM
    ds_a = StreamingPretrainDataset(str(train_p), tok, seq_len=32, shuffle_buffer_size=16, seed=77, cycle=False)
    ds_b = StreamingPretrainDataset(str(train_p), tok, seq_len=32, shuffle_buffer_size=16, seed=77, cycle=False)

    seq_a = [pair for _, pair in zip(range(20), ds_a)]
    seq_b = [pair for _, pair in zip(range(20), ds_b)]
    assert seq_a == seq_b
    tel = ds_a.telemetry()
    assert tel["records_read"] < 2000  # Proves lazy streaming (did NOT read all 2,000 docs for 20 seqs!)
    assert tel["ram_delta_mb"] < 64.0

    # 3. Exact cursor resume (`skip_sequences=8` matches `seq_a[8:20]`)
    ds_res = StreamingPretrainDataset(str(train_p), tok, seq_len=32, shuffle_buffer_size=16, seed=77, skip_sequences=8, cycle=False)
    seq_res = [pair for _, pair in zip(range(12), ds_res)]
    assert seq_res == seq_a[8:20]

    # 4. Split isolation: validation stream never contains TRAIN_DOC
    ds_val = StreamingPretrainDataset(str(val_p), tok, seq_len=32, shuffle_buffer_size=1, seed=77, cycle=False)
    val_text = "".join(tok.decode(x) for x, _ in list(ds_val)[:5])
    assert "VAL_ONLY_DOC" in val_text
    assert "TRAIN_DOC" not in val_text


def test_streaming_sft_masking_and_malformed_rejection(tmp_path):
    tok = GNNMBPETokenizer()
    sft_p = tmp_path / "sft_train.jsonl"
    rows = [
        {"messages": [{"role": "invalid_role", "content": "bad"}]},
        {"messages": [{"role": "system", "content": "Kamu GajahMada."}, {"role": "user", "content": "Halo?"}, {"role": "assistant", "content": "Halo Nusantara!"}]},
        {"messages": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}]},
    ]
    sft_p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ds = StreamingSFTDataset(str(sft_p), tok, seq_len=64, shuffle_buffer_size=1, seed=42, cycle=False)
    out = list(ds)
    assert len(out) == 2  # invalid_role rejected
    for x, y in out:
        assert len(x) == 64 and len(y) == 64
        assert y[0] == -100  # prompt masked
        assert any(t != -100 for t in y)  # assistant tokens active


@_skipif_no_torch
def test_streaming_train_and_sft_with_pytorch(tmp_path):
    tok = GNNMBPETokenizer()
    cfg = build_bumblecore_smoke_config(vocab_size=tok.vocab_size, max_seq_len=32)

    pre_p = tmp_path / "pre.jsonl"
    with open(pre_p, "w", encoding="utf-8") as f:
        for i in range(60):
            f.write(json.dumps({"text": f"Dokumen streaming ke-{i} tentang GajahMada AI dan Nusantara."}) + "\n")

    ds_pre = StreamingPretrainDataset(str(pre_p), tok, seq_len=32, shuffle_buffer_size=8, seed=42, cycle=True)
    ds_val = StreamingPretrainDataset(str(pre_p), tok, seq_len=32, shuffle_buffer_size=1, seed=42, cycle=False)
    trainer = GNNMTrainer(cfg, tok, lr=3e-4, seed=42)
    res = trainer.train_steps(ds_pre, steps=3, batch_size=2, val_pairs=ds_val, eval_interval=3)
    assert res["steps"] == 3
    assert res["data_telemetry"] is not None
    assert res["data_telemetry"]["loader"] == "StreamingPretrainDataset"
    assert res["last_validation"]["status"] == "COMPUTED"

    sft_p = tmp_path / "sft.jsonl"
    with open(sft_p, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"Tanya {i}?"}, {"role": "assistant", "content": f"Jawab {i}."}]}) + "\n")
    ds_sft = StreamingSFTDataset(str(sft_p), tok, seq_len=32, shuffle_buffer_size=4, seed=42, cycle=True)
    res_sft = trainer.train_sft(ds_sft, steps=2, batch_size=2, sft_lr=5e-5)
    assert res_sft["steps"] == 2
    assert res_sft["data_telemetry"]["loader"] == "StreamingSFTDataset"
