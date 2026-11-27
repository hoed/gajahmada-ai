"""
Release 2.6 Batch 2 — Exact Token Budget Enforcement Acceptance Tests (All 7 Cases).

1. Target 100 tokens produces exact 100 tokens (never 101)
2. Target smaller than 1 batch (e.g. 5 tokens) produces exact 5 tokens
3. Target exhausted mid-gradient-accumulation handled cleanly without overshoot
4. Validation tokens do NOT increment consumed training tokens
5. Padding tokens do NOT increment consumed training tokens
6. Resume continues without repeating consumed tokens & hits exact target
7. Resume rejects silent target_tokens change unless allow_resume_mismatch=True
"""
from __future__ import annotations

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_smoke_config
from gajahmada_ai.gnnm.model import HAS_TORCH
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.gnnm.trainer import GNNMTrainer


def _skipif_no_torch(fn):
    return pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")(fn) if pytest else fn


@_skipif_no_torch
def test_all_7_exact_token_budget_cases(tmp_path):
    tok = GNNMBPETokenizer()
    cfg = build_bumblecore_smoke_config(vocab_size=tok.vocab_size, max_seq_len=32)
    pad = tok.pad_id
    # Each sample has 12 non-pad tokens + 4 padding tokens = 16 raw tokens
    pairs = [([10 + (i % 50)] * 12 + [pad] * 4, [11 + (i % 50)] * 12 + [-100] * 4) for i in range(32)]

    # 1. Target 100 tokens MUST produce exact 100 (never 101!)
    t1 = GNNMTrainer(cfg, tok, lr=3e-4, seed=42)
    r1 = t1.train_steps(pairs, steps=50, batch_size=2, target_tokens=100)
    ta1 = r1["token_accounting"]
    assert ta1["tokens_consumed"] == 100, ta1
    assert ta1["token_budget_remaining"] == 0
    assert ta1["token_budget_reached"] is True
    # 5. Padding excluded: raw_input_tokens > tokens_consumed
    assert ta1["raw_input_tokens"] > ta1["tokens_consumed"]

    # 2. Target smaller than 1 batch (5 tokens < 2*12=24) -> exact 5 tokens
    t2 = GNNMTrainer(cfg, tok, lr=3e-4, seed=42)
    r2 = t2.train_steps(pairs, steps=10, batch_size=2, target_tokens=5)
    assert r2["token_accounting"]["tokens_consumed"] == 5
    assert r2["token_accounting"]["token_budget_reached"] is True

    # 3. Target exhausted mid-gradient-accumulation (grad_accum=4, batch_size=2 -> 96/step; target=30)
    t3 = GNNMTrainer(cfg, tok, lr=3e-4, seed=42)
    r3 = t3.train_steps(pairs, steps=10, batch_size=2, grad_accum_steps=4, target_tokens=30)
    assert r3["token_accounting"]["tokens_consumed"] == 30
    assert r3["token_accounting"]["token_budget_reached"] is True

    # 4. Validation tokens do NOT increment consumed training tokens
    before_consumed = t3.tokens_consumed
    val_res = t3.evaluate_loss(pairs[:4])
    assert val_res["status"] == "COMPUTED"
    assert t3.validation_tokens == 48
    assert t3.tokens_consumed == before_consumed

    # 6 & 7. Resume restores token budget, rejects silent target change, and completes exact 100 without repeating
    ckpt_dir = tmp_path / "resume_budget"
    t_a = GNNMTrainer(cfg, tok, lr=3e-4, seed=42)
    t_a.train_steps(pairs, steps=2, batch_size=2, target_tokens=100)
    assert t_a.tokens_consumed == 48
    cursor_at_save = t_a.dataset_cursor
    t_a.save_resumable_checkpoint(str(ckpt_dir))

    t_b = GNNMTrainer(cfg, tok, lr=3e-4, seed=99)
    # 7. Reject silent target_tokens change
    try:
        t_b.resume_from_checkpoint(str(ckpt_dir), expected_target_tokens=200)
        raise AssertionError("Should have rejected TARGET_TOKENS_MISMATCH")
    except ValueError as exc:
        assert "TARGET_TOKENS_MISMATCH" in str(exc)

    # 6. Exact resume continues from cursor_at_save and hits exact 100
    t_b.resume_from_checkpoint(str(ckpt_dir), expected_target_tokens=100)
    assert t_b.tokens_consumed == 48
    assert t_b.dataset_cursor == cursor_at_save
    r_b = t_b.train_steps(pairs, steps=20, batch_size=2)
    assert r_b["token_accounting"]["tokens_consumed"] == 100
    assert r_b["token_accounting"]["token_budget_reached"] is True
