"""
Release 2.6 Batch 5 — Exact Resume & Reproducibility Acceptance Tests (Phase 5).

Compares Scenario A (continuous N=6 steps) vs. Scenario B (K=3 steps -> save -> resume -> 3 steps)
across ALL 7 required dimensions:
1. Final global_step
2. Consumed tokens & full token accounting
3. Dataset cursor
4. Loss history
5. Model weights
6. Optimizer momentum/variance state tensors
7. Scheduler state (`last_lr`, `total_steps`, `warmup_steps`)
Plus honest `APPROXIMATE_ENVIRONMENT_CHANGED` reporting when precision/hardware changes.
"""
from __future__ import annotations

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gajahmada_ai.gnnm.bumblecore.architecture import build_bumblecore_smoke_config
from gajahmada_ai.gnnm.model import HAS_TORCH, torch
from gajahmada_ai.gnnm.tokenizer import GNNMBPETokenizer
from gajahmada_ai.gnnm.trainer import GNNMTrainer


def _skipif_no_torch(fn):
    return pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")(fn) if pytest else fn


@_skipif_no_torch
def test_7_dimension_exact_resume_reproducibility(tmp_path):
    tok = GNNMBPETokenizer()
    cfg = build_bumblecore_smoke_config(vocab_size=tok.vocab_size, max_seq_len=32)
    pad = tok.pad_id
    pairs = [([12 + i] * 12 + [pad] * 4, [13 + i] * 12 + [-100] * 4) for i in range(24)]
    prov = {"dataset_manifest_sha256": "ds-v26", "tokenizer_sha256": "tok-v26", "config_sha256": "cfg-v26"}

    # Scenario A: Continuous N=6 steps with warmup_steps=2
    t_cont = GNNMTrainer(cfg, tok, lr=3e-4, seed=2026, precision="fp32", provenance_hashes=prov)
    t_cont.train_steps(pairs, steps=6, batch_size=2, warmup_steps=2, total_schedule_steps=6, target_tokens=500)

    # Scenario B: K=3 steps -> save -> resume -> 3 steps
    ckpt_dir = tmp_path / "resume_6"
    t_part = GNNMTrainer(cfg, tok, lr=3e-4, seed=2026, precision="fp32", provenance_hashes=prov)
    t_part.train_steps(pairs, steps=3, batch_size=2, warmup_steps=2, total_schedule_steps=6, target_tokens=500)
    t_part.save_resumable_checkpoint(str(ckpt_dir))

    t_res = GNNMTrainer(cfg, tok, lr=3e-4, seed=999, precision="fp32")
    rinfo = t_res.resume_from_checkpoint(
        str(ckpt_dir),
        expected_dataset_hash="ds-v26",
        expected_tokenizer_hash="tok-v26",
        expected_config_hash="cfg-v26",
        expected_target_tokens=500,
    )
    assert rinfo["resume_mode"] == "EXACT_STATE_RESTORED"
    t_res.train_steps(pairs, steps=3, batch_size=2, warmup_steps=2, total_schedule_steps=6)

    # Compare all 7 dimensions
    assert t_res.global_step == t_cont.global_step == 6
    assert t_res.tokens_consumed == t_cont.tokens_consumed
    assert t_res.dataset_cursor == t_cont.dataset_cursor
    assert len(t_res.loss_history) == len(t_cont.loss_history) == 6
    assert all(abs(a - b) < 1e-5 for a, b in zip(t_res.loss_history, t_cont.loss_history))
    assert abs(t_res.last_lr - t_cont.last_lr) < 1e-12
    assert t_res.scheduler_total_steps == t_cont.scheduler_total_steps == 6

    max_w_diff = max((p1 - p2).abs().max().item() for p1, p2 in zip(t_cont.model.parameters(), t_res.model.parameters()))
    assert max_w_diff < 1e-5, f"Model weight diff={max_w_diff}"

    # Optimizer momentum/variance tensors match
    opt_a = t_cont.optimizer.state_dict()["state"]
    opt_b = t_res.optimizer.state_dict()["state"]
    assert set(opt_a.keys()) == set(opt_b.keys())
    for pid in opt_a:
        assert torch.allclose(opt_a[pid]["exp_avg"], opt_b[pid]["exp_avg"], atol=1e-5)
        assert torch.allclose(opt_a[pid]["exp_avg_sq"], opt_b[pid]["exp_avg_sq"], atol=1e-5)

    # Honest environment-changed warning when precision differs
    t_diff_env = GNNMTrainer(cfg, tok, lr=3e-4, seed=1, precision="bf16")
    r_env = t_diff_env.resume_from_checkpoint(str(ckpt_dir))
    assert r_env["resume_mode"] == "APPROXIMATE_ENVIRONMENT_CHANGED"
    assert any("PRECISION_CHANGED" in w for w in r_env["environment_warnings"])
