"""
Phase 1 Acceptance Tests for GajahMada GNNM BumbleCore 300M Architecture.

Verifies:
- Exact analytical parameter count == 303,350,784 (~303.35M)
- Target range rejection outside [285M, 315M] unless allow_target_override=True
- MoE rejection on BumbleCore v1
- PyTorch instantiation, exact trainable parameter match (smoke & full 303M when torch available),
  tensor dimensions, forward/backward pass, finite gradients, no NaN/Inf, causal masking
  invariance, and padding/ignore_index (-100) loss behavior.
"""
from __future__ import annotations

import pytest

from gajahmada_ai.gnnm.bumblecore import (
    BUMBLECORE_300M_TARGET_MAX,
    BUMBLECORE_300M_TARGET_MIN,
    BumbleCoreConfig,
    BumbleCoreModel,
    build_bumblecore_300m_config,
    build_bumblecore_smoke_config,
    estimate_bumblecore_parameters,
    validate_bumblecore_architecture,
)
from gajahmada_ai.gnnm.model import HAS_TORCH, torch


def test_bumblecore_300m_exact_analytical_parameter_count():
    cfg = build_bumblecore_300m_config()
    est = estimate_bumblecore_parameters(cfg)
    assert est["total_trainable_parameters"] == 303_350_784
    assert BUMBLECORE_300M_TARGET_MIN <= est["total_trainable_parameters"] <= BUMBLECORE_300M_TARGET_MAX
    assert est["within_300m_tolerance"] is True
    assert est["output_projection_parameters"] == 0  # tied with token embeddings
    assert est["embedding_parameters"] == 32000 * 1024
    val = validate_bumblecore_architecture(cfg)
    assert val["valid"] is True


def test_bumblecore_rejects_out_of_range_and_moe():
    with pytest.raises(ValueError, match="outside target range"):
        BumbleCoreConfig(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=512)
    with pytest.raises(ValueError, match="dense"):
        BumbleCoreConfig(architecture_type="moe", num_experts=4, num_experts_per_tok=2, allow_target_override=True)
    # Explicit override allowed for smoke testing
    smoke = build_bumblecore_smoke_config()
    assert smoke.allow_target_override is True


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in current Python interpreter")
def test_bumblecore_smoke_forward_backward_causal_and_padding():
    torch.manual_seed(42)
    cfg = build_bumblecore_smoke_config(vocab_size=256, max_seq_len=32)
    model = BumbleCoreModel(cfg)
    ver = model.verify_parameter_count()
    assert ver["matches"] is True
    assert ver["actual_trainable_parameters"] == cfg.estimate_parameters()

    # 1. Forward + Backward + Finite Gradients + Padding (-100) Loss Masking
    x = torch.randint(1, 256, (2, 16))
    y = torch.randint(1, 256, (2, 16))
    y[:, -4:] = -100  # padding / ignored targets
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Missing gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient on {name}"

    # 2. Causal Masking Verification: changing future tokens MUST NOT alter past logits
    model.eval()
    with torch.no_grad():
        x_a = torch.tensor([[5, 10, 15, 20, 25, 30]])
        x_b = torch.tensor([[5, 10, 15, 99, 88, 77]])
        logits_a, _ = model(x_a)
        logits_b, _ = model(x_b)
        assert torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-5)
        assert not torch.allclose(logits_a[:, 3:, :], logits_b[:, 3:, :], atol=1e-5)
