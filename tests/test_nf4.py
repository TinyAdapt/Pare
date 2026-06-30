"""Tests for NF4 (normal float 4-bit) quantization."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.core.functional import _NF4_TABLE, dequantize_nf4, quantize_nf4
from pare.layers.linear import QuantizedLinear


# ---------------------------------------------------------------------------
# 1. NF4 codebook
# ---------------------------------------------------------------------------

class TestNF4Codebook:
    def test_codebook_size(self):
        assert len(_NF4_TABLE) == 16

    def test_codebook_includes_zero(self):
        assert 0.0 in _NF4_TABLE.tolist()

    def test_codebook_symmetric(self):
        assert _NF4_TABLE[0].item() == pytest.approx(-1.0)
        assert _NF4_TABLE[-1].item() == pytest.approx(1.0)

    def test_codebook_monotone_increasing(self):
        vals = _NF4_TABLE.tolist()
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# 2. quantize_nf4 / dequantize_nf4
# ---------------------------------------------------------------------------

class TestNF4Functions:
    def test_output_dtype_indices(self):
        W = torch.randn(8, 16)
        scale = W.abs().amax(dim=-1, keepdim=True)
        idx = quantize_nf4(W, scale)
        assert idx.dtype == torch.int32

    def test_indices_in_range(self):
        torch.manual_seed(0)
        W = torch.randn(32, 64)
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        idx = quantize_nf4(W, scale)
        assert idx.min() >= 0 and idx.max() <= 15

    def test_roundtrip_shape(self):
        torch.manual_seed(1)
        W = torch.randn(16, 32)
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        idx = quantize_nf4(W, scale)
        W_hat = dequantize_nf4(idx, scale)
        assert W_hat.shape == W.shape

    def test_dequant_values_from_codebook(self):
        """Each dequantised value must equal codebook[index] × scale."""
        torch.manual_seed(2)
        W = torch.randn(8, 16)
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        idx = quantize_nf4(W, scale)
        W_hat = dequantize_nf4(idx, scale)

        # Normalised: W_hat / scale should be exactly in the codebook
        W_norm = W_hat / scale
        table = _NF4_TABLE.to(W.device)
        min_dist = (W_norm.unsqueeze(-1) - table).abs().min(dim=-1).values
        assert min_dist.max().item() < 1e-5

    def test_reconstruction_mse_vs_int4(self):
        """NF4 should reconstruct normally-distributed weights better than uniform INT4."""
        torch.manual_seed(42)
        W = torch.randn(64, 128)

        # NF4 per-channel
        scale_nf4 = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        idx = quantize_nf4(W, scale_nf4)
        W_nf4 = dequantize_nf4(idx, scale_nf4)
        mse_nf4 = (W - W_nf4).pow(2).mean().item()

        # INT4 per-channel (for comparison)
        from pare.core.functional import dequantize_tensor, quantize_tensor
        from pare.core.scale import compute_scale
        scale_int4, zero_int4 = compute_scale(W, QuantDtype.INT4, granularity="per_channel")
        q_int4 = quantize_tensor(W, scale_int4, zero_int4, QuantDtype.INT4)
        W_int4 = dequantize_tensor(q_int4.float(), scale_int4, zero_int4).reshape_as(W)
        mse_int4 = (W - W_int4).pow(2).mean().item()

        assert mse_nf4 < mse_int4, (
            f"NF4 MSE {mse_nf4:.6f} should be lower than INT4 MSE {mse_int4:.6f} "
            "for normally-distributed weights"
        )

    def test_zero_weight_roundtrip(self):
        W = torch.zeros(8, 16)
        scale = torch.ones(8, 1) * 1.0
        idx = quantize_nf4(W, scale)
        W_hat = dequantize_nf4(idx, scale)
        # Nearest codebook value to 0 is 0.0
        assert W_hat.abs().max().item() < 1e-6

    def test_scale_effect(self):
        """Larger scale should produce larger reconstructed values."""
        W = torch.ones(4, 8)  # all ones
        scale_small = torch.ones(4, 1) * 0.5
        scale_large = torch.ones(4, 1) * 2.0
        idx_small = quantize_nf4(W, scale_small)
        idx_large = quantize_nf4(W, scale_large)
        W_small = dequantize_nf4(idx_small, scale_small)
        W_large = dequantize_nf4(idx_large, scale_large)
        assert W_large.abs().mean() > W_small.abs().mean()


# ---------------------------------------------------------------------------
# 3. QuantConfig normalisation
# ---------------------------------------------------------------------------

class TestNF4Config:
    def test_nf4_forces_per_channel(self):
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        assert cfg.granularity == "per_channel"

    def test_nf4_bits_unchanged(self):
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        assert cfg.bits == 4

    def test_nf4_effective_dtype(self):
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        assert cfg.effective_dtype == QuantDtype.NF4

    def test_nf4_is_float(self):
        assert QuantDtype.NF4.is_float is True


# ---------------------------------------------------------------------------
# 4. QuantizedLinear with NF4
# ---------------------------------------------------------------------------

class TestNF4Linear:
    def test_from_linear_storage(self):
        """NF4 weights are stored packed (same buffer as INT4)."""
        linear = nn.Linear(64, 32, bias=False)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)

        assert ql.packed_weight is not None
        assert ql.q_weight is None
        assert ql.packed_weight.shape == (32, 32)  # [out, in//2]
        assert ql.packed_weight.dtype == torch.uint8

    def test_from_linear_scale_shape(self):
        linear = nn.Linear(64, 32, bias=False)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)
        assert ql.scale.shape == (32, 1)  # per-channel: [out, 1]
        assert ql.zero.shape == (32, 1)
        assert ql.zero.abs().max().item() == 0.0  # symmetric, no zero-point

    def test_dequantize_shape(self):
        linear = nn.Linear(128, 64, bias=False)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)
        W_hat = ql.dequantize()
        assert W_hat.shape == (64, 128)

    def test_dequantize_from_codebook(self):
        """All dequantised values must lie on the NF4 codebook grid."""
        torch.manual_seed(5)
        linear = nn.Linear(64, 32, bias=False)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)

        W_hat = ql.dequantize()
        scale = ql.scale  # [32, 1]
        W_norm = (W_hat / scale.clamp(min=1e-8)).clamp(-1, 1)
        table = _NF4_TABLE
        min_dist = (W_norm.unsqueeze(-1) - table).abs().min(dim=-1).values
        assert min_dist.max().item() < 1e-5

    def test_reconstruction_mse(self):
        torch.manual_seed(3)
        linear = nn.Linear(128, 64, bias=False)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)
        W_hat = ql.dequantize()
        mse = (linear.weight.float() - W_hat).pow(2).mean().item()
        assert mse < 0.001, f"NF4 reconstruction MSE too high: {mse:.6f}"

    def test_forward_pass(self):
        torch.manual_seed(6)
        linear = nn.Linear(32, 16, bias=True)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)

        x = torch.randn(8, 32)
        y = ql(x)
        assert y.shape == (8, 16)
        assert not y.isnan().any()

    def test_forward_matches_dequant(self):
        """QuantizedLinear forward must equal dequantize() + F.linear."""
        torch.manual_seed(7)
        linear = nn.Linear(64, 32, bias=True)
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, cfg)

        x = torch.randn(4, 64)
        y_fwd = ql(x)
        W_hat = ql.dequantize()
        y_ref = nn.functional.linear(x, W_hat.to(x.dtype), ql.bias)
        assert torch.allclose(y_fwd, y_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. End-to-end: quantize() API
# ---------------------------------------------------------------------------

class TestNF4EndToEnd:
    def test_quantize_model(self):
        """quantize() with NF4 replaces nn.Linear with QuantizedLinear(NF4)."""
        from pare import QuantConfig, quantize

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 16)
                self.fc2 = nn.Linear(16, 8)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = MLP()
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        qmodel = quantize(model, cfg)

        assert isinstance(qmodel.fc1, QuantizedLinear)
        assert isinstance(qmodel.fc2, QuantizedLinear)
        assert qmodel.fc1.config.effective_dtype == QuantDtype.NF4
        assert qmodel.fc2.config.effective_dtype == QuantDtype.NF4

        x = torch.randn(4, 32)
        y = qmodel(x)
        assert y.shape == (4, 8)
        assert not y.isnan().any()

    def test_nf4_reconstruction_low_mse(self):
        """NF4 weight reconstruction MSE should be low on a normal MLP."""
        torch.manual_seed(99)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(128, 64, bias=False)
            def forward(self, x):
                return self.fc(x)

        model = MLP()
        W_orig = model.fc.weight.data.clone()
        cfg = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")

        from pare import quantize
        qmodel = quantize(model, cfg)
        mse = (W_orig - qmodel.fc.dequantize()).pow(2).mean().item()
        assert mse < 0.001, f"NF4 reconstruction MSE too high: {mse:.6f}"
