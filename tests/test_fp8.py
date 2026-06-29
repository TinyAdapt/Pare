"""Tests for FP8 E4M3 weight quantization (W8A16)."""

import pytest
import torch

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.core.functional import (
    _FP8_E4M3_MAX,
    _FP8_E5M2_MAX,
    dequantize_fp8,
    quantize_fp8,
)


# Skip entire module if float8 is not available in this PyTorch build.
fp8_available = hasattr(torch, "float8_e4m3fn")
pytestmark = pytest.mark.skipif(not fp8_available, reason="torch.float8_e4m3fn not available")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestFP8Constants:
    def test_e4m3_max(self):
        assert _FP8_E4M3_MAX == 448.0

    def test_e5m2_max(self):
        assert _FP8_E5M2_MAX == 57344.0


# ---------------------------------------------------------------------------
# quantize_fp8 / dequantize_fp8
# ---------------------------------------------------------------------------

class TestFP8Functions:
    def test_output_dtype_e4m3(self):
        W = torch.randn(8, 16)
        scale = W.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3_MAX
        result = quantize_fp8(W, scale, QuantDtype.FP8_E4M3)
        assert result.dtype == torch.float8_e4m3fn

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="float8_e5m2 cast requires CUDA")
    def test_output_dtype_e5m2(self):
        W = torch.randn(8, 16).cuda()
        scale = W.abs().amax(dim=-1, keepdim=True) / _FP8_E5M2_MAX
        result = quantize_fp8(W, scale, QuantDtype.FP8_E5M2)
        assert result.dtype == torch.float8_e5m2

    def test_invalid_dtype_raises(self):
        W = torch.randn(4, 8)
        scale = torch.ones(4, 1)
        with pytest.raises(ValueError, match="FP8"):
            quantize_fp8(W, scale, QuantDtype.INT8)

    def test_roundtrip_shape_preserved(self):
        torch.manual_seed(0)
        W = torch.randn(32, 64)
        scale = W.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3_MAX
        q = quantize_fp8(W, scale, QuantDtype.FP8_E4M3)
        W_hat = dequantize_fp8(q, scale)
        assert W_hat.shape == W.shape

    def test_dequant_output_is_float32(self):
        torch.manual_seed(1)
        W = torch.randn(16, 32)
        scale = W.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3_MAX
        q = quantize_fp8(W, scale, QuantDtype.FP8_E4M3)
        W_hat = dequantize_fp8(q, scale)
        assert W_hat.dtype == torch.float32

    def test_reconstruction_error_below_int4(self):
        """FP8 should reconstruct normally-distributed weights better than INT4."""
        torch.manual_seed(42)
        W = torch.randn(64, 128)

        # FP8 E4M3 per-channel
        scale_fp8 = W.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3_MAX
        q_fp8 = quantize_fp8(W, scale_fp8, QuantDtype.FP8_E4M3)
        W_fp8 = dequantize_fp8(q_fp8, scale_fp8)
        mse_fp8 = (W - W_fp8).pow(2).mean()

        # INT4 per-channel (for comparison)
        from pare.core.functional import dequantize_tensor, quantize_tensor
        from pare.core.scale import compute_scale
        scale_int4, zero_int4 = compute_scale(W, QuantDtype.INT4, granularity="per_channel")
        q_int4 = quantize_tensor(W, scale_int4, zero_int4, QuantDtype.INT4)
        W_int4 = dequantize_tensor(q_int4.float(), scale_int4, zero_int4).reshape_as(W)
        mse_int4 = (W - W_int4).pow(2).mean()

        assert mse_fp8 < mse_int4, (
            f"FP8 MSE {mse_fp8:.6f} should be lower than INT4 MSE {mse_int4:.6f}"
        )

    def test_zero_weight_roundtrip(self):
        """All-zero weights should reconstruct exactly (no floating-point noise)."""
        W = torch.zeros(8, 16)
        scale = torch.ones(8, 1) / _FP8_E4M3_MAX
        q = quantize_fp8(W, scale, QuantDtype.FP8_E4M3)
        W_hat = dequantize_fp8(q, scale)
        assert W_hat.abs().max() == 0.0

    def test_scale_per_row(self):
        """Each row should use its own scale (per-channel, not per-tensor)."""
        torch.manual_seed(0)
        W = torch.zeros(4, 8)
        W[0] = 1.0   # row 0: large
        W[1] = 0.01  # row 1: small
        scale = W.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3_MAX
        # Row 0 scale should be ~100× row 1 scale
        assert scale[0] > scale[1] * 10


# ---------------------------------------------------------------------------
# QuantConfig FP8
# ---------------------------------------------------------------------------

class TestFP8Config:
    def test_fp8_forces_per_channel(self):
        cfg = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        assert cfg.granularity == "per_channel"

    def test_fp8_forces_sym(self):
        cfg = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        assert cfg.sym is True

    def test_fp8_bits_set_to_8(self):
        cfg = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        assert cfg.bits == 8

    def test_fp8_effective_dtype(self):
        cfg = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        assert cfg.effective_dtype == QuantDtype.FP8_E4M3


# ---------------------------------------------------------------------------
# QuantizedLinear FP8
# ---------------------------------------------------------------------------

class TestFP8Linear:
    def test_from_linear_fp8_e4m3(self):
        """QuantizedLinear.from_linear with FP8 E4M3 roundtrips correctly."""
        import torch.nn as nn
        from pare.layers.linear import QuantizedLinear

        torch.manual_seed(0)
        linear = nn.Linear(64, 32, bias=False)
        config = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, config)

        assert ql.config.effective_dtype == QuantDtype.FP8_E4M3
        # q_weight must be stored as float8
        assert ql.q_weight.dtype == torch.float8_e4m3fn
        # Reconstructed weight should match shape
        W_hat = ql.dequantize()
        assert W_hat.shape == linear.weight.shape
        assert W_hat.dtype == torch.float32

    def test_fp8_reconstruction_mse(self):
        """FP8 reconstruction error should be low on random normal weights."""
        import torch.nn as nn
        from pare.layers.linear import QuantizedLinear

        torch.manual_seed(1)
        linear = nn.Linear(128, 64, bias=False)
        config = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, config)

        W_hat = ql.dequantize()
        mse = (linear.weight.float() - W_hat).pow(2).mean().item()
        assert mse < 1e-4, f"FP8 reconstruction MSE too high: {mse:.6f}"

    def test_fp8_forward_runs(self):
        """Forward pass through an FP8 QuantizedLinear should not crash."""
        import torch.nn as nn
        from pare.layers.linear import QuantizedLinear

        torch.manual_seed(2)
        linear = nn.Linear(32, 16, bias=True)
        config = QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, config)

        x = torch.randn(4, 32)
        out = ql(x)
        assert out.shape == (4, 16)
        assert not out.isnan().any()
