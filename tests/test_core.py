"""Unit tests for pare.core — dtype, scale, functional, pack."""

import pytest
import torch

from pare.core.dtype import QuantDtype
from pare.core.functional import (
    _NF4_CODEBOOK,
    dequantize_nf4,
    dequantize_tensor,
    quantization_error,
    quantize_nf4,
    quantize_tensor,
)
from pare.core.pack import (
    pack_int4,
    pack_int4_signed,
    unpack_int4,
    unpack_int4_signed,
)
from pare.core.scale import compute_scale


# ---------------------------------------------------------------------------
# QuantDtype
# ---------------------------------------------------------------------------

class TestQuantDtype:
    def test_int4_range(self):
        assert QuantDtype.INT4.qmin == 0
        assert QuantDtype.INT4.qmax == 15
        assert QuantDtype.INT4.levels == 16

    def test_int8_range(self):
        assert QuantDtype.INT8.qmin == -128
        assert QuantDtype.INT8.qmax == 127
        assert QuantDtype.INT8.levels == 256

    def test_int2_range(self):
        assert QuantDtype.INT2.qmin == 0
        assert QuantDtype.INT2.qmax == 3
        assert QuantDtype.INT2.levels == 4

    def test_fp8_no_qmin(self):
        with pytest.raises(TypeError):
            _ = QuantDtype.FP8_E4M3.qmin

    def test_from_bits(self):
        assert QuantDtype.from_bits(4) == QuantDtype.INT4
        assert QuantDtype.from_bits(8) == QuantDtype.INT8


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

class TestComputeScale:
    def _weight(self, out=32, in_=64) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(out, in_)

    def test_per_tensor_shapes(self):
        w = self._weight()
        scale, zero = compute_scale(w, QuantDtype.INT8, granularity="per_tensor")
        assert scale.dim() == 0  # scalar
        assert zero.dim() == 0

    def test_per_channel_shapes(self):
        w = self._weight(32, 64)
        scale, zero = compute_scale(w, QuantDtype.INT4, granularity="per_channel")
        assert scale.shape == (32, 1)
        assert zero.shape == (32, 1)

    def test_per_group_shapes(self):
        w = self._weight(32, 128)
        scale, zero = compute_scale(w, QuantDtype.INT4, granularity="per_group", group_size=64)
        assert scale.shape == (32, 2, 1)
        assert zero.shape == (32, 2, 1)

    def test_sym_zero_is_zero(self):
        w = self._weight()
        _, zero = compute_scale(w, QuantDtype.INT8, granularity="per_tensor", sym=True)
        assert zero.item() == 0.0

    def test_asym_zero_in_range(self):
        w = self._weight()
        _, zero = compute_scale(w, QuantDtype.INT4, granularity="per_channel")
        assert (zero >= QuantDtype.INT4.qmin).all()
        assert (zero <= QuantDtype.INT4.qmax).all()

    def test_per_group_indivisible_raises(self):
        w = self._weight(32, 100)
        with pytest.raises(ValueError, match="not divisible"):
            compute_scale(w, QuantDtype.INT4, granularity="per_group", group_size=64)


# ---------------------------------------------------------------------------
# Quantize / Dequantize round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("granularity", ["per_tensor", "per_channel", "per_group"])
    @pytest.mark.parametrize("sym", [False, True])
    def test_int8_round_trip(self, granularity, sym):
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        dtype = QuantDtype.INT8

        scale, zero = compute_scale(
            w, dtype, granularity=granularity, group_size=128, sym=sym
        )
        q = quantize_tensor(w, scale, zero, dtype)
        w_hat = dequantize_tensor(q, scale, zero)

        # Reshape w_hat back to original shape for error computation
        w_hat = w_hat.reshape_as(w)
        mae = (w - w_hat).abs().mean().item()
        assert mae < 0.02, f"INT8 MAE too high: {mae:.4f} ({granularity}, sym={sym})"

    @pytest.mark.parametrize("granularity", ["per_channel", "per_group"])
    def test_int4_round_trip(self, granularity):
        torch.manual_seed(42)
        w = torch.randn(32, 128)
        dtype = QuantDtype.INT4

        scale, zero = compute_scale(
            w, dtype, granularity=granularity, group_size=128
        )
        q = quantize_tensor(w, scale, zero, dtype)
        w_hat = dequantize_tensor(q, scale, zero).reshape_as(w)

        mae = (w - w_hat).abs().mean().item()
        # INT4 per_group=128 typically gives MAE < 0.1 on random normal weights
        assert mae < 0.12, f"INT4 MAE too high: {mae:.4f} ({granularity})"

    def test_q_values_in_range(self):
        torch.manual_seed(0)
        w = torch.randn(32, 128)
        dtype = QuantDtype.INT4
        scale, zero = compute_scale(w, dtype, granularity="per_group", group_size=128)
        q = quantize_tensor(w, scale, zero, dtype)
        assert q.min().item() >= dtype.qmin
        assert q.max().item() <= dtype.qmax

    def test_quantization_error_metrics(self):
        torch.manual_seed(1)
        w = torch.randn(32, 128)
        dtype = QuantDtype.INT8
        scale, zero = compute_scale(w, dtype, granularity="per_tensor")
        q = quantize_tensor(w, scale, zero, dtype)
        w_hat = dequantize_tensor(q, scale, zero).reshape_as(w)
        metrics = quantization_error(w, w_hat)
        assert "mae" in metrics
        assert "snr_db" in metrics
        assert metrics["snr_db"].item() > 30  # INT8 should give >30 dB SNR


# ---------------------------------------------------------------------------
# INT4 packing round-trip
# ---------------------------------------------------------------------------

class TestPack:
    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(0)
        q = torch.randint(0, 16, (32, 128), dtype=torch.int32)
        packed = pack_int4(q)
        assert packed.dtype == torch.uint8
        assert packed.shape == (32, 64)
        recovered = unpack_int4(packed)
        assert recovered.shape == (32, 128)
        assert (recovered == q).all(), "INT4 pack/unpack roundtrip failed"

    def test_signed_roundtrip(self):
        torch.manual_seed(1)
        q = torch.randint(-8, 8, (16, 64), dtype=torch.int32)
        packed = pack_int4_signed(q)
        recovered = unpack_int4_signed(packed)
        assert (recovered == q).all(), "Signed INT4 pack/unpack roundtrip failed"

    def test_pack_odd_dim_raises(self):
        q = torch.zeros(4, 7, dtype=torch.int32)
        with pytest.raises(ValueError, match="even"):
            pack_int4(q)

    def test_memory_halved(self):
        q = torch.zeros(64, 256, dtype=torch.int32)
        packed = pack_int4(q)
        # packed is uint8: 64*128 bytes vs original int32: 64*256*4 bytes
        # The key check: packed has half the elements in last dim
        assert packed.shape == (64, 128)

    def test_values_preserved_boundary(self):
        # Check min (0) and max (15) values survive packing
        q = torch.tensor([[0, 15, 0, 15]], dtype=torch.int32)
        packed = pack_int4(q)
        recovered = unpack_int4(packed)
        assert (recovered == q).all()


# ---------------------------------------------------------------------------
# NF4 quantization
# ---------------------------------------------------------------------------

class TestNF4:
    def test_codebook_length(self):
        assert len(_NF4_CODEBOOK) == 16

    def test_codebook_sorted(self):
        assert _NF4_CODEBOOK == sorted(_NF4_CODEBOOK)

    def test_codebook_endpoints(self):
        assert _NF4_CODEBOOK[0] == -1.0
        assert _NF4_CODEBOOK[-1] == 1.0
        assert _NF4_CODEBOOK[7] == 0.0

    def test_roundtrip_exact_codebook_values(self):
        """Codebook values quantize to themselves exactly."""
        import torch
        cb = torch.tensor(_NF4_CODEBOOK, dtype=torch.float32).unsqueeze(0)  # [1, 16]
        scale = torch.ones(1, 1)
        indices = quantize_nf4(cb, scale)
        assert (indices == torch.arange(16).unsqueeze(0)).all()
        reconstructed = dequantize_nf4(indices, scale)
        assert torch.allclose(reconstructed, cb, atol=1e-6)

    def test_indices_in_range(self):
        torch.manual_seed(0)
        W = torch.randn(32, 64)
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        indices = quantize_nf4(W, scale)
        assert indices.min() >= 0
        assert indices.max() <= 15
        assert indices.shape == W.shape

    def test_reconstruction_error_better_than_int4(self):
        """NF4 should have lower MSE than INT4 on normally-distributed weights."""
        torch.manual_seed(42)
        W = torch.randn(64, 128)

        # NF4
        scale_nf4 = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        indices = quantize_nf4(W, scale_nf4)
        W_nf4 = dequantize_nf4(indices, scale_nf4)
        mse_nf4 = (W - W_nf4).pow(2).mean()

        # INT4 per-channel
        from pare.core.scale import compute_scale
        scale_int4, zero_int4 = compute_scale(W, QuantDtype.INT4, granularity="per_channel")
        q_int4 = quantize_tensor(W, scale_int4, zero_int4, QuantDtype.INT4)
        W_int4 = dequantize_tensor(q_int4.float(), scale_int4, zero_int4).reshape_as(W)
        mse_int4 = (W - W_int4).pow(2).mean()

        assert mse_nf4 < mse_int4, (
            f"NF4 MSE {mse_nf4:.6f} should be lower than INT4 MSE {mse_int4:.6f}"
        )

    def test_from_linear_nf4(self):
        """QuantizedLinear.from_linear with NF4 dtype roundtrips correctly."""
        from pare.config import QuantConfig
        from pare.layers.linear import QuantizedLinear
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 32, bias=False)
        config = QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
        ql = QuantizedLinear.from_linear(linear, config)
        assert ql.config.effective_dtype == QuantDtype.NF4
        W_hat = ql.dequantize()
        assert W_hat.shape == linear.weight.shape
        mse = (linear.weight.float() - W_hat).pow(2).mean().item()
        assert mse < 0.01, f"NF4 reconstruction MSE too high: {mse:.6f}"
