"""Tests for RangeObserver calibration modes."""

import pytest
import torch

from pare.calibration.observer import RangeObserver
from pare.config import QuantConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def activation_batch():
    torch.manual_seed(42)
    # [batch*seq, in_features] with one obvious outlier channel
    x = torch.randn(256, 64) * 0.5
    x[:, 7] *= 20.0   # channel 7 is a heavy outlier
    return x


# ---------------------------------------------------------------------------
# RangeObserver
# ---------------------------------------------------------------------------

class TestRangeObserverAbsmax:
    def test_absmax_equals_max_abs(self, activation_batch):
        obs = RangeObserver(mode="absmax")
        obs.accumulate(activation_batch)
        result = obs.finalize()
        expected = activation_batch.abs().amax(dim=0)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-5)

    def test_accumulates_across_batches(self):
        torch.manual_seed(0)
        obs = RangeObserver(mode="absmax")
        a = torch.randn(10, 32).abs()
        b = torch.randn(10, 32).abs()
        obs.accumulate(a)
        obs.accumulate(b)
        result = obs.finalize()
        expected = torch.cat([a, b], dim=0).amax(dim=0)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_3d_input(self):
        obs = RangeObserver(mode="absmax")
        x = torch.randn(2, 16, 32)
        obs.accumulate(x)
        result = obs.finalize()
        assert result.shape == (32,)

    def test_reset(self, activation_batch):
        obs = RangeObserver(mode="absmax")
        obs.accumulate(activation_batch)
        obs.reset()
        with pytest.raises(RuntimeError):
            obs.finalize()


class TestRangeObserverPercentile:
    def test_percentile_below_absmax(self, activation_batch):
        obs_abs = RangeObserver(mode="absmax")
        obs_pct = RangeObserver(mode="percentile", percentile=99.0)
        obs_abs.accumulate(activation_batch)
        obs_pct.accumulate(activation_batch)
        absmax = obs_abs.finalize()
        pct = obs_pct.finalize()
        # Percentile clipping: result must be <= absmax
        assert (pct <= absmax + 1e-5).all()
        # On the outlier channel (7), percentile should be noticeably lower
        assert pct[7] < absmax[7] * 0.9

    def test_percentile_100_matches_absmax(self, activation_batch):
        obs = RangeObserver(mode="percentile", percentile=100.0)
        obs.accumulate(activation_batch)
        result = obs.finalize()
        expected = activation_batch.abs().amax(dim=0)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_output_shape(self, activation_batch):
        obs = RangeObserver(mode="percentile")
        obs.accumulate(activation_batch)
        assert obs.finalize().shape == (64,)

    def test_percentile_sparse_channel_no_zero(self):
        # A channel that is near-zero for 99% of tokens must not return 0 —
        # that would drive the smooth factor s → 0, overflowing FP16 layernorm weights
        # (ln.weight / s > 65504) and producing NaN in INT8 fake-quant.
        # Truly zero channels (absmax = 0) are fine and allowed to stay 0.
        x = torch.zeros(512, 32)
        x[0, 5] = 10.0   # channel 5 fires exactly once (sparse but non-zero absmax)
        obs = RangeObserver(mode="percentile", percentile=99.0)
        obs.accumulate(x)
        result = obs.finalize()
        # Channel 5: 99th percentile = 0, but absmax = 10.0 → floor = 0.1
        assert result[5] > 0, (
            f"percentile returned 0 for a sparse-but-active channel: {result[5]}"
        )


class TestRangeObserverMSE:
    def test_mse_output_shape(self, activation_batch):
        obs = RangeObserver(mode="mse")
        obs.accumulate(activation_batch)
        result = obs.finalize()
        assert result.shape == (64,)

    def test_mse_below_absmax(self, activation_batch):
        obs_abs = RangeObserver(mode="absmax")
        obs_mse = RangeObserver(mode="mse")
        obs_abs.accumulate(activation_batch)
        obs_mse.accumulate(activation_batch)
        absmax = obs_abs.finalize()
        mse_range = obs_mse.finalize()
        # MSE-optimal clips the tail; result must be <= absmax
        assert (mse_range <= absmax + 1e-5).all()

    def test_mse_positive(self, activation_batch):
        obs = RangeObserver(mode="mse")
        obs.accumulate(activation_batch)
        result = obs.finalize()
        assert (result > 0).all()


class TestRangeObserverConfig:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown calibration mode"):
            RangeObserver(mode="bogus")

    def test_unknown_calibration_in_config(self):
        with pytest.raises(ValueError, match="Unknown calibration"):
            QuantConfig(scheme="smoothquant", calibration="bogus")

    def test_valid_configs(self):
        for mode in ("absmax", "percentile", "mse"):
            cfg = QuantConfig(scheme="smoothquant", calibration=mode)
            assert cfg.calibration == mode
