"""Tests for mixed-precision sensitivity scoring."""

import torch
import torch.nn as nn
import pytest

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.sensitivity import score_layers


def _tiny_model(n_layers: int = 3, in_f: int = 32, out_f: int = 16):
    """A simple Sequential of Linear layers for testing."""
    layers = []
    for i in range(n_layers):
        layers.append((f"layer{i}", nn.Linear(in_f, out_f, bias=False)))
        in_f, out_f = out_f, in_f  # alternate dims so they're distinguishable
    return nn.Sequential(nn.ModuleDict(layers))


def _calib_data(batch: int = 4, seq: int = 8, in_f: int = 32):
    torch.manual_seed(0)
    return [torch.randn(batch, seq, in_f) for _ in range(4)]


# ---------------------------------------------------------------------------
# score_layers
# ---------------------------------------------------------------------------

class TestScoreLayers:
    def test_scores_are_non_negative(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(32, 16, bias=False))
        calib = [torch.randn(2, 8, 32) for _ in range(2)]
        scores = score_layers(model, calib, bits=4, granularity="per_channel",
                              group_size=32, device="cpu")
        for name, err in scores.items():
            assert err >= 0.0, f"Negative score for {name}: {err}"

    def test_all_linear_layers_scored(self):
        torch.manual_seed(1)
        model = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
        )
        calib = [torch.randn(2, 4, 32) for _ in range(2)]
        scores = score_layers(model, calib, bits=4, granularity="per_channel",
                              group_size=16, device="cpu")
        assert len(scores) == 2, f"Expected 2 scored layers, got {len(scores)}"

    def test_high_error_layer_detected(self):
        """A layer with random weights scores higher than one with constant weights.

        Constant weights (all-ones) quantize exactly → delta_W = 0 → error = 0.
        Random weights have nonzero rounding error → error > 0.
        The scorer must correctly rank them.
        """
        torch.manual_seed(42)

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.easy = nn.Linear(64, 32, bias=False)  # constant → perfect quant
                self.hard = nn.Linear(32, 16, bias=False)  # random → nonzero error

            def forward(self, x):
                return self.hard(self.easy(x))

        model = TwoLayer()
        with torch.no_grad():
            model.easy.weight.data.fill_(1.0)   # all-same → 0 quantization error
            model.hard.weight.data = torch.randn(16, 32)  # normal distribution

        calib = [torch.randn(4, 8, 64) for _ in range(4)]
        scores = score_layers(model, calib, bits=4, granularity="per_channel",
                              group_size=32, device="cpu")

        assert "easy" in scores and "hard" in scores
        assert scores["hard"] > scores["easy"], (
            f"Random-weight layer ({scores['hard']:.6f}) should score "
            f"higher than constant-weight ({scores['easy']:.6f})"
        )

    def test_per_group_granularity(self):
        """score_layers should handle per_group without crashing."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(128, 64, bias=False))
        calib = [torch.randn(2, 4, 128) for _ in range(2)]
        scores = score_layers(model, calib, bits=4, granularity="per_group",
                              group_size=64, device="cpu")
        assert len(scores) == 1
        assert list(scores.values())[0] >= 0.0


# ---------------------------------------------------------------------------
# QuantConfig sensitivity fields
# ---------------------------------------------------------------------------

class TestQuantConfigSensitivity:
    def test_defaults(self):
        cfg = QuantConfig(bits=4, scheme="rtn")
        assert cfg.sensitive_bits is None
        assert cfg.sensitivity_threshold == 0.05

    def test_sensitive_bits_set(self):
        cfg = QuantConfig(bits=4, scheme="rtn", sensitive_bits=8, sensitivity_threshold=0.03)
        assert cfg.sensitive_bits == 8
        assert cfg.sensitivity_threshold == 0.03


# ---------------------------------------------------------------------------
# _config_for_layer
# ---------------------------------------------------------------------------

class TestConfigForLayer:
    def test_no_override(self):
        from pare.schemes.rtn import RTNQuantizer
        cfg = QuantConfig(bits=4, scheme="rtn")
        q = RTNQuantizer(cfg)
        assert q._config_for_layer("model.layer0") is cfg

    def test_override_changes_bits(self):
        from pare.schemes.rtn import RTNQuantizer
        cfg = QuantConfig(bits=4, scheme="rtn")
        q = RTNQuantizer(cfg, layer_bits_override={"model.layer0": 8})
        effective = q._config_for_layer("model.layer0")
        assert effective.bits == 8
        assert effective.effective_dtype == QuantDtype.INT8

    def test_non_overridden_layer_unchanged(self):
        from pare.schemes.rtn import RTNQuantizer
        cfg = QuantConfig(bits=4, scheme="rtn")
        q = RTNQuantizer(cfg, layer_bits_override={"model.layer0": 8})
        effective = q._config_for_layer("model.layer1")
        assert effective.bits == 4
        assert effective.effective_dtype == QuantDtype.INT4


# ---------------------------------------------------------------------------
# End-to-end: quantize() with sensitive_bits
# ---------------------------------------------------------------------------

class TestMixedPrecisionQuantize:
    def test_sensitive_layers_get_higher_bits(self):
        """End-to-end: layers above threshold get sensitive_bits, others get bits."""
        from pare import quantize, QuantConfig
        from pare.layers.linear import QuantizedLinear

        torch.manual_seed(0)

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.easy = nn.Linear(64, 32, bias=False)
                self.hard = nn.Linear(32, 16, bias=False)

            def forward(self, x):
                return self.hard(self.easy(x))

        model = TwoLayer()
        # Make 'hard' layer very sensitive by inserting outliers
        with torch.no_grad():
            model.hard.weight.data = torch.randn(16, 32) * 0.01
            model.hard.weight.data[:, 0] = 200.0

        calib = [torch.randn(2, 4, 64) for _ in range(4)]
        cfg = QuantConfig(
            bits=4, scheme="rtn", granularity="per_channel",
            sensitive_bits=8, sensitivity_threshold=0.02,
        )
        quantize(model, cfg, calibration_data=calib, device="cpu")

        # At least one layer should have been upgraded to 8-bit
        layer_bits = {
            name: m.config.bits
            for name, m in model.named_modules()
            if isinstance(m, QuantizedLinear)
        }
        assert any(b == 8 for b in layer_bits.values()), (
            f"No layer was upgraded to 8-bit. Layer bits: {layer_bits}"
        )

    def test_no_sensitivity_all_same_bits(self):
        """Without sensitive_bits, all layers use the same bit-width."""
        from pare import quantize, QuantConfig
        from pare.layers.linear import QuantizedLinear

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))
        cfg = QuantConfig(bits=4, scheme="rtn", granularity="per_channel")
        quantize(model, cfg)

        layer_bits = [m.config.bits for m in model.modules() if isinstance(m, QuantizedLinear)]
        assert all(b == 4 for b in layer_bits)
