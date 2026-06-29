"""Tests for quantization schemes (RTN) and QuantizedLinear."""

import pytest
import torch
import torch.nn as nn

from pare import QuantConfig, quantize
from pare.layers.linear import QuantizedLinear
from pare.schemes.rtn import RTNQuantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_linear():
    torch.manual_seed(0)
    layer = nn.Linear(128, 64, bias=True)
    return layer


@pytest.fixture
def rtn_int4_config():
    return QuantConfig(bits=4, scheme="rtn", granularity="per_group", group_size=64)


@pytest.fixture
def rtn_int8_config():
    return QuantConfig(bits=8, scheme="rtn", granularity="per_channel")


# ---------------------------------------------------------------------------
# QuantizedLinear
# ---------------------------------------------------------------------------

class TestQuantizedLinear:
    def test_from_linear_int4(self, small_linear, rtn_int4_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int4_config)
        assert isinstance(ql, QuantizedLinear)
        assert ql.in_features == 128
        assert ql.out_features == 64

    def test_packed_weight_shape_int4(self, small_linear, rtn_int4_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int4_config)
        # INT4: packed into uint8, last dim halved
        assert ql.packed_weight.shape == (64, 64)
        assert ql.packed_weight.dtype == torch.uint8

    def test_dequantize_shape(self, small_linear, rtn_int4_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int4_config)
        w = ql.dequantize()
        assert w.shape == small_linear.weight.shape

    def test_forward_shape(self, small_linear, rtn_int4_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int4_config)
        x = torch.randn(2, 128)
        out = ql(x)
        assert out.shape == (2, 64)

    def test_forward_close_to_fp16_int8(self, small_linear, rtn_int8_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int8_config)
        x = torch.randn(4, 128)
        fp_out = small_linear(x)
        q_out = ql(x)
        mae = (fp_out - q_out).abs().mean().item()
        assert mae < 0.05, f"INT8 forward MAE too high: {mae:.4f}"

    def test_bias_preserved(self, small_linear, rtn_int4_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int4_config)
        assert ql.bias is not None
        assert torch.allclose(ql.bias.data, small_linear.bias.data)

    def test_no_bias(self, rtn_int4_config):
        layer = nn.Linear(64, 32, bias=False)
        ql = QuantizedLinear.from_linear(layer, rtn_int4_config)
        assert ql.bias is None

    def test_int8_q_weight_stored_as_int8(self, small_linear, rtn_int8_config):
        ql = QuantizedLinear.from_linear(small_linear, rtn_int8_config)
        assert ql.q_weight is not None
        assert ql.q_weight.dtype == torch.int8


# ---------------------------------------------------------------------------
# RTNQuantizer
# ---------------------------------------------------------------------------

class TestRTNQuantizer:
    def test_quantize_layer_returns_quantized_linear(self, small_linear, rtn_int4_config):
        q = RTNQuantizer(rtn_int4_config)
        ql = q.quantize_layer(small_linear, "test.layer")
        assert isinstance(ql, QuantizedLinear)

    def test_should_quantize_excludes_lm_head(self, rtn_int4_config):
        q = RTNQuantizer(rtn_int4_config)
        layer = nn.Linear(64, 32)
        assert not q._should_quantize("lm_head", layer)
        assert q._should_quantize("model.layers.0.mlp.fc1", layer)

    def test_should_quantize_non_linear(self, rtn_int4_config):
        q = RTNQuantizer(rtn_int4_config)
        norm = nn.LayerNorm(64)
        assert not q._should_quantize("model.norm", norm)


# ---------------------------------------------------------------------------
# End-to-end: quantize a tiny model
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.lm_head = nn.Linear(64, 1000)

    def forward(self, x):
        return self.lm_head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class TestEndToEnd:
    def test_quantize_replaces_layers(self):
        model = TinyModel()
        config = QuantConfig(bits=4, scheme="rtn", group_size=64)
        quantize(model, config)
        assert isinstance(model.fc1, QuantizedLinear)
        assert isinstance(model.fc2, QuantizedLinear)
        # lm_head should be excluded
        assert isinstance(model.lm_head, nn.Linear)

    def test_forward_after_quantization(self):
        torch.manual_seed(0)
        model = TinyModel()
        config = QuantConfig(bits=8, scheme="rtn", granularity="per_channel")
        quantize(model, config)
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 1000)
        assert not torch.isnan(out).any()

    def test_quantized_model_smaller_memory_int4(self):
        model = TinyModel()
        fp16_params = sum(p.numel() for p in model.parameters())

        config = QuantConfig(bits=4, scheme="rtn", group_size=64)
        quantize(model, config)

        # QuantizedLinear stores packed uint8 weights — check fc1 specifically
        assert model.fc1.packed_weight is not None
        # Packed weight uses half the elements of the original
        assert model.fc1.packed_weight.numel() == 64 * 128 // 2
