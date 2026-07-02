"""Tests for QuantizedKVCache."""

import pytest
import torch

from pare.kv_cache import (
    KVCacheConfig,
    QuantizedKVCache,
    _dequantize_keys,
    _dequantize_values,
    _quantize_keys,
    _quantize_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, H, G, D = 1, 8, 32, 128  # batch, heads, group_size, head_dim


@pytest.fixture
def key_block():
    torch.manual_seed(0)
    k = torch.randn(B, H, G, D)
    k[:, :, :, 7] *= 10.0   # outlier channel
    return k


@pytest.fixture
def val_block():
    torch.manual_seed(1)
    v = torch.randn(B, H, G, D)
    v[:, :, 3, :] *= 5.0    # outlier token
    return v


# ---------------------------------------------------------------------------
# Primitive: keys
# ---------------------------------------------------------------------------

class TestQuantizeKeys:
    def test_output_shapes(self, key_block):
        kq, ks, kz = _quantize_keys(key_block, bits=4, group_size=G)
        assert kq.shape == (B, H, G, D)
        assert ks.shape == (B, H, 1, D)
        assert kz.shape == (B, H, 1, D)

    def test_quant_dtype(self, key_block):
        kq, _, _ = _quantize_keys(key_block, bits=4, group_size=G)
        assert kq.dtype == torch.uint8

    def test_values_in_4bit_range(self, key_block):
        kq, _, _ = _quantize_keys(key_block, bits=4, group_size=G)
        assert kq.min() >= 0
        assert kq.max() <= 15

    def test_values_in_8bit_range(self, key_block):
        kq, _, _ = _quantize_keys(key_block, bits=8, group_size=G)
        assert kq.min() >= 0
        assert kq.max() <= 255

    def test_roundtrip_error_below_rtn(self, key_block):
        kq, ks, kz = _quantize_keys(key_block, bits=4, group_size=G)
        rec = _dequantize_keys(kq, ks, kz, G)
        mse = ((key_block - rec) ** 2).mean().item()
        # RTN INT4 per-channel on this data: baseline rough bound
        assert mse < 0.5

    def test_multiple_groups(self):
        torch.manual_seed(2)
        k = torch.randn(B, H, G * 3, D)
        kq, ks, kz = _quantize_keys(k, bits=4, group_size=G)
        assert kq.shape == (B, H, G * 3, D)
        assert ks.shape == (B, H, 3, D)
        rec = _dequantize_keys(kq, ks, kz, G)
        assert rec.shape == (B, H, G * 3, D)


# ---------------------------------------------------------------------------
# Primitive: values
# ---------------------------------------------------------------------------

class TestQuantizeValues:
    def test_output_shapes(self, val_block):
        vq, vs, vz = _quantize_values(val_block, bits=4, group_size=G)
        assert vq.shape == (B, H, G, D)
        assert vs.shape == (B, H, G, 1)
        assert vz.shape == (B, H, G, 1)

    def test_quant_dtype(self, val_block):
        vq, _, _ = _quantize_values(val_block, bits=4, group_size=G)
        assert vq.dtype == torch.uint8

    def test_values_in_4bit_range(self, val_block):
        vq, _, _ = _quantize_values(val_block, bits=4, group_size=G)
        assert vq.min() >= 0
        assert vq.max() <= 15

    def test_roundtrip_error(self, val_block):
        vq, vs, vz = _quantize_values(val_block, bits=4, group_size=G)
        rec = _dequantize_values(vq, vs, vz)
        mse = ((val_block - rec) ** 2).mean().item()
        assert mse < 0.5

    def test_per_token_scale_different_per_token(self, val_block):
        # Outlier token (3) should have a larger scale than others
        _, vs, _ = _quantize_values(val_block, bits=4, group_size=G)
        # vs: [B, H, G, 1]
        assert vs[0, 0, 3, 0] > vs[0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Per-channel vs per-token MSE comparison
# ---------------------------------------------------------------------------

class TestAsymmetry:
    def test_key_per_channel_beats_per_token(self, key_block):
        """Keys: per-channel should give lower MSE than per-token."""
        # Per-channel (correct for keys)
        kq, ks, kz = _quantize_keys(key_block, bits=4, group_size=G)
        rec_pc = _dequantize_keys(kq, ks, kz, G)
        mse_pc = ((key_block - rec_pc) ** 2).mean().item()

        # Per-token (wrong for keys: one scale per token across all channels)
        # Simulate per-token by quantizing along head_dim axis
        vq, vs, vz = _quantize_values(key_block, bits=4, group_size=G)
        rec_pt = _dequantize_values(vq, vs, vz)
        mse_pt = ((key_block - rec_pt) ** 2).mean().item()

        assert mse_pc < mse_pt

    def test_value_per_token_beats_per_channel(self):
        """Values: per-token should give lower MSE than per-channel.

        Use a strong outlier token (20×) so the per-channel scale is forced
        high across all tokens, whereas per-token isolates the outlier.
        """
        torch.manual_seed(1)
        v = torch.randn(B, H, G, D)
        v[:, :, 3, :] *= 20.0   # one attention-sink-style outlier token

        vq, vs, vz = _quantize_values(v, bits=4, group_size=G)
        rec_pt = _dequantize_values(vq, vs, vz)
        mse_pt = ((v - rec_pt) ** 2).mean().item()

        kq, ks, kz = _quantize_keys(v, bits=4, group_size=G)
        rec_pc = _dequantize_keys(kq, ks, kz, G)
        mse_pc = ((v - rec_pc) ** 2).mean().item()

        assert mse_pt < mse_pc


# ---------------------------------------------------------------------------
# QuantizedKVCache
# ---------------------------------------------------------------------------

class TestQuantizedKVCache:
    def _make_tokens(self, n, seed=0):
        torch.manual_seed(seed)
        k = torch.randn(B, H, n, D, dtype=torch.float16)
        v = torch.randn(B, H, n, D, dtype=torch.float16)
        return k, v

    def test_residual_only_before_overflow(self):
        cfg = KVCacheConfig(bits=4, group_size=32, residual_length=128)
        cache = QuantizedKVCache(num_layers=1, config=cfg)
        k, v = self._make_tokens(64)
        fk, fv = cache.update(0, k, v)
        # Still inside residual — no compressed state
        assert cache._compressed[0] is None
        assert fk.shape == (B, H, 64, D)
        assert fv.shape == (B, H, 64, D)

    def test_compression_triggers_after_overflow(self):
        cfg = KVCacheConfig(bits=4, group_size=32, residual_length=64)
        cache = QuantizedKVCache(num_layers=1, config=cfg)
        k, v = self._make_tokens(128)
        cache.update(0, k, v)
        # 128 tokens; residual_length=64 → 64 tokens compressed in 2 groups
        assert cache._compressed[0] is not None
        assert cache._compressed[0].k_quant.shape[2] == 64
        assert cache._residual_k[0].shape[2] == 64

    def test_output_shape_matches_total_tokens(self):
        cfg = KVCacheConfig(bits=4, group_size=32, residual_length=64)
        cache = QuantizedKVCache(num_layers=1, config=cfg)
        # Prefill 96 tokens
        k, v = self._make_tokens(96)
        fk, fv = cache.update(0, k, v)
        assert fk.shape == (B, H, 96, D)
        assert fv.shape == (B, H, 96, D)
        # Generate 3 more tokens
        for i in range(3):
            k1, v1 = self._make_tokens(1, seed=i + 10)
            fk, fv = cache.update(0, k1, v1)
        assert fk.shape == (B, H, 99, D)

    def test_output_dtype_preserved(self):
        cache = QuantizedKVCache(num_layers=1)
        k, v = self._make_tokens(200)   # triggers compression
        fk, fv = cache.update(0, k, v)
        assert fk.dtype == torch.float16
        assert fv.dtype == torch.float16

    def test_seq_len_tracking(self):
        cfg = KVCacheConfig(bits=4, group_size=32, residual_length=64)
        cache = QuantizedKVCache(num_layers=1, config=cfg)
        k, v = self._make_tokens(100)
        cache.update(0, k, v)
        assert cache.seq_len(0) == 100

    def test_reconstruction_close_to_fp16(self):
        cfg = KVCacheConfig(bits=8, group_size=32, residual_length=0)
        cache = QuantizedKVCache(num_layers=1, config=cfg)
        k, v = self._make_tokens(64)
        fk, fv = cache.update(0, k, v)
        # INT8 with residual_length=0 → all tokens compressed
        # Reconstruction error should be small
        mse_k = ((fk.float() - k.float()) ** 2).mean().item()
        mse_v = ((fv.float() - v.float()) ** 2).mean().item()
        assert mse_k < 0.01
        assert mse_v < 0.01

    def test_reset_single_layer(self):
        cache = QuantizedKVCache(num_layers=2)
        k, v = self._make_tokens(256)
        cache.update(0, k, v)
        cache.update(1, k, v)
        cache.reset(layer_idx=0)
        assert cache._compressed[0] is None
        assert cache._residual_k[0] is None
        assert cache._residual_k[1] is not None

    def test_reset_all_layers(self):
        cache = QuantizedKVCache(num_layers=2)
        k, v = self._make_tokens(256)
        cache.update(0, k, v)
        cache.reset()
        assert all(c is None for c in cache._compressed)
        assert all(r is None for r in cache._residual_k)

    def test_memory_bytes(self):
        cfg = KVCacheConfig(bits=4, group_size=32, residual_length=64)
        cache = QuantizedKVCache(num_layers=1, config=cfg)
        k, v = self._make_tokens(128)
        cache.update(0, k, v)
        mem = cache.memory_bytes(0)
        assert mem["total"] > 0
        assert mem["compressed"] > 0
        assert mem["residual"] > 0
        assert mem["compressed"] + mem["residual"] == mem["total"]

    def test_incremental_matches_batch(self):
        """Token-by-token generation should produce same keys as one-shot prefill."""
        cfg = KVCacheConfig(bits=4, group_size=32, residual_length=64)

        # Batch: feed all 128 tokens at once
        cache_batch = QuantizedKVCache(num_layers=1, config=cfg)
        k_all, v_all = self._make_tokens(128, seed=99)
        fk_batch, fv_batch = cache_batch.update(0, k_all, v_all)

        # Incremental: feed 32 tokens at a time
        cache_inc = QuantizedKVCache(num_layers=1, config=cfg)
        for i in range(4):
            k_i = k_all[:, :, i * 32:(i + 1) * 32, :]
            v_i = v_all[:, :, i * 32:(i + 1) * 32, :]
            fk_inc, fv_inc = cache_inc.update(0, k_i, v_i)

        # Final outputs should match (same tokens, same quantization)
        assert torch.allclose(fk_batch.float(), fk_inc.float(), atol=1e-4)
        assert torch.allclose(fv_batch.float(), fv_inc.float(), atol=1e-4)
