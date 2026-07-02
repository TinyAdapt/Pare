"""Quantized KV cache for memory-efficient long-context inference.

Implements the KIVI asymmetric quantization scheme:

    Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
    ICML 2024. arXiv:2402.02750.

Key insight: Keys and values have different outlier structures.
- Keys: consistent outlier *channels* across all tokens → per-channel scales
- Values: outlier *tokens* (attention sinks) across all channels → per-token scales

Both use unsigned asymmetric quantization (min-max, no symmetric constraint)
within sliding groups of G tokens along the sequence dimension.  The most
recent ``residual_length`` tokens are kept in FP16 to avoid quantizing the
frequently-accessed recent context.

Usage::

    from pare.kv_cache import QuantizedKVCache, KVCacheConfig

    cache = QuantizedKVCache(num_layers=32, config=KVCacheConfig(bits=4))

    # Inside your attention loop:
    k, v = cache.update(layer_idx, new_keys, new_values)
    # k, v are full-precision tensors ready for attention
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KVCacheConfig:
    """Configuration for the quantized KV cache.

    Args:
        bits:             Bit-width for compressed tokens (4 or 8).
        group_size:       Number of tokens per quantization group (G in the
                          paper; default 32).  Must divide the number of tokens
                          being compressed.
        residual_length:  Number of most-recent tokens to keep in FP16.
                          Everything older is compressed in G-token groups.
    """
    bits: int = 4
    group_size: int = 32
    residual_length: int = 128


# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

class _CompressedKV(NamedTuple):
    k_quant: Tensor   # [B, H, n_comp, D]  uint8
    k_scale: Tensor   # [B, H, n_groups, D] float32
    k_zero:  Tensor   # [B, H, n_groups, D] float32
    v_quant: Tensor   # [B, H, n_comp, D]  uint8
    v_scale: Tensor   # [B, H, n_comp, 1]  float32
    v_zero:  Tensor   # [B, H, n_comp, 1]  float32


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def _quantize_keys(k: Tensor, bits: int, group_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Per-channel unsigned asymmetric quantization of a key block.

    Args:
        k:          [B, H, n, D] — n must be divisible by group_size.
        bits:       Target bit-width (4 or 8).
        group_size: Tokens per group (G).

    Returns:
        k_quant [B, H, n, D] uint8,
        k_scale [B, H, n_groups, D] float32,
        k_zero  [B, H, n_groups, D] float32 (minimum value, acts as zero-point).
    """
    q_max = float((1 << bits) - 1)
    B, H, n, D = k.shape
    G = group_size
    n_groups = n // G

    k_g = k.reshape(B, H, n_groups, G, D)          # [B, H, ng, G, D]
    k_min = k_g.amin(dim=3, keepdim=True)           # [B, H, ng, 1, D]
    k_max = k_g.amax(dim=3, keepdim=True)
    scale = ((k_max - k_min) / q_max).clamp(min=1e-8)

    q = ((k_g - k_min) / scale).clamp(0, q_max).round().to(torch.uint8)

    return (
        q.reshape(B, H, n, D),
        scale.squeeze(3),                            # [B, H, ng, D]
        k_min.squeeze(3),                            # [B, H, ng, D]
    )


def _dequantize_keys(
    k_quant: Tensor, k_scale: Tensor, k_zero: Tensor, group_size: int
) -> Tensor:
    """Reconstruct fp32 keys from compressed representation."""
    B, H, n, D = k_quant.shape
    G = group_size
    n_groups = n // G

    q_g = k_quant.float().reshape(B, H, n_groups, G, D)
    scale_e = k_scale.unsqueeze(3)                  # [B, H, ng, 1, D]
    zero_e  = k_zero.unsqueeze(3)

    return (q_g * scale_e + zero_e).reshape(B, H, n, D)


def _quantize_values(v: Tensor, bits: int, group_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token unsigned asymmetric quantization of a value block.

    Args:
        v:          [B, H, n, D] — n must be divisible by group_size.
        bits:       Target bit-width (4 or 8).
        group_size: Tokens per group (G).

    Returns:
        v_quant [B, H, n, D] uint8,
        v_scale [B, H, n, 1] float32,
        v_zero  [B, H, n, 1] float32.
    """
    q_max = float((1 << bits) - 1)
    B, H, n, D = v.shape
    G = group_size
    n_groups = n // G

    v_g = v.reshape(B, H, n_groups, G, D)           # [B, H, ng, G, D]
    v_min = v_g.amin(dim=4, keepdim=True)            # [B, H, ng, G, 1]
    v_max = v_g.amax(dim=4, keepdim=True)
    scale = ((v_max - v_min) / q_max).clamp(min=1e-8)

    q = ((v_g - v_min) / scale).clamp(0, q_max).round().to(torch.uint8)

    return (
        q.reshape(B, H, n, D),
        scale.reshape(B, H, n, 1),                  # [B, H, n, 1]
        v_min.reshape(B, H, n, 1),                  # [B, H, n, 1]
    )


def _dequantize_values(v_quant: Tensor, v_scale: Tensor, v_zero: Tensor) -> Tensor:
    """Reconstruct fp32 values from compressed representation."""
    return v_quant.float() * v_scale + v_zero


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class QuantizedKVCache:
    """Per-layer quantized KV cache.

    Keys are compressed with per-channel scales (one scale per head_dim
    channel, shared across G tokens).  Values are compressed with per-token
    scales (one scale per token, shared across head_dim).  The most recent
    ``residual_length`` tokens are always kept in full precision.

    Args:
        num_layers:  Number of transformer layers.
        config:      ``KVCacheConfig`` (bits, group_size, residual_length).

    Note:
        Quantized weights are stored as uint8.  For 4-bit quantization this
        uses 2× more memory than packed INT4; packing is a future optimisation.

    Example::

        cache = QuantizedKVCache(num_layers=32)

        for step in range(max_new_tokens):
            new_k, new_v = model.attention_forward(...)   # [B, H, 1, D]
            k, v = cache.update(layer_idx, new_k, new_v)
            attn_output = scaled_dot_product_attention(q, k, v)
    """

    def __init__(
        self,
        num_layers: int,
        config: KVCacheConfig | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.config = config or KVCacheConfig()

        self._compressed: list[_CompressedKV | None] = [None] * num_layers
        self._residual_k: list[Tensor | None] = [None] * num_layers
        self._residual_v: list[Tensor | None] = [None] * num_layers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, layer_idx: int, new_k: Tensor, new_v: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Append new tokens and return the full (key, value) for attention.

        Args:
            layer_idx:  Which transformer layer (0-indexed).
            new_k:      New key tokens, shape [B, H, new_tokens, D].
            new_v:      New value tokens, shape [B, H, new_tokens, D].

        Returns:
            (full_k, full_v) — full sequence in the original dtype, ready for
            scaled dot-product attention.
        """
        orig_dtype = new_k.dtype
        cfg = self.config

        # Work in float32 for numerical precision
        rk = self._residual_k[layer_idx]
        rv = self._residual_v[layer_idx]
        nk = new_k.float()
        nv = new_v.float()

        rk = nk if rk is None else torch.cat([rk, nk], dim=2)
        rv = nv if rv is None else torch.cat([rv, nv], dim=2)

        # Quantize complete groups that overflow the residual budget
        r_len = rk.shape[2]
        overflow = max(0, r_len - cfg.residual_length)
        n_full_groups = overflow // cfg.group_size

        if n_full_groups > 0:
            n_to_compress = n_full_groups * cfg.group_size
            self._compressed[layer_idx] = self._append_compressed(
                layer_idx,
                rk[:, :, :n_to_compress, :],
                rv[:, :, :n_to_compress, :],
            )
            rk = rk[:, :, n_to_compress:, :]
            rv = rv[:, :, n_to_compress:, :]

        self._residual_k[layer_idx] = rk
        self._residual_v[layer_idx] = rv

        full_k = self._reconstruct(layer_idx, rk, is_key=True)
        full_v = self._reconstruct(layer_idx, rv, is_key=False)

        return full_k.to(orig_dtype), full_v.to(orig_dtype)

    def reset(self, layer_idx: int | None = None) -> None:
        """Clear cache state for one layer or all layers."""
        if layer_idx is None:
            self._compressed = [None] * self.num_layers
            self._residual_k = [None] * self.num_layers
            self._residual_v = [None] * self.num_layers
        else:
            self._compressed[layer_idx] = None
            self._residual_k[layer_idx] = None
            self._residual_v[layer_idx] = None

    def seq_len(self, layer_idx: int) -> int:
        """Total number of cached tokens (compressed + residual) for one layer."""
        c = self._compressed[layer_idx]
        rk = self._residual_k[layer_idx]
        n_comp = c.k_quant.shape[2] if c is not None else 0
        n_res  = rk.shape[2] if rk is not None else 0
        return n_comp + n_res

    def memory_bytes(self, layer_idx: int) -> dict[str, int]:
        """Approximate memory usage in bytes for one layer."""
        c  = self._compressed[layer_idx]
        rk = self._residual_k[layer_idx]
        rv = self._residual_v[layer_idx]

        def _nb(t: Tensor | None) -> int:
            return t.numel() * t.element_size() if t is not None else 0

        comp = 0
        if c is not None:
            comp = sum(_nb(t) for t in c)

        # Residual is float32 (4 bytes/element)
        res = _nb(rk) + _nb(rv)
        return {"compressed": comp, "residual": res, "total": comp + res}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_compressed(
        self, layer_idx: int, k_chunk: Tensor, v_chunk: Tensor
    ) -> _CompressedKV:
        G = self.config.group_size
        kq, ks, kz = _quantize_keys(k_chunk, self.config.bits, G)
        vq, vs, vz = _quantize_values(v_chunk, self.config.bits, G)
        new = _CompressedKV(kq, ks, kz, vq, vs, vz)

        prev = self._compressed[layer_idx]
        if prev is None:
            return new

        return _CompressedKV(
            k_quant=torch.cat([prev.k_quant, new.k_quant], dim=2),
            k_scale=torch.cat([prev.k_scale, new.k_scale], dim=2),
            k_zero =torch.cat([prev.k_zero,  new.k_zero],  dim=2),
            v_quant=torch.cat([prev.v_quant, new.v_quant], dim=2),
            v_scale=torch.cat([prev.v_scale, new.v_scale], dim=2),
            v_zero =torch.cat([prev.v_zero,  new.v_zero],  dim=2),
        )

    def _reconstruct(self, layer_idx: int, residual: Tensor, is_key: bool) -> Tensor:
        c = self._compressed[layer_idx]
        if c is None:
            return residual

        if is_key:
            dequant = _dequantize_keys(c.k_quant, c.k_scale, c.k_zero, self.config.group_size)
        else:
            dequant = _dequantize_values(c.v_quant, c.v_scale, c.v_zero)

        return torch.cat([dequant, residual], dim=2)
