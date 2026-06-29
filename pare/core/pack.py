"""Bit-packing utilities for sub-byte quantized weights.

PyTorch has no native int4 dtype, so we pack two 4-bit values into each
uint8 byte.  The convention used throughout Pare:

  low  nibble  (bits 0–3) ← element at even index
  high nibble  (bits 4–7) ← element at odd  index

Packing always operates on the last dimension of the tensor (the
``in_features`` axis of a weight matrix).  The last dimension must be
even for INT4.

For 2-bit and 3-bit packing the approach generalises: pack the maximum
number of values per byte (4 values for 2-bit, 2 values for 3-bit with
one bit wasted).  Helpers are provided for 4-bit and 8-bit;
lower bit-widths may be added in future versions.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# INT4 (4-bit)
# ---------------------------------------------------------------------------

def pack_int4(q: Tensor) -> Tensor:
    """Pack a tensor of INT4 values (stored as int32) into uint8.

    Args:
        q: Integer tensor with values in [0, 15] (unsigned INT4).
           Shape: ``(..., in_features)`` where ``in_features`` is even.

    Returns:
        uint8 tensor of shape ``(..., in_features // 2)``.
        Even indices of the last dim go into low nibbles,
        odd indices into high nibbles.
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even for INT4 packing, got {q.shape[-1]}"
        )
    q = q.to(torch.int32)
    low = q[..., 0::2] & 0xF        # even elements → low nibble
    high = (q[..., 1::2] & 0xF) << 4  # odd elements → high nibble
    return (low | high).to(torch.uint8)


def unpack_int4(packed: Tensor) -> Tensor:
    """Unpack uint8 tensor back to INT4 values (unsigned, stored as int32).

    Args:
        packed: uint8 tensor of shape ``(..., in_features // 2)``.

    Returns:
        int32 tensor of shape ``(..., in_features)`` with values in [0, 15].
    """
    packed = packed.to(torch.int32)
    low = packed & 0xF                # low nibble → even positions
    high = (packed >> 4) & 0xF       # high nibble → odd positions

    # Interleave: [low_0, high_0, low_1, high_1, ...]
    # Stack along a new last dim and flatten the last two dims.
    out = torch.stack([low, high], dim=-1)       # (..., in//2, 2)
    return out.reshape(*packed.shape[:-1], -1)   # (..., in)


def pack_int4_signed(q: Tensor) -> Tensor:
    """Pack signed INT4 values (range -8..7) into uint8.

    Shifts values by 8 so they fall in [0, 15] before packing.
    Use ``unpack_int4_signed`` to recover the original signed values.
    """
    return pack_int4(q + 8)


def unpack_int4_signed(packed: Tensor) -> Tensor:
    """Unpack uint8 → signed int32 in range [-8, 7]."""
    return unpack_int4(packed) - 8


# ---------------------------------------------------------------------------
# Kernel-layout repacking (for Triton matmul_w4a16)
# ---------------------------------------------------------------------------

def repack_int4_for_kernel(packed: Tensor, group_size: int = 128) -> Tensor:
    """Repack storage-format INT4 weights into Triton kernel layout.

    The standard ``pack_int4`` storage format pairs *adjacent* columns:
        byte j → (q[n, 2j], q[n, 2j+1])

    The Triton kernel needs *first-half / second-half* pairs within each
    ``group_size``-wide tile so that both X halves can be loaded contiguously:
        byte j → (q[n, k + j_local], q[n, k + j_local + group_size//2])
    where k = tile start, j_local = byte offset within tile.

    This repacking is done once offline; it does not change the number of bytes.

    Args:
        packed:     Storage-layout packed weights, shape [N, K//2] uint8.
        group_size: K-tile width used by the kernel (must equal the autotune
                    BLOCK_K, typically equal to the quantization group_size).

    Returns:
        Kernel-layout packed weights, same shape [N, K//2] uint8.
    """
    N, K_half = packed.shape
    K = K_half * 2
    device = packed.device

    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    half = group_size // 2
    n_tiles = K // group_size

    # Unpack to [N, K] int32
    p = packed.to(torch.int32)
    q = torch.empty(N, K, dtype=torch.int32, device=device)
    q[:, 0::2] = p & 0xF           # even cols: low nibbles
    q[:, 1::2] = (p >> 4) & 0xF   # odd cols:  high nibbles

    # Repack: within each group_size-wide tile, lo = first half, hi = second half
    repacked = torch.empty(N, K_half, dtype=torch.uint8, device=device)
    for t in range(n_tiles):
        k_start = t * group_size
        b_start = t * half          # byte offset in repacked
        lo = q[:, k_start : k_start + half].to(torch.uint8)
        hi = q[:, k_start + half : k_start + group_size].to(torch.uint8)
        repacked[:, b_start : b_start + half] = lo | (hi << 4)

    return repacked


# ---------------------------------------------------------------------------
# Utility: storage size report
# ---------------------------------------------------------------------------

def packed_size_bytes(shape: tuple[int, ...], bits: int) -> int:
    """Return the number of bytes needed to store a tensor after packing."""
    n_elements = 1
    for s in shape:
        n_elements *= s
    return (n_elements * bits + 7) // 8
