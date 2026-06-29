"""Core quantize/dequantize operations on float tensors.

Both functions handle the three granularities by detecting the shape of
``scale``:
  scalar            → per-tensor
  [out, 1]          → per-channel
  [out, n_groups, 1] → per-group  (x must be 2-D; group_size is inferred)

NF4 (normal float 4-bit) uses a non-uniform codebook and separate functions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pare.core.dtype import QuantDtype

# ---------------------------------------------------------------------------
# NF4 codebook (Dettmers et al. 2023, QLoRA, Table 1)
# 16 quantile values of a standard normal distribution, scaled to [-1, 1].
# Index 0 = most negative, index 7 = 0.0, index 15 = 1.0.
# ---------------------------------------------------------------------------
_NF4_CODEBOOK: list[float] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
# Register as a buffer-friendly tensor (float32); index is the 4-bit integer.
_NF4_TABLE = torch.tensor(_NF4_CODEBOOK, dtype=torch.float32)


def quantize_tensor(
    x: Tensor,
    scale: Tensor,
    zero: Tensor,
    dtype: QuantDtype,
) -> Tensor:
    """Map float ``x`` to integer grid defined by (scale, zero, dtype).

    Formula:  q = clamp( round(x / scale + zero), qmin, qmax )

    Returns an integer tensor with values in [dtype.qmin, dtype.qmax],
    stored as torch.int32 (PyTorch has no int4 dtype — use pack_int4 for
    storage-efficient representation).

    For NF4, use ``quantize_nf4`` / ``dequantize_nf4`` directly.
    """
    if dtype.is_float:
        raise ValueError(
            f"quantize_tensor is for integer dtypes; got {dtype}. "
            "For NF4 use quantize_nf4(); for FP8 cast with x.to(torch.float8_e4m3fn)."
        )

    x, scale, zero = _align_shapes(x, scale, zero)

    q = torch.clamp(
        torch.round(x / scale + zero),
        min=dtype.qmin,
        max=dtype.qmax,
    )
    return q.to(torch.int32)


def dequantize_tensor(
    q: Tensor,
    scale: Tensor,
    zero: Tensor,
) -> Tensor:
    """Reconstruct float approximation from quantized integer tensor.

    Formula:  x̂ = (q - zero) * scale

    This is the exact inverse of ``quantize_tensor`` up to the rounding
    that was applied during quantization.
    """
    q_float, scale, zero = _align_shapes(q.float(), scale, zero)
    return (q_float - zero) * scale


# ---------------------------------------------------------------------------
# NF4 quantization (non-uniform codebook, nearest-neighbour lookup)
# ---------------------------------------------------------------------------

def quantize_nf4(x: Tensor, scale: Tensor) -> Tensor:
    """Quantize ``x`` to 4-bit NF4 indices via nearest-codebook lookup.

    Args:
        x:     Weight tensor [out, in] in float32.  Values are expected to
               lie in [-1, 1] after dividing by ``scale``; the caller is
               responsible for computing ``scale = x.abs().max(dim=-1)``.
        scale: Per-row absmax scale [out, 1].

    Returns:
        Integer index tensor [out, in] with values in [0, 15] (torch.int32).
        Pack with ``pack_int4`` for storage-efficient uint8 representation.
    """
    table = _NF4_TABLE.to(x.device)
    x_norm = x / scale.clamp(min=1e-8)             # normalise to [-1, 1]
    # Nearest-neighbour: find index of closest codebook value per element.
    diff = (x_norm.unsqueeze(-1) - table).abs()     # [..., 16]
    indices = diff.argmin(dim=-1)                   # [out, in]
    return indices.to(torch.int32)


def dequantize_nf4(indices: Tensor, scale: Tensor) -> Tensor:
    """Reconstruct float weights from NF4 indices.

    Args:
        indices: Integer tensor [out, in] with values in [0, 15].
        scale:   Per-row absmax scale [out, 1] used during quantization.

    Returns:
        Reconstructed weight tensor [out, in] in float32.
    """
    table = _NF4_TABLE.to(indices.device)
    x_norm = table[indices.long()]                  # [out, in] — codebook lookup
    return x_norm * scale


# ---------------------------------------------------------------------------
# FP8 quantization (weight-only, W8A16 style)
# ---------------------------------------------------------------------------

_FP8_E4M3_MAX: float = 448.0    # max finite value for torch.float8_e4m3fn
_FP8_E5M2_MAX: float = 57344.0  # max finite value for torch.float8_e5m2


def quantize_fp8(x: Tensor, scale: Tensor, fp8_dtype: "QuantDtype") -> Tensor:
    """Quantize ``x`` to FP8 via a saturating cast after scale normalisation.

    Args:
        x:         Weight tensor [out, in] in float32.
        scale:     Per-row absmax scale [out, 1] = absmax(W, dim=-1) / fp8_max.
                   The normalisation maps W into [-fp8_max, fp8_max] before casting.
        fp8_dtype: ``QuantDtype.FP8_E4M3`` or ``QuantDtype.FP8_E5M2``.

    Returns:
        Tensor in the requested float8 dtype with values in [-fp8_max, fp8_max].
    """
    from pare.core.dtype import QuantDtype
    if fp8_dtype == QuantDtype.FP8_E4M3:
        torch_dtype = torch.float8_e4m3fn
    elif fp8_dtype == QuantDtype.FP8_E5M2:
        torch_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"quantize_fp8 requires FP8_E4M3 or FP8_E5M2, got {fp8_dtype}")
    x_norm = x / scale.clamp(min=1e-8)
    return x_norm.to(torch_dtype)


def dequantize_fp8(x_fp8: Tensor, scale: Tensor) -> Tensor:
    """Reconstruct float32 weights from an FP8 tensor and its per-row scale.

    Args:
        x_fp8: FP8 weight tensor [out, in] (float8_e4m3fn or float8_e5m2).
        scale: Per-row scale [out, 1] used during quantization.

    Returns:
        Reconstructed float32 tensor [out, in].
    """
    return x_fp8.float() * scale


def quantization_error(x: Tensor, x_hat: Tensor) -> dict[str, Tensor]:
    """Compute common quantization error metrics between x and its reconstruction."""
    err = x - x_hat
    return {
        "mae": err.abs().mean(),
        "rmse": err.pow(2).mean().sqrt(),
        "max_err": err.abs().max(),
        "snr_db": 10 * torch.log10(x.pow(2).mean() / err.pow(2).mean().clamp(1e-12)),
    }


# ---------------------------------------------------------------------------
# Shape alignment helper
# ---------------------------------------------------------------------------

def _align_shapes(
    x: Tensor, scale: Tensor, zero: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Reshape x (and broadcast scale/zero) for the three granularities.

    Per-group case: scale is [out, n_groups, 1], x is [out, in_features].
    We reshape x to [out, n_groups, group_size] so the broadcast works.
    """
    if scale.dim() == 3:
        # per_group
        out, n_groups, _ = scale.shape
        group_size = x.shape[-1] // n_groups
        x = x.reshape(out, n_groups, group_size)
        # scale/zero already [out, n_groups, 1] → broadcasts over group_size
    elif scale.dim() == 2:
        # per_channel: scale is [out, 1], x is [out, in]
        pass  # broadcast works directly
    else:
        # per_tensor: scalar or 0-d
        pass

    return x, scale, zero
