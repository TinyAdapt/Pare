"""Scale and zero-point computation for uniform quantization.

Three granularities:
  per_tensor  — one (scale, zero) for the whole tensor
  per_channel — one (scale, zero) per output channel (dim 0)
  per_group   — one (scale, zero) per contiguous group of `group_size`
                elements along the last dimension

For a weight matrix W of shape [out_features, in_features] the standard
convention is:
  per_channel  →  shape [out_features, 1]
  per_group    →  shape [out_features, in_features // group_size, 1]
                  (then squeezed / broadcast as needed during quant)
"""

from __future__ import annotations

import torch
from torch import Tensor

from pare.core.dtype import QuantDtype


def compute_scale(
    x: Tensor,
    dtype: QuantDtype,
    *,
    granularity: str = "per_group",
    group_size: int = 128,
    sym: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute scale and zero-point for quantizing ``x``.

    Args:
        x:           Input float tensor (usually a weight matrix).
        dtype:       Target quantization dtype (must be an integer type).
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        group_size:  Number of elements per group (only for ``"per_group"``).
        sym:         If True, use symmetric quantization (zero = 0 always).

    Returns:
        (scale, zero) as float tensors broadcastable against ``x``.
        zero is always an integer-valued float (not yet cast to int).
    """
    if dtype.is_float:
        raise ValueError(f"compute_scale is for integer dtypes; got {dtype}")

    qmin = float(dtype.qmin)
    qmax = float(dtype.qmax)

    if granularity == "per_tensor":
        return _scale_per_tensor(x, qmin, qmax, sym)
    elif granularity == "per_channel":
        return _scale_per_channel(x, qmin, qmax, sym)
    elif granularity == "per_group":
        return _scale_per_group(x, qmin, qmax, group_size, sym)
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------

def _minmax_to_scale_zero(
    x_min: Tensor,
    x_max: Tensor,
    qmin: float,
    qmax: float,
    sym: bool,
) -> tuple[Tensor, Tensor]:
    """Convert per-group/channel min & max into (scale, zero_point)."""
    if sym:
        # Symmetric: grid is centered at 0, no zero-point.
        # Use the larger absolute value so the grid covers both sides.
        abs_max = torch.maximum(x_min.abs(), x_max.abs())
        abs_max = abs_max.clamp(min=1e-8)
        # For unsigned INT4 (qmin=0, qmax=15) symmetric: shift to [-8, 7]
        # For signed INT8 (qmin=-128, qmax=127): use qmax directly
        half = max(abs(qmin), abs(qmax))
        scale = abs_max / half
        zero = torch.zeros_like(scale)
    else:
        # Asymmetric: stretch grid to exactly cover [x_min, x_max].
        x_min = x_min.clamp(max=0.0)   # ensure min ≤ 0 (include 0 in range)
        x_max = x_max.clamp(min=0.0)   # ensure max ≥ 0
        scale = (x_max - x_min) / (qmax - qmin)
        scale = scale.clamp(min=1e-8)
        zero = torch.round(-x_min / scale + qmin)
        zero = zero.clamp(qmin, qmax)
    return scale, zero


def _scale_per_tensor(
    x: Tensor, qmin: float, qmax: float, sym: bool
) -> tuple[Tensor, Tensor]:
    x_flat = x.reshape(-1)
    x_min = x_flat.min().unsqueeze(0)
    x_max = x_flat.max().unsqueeze(0)
    scale, zero = _minmax_to_scale_zero(x_min, x_max, qmin, qmax, sym)
    return scale.squeeze(), zero.squeeze()


def _scale_per_channel(
    x: Tensor, qmin: float, qmax: float, sym: bool
) -> tuple[Tensor, Tensor]:
    # x: [out_features, in_features] or [out_features, ...]
    # reduce over all dims except 0
    reduce_dims = list(range(1, x.dim()))
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    return _minmax_to_scale_zero(x_min, x_max, qmin, qmax, sym)


def _scale_per_group(
    x: Tensor, qmin: float, qmax: float, group_size: int, sym: bool
) -> tuple[Tensor, Tensor]:
    # x: [out_features, in_features]
    # Reshape to [out_features, n_groups, group_size] for per-group stats.
    if x.dim() != 2:
        raise ValueError(
            f"per_group expects a 2-D weight matrix, got shape {tuple(x.shape)}"
        )
    out_features, in_features = x.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )
    n_groups = in_features // group_size
    x_grouped = x.reshape(out_features, n_groups, group_size)

    x_min = x_grouped.amin(dim=-1, keepdim=True)   # [out, n_groups, 1]
    x_max = x_grouped.amax(dim=-1, keepdim=True)

    scale, zero = _minmax_to_scale_zero(x_min, x_max, qmin, qmax, sym)
    # scale/zero: [out_features, n_groups, 1] — broadcast over group_size dim
    return scale, zero
