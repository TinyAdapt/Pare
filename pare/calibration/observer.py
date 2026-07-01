"""Activation observers for calibration.

``ActivationObserver``  — per-channel mean and max |x| (used by AWQ and
                          SmoothQuant with the default absmax mode).

``RangeObserver``       — per-channel range estimation with three modes:
                          absmax | percentile | mse.  Drop-in replacement
                          for the max_abs() path in SmoothQuant when a
                          better calibration strategy is configured.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ActivationObserver:
    """Online accumulator for per-channel activation statistics.

    Tracks both mean |x| (used by AWQ) and max |x| (used by SmoothQuant)
    in a single pass over calibration data.

    Accumulates over any number of batches of shape
    ``[batch, seq_len, in_features]`` or ``[batch, in_features]``.
    """

    def __init__(self) -> None:
        self._sum: Tensor | None = None
        self._max: Tensor | None = None
        self._n_tokens: int = 0

    def accumulate(self, x: Tensor) -> None:
        x = x.detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])   # [n_tokens, in_features]
        elif x.dim() != 2:
            raise ValueError(f"Expected 2-D or 3-D activation, got {tuple(x.shape)}")

        x_abs = x.abs()

        if self._sum is None:
            self._sum = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)
            self._max = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)

        self._sum.add_(x_abs.sum(dim=0))
        torch.maximum(self._max, x_abs.amax(dim=0), out=self._max)
        self._n_tokens += x.shape[0]

    def finalize(self) -> Tensor:
        """Return mean |x| per input channel, shape [in_features]. Used by AWQ."""
        if self._sum is None or self._n_tokens == 0:
            raise RuntimeError("ActivationObserver has no samples")
        return self._sum / self._n_tokens

    def max_abs(self) -> Tensor:
        """Return max |x| per input channel, shape [in_features]. Used by SmoothQuant."""
        if self._max is None or self._n_tokens == 0:
            raise RuntimeError("ActivationObserver has no samples")
        return self._max.clone()

    def reset(self) -> None:
        self._sum = None
        self._max = None
        self._n_tokens = 0


class RangeObserver:
    """Per-channel activation range estimator for calibration.

    Three modes:
        absmax:      Running maximum |x| per channel. Fast; can be skewed by
                     a single outlier token.
        percentile:  p-th percentile of |x| across all calibration tokens per
                     channel. Clips the top (100-p)% of values so that the
                     remaining range gets finer resolution.
        mse:         Searches over candidate clip ranges and picks the one that
                     minimises per-channel INT8 reconstruction MSE. Best quality;
                     slightly slower to calibrate.

    Args:
        mode:           ``"absmax"`` | ``"percentile"`` | ``"mse"``.
        percentile:     Percentile to use in ``"percentile"`` mode (default 99.99).
        max_samples:    Maximum number of token activations to buffer for
                        ``"percentile"`` and ``"mse"`` modes. Excess tokens are
                        discarded (first-come-first-served, then ignored).

    Usage::

        obs = RangeObserver(mode="percentile", percentile=99.99)
        hook = layer.register_forward_hook(
            lambda m, inp, out: obs.accumulate(inp[0])
        )
        # ... run calibration forward passes ...
        hook.remove()
        x_range = obs.finalize()   # [in_features]
    """

    def __init__(
        self,
        mode: str = "absmax",
        percentile: float = 99.99,
        max_samples: int = 1024,
    ) -> None:
        if mode not in {"absmax", "percentile", "mse"}:
            raise ValueError(f"Unknown calibration mode: {mode!r}")
        self.mode = mode
        self.percentile = percentile
        self.max_samples = max_samples

        self._max: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._n_buffered: int = 0

    def accumulate(self, x: Tensor) -> None:
        x = x.detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() != 2:
            raise ValueError(f"Expected 2-D or 3-D activation, got {tuple(x.shape)}")

        x_abs = x.abs()

        if self._max is None:
            self._max = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)

        torch.maximum(self._max, x_abs.amax(dim=0), out=self._max)

        if self.mode != "absmax" and self._n_buffered < self.max_samples:
            remaining = self.max_samples - self._n_buffered
            n_tokens = x_abs.shape[0]
            if n_tokens > remaining:
                idx = torch.randperm(n_tokens, device=x.device)[:remaining]
                x_abs = x_abs[idx]
            self._buffer.append(x_abs.cpu())
            self._n_buffered += x_abs.shape[0]

    def finalize(self) -> Tensor:
        if self._max is None:
            raise RuntimeError("RangeObserver has no samples")

        if self.mode == "absmax":
            return self._max.clone()

        samples = torch.cat(self._buffer, dim=0)  # [n_samples, in_features]

        if self.mode == "percentile":
            result = torch.quantile(samples, self.percentile / 100.0, dim=0)
            return result.to(self._max.device)

        return self._mse_optimal(samples).to(self._max.device)

    def _mse_optimal(self, samples: Tensor, n_candidates: int = 100) -> Tensor:
        q_max = 127.0  # INT8 signed symmetric range
        ch_max = samples.max(dim=0).values.clamp(min=1e-8)  # [n_channels]
        best_mse = torch.full_like(ch_max, float("inf"))
        best_range = ch_max.clone()

        for alpha in torch.linspace(0.5, 1.0, n_candidates):
            r = alpha * ch_max                             # [n_channels]
            scale = (r / q_max).clamp(min=1e-8)           # [n_channels]
            q = torch.clamp(torch.round(samples / scale), -q_max, q_max) * scale
            mse = ((samples - q) ** 2).mean(dim=0)        # [n_channels]
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_range = torch.where(better, r, best_range)

        return best_range

    def reset(self) -> None:
        self._max = None
        self._buffer = []
        self._n_buffered = 0
