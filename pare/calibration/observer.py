"""Per-channel activation magnitude observer for AWQ.

AWQ needs to know which input channels carry large activations so it can
protect those weight columns during quantization.  This module accumulates
the per-channel mean |x| online — just like HessianAccumulator but O(in)
instead of O(in²).

Usage::

    obs = ActivationObserver()
    hook = layer.register_forward_hook(
        lambda m, inp, out: obs.accumulate(inp[0])
    )
    # ... run calibration forward passes ...
    hook.remove()
    x_max = obs.finalize()   # [in_features]  mean |x| per input channel
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
