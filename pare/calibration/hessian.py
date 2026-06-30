"""Per-layer Hessian accumulation for GPTQ.

The GPTQ objective for one weight row w is:

    (w - q)^T H (w - q)

where H = 2/n * X X^T, X shaped [in_features, n_tokens].

We accumulate H incrementally over batches so we never store all
activations in memory — only the running [in_features, in_features] sum.
"""

from __future__ import annotations

import torch
from torch import Tensor


class HessianAccumulator:
    """Accumulates the per-layer Hessian H = 2/n * X X^T online.

    Usage::

        acc = HessianAccumulator()
        for batch_activations in ...:
            acc.accumulate(batch_activations)
        H = acc.finalize()   # [in_features, in_features]
    """

    def __init__(self) -> None:
        self.H: Tensor | None = None
        self.n_tokens: int = 0

    def accumulate(self, x: Tensor) -> None:
        """Add one batch of activations to the running sum.

        Args:
            x: Input activations, shape [batch, seq_len, in_features]
               or [batch, in_features].  Will be cast to float32.
        """
        x = x.detach().float()

        # Flatten batch and sequence dims → [n_tokens, in_features]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() != 2:
            raise ValueError(f"Expected 2-D or 3-D activation, got shape {tuple(x.shape)}")

        n = x.shape[0]  # number of tokens in this batch

        if self.H is None:
            in_features = x.shape[1]
            self.H = torch.zeros(
                in_features, in_features,
                device=x.device, dtype=torch.float32,
            )

        # H_unnorm += X_row^T @ X_row  (= X_col @ X_col^T  where X_col = x.T)
        self.H.addmm_(x.T, x)   # in-place fused multiply-add, no extra alloc
        self.n_tokens += n

    def finalize(self) -> Tensor:
        """Return the normalised Hessian H = 2/n * ΣX^TX.

        Raises RuntimeError if no samples have been accumulated.
        """
        if self.H is None or self.n_tokens == 0:
            raise RuntimeError("HessianAccumulator has no samples — did you forget to call accumulate()?")
        return self.H * (2.0 / self.n_tokens)

    def reset(self) -> None:
        self.H = None
        self.n_tokens = 0
