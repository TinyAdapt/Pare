"""Calibration runner: register forward hooks, collect per-layer Hessians.

For GPTQ we need the input activations X to each Linear layer so we can
form H = 2/n * X X^T.  This module runs the model once on a small
calibration corpus and returns a per-layer Hessian dict.

Usage::

    runner = CalibrationRunner(quantizer)
    hessians = runner.collect(model, calibration_data, device="cuda")
    # hessians["model.layers.0.self_attn.q_proj"] → Tensor [in, in]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from pare.calibration.hessian import HessianAccumulator

if TYPE_CHECKING:
    from pare.schemes.base import BaseQuantizer


class CalibrationRunner:
    """Collects per-layer Hessians via forward hooks.

    Args:
        quantizer: Used only to call ``_should_quantize(name, module)``
                   so that hooks are registered on exactly the same set
                   of layers that will later be quantized.
    """

    def __init__(self, quantizer: "BaseQuantizer") -> None:
        self.quantizer = quantizer

    def collect(
        self,
        model: nn.Module,
        calibration_data: list[Tensor],
        device: str | torch.device = "cpu",
    ) -> dict[str, Tensor]:
        """Run the model on calibration data and return per-layer Hessians.

        Args:
            model:            The unquantized model.
            calibration_data: List of input_ids tensors, each shaped
                              [batch, seq_len].  Typically 128 sequences
                              of length 2048.
            device:           Device to run calibration on.

        Returns:
            Dict mapping fully-qualified layer name → Hessian tensor
            of shape [in_features, in_features].
        """
        if not calibration_data:
            raise ValueError("calibration_data is empty")

        model = model.to(device)
        model.eval()

        accumulators: dict[str, HessianAccumulator] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        # Register a hook on every layer we intend to quantize.
        for name, module in model.named_modules():
            if self.quantizer._should_quantize(name, module):
                acc = HessianAccumulator()
                accumulators[name] = acc
                hooks.append(
                    module.register_forward_hook(_make_hook(acc))
                )

        try:
            with torch.no_grad():
                for input_ids in calibration_data:
                    input_ids = input_ids.to(device)
                    # Causal LMs accept input_ids as the first positional arg.
                    model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return {name: acc.finalize() for name, acc in accumulators.items()}


def _make_hook(acc: HessianAccumulator):
    """Return a forward hook that feeds the layer input into acc."""
    def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
        # inputs[0] is the activation tensor entering nn.Linear:
        # shape [batch, seq_len, in_features] or [batch, in_features]
        acc.accumulate(inputs[0])
    return hook
