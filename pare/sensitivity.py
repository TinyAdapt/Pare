"""Per-layer sensitivity scoring for mixed-precision quantization.

Sensitivity metric for a linear layer (weight W, calibration inputs X):

    e = ‖(W − Ŵ) Xᵀ‖_F  /  ‖W Xᵀ‖_F

where Ŵ is the RTN-quantized reconstruction of W.  RTN is used as a fast
proxy; it tracks GPTQ/AWQ error well enough to identify outlier layers.

The activation weighting (multiplication by Xᵀ) matters: a column of W that
multiplies a near-zero input barely affects the layer's output, so the error
there is less important than the weight MSE alone would suggest.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def score_layers(
    model: nn.Module,
    calibration_data: list,
    bits: int,
    granularity: str,
    group_size: int,
    device: str,
    n_samples: int = 256,
) -> dict[str, float]:
    """Compute per-layer relative output reconstruction error.

    Args:
        model:            Original (FP16/FP32) model — must not be quantized yet.
        calibration_data: List of input_ids tensors shaped [batch, seq_len].
        bits:             Target quantization bit-width (used for RTN proxy).
        granularity:      ``"per_channel"`` or ``"per_group"``.
        group_size:       Group size for per-group granularity.
        device:           Device the model is on.
        n_samples:        Max number of token activations to collect per layer.

    Returns:
        ``{layer_name: relative_error}`` for every ``nn.Linear`` that fired
        during the forward passes.  Layers that were not reached (e.g. unused
        experts) are absent from the dict.
    """
    from pare.core.dtype import QuantDtype
    from pare.core.functional import dequantize_tensor, quantize_tensor
    from pare.core.scale import compute_scale

    dtype = QuantDtype.from_bits(bits)
    activations: dict[str, Tensor] = {}
    hooks: list = []

    def _make_hook(layer_name: str):
        def hook(mod: nn.Module, inp: tuple, out: Tensor) -> None:
            if layer_name in activations:
                return  # already have enough samples
            x = inp[0].detach().float()          # [batch, seq, in] or [batch, in]
            x_flat = x.reshape(-1, x.shape[-1])  # [tokens, in]
            activations[layer_name] = x_flat[:n_samples].cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in calibration_data[:8]:
            try:
                model(batch.to(device))
            except Exception:
                pass
            if all(len(v) >= n_samples for v in activations.values()):
                break

    for h in hooks:
        h.remove()

    scores: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in activations:
            continue

        W = module.weight.data.float()  # [out, in]
        X = activations[name].to(W.device)  # [n, in]

        try:
            scale, zero = compute_scale(
                W, dtype, granularity=granularity, group_size=group_size
            )
            q = quantize_tensor(W, scale, zero, dtype)
            W_hat = dequantize_tensor(q, scale, zero).reshape_as(W)
        except Exception:
            scores[name] = 0.0
            continue

        delta_W = W - W_hat   # [out, in]
        # Output-referred error: project both through calibration inputs.
        Xt = X.T              # [in, n]
        err = (delta_W @ Xt).norm()
        ref = (W @ Xt).norm().clamp(min=1e-8)
        scores[name] = (err / ref).item()

    return scores
