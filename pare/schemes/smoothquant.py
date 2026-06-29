"""SmoothQuant — W8A8 quantization via activation smoothing (Xiao et al., 2022).

Algorithm summary
-----------------
Activations are hard to quantize: some input channels have systematically large
values (outliers) that force a wide quantization range, leaving little precision
for the majority of values.  Weights are easy to quantize: their distribution
is relatively uniform across channels.

SmoothQuant migrates quantization difficulty from activations to weights:

  y = Wx = (W · diag(s)) · (diag(s)⁻¹ · x) = Ŵ · x̂

The smooth factor s_j balances the per-channel dynamic ranges:

  s_j = max|X_j|^α / max|W_{:,j}|^(1-α)

  max|X_j|     — per-channel maximum absolute activation over calibration data
  max|W_{:,j}| — per-column maximum absolute weight
  α = 0.5      — geometric mean; balances difficulty evenly (paper default)

Ŵ = W · diag(s) is absorbed into the weight matrix offline.
x̂ = x / s is obtained at runtime by fusing 1/s into the predecessor LayerNorm
(no extra kernel call).

At inference, both Ŵ and x̂ are quantized to INT8:
  - Ŵ: per-channel static, computed offline
  - x̂: per-token dynamic (scale = max|x̂_token| / 127), computed at runtime

This enables INT8 × INT8 matrix multiplication on tensor cores.
For now, activations are fake-quantized (quantize then dequantize) to validate
correctness without a custom kernel.

Reference: https://arxiv.org/abs/2211.10438
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from pare.config import QuantConfig
from pare.core.functional import quantize_tensor
from pare.core.scale import compute_scale
from pare.layers.linear import QuantizedLinear
from pare.schemes.awq import _scale_fc_fc, _scale_ln_fcs
from pare.schemes.base import BaseQuantizer


class SmoothQuantQuantizer(BaseQuantizer):
    """SmoothQuant W8A8 quantizer.

    Requires calibration data to estimate per-channel activation ranges.
    Use INT8 (bits=8) with per-channel weight quantization::

        from pare import quantize, QuantConfig
        config = QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel")
        model = quantize(model, config, calibration_data=calib_ids, device="cuda")
    """

    def quantize_model(  # type: ignore[override]
        self,
        model: nn.Module,
        calibration_data: list[Tensor] | None = None,
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "SmoothQuant requires calibration_data. Pass a list of input_ids:\n\n"
                "    quantize(model, config, calibration_data=calib_ids)"
            )

        from pare.calibration.layerwise import is_supported

        if is_supported(model):
            return _LayerwiseSmoothQuant().run(model, calibration_data, self, device)

        print(
            "[pare] SmoothQuant: unsupported architecture — "
            "no model.model.layers found. Falling back to RTN."
        )
        from pare.model.patcher import ModelPatcher
        return ModelPatcher(self).patch(model)

    def quantize_layer(self, linear: nn.Linear, name: str) -> QuantizedLinear:
        """INT8 per-channel quantization with activation quantization enabled."""
        cfg = self._config_for_layer(name)
        W = linear.weight.data.float()
        scale, zero = compute_scale(
            W, cfg.effective_dtype,
            granularity=cfg.granularity,
            group_size=cfg.group_size,
            sym=cfg.sym,
        )
        q_int = quantize_tensor(W, scale, zero, cfg.effective_dtype).reshape(W.shape[0], -1)
        return QuantizedLinear(
            q_weight=q_int, scale=scale, zero=zero, config=cfg,
            bias=linear.bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
            quantize_inputs=True,
        )


# ---------------------------------------------------------------------------
# Layerwise SmoothQuant processor
# ---------------------------------------------------------------------------

class _LayerwiseSmoothQuant:
    """Block-by-block SmoothQuant for models with model.model.layers."""

    def run(
        self,
        model: nn.Module,
        calibration_data: list[Tensor],
        quantizer: SmoothQuantQuantizer,
        device: str | torch.device,
    ) -> nn.Module:
        from pare.calibration.layerwise import (
            _call_layer, _capture_embeddings,
            _find_layers_path, _precompute_pe, _set_submodule,
        )

        device    = torch.device(device)
        layers    = model.model.layers
        seq_len   = calibration_data[0].shape[1]
        n_layers  = len(layers)

        inps = _capture_embeddings(model, layers, calibration_data, device)
        pe   = _precompute_pe(model, layers, inps, seq_len, device)
        model.cpu()
        torch.cuda.empty_cache()

        layers_path = _find_layers_path(model)

        for li, layer in enumerate(layers):
            layer.to(device)

            named_linears = {
                name: mod
                for name, mod in layer.named_modules()
                if isinstance(mod, nn.Linear)
            }

            # ── 1. Collect max |x| for each linear layer's input ──────────
            act_stats = _collect_max_abs(layer, named_linears, inps, pe, device)

            # ── 2. Compute smooth factors and fuse into predecessor layers ─
            _apply_smooth_groups(layer, named_linears, act_stats, quantizer.config)

            # ── 3. INT8 quantize all linear layers (quantize_inputs=True) ─
            layer_prefix = f"{layers_path}.{li}"
            for rel_name, linear in named_linears.items():
                full_name = f"{layer_prefix}.{rel_name}"
                if quantizer._should_quantize(full_name, linear):
                    q_layer = quantizer.quantize_layer(linear, full_name)
                    _set_submodule(layer, rel_name, q_layer)

            # ── 4. Collect outputs for the next layer ──────────────────────
            outs: list[Tensor] = []
            with torch.no_grad():
                for x in inps:
                    out = _call_layer(layer, x.to(device), pe, device)
                    outs.append(out.cpu())
            inps = outs

            layer.cpu()
            torch.cuda.empty_cache()
            print(f"[pare] SmoothQuant layer {li + 1}/{n_layers} done", flush=True)

        return model


# ---------------------------------------------------------------------------
# Activation statistics collection (max |x| per channel)
# ---------------------------------------------------------------------------

def _collect_max_abs(
    layer: nn.Module,
    named_linears: dict[str, nn.Linear],
    inps: list[Tensor],
    pe,
    device: torch.device,
) -> dict[str, Tensor]:
    """Collect per-channel max |x| for each linear layer's input activation."""
    from pare.calibration.observer import ActivationObserver
    from pare.calibration.layerwise import _call_layer

    observers = {n: ActivationObserver() for n in named_linears}
    hooks = [
        mod.register_forward_hook(
            (lambda obs: lambda m, inp, out: obs.accumulate(inp[0].detach()))(observers[n])
        )
        for n, mod in named_linears.items()
    ]

    with torch.no_grad():
        for x in inps:
            _call_layer(layer, x.to(device), pe, device)

    for h in hooks:
        h.remove()

    return {n: obs.max_abs() for n, obs in observers.items()}


# ---------------------------------------------------------------------------
# Smooth factor computation and group fusion
# ---------------------------------------------------------------------------

def _apply_smooth_groups(
    layer: nn.Module,
    named_linears: dict[str, nn.Linear],
    act_stats: dict[str, Tensor],
    config: QuantConfig,
) -> None:
    """Compute smooth factors and fuse into predecessor layers.

    Handles the same four groups as AWQ but uses the closed-form SmoothQuant
    formula instead of a grid search.
    """
    alpha = config.smooth_alpha
    attn  = layer.self_attn
    mlp   = layer.mlp

    # ── Group 1: input_layernorm → [q_proj, k_proj, v_proj] ──────────
    if hasattr(layer, "input_layernorm") and hasattr(attn, "q_proj"):
        x_max = act_stats.get("self_attn.q_proj")
        if x_max is not None:
            fcs = [attn.q_proj, attn.k_proj, attn.v_proj]
            s   = _smooth_factor(x_max, fcs, alpha)
            _scale_ln_fcs(layer.input_layernorm, fcs, s)

    # ── Group 2: v_proj → [o_proj] (MHA only, skipped for GQA) ──────
    if (hasattr(attn, "v_proj") and hasattr(attn, "o_proj")
            and attn.v_proj.out_features == attn.o_proj.in_features):
        x_max = act_stats.get("self_attn.o_proj")
        if x_max is not None:
            s = _smooth_factor(x_max, [attn.o_proj], alpha)
            _scale_fc_fc(attn.v_proj, attn.o_proj, s)

    # ── Group 3: post_attention_layernorm → [gate_proj, up_proj] ─────
    if hasattr(layer, "post_attention_layernorm") and hasattr(mlp, "gate_proj"):
        x_max = act_stats.get("mlp.gate_proj")
        if x_max is not None:
            fcs = [mlp.gate_proj, mlp.up_proj]
            s   = _smooth_factor(x_max, fcs, alpha)
            _scale_ln_fcs(layer.post_attention_layernorm, fcs, s)

    # ── Group 4: up_proj → [down_proj] ───────────────────────────────
    if (hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")
            and mlp.up_proj.out_features == mlp.down_proj.in_features):
        x_max = act_stats.get("mlp.down_proj")
        if x_max is not None:
            s = _smooth_factor(x_max, [mlp.down_proj], alpha)
            _scale_fc_fc(mlp.up_proj, mlp.down_proj, s)


def _smooth_factor(
    x_max: Tensor,
    fcs: list[nn.Linear],
    alpha: float,
) -> Tensor:
    """Compute the SmoothQuant scale: s_j = max|x_j|^α / max|W_{:,j}|^(1-α).

    When there are multiple successor layers sharing the same input, we take
    the per-column max across all of them for the weight term.
    """
    x_max = x_max.clamp(min=1e-6)

    # Per-column max absolute weight, averaged over all fcs sharing this input.
    w_max = torch.stack(
        [fc.weight.data.float().abs().amax(dim=0).to(x_max.device) for fc in fcs]
    ).amax(dim=0).clamp(min=1e-6)

    s = (x_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-6)
    return s
