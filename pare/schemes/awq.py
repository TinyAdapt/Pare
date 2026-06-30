"""AWQ — Activation-aware Weight Quantization (Lin et al., 2023).

Algorithm summary
-----------------
For each Linear layer, identify salient input channels by measuring
per-channel mean |x| across calibration data.  Scale those channels UP
before quantizing (so the rounding error is proportionally smaller where
it matters), and absorb the inverse scale into the predecessor layer so
the computation is mathematically equivalent.

Scale search
------------
For a group of linear layers sharing an input x:

  s_j = mean(|x_j|)^α     α ∈ {0/N, 1/N, …, (N-1)/N}  (grid search, N=20)
  s   = s / sqrt(s_max · s_min)           (geometric-mean normalisation)
  α*  = argmin_α  mean( (Q(W·diag(s))·diag(1/s) − W)² · mean|x| )

Scale fusion (Llama / Mistral / Qwen block structure)
------------------------------------------------------
Four groups per transformer block, processed in order:

  1. input_layernorm  → [q_proj, k_proj, v_proj]  via scale_ln_fcs
  2. v_proj           → [o_proj]                  via scale_fc_fc
  3. post_attn_ln     → [gate_proj, up_proj]       via scale_ln_fcs
  4. up_proj          → [down_proj]                via scale_fc_fc

After all scales are fused, every linear layer in the block is RTN-quantized.

Reference: https://arxiv.org/abs/2306.00978
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from pare.config import QuantConfig
from pare.core.functional import dequantize_tensor, quantize_tensor
from pare.core.scale import compute_scale
from pare.layers.linear import QuantizedLinear
from pare.schemes.base import BaseQuantizer


class AWQQuantizer(BaseQuantizer):
    """AWQ quantizer.  Requires calibration data to collect activation stats.

    Call via the top-level ``quantize()`` function::

        from pare import quantize, QuantConfig
        config = QuantConfig(bits=4, scheme="awq", group_size=128)
        model = quantize(model, config, calibration_data=calib_ids, device="cuda")
    """

    def quantize_model(  # type: ignore[override]
        self,
        model: nn.Module,
        calibration_data: list[Tensor] | None = None,
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        """Apply AWQ scale search + fusion then RTN-quantize the model.

        For Llama/Mistral/Qwen-family models (those with ``model.model.layers``)
        this uses a layerwise strategy to stay within GPU memory limits.
        Falls back to RTN-only for unsupported architectures.
        """
        if calibration_data is None:
            raise ValueError(
                "AWQ requires calibration_data. Pass a list of input_ids tensors:\n\n"
                "    quantize(model, config, calibration_data=calib_ids)"
            )

        from pare.calibration.layerwise import is_supported

        if is_supported(model):
            return _LayerwiseAWQ().run(model, calibration_data, self, device)

        # Unsupported architecture: warn and fall back to RTN.
        print(
            "[pare] AWQ: unsupported architecture — "
            "no model.model.layers found. Falling back to RTN."
        )
        from pare.model.patcher import ModelPatcher
        patcher = ModelPatcher(self)
        return patcher.patch(model)

    # ------------------------------------------------------------------
    # Layer-level quantization — called after scale fusion
    # ------------------------------------------------------------------

    def quantize_layer(self, linear: nn.Linear, name: str) -> QuantizedLinear:
        """RTN-quantize a single linear layer (scales already fused into weights)."""
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
        )


# ---------------------------------------------------------------------------
# Layerwise AWQ processor
# ---------------------------------------------------------------------------

class _LayerwiseAWQ:
    """Block-by-block AWQ for transformer models with model.model.layers."""

    def run(
        self,
        model: nn.Module,
        calibration_data: list[Tensor],
        quantizer: AWQQuantizer,
        device: str | torch.device,
    ) -> nn.Module:
        from pare.calibration.layerwise import (
            _Catcher, _call_layer, _capture_embeddings,
            _find_layers_path, _precompute_pe, _set_submodule,
        )

        device = torch.device(device)
        layers = model.model.layers
        seq_len = calibration_data[0].shape[1]

        inps = _capture_embeddings(model, layers, calibration_data, device)
        pe   = _precompute_pe(model, layers, inps, seq_len, device)
        model.cpu()
        torch.cuda.empty_cache()

        layers_path = _find_layers_path(model)
        n_layers    = len(layers)

        for li, layer in enumerate(layers):
            layer.to(device)

            # ── 1. Collect per-channel mean |x| for all linear inputs ──────
            named_linears = {
                name: mod
                for name, mod in layer.named_modules()
                if isinstance(mod, nn.Linear)
            }
            act_stats = _collect_act_stats(layer, named_linears, inps, pe, device)

            # ── 2. Process the four groups: search scale, fuse, update weights ─
            _apply_awq_groups(layer, act_stats, quantizer.config)

            # ── 3. RTN-quantize all linear layers in this block ────────────
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
            print(f"[pare] AWQ layer {li + 1}/{n_layers} done", flush=True)

        return model


# ---------------------------------------------------------------------------
# Activation statistics collection
# ---------------------------------------------------------------------------

def _collect_act_stats(
    layer: nn.Module,
    named_linears: dict[str, nn.Linear],
    inps: list[Tensor],
    pe,
    device: torch.device,
) -> dict[str, Tensor]:
    """Run calibration data through layer and return mean |x| per linear input."""
    from pare.calibration.observer import ActivationObserver
    from pare.calibration.layerwise import _call_layer

    observers: dict[str, ActivationObserver] = {n: ActivationObserver() for n in named_linears}
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

    return {n: obs.finalize() for n, obs in observers.items()}


# ---------------------------------------------------------------------------
# Group scale search and fusion
# ---------------------------------------------------------------------------

def _apply_awq_groups(
    layer: nn.Module,
    act_stats: dict[str, Tensor],
    config: QuantConfig,
) -> None:
    """Search and fuse AWQ scales for all four groups in one transformer block.

    Targets Llama/Mistral/Qwen block naming: input_layernorm, post_attention_layernorm,
    self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj.
    """
    attn = layer.self_attn
    mlp  = layer.mlp

    # ── Group 1: input_layernorm → [q_proj, k_proj, v_proj] ──────────
    if hasattr(layer, "input_layernorm") and hasattr(attn, "q_proj"):
        x_max = act_stats.get("self_attn.q_proj")
        if x_max is not None:
            fcs = [attn.q_proj, attn.k_proj, attn.v_proj]
            s   = _search_scale(fcs, x_max, config)
            _scale_ln_fcs(layer.input_layernorm, fcs, s)

    # ── Group 2: v_proj → [o_proj] ───────────────────────────────────
    # Skipped for GQA models (e.g. Qwen2.5, Mistral) where v_proj outputs
    # num_kv_heads * head_dim but o_proj takes num_heads * head_dim.
    # The scale can only be fused when these dimensions match (MHA).
    if (hasattr(attn, "v_proj") and hasattr(attn, "o_proj")
            and attn.v_proj.out_features == attn.o_proj.in_features):
        x_max = act_stats.get("self_attn.o_proj")
        if x_max is not None:
            s = _search_scale([attn.o_proj], x_max, config)
            _scale_fc_fc(attn.v_proj, attn.o_proj, s)

    # ── Group 3: post_attention_layernorm → [gate_proj, up_proj] ─────
    if hasattr(layer, "post_attention_layernorm") and hasattr(mlp, "gate_proj"):
        x_max = act_stats.get("mlp.gate_proj")
        if x_max is not None:
            fcs = [mlp.gate_proj, mlp.up_proj]
            s   = _search_scale(fcs, x_max, config)
            _scale_ln_fcs(layer.post_attention_layernorm, fcs, s)

    # ── Group 4: up_proj → [down_proj] ───────────────────────────────
    # Only feasible when up_proj.out == down_proj.in (SwiGLU without gating
    # dimension mismatch). Standard in Llama/Mistral/Qwen MLP blocks.
    if (hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")
            and mlp.up_proj.out_features == mlp.down_proj.in_features):
        x_max = act_stats.get("mlp.down_proj")
        if x_max is not None:
            s = _search_scale([mlp.down_proj], x_max, config)
            _scale_fc_fc(mlp.up_proj, mlp.down_proj, s)


def _search_scale(
    fcs: list[nn.Linear],
    x_max: Tensor,
    config: QuantConfig,
    n_grid: int = 20,
) -> Tensor:
    """Grid-search the best per-channel scale for a group of linear layers.

    Args:
        fcs:    Linear layers that share the same input distribution x_max.
        x_max:  Per-channel mean |x|, shape [in_features].
        config: QuantConfig (bits, granularity, group_size, sym).
        n_grid: Number of α values to try (paper default: 20).

    Returns:
        Best scale tensor, shape [in_features], on same device as x_max.
    """
    best_scale = torch.ones_like(x_max)
    best_loss  = float("inf")

    x_max = x_max.clamp(min=1e-6)

    for i in range(n_grid):
        alpha = i / n_grid

        s = x_max.pow(alpha).clamp(min=1e-4)
        # Geometric-mean normalisation: keeps scale centred around 1.
        s = s / (s.max() * s.min()).sqrt()

        # Measure weighted reconstruction loss across all layers in the group.
        loss = 0.0
        for fc in fcs:
            W = fc.weight.data.float()
            loss += _reconstruction_loss(W, s, x_max, config)
        loss /= len(fcs)

        if loss < best_loss:
            best_loss  = loss
            best_scale = s.clone()

    return best_scale


def _reconstruction_loss(
    W: Tensor,
    s: Tensor,
    x_max: Tensor,
    config: QuantConfig,
) -> float:
    """Weighted quantization reconstruction error for scale candidate s.

    Loss = mean( (Q(W·diag(s))·diag(1/s) − W)² · x_max )
    """
    W_scaled = W * s.to(W.device).view(1, -1)
    W_q      = _fake_quant(W_scaled, config)
    W_dq     = W_q / s.to(W.device).view(1, -1)
    err      = (W_dq - W).pow(2) * x_max.to(W.device).view(1, -1)
    return err.mean().item()


def _fake_quant(W: Tensor, config: QuantConfig) -> Tensor:
    """Quantize W to config.bits and immediately dequantize back to float.

    Used only during scale search — produces the same values that RTN
    would store, without actually packing integers.
    """
    scale, zero = compute_scale(
        W, config.effective_dtype,
        granularity=config.granularity,
        group_size=config.group_size,
        sym=config.sym,
    )
    q    = quantize_tensor(W, scale, zero, config.effective_dtype)
    W_dq = dequantize_tensor(q.reshape(W.shape[0], -1), scale, zero)
    return W_dq.reshape(W.shape)


# ---------------------------------------------------------------------------
# Scale fusion helpers
# ---------------------------------------------------------------------------

def _scale_ln_fcs(
    ln: nn.Module,
    fcs: list[nn.Linear],
    s: Tensor,
) -> None:
    """Fuse scale into a LayerNorm predecessor and its successor linears.

    LayerNorm output becomes x/s; successor weights become W·diag(s).
    Product is unchanged: (W·diag(s)) · (x/s) = Wx.
    """
    s = s.to(ln.weight.device)
    ln.weight.data.div_(s)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data.div_(s)
    for fc in fcs:
        fc.weight.data.mul_(s.to(fc.weight.device).view(1, -1))


def _scale_fc_fc(
    fc1: nn.Linear,
    fc2: nn.Linear,
    s: Tensor,
) -> None:
    """Fuse scale into a linear-predecessor / linear-successor pair.

    fc1's output rows are scaled down by s; fc2's input columns are scaled
    up by s.  Product is unchanged: (W2·diag(s)) · (W1/s · x) = W2·W1·x.

    The ``[-n:]`` slice handles the case where fc1 may have more output
    features than fc2 has input features (e.g. combined QKV projections).
    """
    n = s.shape[0]
    s = s.to(fc1.weight.device)
    fc1.weight.data[-n:].div_(s.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.data[-n:].div_(s)
    fc2.weight.data.mul_(s.to(fc2.weight.device).view(1, -1))
