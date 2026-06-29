"""Layerwise GPTQ for transformer block models (Llama, Mistral, Qwen, etc.).

Problem: For 7B+ models, accumulating Hessians for all layers simultaneously
requires ~39 GB on top of the 14 GB model — OOM on 48 GB GPUs.

Solution: process one transformer block at a time:
  1. Intercept the first block with a Catcher to capture embedding outputs.
  2. Precompute position_embeddings once from the shared rotary_emb.
  3. For each block i:
     a. Load block to GPU.
     b. Register hooks on its linear sublayers; run all calibration inputs.
     c. GPTQ-quantize each sublayer immediately (peak VRAM: ~2 GB).
     d. Collect outputs (= inputs to block i+1).
     e. Offload block to CPU.

Supported architectures: any HF model with model.model.layers (Llama, Mistral,
Qwen2, Phi-3, Gemma, ...).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from pare.calibration.hessian import HessianAccumulator


def is_supported(model: nn.Module) -> bool:
    """Return True if this model has the block structure we support."""
    return (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and len(model.model.layers) > 0
        and isinstance(model.model.layers[0], nn.Module)
    )


class LayerwiseGPTQ:
    """Block-by-block GPTQ. Keeps peak GPU usage to O(one_block + calib_acts)."""

    def run(
        self,
        model: nn.Module,
        calibration_data: list[Tensor],
        quantizer,
        device: str | torch.device,
    ) -> nn.Module:
        """Quantize model in-place using layerwise Hessian collection.

        Args:
            model:            Model to quantize (Llama/Mistral-family HF model).
            calibration_data: List of input_ids tensors [1, seq_len].
            quantizer:        GPTQQuantizer instance; quantize_layer is called
                              for each sublayer.
            device:           GPU device string, e.g. "cuda".

        Returns:
            The same model, quantized in-place.
        """
        device = torch.device(device)
        layers = model.model.layers
        seq_len = calibration_data[0].shape[1]

        # ── 1. Capture embedding outputs (inputs to layer 0) ───────────────
        inps = _capture_embeddings(model, layers, calibration_data, device)

        # ── 2. Precompute shared position_embeddings ───────────────────────
        pe = _precompute_pe(model, layers, inps, seq_len, device)
        model.cpu()
        torch.cuda.empty_cache()

        # ── 3. Resolve the dotted path to model.model.layers ──────────────
        layers_path = _find_layers_path(model)

        # ── 4. Process one block at a time ─────────────────────────────────
        n_layers = len(layers)
        for li, layer in enumerate(layers):
            layer_prefix = f"{layers_path}.{li}"
            layer.to(device)

            # Register hooks on every quantizable linear sublayer in this block.
            subs: dict[str, nn.Linear] = {}
            accs: dict[str, HessianAccumulator] = {}
            hooks = []

            for rel_name, mod in layer.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                full_name = f"{layer_prefix}.{rel_name}"
                if not quantizer._should_quantize(full_name, mod):
                    continue
                acc = HessianAccumulator()
                subs[full_name] = mod
                accs[full_name] = acc
                hooks.append(mod.register_forward_hook(_make_hook(acc)))

            with torch.no_grad():
                for x in inps:
                    _call_layer(layer, x.to(device), pe, device)

            for h in hooks:
                h.remove()

            # Quantize each sublayer immediately — no simultaneous Hessian storage.
            for full_name, linear in subs.items():
                H = accs[full_name].finalize()
                # Expose the Hessian so quantize_layer can find it.
                quantizer._hessians[full_name] = H.cpu()
                q_layer = quantizer.quantize_layer(linear, full_name)
                _set_submodule(layer, full_name[len(layer_prefix) + 1:], q_layer)
                # Clean up immediately to keep memory flat.
                del quantizer._hessians[full_name], H
                torch.cuda.empty_cache()

            # Collect this block's outputs → inputs for the next block.
            outs: list[Tensor] = []
            with torch.no_grad():
                for x in inps:
                    out = _call_layer(layer, x.to(device), pe, device)
                    outs.append(out.cpu())
            inps = outs

            layer.cpu()
            torch.cuda.empty_cache()
            print(f"[pare] layer {li + 1}/{n_layers} quantized", flush=True)

        return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Catcher(nn.Module):
    """Replaces layer 0 temporarily to intercept embedding outputs."""
    def __init__(self, layer: nn.Module, store: list[Tensor]) -> None:
        super().__init__()
        self._layer = layer
        self._store = store

    def forward(self, x: Tensor, **_kw) -> Tensor:
        self._store.append(x.cpu())
        raise StopIteration


def _capture_embeddings(
    model: nn.Module,
    layers: nn.ModuleList,
    calibration_data: list[Tensor],
    device: torch.device,
) -> list[Tensor]:
    """Run the embedding + pre-block path, capture inputs to layer 0."""
    inps: list[Tensor] = []
    original = layers[0]
    layers[0] = _Catcher(original, inps)
    model.to(device)
    try:
        with torch.no_grad():
            for input_ids in calibration_data:
                try:
                    model(input_ids.to(device))
                except StopIteration:
                    pass
    finally:
        layers[0] = original
    return inps


def _precompute_pe(
    model: nn.Module,
    layers: nn.ModuleList,
    inps: list[Tensor],
    seq_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    """Precompute (cos, sin) position embeddings for seq_len tokens.

    In transformers >= 4.43, rotary_emb is a shared module at model.model
    level. Older versions attach it per attention layer. Returns None if
    the model doesn't use RoPE.
    """
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        # Try old-style per-layer rotary (transformers < 4.43).
        rotary = getattr(getattr(layers[0], "self_attn", None), "rotary_emb", None)
    if rotary is None:
        return None

    with torch.no_grad():
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = rotary(inps[0].to(device), position_ids)
    return cos.cpu(), sin.cpu()


def _call_layer(
    layer: nn.Module,
    x: Tensor,
    pe: tuple[Tensor, Tensor] | None,
    device: torch.device,
) -> Tensor:
    """Call a decoder layer, handling both new and old transformers APIs.

    - New (>= 4.46): requires position_embeddings=(cos, sin) kwarg; returns Tensor.
    - Old (< 4.46): uses position_ids internally; returns (hidden_states, ...) tuple.
    """
    kwargs: dict = {"use_cache": False}
    if pe is not None:
        kwargs["position_embeddings"] = (pe[0].to(device), pe[1].to(device))

    out = layer(x, **kwargs)

    # Normalise output: newer transformers returns a plain Tensor,
    # older versions return a tuple (hidden_states, ...).
    if isinstance(out, tuple):
        out = out[0]
    return out


def _find_layers_path(model: nn.Module) -> str:
    """Return the dotted name of model.model.layers in named_modules()."""
    target = model.model.layers
    for name, mod in model.named_modules():
        if mod is target:
            return name
    raise RuntimeError("Could not find model.model.layers in named_modules()")


def _set_submodule(parent: nn.Module, rel_path: str, new_mod: nn.Module) -> None:
    """Set parent.a.b.c = new_mod given rel_path='a.b.c'."""
    parts = rel_path.split(".")
    m = parent
    for part in parts[:-1]:
        m = getattr(m, part)
    setattr(m, parts[-1], new_mod)


def _make_hook(acc: HessianAccumulator):
    def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
        acc.accumulate(inputs[0].detach())
    return hook
