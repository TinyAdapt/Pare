"""Model I/O — save and load quantized models via safetensors.

Format
------
Saved to a directory containing two files:

  model.safetensors   — all tensor data: quantized buffers for QuantizedLinear
                        layers, plus all non-quantized parameters (embeddings,
                        LayerNorms, lm_head, etc.)

  pare_config.json  — metadata: which layers are quantized, their QuantConfig
                        fields (bits, scheme, granularity, …), and shape info
                        needed to reconstruct QuantizedLinear without the
                        original nn.Linear.

Usage::

    from pare.model.io import save_quantized, load_quantized

    # After quantizing with pare.quantize():
    save_quantized(model, "./llama2-7b-gptq-int4")

    # Later, load for inference (pass uninitialized / FP16 original model):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16
    )
    model = load_quantized(model, "./llama2-7b-gptq-int4")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    pass

_PARE_CONFIG_FILE = "pare_config.json"
_TENSORS_FILE       = "model.safetensors"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_quantized(model: nn.Module, path: str | Path) -> None:
    """Save a quantized model to ``path``.

    Creates ``path/model.safetensors`` (all tensors) and
    ``path/pare_config.json`` (layer metadata).

    Args:
        model: A model that has been quantized with ``pare.quantize()``.
               Must contain at least one ``QuantizedLinear`` layer.
        path:  Directory to write into (created if it does not exist).
    """
    from safetensors.torch import save_file

    from pare.layers.linear import QuantizedLinear

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect metadata for every QuantizedLinear ──────────────────
    quantized_layers: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_layers[name] = {
                "bits":             module.config.bits,
                "scheme":           module.config.scheme,
                "granularity":      module.config.granularity,
                "group_size":       module.config.group_size,
                "sym":              module.config.sym,
                "smooth_alpha":     module.config.smooth_alpha,
                "in_features":      module.in_features,
                "out_features":     module.out_features,
                "quantize_inputs":  module.quantize_inputs,
                "has_bias":         module.bias is not None,
            }

    if not quantized_layers:
        raise ValueError(
            "No QuantizedLinear layers found in model. "
            "Quantize the model first with pare.quantize()."
        )

    # ── 2. Collect all tensors from state_dict ─────────────────────────
    # state_dict() skips None buffers (registered with None value) automatically.
    # Non-quantized layers contribute weight/bias under their original names.
    # QuantizedLinear layers contribute packed_weight/q_weight/scale/zero/bias.
    #
    # Tied-weight deduplication: some models (e.g. GPT-2) share storage between
    # lm_head.weight and the embedding weight. safetensors rejects duplicate
    # storage pointers, so we keep only the first occurrence of each storage.
    # On load the architectural weight-tying is preserved by the model itself.
    seen_ptrs: set[int] = set()
    tensors: dict[str, Tensor] = {}
    for k, v in model.state_dict().items():
        v = v.contiguous().cpu()
        ptr = v.untyped_storage().data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            tensors[k] = v

    # ── 3. Write files ─────────────────────────────────────────────────
    save_file(tensors, path / _TENSORS_FILE)

    meta = {
        "pare_version":     _pare_version(),
        "quantized_layers": quantized_layers,
    }
    with open(path / _PARE_CONFIG_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    n = len(quantized_layers)
    size_mb = (path / _TENSORS_FILE).stat().st_size / 1e6
    print(f"[pare] Saved {n} quantized layers to {path}  ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_quantized(model: nn.Module, path: str | Path) -> nn.Module:
    """Load a quantized model saved with ``save_quantized``.

    Reconstructs ``QuantizedLinear`` layers in place, then loads all
    non-quantized tensors (embeddings, LayerNorms, lm_head, etc.).

    Args:
        model: An unquantized model of the **same architecture** as the one
               that was saved — e.g. loaded via ``AutoModelForCausalLM.from_config``
               (no weights) or ``from_pretrained`` (FP16 weights that will be
               overwritten by the saved tensors).
        path:  Directory created by ``save_quantized``.

    Returns:
        The same ``model`` object with quantized layers in place.
    """
    from safetensors.torch import load_file

    from pare.config import QuantConfig
    from pare.layers.linear import QuantizedLinear

    path = Path(path)

    # ── 1. Load metadata ───────────────────────────────────────────────
    with open(path / _PARE_CONFIG_FILE) as f:
        meta = json.load(f)
    ql_meta: dict[str, dict] = meta["quantized_layers"]

    # ── 2. Load all tensors ────────────────────────────────────────────
    tensors: dict[str, Tensor] = load_file(path / _TENSORS_FILE)

    # ── 3. Reconstruct QuantizedLinear layers ──────────────────────────
    for name, lm in ql_meta.items():
        config = QuantConfig(
            bits=lm["bits"],
            scheme=lm["scheme"],
            granularity=lm["granularity"],
            group_size=lm["group_size"],
            sym=lm["sym"],
            smooth_alpha=lm.get("smooth_alpha", 0.5),
        )
        prefix = name + "."
        ql = QuantizedLinear.from_tensors(
            packed_weight  = tensors.get(prefix + "packed_weight"),
            q_weight       = tensors.get(prefix + "q_weight"),
            scale          = tensors[prefix + "scale"],
            zero           = tensors[prefix + "zero"],
            config         = config,
            bias           = tensors.get(prefix + "bias"),
            in_features    = lm["in_features"],
            out_features   = lm["out_features"],
            quantize_inputs= lm.get("quantize_inputs", False),
        )
        _set_submodule(model, name, ql)

    # ── 4. Load non-quantized tensors (embeddings, norms, lm_head …) ──
    ql_prefixes = tuple(n + "." for n in ql_meta)
    non_ql = {k: v for k, v in tensors.items() if not k.startswith(ql_prefixes)}
    missing, unexpected = model.load_state_dict(non_ql, strict=False)

    if unexpected:
        # These are keys in the file that don't exist in the model at all.
        print(f"[pare] load_quantized: {len(unexpected)} unexpected keys (ignored)")

    n = len(ql_meta)
    print(f"[pare] Loaded {n} quantized layers from {path}")
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_submodule(parent: nn.Module, rel_path: str, new_mod: nn.Module) -> None:
    """Replace a submodule by dotted relative path."""
    parts = rel_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_mod)


def _pare_version() -> str:
    try:
        from pare import __version__
        return __version__
    except Exception:
        return "unknown"
