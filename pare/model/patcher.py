"""ModelPatcher — walks a model and swaps nn.Linear → QuantizedLinear.

The tricky part of in-place module replacement is that you can't call
``setattr`` on a module while iterating ``named_modules()`` (it mutates
the tree).  The pattern used here:

  1. Collect all (parent_module, child_attr_name, child_module, full_name)
     tuples in a list.
  2. Iterate the list and call setattr on the parent.

This is the same approach used by bitsandbytes, AutoGPTQ, and torchao.

Conv1D note
-----------
HuggingFace GPT-2 (and some other models) use ``transformers.pytorch_utils.Conv1D``
instead of ``nn.Linear``.  Conv1D stores its weight transposed: shape
[in_features, out_features] vs nn.Linear's [out_features, in_features], and
its forward pass is ``x @ weight + bias`` (no transpose).

We transparently convert Conv1D → nn.Linear in ``_to_linear`` so that all
downstream quantization code only ever sees nn.Linear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from pare.schemes.base import BaseQuantizer


class ModelPatcher:
    """Replaces ``nn.Linear`` (and Conv1D) layers with quantized equivalents.

    Args:
        quantizer: A ``BaseQuantizer`` subclass that implements
                   ``quantize_layer`` and ``_should_quantize``.
    """

    def __init__(self, quantizer: "BaseQuantizer") -> None:
        self.quantizer = quantizer

    def patch(self, model: nn.Module) -> nn.Module:
        """Replace all matching linear layers in ``model`` in-place."""
        replacements = self._collect_replacements(model)

        replaced = 0
        skipped = 0
        for parent, attr, child, full_name in replacements:
            if self.quantizer._should_quantize(full_name, child):
                linear = _to_linear(child)   # no-op for nn.Linear; converts Conv1D
                q_layer = self.quantizer.quantize_layer(linear, full_name)
                setattr(parent, attr, q_layer)
                replaced += 1
            else:
                skipped += 1

        print(f"[pare] Quantized {replaced} layers, skipped {skipped}.")
        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_replacements(
        model: nn.Module,
    ) -> list[tuple[nn.Module, str, nn.Module, str]]:
        """Return (parent, attr_name, child, full_name) for every quantizable layer."""
        from pare.schemes.base import _quantizable_module_types
        quantizable = _quantizable_module_types()

        results = []
        for full_name, module in model.named_modules():
            for attr_name, child in module.named_children():
                if isinstance(child, quantizable):
                    child_full_name = f"{full_name}.{attr_name}".lstrip(".")
                    results.append((module, attr_name, child, child_full_name))
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_linear(module: nn.Module) -> nn.Linear:
    """Return an equivalent nn.Linear, converting Conv1D if necessary.

    Conv1D.weight is [in_features, out_features] (transposed vs nn.Linear).
    Calling ``x @ W_conv`` equals ``nn.Linear`` with ``weight = W_conv.T``.
    """
    if isinstance(module, nn.Linear):
        return module

    # transformers.Conv1D — graceful import so transformers stays optional.
    in_features, out_features = module.weight.shape   # [in, out] for Conv1D
    linear = nn.Linear(in_features, out_features, bias=module.bias is not None)
    linear.weight.data = module.weight.T.detach().clone()
    if module.bias is not None:
        linear.bias.data = module.bias.detach().clone()
    return linear
