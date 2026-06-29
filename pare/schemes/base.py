"""Abstract base class for all quantization schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from pare.config import QuantConfig
    from pare.layers.linear import QuantizedLinear


def _quantizable_module_types() -> tuple[type, ...]:
    """Return a tuple of module types that can be quantized.

    Always includes ``nn.Linear``.  If ``transformers`` is installed, also
    includes ``transformers.pytorch_utils.Conv1D`` so that GPT-2-style models
    (which use Conv1D instead of Linear) are handled transparently.
    """
    types: list[type] = [nn.Linear]
    try:
        from transformers.pytorch_utils import Conv1D
        types.append(Conv1D)
    except ImportError:
        pass
    return tuple(types)


class BaseQuantizer(ABC):
    """Common interface for RTN, GPTQ, AWQ, SmoothQuant.

    Subclasses implement ``quantize_layer`` for a single ``nn.Linear``.
    The ``quantize_model`` method in this base class handles the full
    model traversal so subclasses don't repeat that logic.
    """

    def __init__(
        self,
        config: "QuantConfig",
        layer_bits_override: "dict[str, int] | None" = None,
    ) -> None:
        self.config = config
        self._layer_bits_override: dict[str, int] = layer_bits_override or {}

    def _config_for_layer(self, name: str) -> "QuantConfig":
        """Return the effective QuantConfig for this layer.

        If ``name`` appears in the sensitivity override dict, returns a copy
        of ``self.config`` with ``bits`` set to the overridden value and
        ``dtype`` re-inferred.  Otherwise returns ``self.config`` unchanged.
        """
        if name not in self._layer_bits_override:
            return self.config
        from dataclasses import replace
        override_bits = self._layer_bits_override[name]
        return replace(self.config, bits=override_bits, dtype=None)

    @abstractmethod
    def quantize_layer(
        self,
        linear: nn.Linear,
        name: str,
    ) -> "QuantizedLinear":
        """Quantize a single linear layer and return its replacement.

        Args:
            linear: The original ``nn.Linear`` module.
            name:   Fully-qualified module name (e.g. ``"model.layers.0.self_attn.q_proj"``).
        """

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Replace all matching ``nn.Linear`` layers in-place.

        Delegates matching / exclusion logic to ``_should_quantize``.
        Returns the same model object (modified in-place).
        """
        from pare.model.patcher import ModelPatcher
        patcher = ModelPatcher(self)
        return patcher.patch(model)

    def _should_quantize(self, name: str, module: nn.Module) -> bool:
        """Return True if this module should be quantized."""
        if not isinstance(module, _quantizable_module_types()):
            return False
        cfg = self.config
        if any(excl in name for excl in cfg.exclude):
            return False
        if cfg.modules is not None:
            import re
            return any(re.search(pattern, name) for pattern in cfg.modules)
        return True
