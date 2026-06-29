"""Round-to-Nearest (RTN) quantizer — the simplest PTQ baseline.

RTN requires no calibration data.  It computes the scale from the
weight statistics alone, rounds weights to the nearest grid point, and
stores them packed.  It is the starting point for understanding how much
more sophisticated methods (GPTQ, AWQ) gain over this naive baseline.
"""

from __future__ import annotations

import torch.nn as nn

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.core.functional import quantize_tensor
from pare.core.scale import compute_scale
from pare.schemes.base import BaseQuantizer


class RTNQuantizer(BaseQuantizer):
    """Round-to-Nearest post-training quantizer.

    No calibration needed — scale is computed from the weight min/max.
    This makes it fast but usually 0.5–2 PPL points worse than GPTQ on
    INT4 with the same group size.
    """

    def __init__(self, config: QuantConfig, layer_bits_override: dict | None = None) -> None:
        super().__init__(config, layer_bits_override=layer_bits_override)

    def quantize_layer(
        self,
        linear: nn.Linear,
        name: str,
    ) -> "QuantizedLinear":  # type: ignore[override]
        from pare.layers.linear import QuantizedLinear

        cfg = self._config_for_layer(name)
        dtype = cfg.effective_dtype

        # NF4 and FP8 have non-uniform or float grids — delegate to from_linear()
        # which handles their scale computation and storage correctly.
        if dtype in (QuantDtype.NF4, QuantDtype.FP8_E4M3, QuantDtype.FP8_E5M2):
            return QuantizedLinear.from_linear(linear, cfg)

        weight = linear.weight.data.float()

        scale, zero = compute_scale(
            weight,
            dtype,
            granularity=cfg.granularity,
            group_size=cfg.group_size,
            sym=cfg.sym,
        )
        q_weight = quantize_tensor(weight, scale, zero, dtype)

        return QuantizedLinear(
            q_weight=q_weight,
            scale=scale,
            zero=zero,
            config=cfg,
            bias=linear.bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
