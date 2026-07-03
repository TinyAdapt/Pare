"""Pare — production-ready quantization for large language and multimodal models."""

from __future__ import annotations

import torch.nn as nn

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.kv_cache import KVCacheConfig, QuantizedKVCache
from pare.model.io import load_quantized, save_quantized

__version__ = "0.1.3"
__all__ = [
    "QuantConfig", "QuantDtype", "quantize",
    "save_quantized", "load_quantized",
    "KVCacheConfig", "QuantizedKVCache",
    "__version__",
]


def quantize(
    model: nn.Module,
    config: QuantConfig,
    calibration_data: "list | None" = None,
    device: "str" = "cpu",
) -> nn.Module:
    """Quantize a model in-place using the scheme specified in ``config``.

    Args:
        model:            Any ``nn.Module`` (HuggingFace, custom, etc.).
        config:           A ``QuantConfig`` describing the target dtype, scheme,
                          granularity, and scheme-specific hyperparameters.
        calibration_data: Required for GPTQ, AWQ, and SmoothQuant.
                          A list of input_ids tensors, each shaped [batch, seq_len].
                          Ignored for RTN.
        device:           Device to run calibration forward passes on (GPTQ only).

    Returns:
        The same model object with ``nn.Linear`` layers replaced by
        ``QuantizedLinear`` instances.

    Examples::

        # RTN — no calibration needed
        from pare import quantize, QuantConfig
        config = QuantConfig(bits=4, scheme="rtn", group_size=128)
        model = quantize(model, config)

        # GPTQ — calibration data required
        config = QuantConfig(bits=4, scheme="gptq", group_size=128)
        model = quantize(model, config, calibration_data=calib_ids, device="cuda")
    """
    # Mixed-precision sensitivity: score layers BEFORE quantization so we
    # still have the original FP16 weights.  RTN is used as a fast proxy.
    layer_bits_override: dict[str, int] = {}
    if config.sensitive_bits is not None and calibration_data is not None:
        from pare.sensitivity import score_layers
        scores = score_layers(
            model,
            calibration_data,
            bits=config.bits,
            granularity=config.granularity,
            group_size=config.group_size,
            device=device,
        )
        layer_bits_override = {
            name: config.sensitive_bits
            for name, err in scores.items()
            if err > config.sensitivity_threshold
        }
        n_total = len(scores)
        n_sensitive = len(layer_bits_override)
        print(
            f"[pare] Sensitivity: {n_total} layers scored, "
            f"{n_sensitive} above {config.sensitivity_threshold:.0%} threshold "
            f"→ {config.bits}-bit→{config.sensitive_bits}-bit"
        )

    scheme = config.scheme
    if scheme == "rtn":
        from pare.schemes.rtn import RTNQuantizer
        quantizer = RTNQuantizer(config, layer_bits_override=layer_bits_override)
        return quantizer.quantize_model(model)

    elif scheme == "gptq":
        from pare.schemes.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer(config, layer_bits_override=layer_bits_override)
        return quantizer.quantize_model(model, calibration_data=calibration_data, device=device)

    elif scheme == "awq":
        from pare.schemes.awq import AWQQuantizer
        quantizer = AWQQuantizer(config, layer_bits_override=layer_bits_override)
        return quantizer.quantize_model(model, calibration_data=calibration_data, device=device)

    elif scheme == "smoothquant":
        from pare.schemes.smoothquant import SmoothQuantQuantizer
        quantizer = SmoothQuantQuantizer(config, layer_bits_override=layer_bits_override)
        return quantizer.quantize_model(model, calibration_data=calibration_data, device=device)

    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")
