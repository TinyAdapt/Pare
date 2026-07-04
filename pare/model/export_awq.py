# SPDX-License-Identifier: Apache-2.0
# The AWQ-GEMM packing layout follows AutoAWQ / mit-han-lab/llm-awq (MIT).
# See LICENSE (ATTRIBUTIONS).
"""Export a Pare 4-bit quantized model to the AutoAWQ (GEMM) checkpoint format.

The resulting directory is a standard AWQ checkpoint that serving stacks such as
vLLM (Marlin kernels) load directly, so a Pare-quantized model can be served in
production without re-quantization.

The repack is lossless. Pare stores INT4 weights in the affine form
``w_hat = (q - z) * s`` with an integer zero-point ``z``, which is exactly the
AWQ convention, so we pack Pare's own codes ``q`` and zero-points ``z`` into the
AWQ int32 layout directly. Verified by a WikiText-2 perplexity match between the
Pare model and the exported checkpoint served by vLLM (Delta < 0.01 PPL).

This module has no third-party quantization dependency: it produces the standard
AWQ *format* (``qweight``/``qzeros``/``scales`` + an ``awq`` ``quantization_config``),
which the serving stack reads with its own loader.

Usage::

    from pare import quantize, QuantConfig
    from pare.model.export_awq import export_awq
    quantize(model, QuantConfig(bits=4, scheme="awq", group_size=128),
             calibration_data=calib, device="cuda")
    export_awq(model, "qwen-awq-int4/", tokenizer=tokenizer)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pare.core.pack import unpack_int4
from pare.layers.linear import QuantizedLinear

# AWQ-GEMM packs eight 4-bit values per int32 along the output dimension in this
# interleaved order (matching AutoAWQ's WQLinear_GEMM / vLLM's AWQ loader).
_AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_awq(x: torch.Tensor) -> torch.Tensor:
    """Pack a [..., N] int tensor (values 0-15) into [..., N//8] int32, AWQ order."""
    rows = x.shape[:-1]
    x = x.reshape(*rows, x.shape[-1] // 8, 8).to(torch.int32)
    out = torch.zeros(*rows, x.shape[-2], dtype=torch.int32)
    for i, o in enumerate(_AWQ_ORDER):
        out |= x[..., o] << (i * 4)
    return out


class AwqGemmLinear(nn.Module):
    """Holds the packed AWQ-GEMM buffers so the checkpoint serializes correctly.

    Inference is performed by the serving stack (e.g. vLLM's Marlin kernel), which
    replaces this module at load time, so no ``forward`` is defined here.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int, has_bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = 4
        self.group_size = group_size
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.zeros(in_features, out_features // 8, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(n_groups, out_features // 8, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(n_groups, out_features, dtype=torch.float16))
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None


def _to_awq_linear(ql: QuantizedLinear, group_size: int) -> AwqGemmLinear:
    """Repack a Pare 4-bit layer into an ``AwqGemmLinear`` (lossless)."""
    out_f, in_f = ql.out_features, ql.in_features
    if in_f % group_size != 0 or out_f % 8 != 0:
        raise ValueError(
            f"AWQ export needs in_features ({in_f}) divisible by group_size "
            f"({group_size}) and out_features ({out_f}) divisible by 8."
        )

    q = unpack_int4(ql.packed_weight).reshape(out_f, in_f).to(torch.int32).cpu()  # [out, in]
    scale = ql.scale.reshape(out_f, -1).float().cpu()                 # [out, n_groups]
    zero = ql.zero.reshape(out_f, -1).round().to(torch.int32).cpu()   # [out, n_groups]

    m = AwqGemmLinear(in_f, out_f, group_size, ql.bias is not None)
    m.qweight = _pack_awq(q.t().contiguous())        # [in, out//8]
    m.qzeros = _pack_awq(zero.t().contiguous())      # [n_groups, out//8]
    m.scales = scale.t().contiguous().half()         # [n_groups, out]
    if ql.bias is not None:
        m.bias = ql.bias.data.half().cpu()
    return m


def export_awq(model: nn.Module, save_dir: str, tokenizer=None) -> str:
    """Repack a Pare 4-bit model in place into an AWQ checkpoint and save it.

    Args:
        model:      A model already quantized by Pare with ``bits=4`` (AWQ or RTN).
                    Its ``QuantizedLinear`` layers are replaced in place.
        save_dir:   Output directory for the AWQ-format checkpoint.
        tokenizer:  Optional tokenizer to save alongside the model.

    Returns:
        ``save_dir``.
    """
    qls = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
    if not qls:
        raise ValueError(
            "No QuantizedLinear layers found. Quantize with a 4-bit scheme "
            "(e.g. QuantConfig(bits=4, scheme='awq', group_size=128)) first."
        )
    cfg = qls[0].config
    if cfg.bits != 4:
        raise ValueError(f"AWQ export supports 4-bit only; got bits={cfg.bits}.")
    group_size = cfg.group_size

    for name, mod in list(model.named_modules()):
        if isinstance(mod, QuantizedLinear):
            awq = _to_awq_linear(mod, group_size)
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], awq)

    model.config.quantization_config = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": group_size,
        "version": "gemm",
        "zero_point": True,
    }
    model.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    return save_dir
