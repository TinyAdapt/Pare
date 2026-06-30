"""QuantizedLinear — a drop-in replacement for nn.Linear.

Storage layout
--------------
Weights are stored in quantized (packed) form to reduce memory.
On forward pass they are dequantized back to float for the matmul.
This is "dequantize-on-the-fly" (W4A16 style): weights are INT4,
activations stay in FP16/BF16.

The optional Triton kernel (use_kernel=True) fuses dequant+matmul for
never materializes the full FP16 weight matrix.

Shapes (per-group example, bits=4, group_size=128)
----------------------------------------------------
  linear.weight   : [out, in]             float16
  q_weight        : [out, in]             int32   (before packing)
  packed          : [out, in // 2]        uint8   (after pack_int4)
  scale           : [out, n_groups, 1]    float32
  zero            : [out, n_groups, 1]    float32 (integer-valued)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.core.functional import (
    dequantize_fp8,
    dequantize_nf4,
    dequantize_tensor,
    quantize_fp8,
    quantize_nf4,
)
from pare.core.pack import pack_int4, unpack_int4


class QuantizedLinear(nn.Module):
    """Quantized replacement for ``nn.Linear``.

    Args:
        q_weight:     Integer weight tensor (int32, values in [qmin, qmax]).
        scale:        Scale tensor, broadcastable against the weight shape.
        zero:         Zero-point tensor, same shape as scale.
        config:       The ``QuantConfig`` used to produce this layer.
        bias:         Optional bias parameter (kept in float).
        in_features:  Original linear layer ``in_features``.
        out_features: Original linear layer ``out_features``.
    """

    def __init__(
        self,
        q_weight: Tensor,
        scale: Tensor,
        zero: Tensor,
        config: QuantConfig,
        bias: nn.Parameter | None,
        in_features: int,
        out_features: int,
        quantize_inputs: bool = False,
        use_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.use_kernel = use_kernel

        # Weights from quantize_tensor may be in grouped shape [out, n_groups, group_size]
        # when per_group granularity is used. Flatten to [out, in] before storing so
        # unpack_int4 + _align_shapes can do the reshape correctly on dequantize.
        q_flat = q_weight.reshape(out_features, in_features)

        _fp8_dtypes = (QuantDtype.FP8_E4M3, QuantDtype.FP8_E5M2)
        if config.bits == 4 and config.effective_dtype not in _fp8_dtypes:
            self.register_buffer("packed_weight", pack_int4(q_flat))
            self.register_buffer("q_weight", None)
        elif config.effective_dtype in _fp8_dtypes:
            # Store float8 tensor directly — do NOT cast to int8.
            self.register_buffer("q_weight", q_flat)
            self.register_buffer("packed_weight", None)
        else:
            self.register_buffer("q_weight", q_flat.to(torch.int8))
            self.register_buffer("packed_weight", None)

        self.register_buffer("scale", scale.float())
        self.register_buffer("zero", zero.float())
        self.quantize_inputs = quantize_inputs

        if bias is not None:
            self.bias = nn.Parameter(bias.data.clone())
        else:
            self.bias = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dequantize(self) -> Tensor:
        """Reconstruct the full float weight matrix."""
        if self.config.effective_dtype == QuantDtype.NF4:
            indices = unpack_int4(self.packed_weight)           # [out, in] int32
            return dequantize_nf4(indices, self.scale).reshape(self.out_features, self.in_features)

        if self.config.effective_dtype in (QuantDtype.FP8_E4M3, QuantDtype.FP8_E5M2):
            return dequantize_fp8(self.q_weight, self.scale).reshape(
                self.out_features, self.in_features
            )

        if self.config.bits == 4:
            q = unpack_int4(self.packed_weight)
        else:
            q = self.q_weight.to(torch.int32)

        w_float = dequantize_tensor(q, self.scale, self.zero)
        return w_float.reshape(self.out_features, self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        if (
            self.use_kernel
            and self.config.effective_dtype == QuantDtype.INT4
            and not self.quantize_inputs
            and x.is_cuda
        ):
            return self._forward_kernel(x)

        weight = self.dequantize().to(x.dtype)
        if self.quantize_inputs:
            # SmoothQuant W+A: per-token dynamic INT8 quantization of activations.
            # The smooth factor s has already been absorbed into the upstream LayerNorm,
            # so x here is already x/s. We fake-quantize to simulate INT8 precision;
            # Fused INT8 matmul available via use_kernel=True.
            act_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
            x_q = (x / act_scale).round().clamp(-128, 127)
            x = (x_q * act_scale).to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)

    def _forward_kernel(self, x: Tensor) -> Tensor:
        """Forward pass using the fused Triton INT4 dequant+matmul kernel."""
        from pare.kernels.matmul_int4 import matmul_w4a16
        from pare.core.pack import repack_int4_for_kernel

        group_size = self.config.group_size if self.config.granularity == "per_group" else self.in_features

        # Lazily repack storage-layout weights into kernel-layout (done once per layer).
        if not hasattr(self, "_packed_weight_kernel"):
            self._packed_weight_kernel = repack_int4_for_kernel(
                self.packed_weight, group_size=group_size
            )

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float16)
        y = matmul_w4a16(x_2d, self._packed_weight_kernel, self.scale, self.zero, group_size)
        y = y.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig_shape[:-1], self.out_features)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_tensors(
        cls,
        packed_weight: Tensor | None,
        q_weight: Tensor | None,
        scale: Tensor,
        zero: Tensor,
        config: QuantConfig,
        bias: Tensor | None,
        in_features: int,
        out_features: int,
        quantize_inputs: bool = False,
    ) -> "QuantizedLinear":
        """Reconstruct from already-packed/quantized tensors loaded from disk.

        Bypasses the packing step in ``__init__`` — used by ``load_quantized``.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.config = config
        instance.in_features = in_features
        instance.out_features = out_features
        instance.quantize_inputs = quantize_inputs

        instance.register_buffer("packed_weight", packed_weight)
        instance.register_buffer(
            "q_weight",
            q_weight.to(torch.int8) if q_weight is not None else None,
        )
        instance.register_buffer("scale", scale.float())
        instance.register_buffer("zero", zero.float())

        if bias is not None:
            instance.bias = nn.Parameter(bias.clone())
        else:
            instance.bias = None

        return instance

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: QuantConfig,
    ) -> "QuantizedLinear":
        """Quantize an ``nn.Linear`` with RTN and return a QuantizedLinear.

        For GPTQ/AWQ use the respective quantizer classes directly.
        """
        from pare.core.functional import quantize_tensor
        from pare.core.scale import compute_scale

        dtype = config.effective_dtype
        weight = linear.weight.data.float()

        if dtype == QuantDtype.NF4:
            # NF4: per-row absmax scale; no zero-point (symmetric codebook).
            scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [out, 1]
            zero  = torch.zeros_like(scale)
            q_weight = quantize_nf4(weight, scale)   # indices in [0, 15]
            return cls(
                q_weight=q_weight,
                scale=scale,
                zero=zero,
                config=config,
                bias=linear.bias,
                in_features=linear.in_features,
                out_features=linear.out_features,
            )

        if dtype in (QuantDtype.FP8_E4M3, QuantDtype.FP8_E5M2):
            from pare.core.functional import _FP8_E4M3_MAX, _FP8_E5M2_MAX
            fp8_max = _FP8_E4M3_MAX if dtype == QuantDtype.FP8_E4M3 else _FP8_E5M2_MAX
            # Per-row absmax scale normalises W into [-fp8_max, fp8_max] before cast.
            scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
            zero  = torch.zeros_like(scale)
            q_weight = quantize_fp8(weight, scale, dtype)
            return cls(
                q_weight=q_weight,
                scale=scale,
                zero=zero,
                config=config,
                bias=linear.bias,
                in_features=linear.in_features,
                out_features=linear.out_features,
            )

        scale, zero = compute_scale(
            weight,
            dtype,
            granularity=config.granularity,
            group_size=config.group_size,
            sym=config.sym,
        )
        q_weight = quantize_tensor(weight, scale, zero, dtype)
        return cls(
            q_weight=q_weight,
            scale=scale,
            zero=zero,
            config=config,
            bias=linear.bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        dtype_name = self.config.effective_dtype.name
        mode = "W+A" if self.quantize_inputs else "W"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"dtype={dtype_name}, scheme={self.config.scheme}, "
            f"granularity={self.config.granularity}, mode={mode}"
        )
