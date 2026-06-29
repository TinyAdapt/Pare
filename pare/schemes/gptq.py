"""GPTQ — Accurate Post-Training Quantization (Frantar et al., 2022).

Algorithm summary
-----------------
For each Linear layer (in order), given weight W [out, in] and Hessian
H = 2/n * X X^T [in, in]:

  1. Add damping:  H += λI   (λ = damp_percent * mean(diag(H)))
  2. Cholesky of H: L = cholesky(H)
     Invert via Cholesky:  H⁻¹ = cholesky_inverse(L)
     Upper Cholesky of H⁻¹:  C = cholesky(H⁻¹, upper=True)
     (C is reused as 'Hinv' in the update loop)
  3. Column-block loop (block_size = 128):
       For each column k in block:
         q = quant(w_k)                   # round to nearest grid point
         e = (w_k - q) / C[k,k]          # normalised error
         Update remaining cols in block:  w[k+1:] -= e * C[k, k+1:]
       Lazy update cols right of block:   W[:, block_end:] -= E @ C[block:block_end, block_end:]
  4. Optional activation ordering: sort columns by descending H diagonal
     before the loop; undo permutation on Q_int afterward.

Reference: https://arxiv.org/abs/2210.17323
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.core.scale import compute_scale
from pare.layers.linear import QuantizedLinear
from pare.schemes.base import BaseQuantizer

_BLOCK_SIZE = 128   # number of columns per lazy-update block (paper default)


class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantizer.  Requires calibration data to build per-layer Hessians.

    Call via the top-level ``quantize()`` function::

        from pare import quantize, QuantConfig
        config = QuantConfig(bits=4, scheme="gptq", group_size=128)
        model = quantize(model, config, calibration_data=calib_ids)
    """

    def __init__(self, config: QuantConfig, layer_bits_override: dict | None = None) -> None:
        super().__init__(config, layer_bits_override=layer_bits_override)
        self._hessians: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Model-level entry point (overrides base to inject calibration data)
    # ------------------------------------------------------------------

    def quantize_model(  # type: ignore[override]
        self,
        model: nn.Module,
        calibration_data: list[Tensor] | None = None,
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        """Collect Hessians then quantize every matching Linear layer.

        For transformer block models (Llama, Mistral, Qwen, ...) this uses a
        layerwise strategy that keeps peak GPU memory to ~2 GB regardless of
        model size.  For other architectures it falls back to the full-model
        forward pass (fine for small models like GPT-2).

        Args:
            model:            The model to quantize (modified in-place).
            calibration_data: List of input_ids tensors ([batch, seq_len]).
                              Required for GPTQ — raises ValueError if None.
            device:           Device for calibration forward passes.
        """
        if calibration_data is None:
            raise ValueError(
                "GPTQ requires calibration_data. Pass a list of input_ids "
                "tensors to quantize():\n\n"
                "    quantize(model, config, calibration_data=calib_ids)"
            )

        from pare.calibration.layerwise import is_supported, LayerwiseGPTQ

        if is_supported(model):
            return LayerwiseGPTQ().run(model, calibration_data, self, device)

        # ── Fallback: full-model forward (small models / unsupported arch) ──
        from pare.calibration.runner import CalibrationRunner
        from pare.model.patcher import ModelPatcher

        runner = CalibrationRunner(self)
        self._hessians = runner.collect(model, calibration_data, device=device)
        self._hessians = {name: H.cpu() for name, H in self._hessians.items()}

        patcher = ModelPatcher(self)
        return patcher.patch(model)

    # ------------------------------------------------------------------
    # Layer-level quantization (called by ModelPatcher)
    # ------------------------------------------------------------------

    def quantize_layer(self, linear: nn.Linear, name: str) -> QuantizedLinear:
        H = self._hessians.get(name)
        if H is None:
            raise RuntimeError(
                f"No Hessian found for layer '{name}'. "
                "This should not happen — check that CalibrationRunner and "
                "ModelPatcher use the same _should_quantize predicate."
            )
        cfg = self._config_for_layer(name)
        weight = linear.weight.data.float()

        # Pre-compute scales from the ORIGINAL (unmodified) weight matrix.
        # These are fixed; the GPTQ loop uses them for column-level rounding.
        scale, zero = compute_scale(
            weight,
            cfg.effective_dtype,
            granularity=cfg.granularity,
            group_size=cfg.group_size,
            sym=cfg.sym,
        )

        q_int = _gptq_one_layer(
            W=weight.clone(),
            H=H.clone().to(weight.device),
            scale=scale,
            zero=zero,
            dtype=cfg.effective_dtype,
            granularity=cfg.granularity,
            group_size=cfg.group_size,
            damp_percent=cfg.damp_percent,
            act_order=cfg.act_order,
        )

        return QuantizedLinear(
            q_weight=q_int,
            scale=scale,
            zero=zero,
            config=cfg,
            bias=linear.bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _gptq_one_layer(
    W: Tensor,
    H: Tensor,
    scale: Tensor,
    zero: Tensor,
    dtype: QuantDtype,
    granularity: str,
    group_size: int,
    damp_percent: float,
    act_order: bool,
) -> Tensor:
    """Run the GPTQ column-wise quantization for one Linear layer.

    Args:
        W:             Weight matrix [out, in] in float32.
        H:             Hessian [in, in] in float32.
        scale / zero:  Pre-computed from the original W (not modified here).
        dtype:         Target integer dtype.
        granularity:   "per_tensor" | "per_channel" | "per_group".
        group_size:    Elements per group (per_group only).
        damp_percent:  Fraction of mean(diag(H)) added as diagonal damping.
        act_order:     If True, sort columns by descending Hessian diagonal.

    Returns:
        Q_int [out, in] int32 — integer-quantized weights in ORIGINAL column order.
    """
    out_features, in_features = W.shape
    qmin = float(dtype.qmin)
    qmax = float(dtype.qmax)

    # --- Handle dead input neurons (never activated during calibration) ---
    dead = H.diag() == 0.0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # --- Diagonal damping for numerical stability ---
    damp = damp_percent * H.diag().mean()
    H.diagonal().add_(damp)

    # --- Optional activation ordering: process most-sensitive columns first ---
    perm: Tensor | None = None
    if act_order:
        perm = H.diag().argsort(descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        # Scale/zero keep their ORIGINAL column association (see docstring).

    # --- Cholesky: H → H⁻¹ → upper Cholesky factor of H⁻¹ ---
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        C = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        # Cholesky failed even after damping; increase damping and retry.
        H.diagonal().add_(damp * 10.0)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        C = torch.linalg.cholesky(Hinv, upper=True)

    # Q_int stores integer quantized values in (possibly permuted) column order.
    Q_int = torch.zeros(out_features, in_features, dtype=torch.int32, device=W.device)

    for block_start in range(0, in_features, _BLOCK_SIZE):
        block_end = min(block_start + _BLOCK_SIZE, in_features)
        count = block_end - block_start

        W_block = W[:, block_start:block_end].clone()   # [out, count]
        E_block = torch.zeros_like(W_block)             # quantization errors
        C_block = C[block_start:block_end, block_start:block_end]  # [count, count]

        for k in range(count):
            col_perm = block_start + k  # index in (permuted) weight matrix

            # Map to original column to look up the correct scale/zero group.
            col_orig = int(perm[col_perm].item()) if perm is not None else col_perm

            w = W_block[:, k]                        # [out]
            d = C_block[k, k]                        # Cholesky diagonal element

            s, z = _col_scale_zero(scale, zero, col_orig, granularity, group_size)

            # Integer quantization of this column.
            q_int = (w / s + z).round().clamp(qmin, qmax).to(torch.int32)
            Q_int[:, col_perm] = q_int

            # Dequantize to compute the correction signal.
            q_fp = (q_int.float() - z) * s          # [out]

            # Normalised error: e = (w - q) / C[k,k]
            e = (w - q_fp) / d                       # [out]
            E_block[:, k] = e

            # Update remaining columns in the current block.
            if k + 1 < count:
                W_block[:, k + 1:] -= e.unsqueeze(1).matmul(
                    C_block[k : k + 1, k + 1:]      # [1, count-k-1]
                )

        # Lazy update: propagate block errors to all columns to the right.
        if block_end < in_features:
            W[:, block_end:] -= E_block.matmul(C[block_start:block_end, block_end:])

    # --- Undo column permutation so Q_int is in original weight ordering ---
    if perm is not None:
        invperm = torch.argsort(perm)
        Q_int = Q_int[:, invperm]

    return Q_int


def _col_scale_zero(
    scale: Tensor,
    zero: Tensor,
    col_orig: int,
    granularity: str,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Extract the scale and zero for a single column of the weight matrix.

    Args:
        scale / zero: Pre-computed tensors (shapes depend on granularity).
        col_orig:     ORIGINAL column index (before any permutation).
        granularity:  "per_tensor" | "per_channel" | "per_group".
        group_size:   Used only for per_group.

    Returns:
        (s, z) broadcastable against a column vector [out_features].
    """
    if granularity == "per_tensor":
        return scale, zero          # scalar or 0-d

    if granularity == "per_channel":
        # scale: [out, 1] → take first (and only) column
        return scale[:, 0], zero[:, 0]   # [out]

    # per_group: scale is [out, n_groups, 1]
    g = col_orig // group_size
    return scale[:, g, 0], zero[:, g, 0]   # [out]
