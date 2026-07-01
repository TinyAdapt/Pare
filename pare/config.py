"""QuantConfig — the single configuration object for Pare.

Designed as a plain Python dataclass: no HuggingFace dependency.
Works with any PyTorch model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pare.core.dtype import QuantDtype


@dataclass
class QuantConfig:
    """Configuration for a quantization run.

    Args:
        bits:        Target bit-width (2, 3, 4, 8).
        dtype:       Explicit QuantDtype; inferred from ``bits`` if None.
        scheme:      Algorithm: ``"rtn"`` | ``"gptq"`` | ``"awq"`` |
                     ``"smoothquant"``.
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        group_size:  Elements per quantization group (per_group only).
        sym:         Symmetric quantization (zero_point = 0).
        modules:     List of regex patterns matching module names to quantize.
                     If None, all ``nn.Linear`` layers are quantized.
        exclude:     Module name substrings / patterns to skip.

    Calibration:
        n_calibration_samples:  Number of calibration sequences.
        calibration_seq_len:    Token length of each calibration sequence.

    GPTQ-specific:
        damp_percent:  Diagonal damping for Hessian inversion (fraction of
                       mean diagonal value).  Prevents numerical issues.
        act_order:     Sort columns by decreasing Hessian diagonal before
                       quantizing (usually improves quality slightly).

    AWQ-specific:
        awq_n_grid:    Number of scale candidates to search per channel.
    """

    # Core
    bits: int = 4
    dtype: QuantDtype | None = None
    scheme: str = "awq"
    granularity: str = "per_group"
    group_size: int = 128
    sym: bool = False
    modules: list[str] | None = None
    exclude: list[str] = field(default_factory=lambda: ["lm_head"])

    # Calibration
    n_calibration_samples: int = 128
    calibration_seq_len: int = 2048
    calibration: str = "absmax"         # "absmax" | "percentile" | "mse"
    calibration_percentile: float = 99.99

    # GPTQ
    damp_percent: float = 0.01
    act_order: bool = False

    # AWQ
    awq_n_grid: int = 20

    # SmoothQuant
    smooth_alpha: float = 0.5   # migration strength: 0 = no migration, 1 = full migration to weights

    # Mixed-precision sensitivity
    # Layers whose activation-weighted reconstruction error exceeds
    # sensitivity_threshold are re-quantized at sensitive_bits instead of bits.
    sensitive_bits: int | None = None
    sensitivity_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.dtype is None:
            self.dtype = QuantDtype.from_bits(self.bits)
        if self.scheme not in {"rtn", "gptq", "awq", "smoothquant"}:
            raise ValueError(f"Unknown scheme: {self.scheme!r}")
        if self.calibration not in {"absmax", "percentile", "mse"}:
            raise ValueError(f"Unknown calibration: {self.calibration!r}")
        if self.granularity not in {"per_tensor", "per_channel", "per_group"}:
            raise ValueError(f"Unknown granularity: {self.granularity!r}")
        if self.bits not in {2, 3, 4, 8}:
            raise ValueError(f"Unsupported bits: {self.bits}")
        # NF4 forces per_channel granularity (absmax per row) and bits=4.
        if self.dtype == QuantDtype.NF4:
            self.bits = 4
            self.granularity = "per_channel"
        # FP8 is always per-channel symmetric (no zero-point).
        if self.dtype in (QuantDtype.FP8_E4M3, QuantDtype.FP8_E5M2):
            self.bits = 8
            self.granularity = "per_channel"
            self.sym = True

    @property
    def effective_dtype(self) -> QuantDtype:
        assert self.dtype is not None
        return self.dtype
