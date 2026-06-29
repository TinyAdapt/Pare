"""Quantization data types with their integer range metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class _DTypeSpec:
    bits: int
    is_float: bool
    signed: bool

    @property
    def qmin(self) -> int:
        if self.is_float:
            raise TypeError("float dtypes have no integer qmin/qmax")
        return -(2 ** (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        if self.is_float:
            raise TypeError("float dtypes have no integer qmin/qmax")
        return (2 ** (self.bits - 1) - 1) if self.signed else (2**self.bits - 1)

    @property
    def levels(self) -> int:
        return 2**self.bits


class QuantDtype(Enum):
    """Supported quantization dtypes.

    Integer types: INT2–INT8.  Per-group asymmetric INT4 is the workhorse
    for weight-only quantization (GPTQ, AWQ).  INT8 is the standard for
    weight+activation schemes (SmoothQuant, LLM.int8).

    Float types: FP8 variants native on A100/H100.  NF4 is the 4-bit normal
    float from QLoRA — not a uniform grid, but a non-uniform one optimized
    for normally-distributed weights.
    """

    INT2 = _DTypeSpec(bits=2, is_float=False, signed=False)
    INT3 = _DTypeSpec(bits=3, is_float=False, signed=False)
    INT4 = _DTypeSpec(bits=4, is_float=False, signed=False)
    INT8 = _DTypeSpec(bits=8, is_float=False, signed=True)
    FP8_E4M3 = _DTypeSpec(bits=8, is_float=True, signed=True)
    FP8_E5M2 = _DTypeSpec(bits=8, is_float=True, signed=True)
    NF4 = _DTypeSpec(bits=4, is_float=True, signed=True)

    # ------------------------------------------------------------------
    # Convenience properties forwarded from the spec
    # ------------------------------------------------------------------

    @property
    def bits(self) -> int:
        return self.value.bits

    @property
    def is_float(self) -> bool:
        return self.value.is_float

    @property
    def signed(self) -> bool:
        return self.value.signed

    @property
    def qmin(self) -> int:
        return self.value.qmin

    @property
    def qmax(self) -> int:
        return self.value.qmax

    @property
    def levels(self) -> int:
        return self.value.levels

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_bits(cls, bits: int, *, signed: bool = False) -> "QuantDtype":
        """Return the standard integer dtype for a given bit-width."""
        mapping = {
            (2, False): cls.INT2,
            (3, False): cls.INT3,
            (4, False): cls.INT4,
            (8, True): cls.INT8,
            (8, False): cls.INT8,  # treat unsigned-8 as INT8 (symmetric uses signed)
        }
        key = (bits, signed or bits == 8)
        if key not in mapping:
            raise ValueError(f"No standard dtype for bits={bits}, signed={signed}")
        return mapping[key]
