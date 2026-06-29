"""Triton INT4 kernel correctness + throughput benchmark.

Tests:
  1. Correctness: kernel output matches dequant-on-the-fly within tolerance.
  2. Throughput: kernel vs dequant-on-the-fly at batch sizes 1, 4, 16, 32.

Gate: ≥ 1.5× speedup at BS=4.

Usage:
    python benchmarks/bench_triton_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from pare.config import QuantConfig
from pare.core.dtype import QuantDtype
from pare.core.pack import repack_int4_for_kernel
from pare.kernels.matmul_int4 import matmul_w4a16, triton_available
from pare.layers.linear import QuantizedLinear


DEVICE = "cuda"
DTYPE  = torch.float16

# Representative LLaMA-2-7B attention projection shape
OUT_FEATURES = 4096
IN_FEATURES  = 4096
GROUP_SIZE   = 128
SEQ_LEN      = 128
N_WARMUP     = 5
N_TIMED      = 20


def make_layer(out: int = OUT_FEATURES, in_: int = IN_FEATURES) -> QuantizedLinear:
    torch.manual_seed(0)
    linear = nn.Linear(in_, out, bias=False)
    cfg = QuantConfig(bits=4, scheme="rtn", granularity="per_group", group_size=GROUP_SIZE)
    ql = QuantizedLinear.from_linear(linear, cfg)
    return ql.to(DEVICE)


def bench(fn, warmup: int = N_WARMUP, timed: int = N_TIMED) -> float:
    """Return mean latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / timed * 1000  # ms


# ---------------------------------------------------------------------------
# 1. Correctness check
# ---------------------------------------------------------------------------

def check_correctness():
    print("=" * 60)
    print("Correctness check")
    print("=" * 60)
    ql = make_layer()

    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, SEQ_LEN, IN_FEATURES, device=DEVICE, dtype=DTYPE)

        # Reference: dequant-on-the-fly
        with torch.no_grad():
            y_ref = ql.forward(x)  # use_kernel=False (default)

        # Kernel path — uses kernel-layout repacked weight
        x_2d = x.reshape(-1, IN_FEATURES)
        w_kernel = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)
        with torch.no_grad():
            y_kernel = matmul_w4a16(
                x_2d,
                w_kernel,
                ql.scale,
                ql.zero,
                group_size=GROUP_SIZE,
            ).reshape(batch_size, SEQ_LEN, OUT_FEATURES)

        max_err = (y_ref - y_kernel).abs().max().item()
        rel_err = (y_ref - y_kernel).norm() / y_ref.norm().clamp(min=1e-8)
        status = "PASS" if max_err < 0.1 else "FAIL"
        print(f"  BS={batch_size:2d}  max_err={max_err:.4f}  rel_err={rel_err:.4f}  [{status}]")

    print()


# ---------------------------------------------------------------------------
# 2. Throughput benchmark
# ---------------------------------------------------------------------------

def bench_throughput():
    print("=" * 60)
    print(f"Throughput benchmark  (out={OUT_FEATURES}, in={IN_FEATURES}, gs={GROUP_SIZE})")
    print(f"Warmup={N_WARMUP}  Timed={N_TIMED} runs each")
    print("=" * 60)
    print(f"{'BS':>4}  {'Dequant ms':>12}  {'Kernel ms':>12}  {'Speedup':>10}  {'Gate ≥1.5×':>12}")
    print("-" * 60)

    ql = make_layer()
    w_kernel = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)

    gate_passed = False
    for batch_size in [1, 4, 16, 32]:
        x = torch.randn(batch_size, SEQ_LEN, IN_FEATURES, device=DEVICE, dtype=DTYPE)
        x_2d = x.reshape(-1, IN_FEATURES).contiguous()

        with torch.no_grad():
            ms_dequant = bench(lambda: ql.forward(x))
            ms_kernel  = bench(lambda: matmul_w4a16(
                x_2d, w_kernel, ql.scale, ql.zero, GROUP_SIZE
            ))

        speedup = ms_dequant / ms_kernel
        if batch_size == 4 and speedup >= 1.5:
            gate_passed = True
        gate_str = "✓ PASS" if speedup >= 1.5 else "✗ FAIL" if batch_size == 4 else ""
        print(f"{batch_size:>4}  {ms_dequant:>12.3f}  {ms_kernel:>12.3f}  {speedup:>10.2f}×  {gate_str:>12}")

    print()
    if gate_passed:
        print("✓  Gate PASSED: ≥ 1.5× speedup at BS=4")
    else:
        print("✗  Gate FAILED: < 1.5× speedup at BS=4")
    print()


# ---------------------------------------------------------------------------
# 3. Memory savings
# ---------------------------------------------------------------------------

def show_memory():
    print("=" * 60)
    print("Memory comparison (per layer)")
    print("=" * 60)
    ql = make_layer()
    fp16_bytes   = OUT_FEATURES * IN_FEATURES * 2
    int4_bytes   = ql.packed_weight.numel()  # uint8, 2 INT4 per byte
    scale_bytes  = ql.scale.numel() * 4
    total_int4   = int4_bytes + scale_bytes

    print(f"  FP16 weights:      {fp16_bytes / 1e6:.2f} MB")
    print(f"  INT4 packed:       {int4_bytes / 1e6:.2f} MB")
    print(f"  Scales/zeros:      {scale_bytes / 1e6:.2f} MB")
    print(f"  INT4 total:        {total_int4 / 1e6:.2f} MB")
    print(f"  Compression ratio: {fp16_bytes / total_int4:.2f}×")
    print()


if __name__ == "__main__":
    if not triton_available():
        print("ERROR: Triton not available. Install with: pip install triton")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}\n")

    check_correctness()
    bench_throughput()
    show_memory()
