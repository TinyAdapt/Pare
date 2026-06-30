"""Tests for the Triton INT4 W4A16 matmul kernel.

Correctness tests run on CPU (no Triton); integration + throughput tests
require CUDA and Triton, and are skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pare.config import QuantConfig
from pare.core.pack import pack_int4, repack_int4_for_kernel, unpack_int4
from pare.kernels.matmul_int4 import triton_available
from pare.layers.linear import QuantizedLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cuda_and_triton = pytest.mark.skipif(
    not triton_available(),
    reason="Triton + CUDA not available",
)

GROUP_SIZE = 128


def _make_ql(out: int = 256, in_: int = 256, seed: int = 0) -> QuantizedLinear:
    torch.manual_seed(seed)
    lin = nn.Linear(in_, out, bias=False)
    cfg = QuantConfig(bits=4, scheme="rtn", granularity="per_group", group_size=GROUP_SIZE)
    return QuantizedLinear.from_linear(lin, cfg)


# ---------------------------------------------------------------------------
# 1. repack_int4_for_kernel — CPU unit tests
# ---------------------------------------------------------------------------

class TestRepackInt4:
    def test_output_shape(self):
        N, K = 16, 256
        q = torch.randint(0, 16, (N, K), dtype=torch.int32)
        packed = pack_int4(q)               # [N, K//2]
        repacked = repack_int4_for_kernel(packed, group_size=128)
        assert repacked.shape == packed.shape

    def test_output_dtype(self):
        q = torch.randint(0, 16, (8, 128), dtype=torch.int32)
        repacked = repack_int4_for_kernel(pack_int4(q), group_size=128)
        assert repacked.dtype == torch.uint8

    def test_roundtrip_via_unpack(self):
        """repack then decode should give same weights as original."""
        torch.manual_seed(42)
        N, K, G = 8, 256, 128
        q_orig = torch.randint(0, 16, (N, K), dtype=torch.int32)
        packed  = pack_int4(q_orig)
        repacked = repack_int4_for_kernel(packed, group_size=G)

        # Decode the repacked format manually (lo = first half, hi = second half)
        q_decoded = torch.zeros(N, K, dtype=torch.int32)
        HALF = G // 2
        n_tiles = K // G
        for t in range(n_tiles):
            b_start = t * HALF
            k_start = t * G
            lo_bytes = repacked[:, b_start : b_start + HALF].int()
            q_decoded[:, k_start : k_start + HALF]      = lo_bytes & 0xF
            q_decoded[:, k_start + HALF : k_start + G]  = (lo_bytes >> 4) & 0xF

        assert torch.equal(q_orig, q_decoded)

    def test_different_groups_independent(self):
        """Weights from different groups should not bleed across the repack boundary."""
        N, K, G = 4, 256, 128
        q = torch.zeros(N, K, dtype=torch.int32)
        q[:, :G] = 5          # first group: all 5
        q[:, G:] = 9          # second group: all 9
        packed   = pack_int4(q)
        repacked = repack_int4_for_kernel(packed, G)

        # First group bytes (positions 0..G//2-1): lo=5, hi=5
        first = repacked[:, : G // 2]
        assert (first & 0xF).eq(5).all()
        assert ((first >> 4) & 0xF).eq(5).all()

        # Second group bytes (positions G//2..G-1): lo=9, hi=9
        second = repacked[:, G // 2 : G]
        assert (second & 0xF).eq(9).all()
        assert ((second >> 4) & 0xF).eq(9).all()

    def test_idempotent_on_diagonal_weights(self):
        """All-zero weights should repack to all-zero bytes."""
        N, K = 8, 256
        q = torch.zeros(N, K, dtype=torch.int32)
        packed   = pack_int4(q)
        repacked = repack_int4_for_kernel(packed, 128)
        assert repacked.eq(0).all()


# ---------------------------------------------------------------------------
# 2. matmul_w4a16 — correctness (CUDA + Triton)
# ---------------------------------------------------------------------------

class TestMatmulW4A16Correctness:
    @cuda_and_triton
    def test_output_shape(self):
        from pare.kernels.matmul_int4 import matmul_w4a16
        ql = _make_ql(256, 256).cuda()
        w = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)
        x = torch.randn(32, 256, device="cuda", dtype=torch.float16)
        y = matmul_w4a16(x, w, ql.scale, ql.zero, GROUP_SIZE)
        assert y.shape == (32, 256)

    @cuda_and_triton
    def test_output_dtype(self):
        from pare.kernels.matmul_int4 import matmul_w4a16
        ql = _make_ql(128, 256).cuda()
        w = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)
        x = torch.randn(16, 256, device="cuda", dtype=torch.float16)
        y = matmul_w4a16(x, w, ql.scale, ql.zero, GROUP_SIZE)
        assert y.dtype == torch.float16

    @cuda_and_triton
    @pytest.mark.parametrize("m", [1, 4, 16, 64])
    def test_matches_dequant_on_the_fly(self, m):
        """Kernel output must match dequant-then-matmul within FP16 tolerance."""
        from pare.kernels.matmul_int4 import matmul_w4a16
        torch.manual_seed(1)
        ql = _make_ql(256, 256).cuda()
        w_kernel = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)

        x = torch.randn(m, 256, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            y_ref = ql.forward(x).float()   # dequant-on-the-fly (fp16→fp32 for comparison)
            y_kernel = matmul_w4a16(x, w_kernel, ql.scale, ql.zero, GROUP_SIZE).float()

        # Allow ≤0.05 absolute error (float16 accumulation noise)
        max_err = (y_ref - y_kernel).abs().max().item()
        assert max_err < 0.05, f"max_err={max_err:.5f} too large at m={m}"

    @cuda_and_triton
    def test_zero_input_gives_zero_output(self):
        from pare.kernels.matmul_int4 import matmul_w4a16
        ql = _make_ql(128, 256).cuda()
        w = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)
        x = torch.zeros(8, 256, device="cuda", dtype=torch.float16)
        y = matmul_w4a16(x, w, ql.scale, ql.zero, GROUP_SIZE)
        assert y.abs().max().item() == 0.0

    @cuda_and_triton
    def test_non_divisible_m(self):
        """M not divisible by BLOCK_M should still produce correct output."""
        from pare.kernels.matmul_int4 import matmul_w4a16
        torch.manual_seed(7)
        ql = _make_ql(256, 256).cuda()
        w  = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)

        for m in [1, 3, 7, 31]:
            x = torch.randn(m, 256, device="cuda", dtype=torch.float16)
            y_ref    = ql.forward(x).float()
            y_kernel = matmul_w4a16(x, w, ql.scale, ql.zero, GROUP_SIZE).float()
            max_err  = (y_ref - y_kernel).abs().max().item()
            assert max_err < 0.05, f"m={m}: max_err={max_err:.5f}"


# ---------------------------------------------------------------------------
# 3. QuantizedLinear.use_kernel integration
# ---------------------------------------------------------------------------

class TestQuantizedLinearKernel:
    @cuda_and_triton
    def test_use_kernel_flag(self):
        """use_kernel=True dispatches to the Triton kernel path."""
        ql = _make_ql(256, 256)
        ql.use_kernel = True
        ql = ql.cuda()

        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            y = ql(x)
        assert y.shape == (4, 256)
        assert not y.isnan().any()

    @cuda_and_triton
    def test_kernel_matches_dequant(self):
        """kernel path must agree with standard dequant path within tolerance."""
        torch.manual_seed(3)
        ql_std    = _make_ql(256, 256).cuda()
        ql_kernel = _make_ql(256, 256).cuda()
        ql_kernel.use_kernel = True

        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            y_std    = ql_std(x).float()
            y_kernel = ql_kernel(x).float()

        max_err = (y_std - y_kernel).abs().max().item()
        assert max_err < 0.05, f"max_err={max_err:.5f}"

    @cuda_and_triton
    def test_lazy_repack_cached(self):
        """_packed_weight_kernel is created on first call and reused."""
        ql = _make_ql(128, 256)
        ql.use_kernel = True
        ql = ql.cuda()

        x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
        assert not hasattr(ql, "_packed_weight_kernel")

        with torch.no_grad():
            ql(x)
        assert hasattr(ql, "_packed_weight_kernel")

        # Second call reuses the same buffer
        buf_id = id(ql._packed_weight_kernel)
        with torch.no_grad():
            ql(x)
        assert id(ql._packed_weight_kernel) == buf_id

    @cuda_and_triton
    def test_3d_input_reshape(self):
        """[batch, seq, in_features] input should be correctly handled."""
        ql = _make_ql(256, 256)
        ql.use_kernel = True
        ql = ql.cuda()

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            y = ql(x)
        assert y.shape == (2, 64, 256)

    @cuda_and_triton
    def test_cpu_fallback(self):
        """use_kernel=True on CPU must fall back to the dequant path without error."""
        ql = _make_ql(64, 128)
        ql.use_kernel = True   # kernel flag set, but tensor is on CPU

        x = torch.randn(4, 128)
        with torch.no_grad():
            y = ql(x)   # should use dequant path (x.is_cuda is False)
        assert y.shape == (4, 64)


# ---------------------------------------------------------------------------
# 4. Throughput gate (informational — not a hard test failure)
# ---------------------------------------------------------------------------

class TestThroughputGate:
    @cuda_and_triton
    @pytest.mark.slow
    def test_speedup_at_bs4(self):
        """Triton kernel should be ≥ 1.5× faster than dequant-on-the-fly at BS=4."""
        import time
        from pare.kernels.matmul_int4 import matmul_w4a16

        N_WARMUP, N_TIMED = 3, 10
        ql = _make_ql(4096, 4096).cuda()
        w  = repack_int4_for_kernel(ql.packed_weight, GROUP_SIZE)
        x  = torch.randn(4 * 128, 4096, device="cuda", dtype=torch.float16)

        def _bench(fn):
            for _ in range(N_WARMUP):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_TIMED):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / N_TIMED * 1000

        with torch.no_grad():
            ms_dq = _bench(lambda: ql.forward(x))
            ms_k  = _bench(lambda: matmul_w4a16(x, w, ql.scale, ql.zero, GROUP_SIZE))

        speedup = ms_dq / ms_k
        assert speedup >= 1.5, (
            f"Triton kernel speedup {speedup:.2f}× < 1.5× gate at BS=4 "
            f"(dequant={ms_dq:.2f}ms, kernel={ms_k:.2f}ms)"
        )
