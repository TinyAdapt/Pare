"""Triton INT4 W4A16 matmul kernel.

Computes Y = X @ W.T where W is stored as asymmetric INT4 in a kernel-friendly
packed layout (see ``pare.core.pack.repack_int4_for_kernel``).

Key optimisations vs the dequantize-on-the-fly path
----------------------------------------------------
1. Both halves of X are loaded as contiguous [BLOCK_M, BLOCK_K//2] slices,
   not as strided every-other accesses — full memory bandwidth utilisation.
2. W is read from its packed uint8 form directly (4× smaller than FP16),
   dequantised in registers, never written back as a full FP16 matrix.
3. Software pipelining (num_stages=4) overlaps the next tile's memory loads
   with the current tile's tensor-core computation.
4. @triton.autotune selects the best (BLOCK_M, BLOCK_N) per shape; BLOCK_K is
   fixed to group_size to ensure exactly one scale per K-tile.

Weight layout (kernel-friendly, vs storage layout)
---------------------------------------------------
Storage (``pack_int4``): byte j at row n holds pair (q[n, 2j], q[n, 2j+1]).
Kernel  (``repack_int4_for_kernel``): within each group_size-wide tile starting
at column k, byte j_local at row n holds:
    low  nibble → q[n, k + j_local]                  (first  half of tile)
    high nibble → q[n, k + j_local + group_size//2]  (second half of tile)

This means X_lo = X[:, k : k+BLOCK_K//2] and X_hi = X[:, k+BLOCK_K//2 : k+BLOCK_K]
are both contiguous loads; no register splitting required.

Usage
-----
1. Call ``repack_int4_for_kernel(packed_weight, group_size)`` once offline.
2. Pass the result to ``matmul_w4a16`` instead of the storage-layout weight.

Speedup target: ≥ 1.5× over dequant-on-the-fly at batch size ≥ 4 on A40.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128}, num_stages=4, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _w4a16_kernel(
        # ---- pointers ----
        x_ptr,           # [M, K]        fp16 activations
        w_ptr,           # [N, K//2]     uint8 kernel-layout packed weights
        scales_ptr,      # [N, n_groups] fp32
        zeros_ptr,       # [N, n_groups] fp32
        y_ptr,           # [M, N]        fp16 output
        # ---- dimensions ----
        M, N, K,
        # ---- strides ----
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sg,
        stride_ym, stride_yn,
        # ---- compile-time constants ----
        group_size: tl.constexpr,   # BLOCK_K = group_size (one scale per tile)
        BLOCK_M:    tl.constexpr,
        BLOCK_N:    tl.constexpr,
    ):
        """Each program computes one [BLOCK_M, BLOCK_N] tile of Y = X @ W.T.

        BLOCK_K is fixed to group_size so that every K-tile belongs to exactly
        one quantisation group (no cross-group scale boundary within a tile).
        """
        BLOCK_K: tl.constexpr = group_size          # e.g. 128
        HALF_K:  tl.constexpr = group_size // 2     # e.g. 64

        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        half_idx = tl.arange(0, HALF_K)   # [0, 1, ..., HALF_K-1]

        for k_tile in range(tl.cdiv(K, BLOCK_K)):
            k_base = k_tile * BLOCK_K

            # ------------------------------------------------------------------
            # Load X in two contiguous [BLOCK_M, HALF_K] slices.
            # X_lo covers columns [k_base, k_base + HALF_K).
            # X_hi covers columns [k_base + HALF_K, k_base + BLOCK_K).
            # Both are stride-1 in K → fully coalesced memory access.
            # ------------------------------------------------------------------
            offs_k_lo = k_base + half_idx               # contiguous first half
            offs_k_hi = k_base + HALF_K + half_idx      # contiguous second half
            mask_lo = offs_k_lo < K
            mask_hi = offs_k_hi < K

            x_lo = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k_lo[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_lo[None, :],
                other=0.0,
            ).to(tl.float16)   # [BLOCK_M, HALF_K]

            x_hi = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k_hi[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_hi[None, :],
                other=0.0,
            ).to(tl.float16)   # [BLOCK_M, HALF_K]

            # ------------------------------------------------------------------
            # Load packed weights: [BLOCK_N, HALF_K] uint8 (kernel layout).
            # Byte at (n, j_local): lo nibble → w[n, k_base + j_local]
            #                        hi nibble → w[n, k_base + HALF_K + j_local]
            # ------------------------------------------------------------------
            offs_kp = k_base // 2 + half_idx   # byte indices in packed weight row
            mask_kp = offs_kp < K // 2

            w_packed = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_kp[None, :],
                other=0,
            )   # [BLOCK_N, HALF_K] uint8

            w_lo = (w_packed & 0xF).to(tl.float32)           # first  half weights
            w_hi = ((w_packed >> 4) & 0xF).to(tl.float32)    # second half weights

            # ------------------------------------------------------------------
            # Per-group dequantisation.  Exactly one group per K-tile.
            # ------------------------------------------------------------------
            g = k_tile   # group index = tile index when BLOCK_K == group_size

            scale = tl.load(
                scales_ptr + offs_n * stride_sn + g * stride_sg,
                mask=mask_n, other=1.0,
            ).to(tl.float32)[:, None]   # [BLOCK_N, 1]

            zero = tl.load(
                zeros_ptr + offs_n * stride_sn + g * stride_sg,
                mask=mask_n, other=0.0,
            ).to(tl.float32)[:, None]   # [BLOCK_N, 1]

            w_lo_f = ((w_lo - zero) * scale).to(tl.float16)  # [BLOCK_N, HALF_K]
            w_hi_f = ((w_hi - zero) * scale).to(tl.float16)

            # ------------------------------------------------------------------
            # Accumulate:  Y += X_lo @ W_lo.T  +  X_hi @ W_hi.T
            # Each tl.dot: [BLOCK_M, HALF_K] × [HALF_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
            # HALF_K = 64 gives good tensor-core utilisation on A40.
            # ------------------------------------------------------------------
            acc = tl.dot(x_lo, tl.trans(w_lo_f), acc=acc, out_dtype=tl.float32)
            acc = tl.dot(x_hi, tl.trans(w_hi_f), acc=acc, out_dtype=tl.float32)

        # Write output tile.
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def matmul_w4a16(
    x: "torch.Tensor",
    packed_weight: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int,
) -> "torch.Tensor":
    """Fused INT4 dequant + matmul via Triton.

    Args:
        x:             Activation tensor [M, K] in float16.  M = batch × seq_len.
        packed_weight: *Kernel-layout* packed uint8 weight [N, K//2].
                       Must be the output of ``repack_int4_for_kernel``, NOT the
                       raw output of ``pack_int4``.
        scales:        Scale tensor, shape [N], [N, n_groups], or [N, n_groups, 1].
        zeros:         Zero-point tensor, same shape as scales.
        group_size:    INT4 quantisation group size (e.g. 128); also used as
                       BLOCK_K in the kernel — must equal the repack group_size.

    Returns:
        Output [M, N] float16.

    Raises:
        RuntimeError: If Triton is not installed or no CUDA device is available.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. `pip install triton`.")
    if not x.is_cuda:
        raise RuntimeError("matmul_w4a16 requires CUDA tensors.")
    assert x.is_contiguous(), "x must be contiguous"
    assert x.dtype == torch.float16, f"x must be float16, got {x.dtype}"

    M, K = x.shape
    N, K_half = packed_weight.shape
    assert K == K_half * 2, f"packed_weight has {K_half} columns, expected K//2={K//2}"

    def _to_2d(t: "torch.Tensor") -> "torch.Tensor":
        t = t.float()
        if t.dim() == 1:
            return t.unsqueeze(1)
        if t.dim() == 3:
            return t.squeeze(2)
        return t

    scales_2d = _to_2d(scales).contiguous()
    zeros_2d  = _to_2d(zeros).contiguous()

    y = torch.empty(M, N, device=x.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _w4a16_kernel[grid](
        x, packed_weight, scales_2d, zeros_2d, y,
        M, N, K,
        x.stride(0),             x.stride(1),
        packed_weight.stride(0), packed_weight.stride(1),
        scales_2d.stride(0),     scales_2d.stride(1),
        y.stride(0),             y.stride(1),
        group_size=group_size,
    )
    return y


def triton_available() -> bool:
    """Return True if Triton is installed and a CUDA device is available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()
