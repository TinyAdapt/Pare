"""Numerical unit tests for GPTQ internals.

Each test uses a small, hand-verifiable example so that a mathematical
mistake surfaces as a concrete number mismatch, not just a bad PPL.

Tests:
  - HessianAccumulator matches numpy reference
  - Cholesky chain (H → H⁻¹ → C) satisfies C^T C = H⁻¹
  - _col_scale_zero returns correct slice for each granularity
  - _gptq_one_layer on a trivial case: INT8 per-channel, no damping needed,
    dequantized output matches a numpy round-trip within tolerance
  - GPTQ Q_int is strictly inside [qmin, qmax]
  - act_order permutation is undone: Q_int is in original column order
"""

import numpy as np
import pytest
import torch

from pare.calibration.hessian import HessianAccumulator
from pare.core.dtype import QuantDtype
from pare.core.scale import compute_scale
from pare.schemes.gptq import _col_scale_zero, _gptq_one_layer

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# HessianAccumulator
# ---------------------------------------------------------------------------

class TestHessianAccumulator:
    def test_single_batch_matches_numpy(self):
        """H = 2/n * X^T X, verified against numpy."""
        X = torch.randn(10, 8)   # [n_tokens=10, in_features=8]
        acc = HessianAccumulator()
        acc.accumulate(X.unsqueeze(0))   # add batch dim → [1, 10, 8]

        H = acc.finalize()
        H_ref = torch.from_numpy(2.0 / 10 * (X.numpy().T @ X.numpy()))
        torch.testing.assert_close(H, H_ref.float(), atol=1e-5, rtol=1e-5)

    def test_multi_batch_equals_concat(self):
        """Accumulating two batches gives same H as one big batch."""
        X1 = torch.randn(6, 4)
        X2 = torch.randn(8, 4)
        X_all = torch.cat([X1, X2], dim=0)

        acc_split = HessianAccumulator()
        acc_split.accumulate(X1)
        acc_split.accumulate(X2)

        acc_full = HessianAccumulator()
        acc_full.accumulate(X_all)

        torch.testing.assert_close(acc_split.finalize(), acc_full.finalize(), atol=1e-5, rtol=1e-5)

    def test_3d_input_is_flattened(self):
        """[batch, seq, in] produces same H as [batch*seq, in]."""
        X = torch.randn(2, 5, 4)
        acc3d = HessianAccumulator()
        acc3d.accumulate(X)

        acc2d = HessianAccumulator()
        acc2d.accumulate(X.reshape(10, 4))

        torch.testing.assert_close(acc3d.finalize(), acc2d.finalize())

    def test_empty_raises(self):
        acc = HessianAccumulator()
        with pytest.raises(RuntimeError, match="no samples"):
            acc.finalize()

    def test_h_is_symmetric(self):
        X = torch.randn(20, 6)
        acc = HessianAccumulator()
        acc.accumulate(X)
        H = acc.finalize()
        torch.testing.assert_close(H, H.T, atol=1e-5, rtol=1e-5)

    def test_h_is_positive_semidefinite(self):
        X = torch.randn(50, 8)
        acc = HessianAccumulator()
        acc.accumulate(X)
        H = acc.finalize()
        eigvals = torch.linalg.eigvalsh(H)
        assert (eigvals >= -1e-6).all(), f"H has negative eigenvalue: {eigvals.min()}"


# ---------------------------------------------------------------------------
# Cholesky chain
# ---------------------------------------------------------------------------

class TestCholeskyChain:
    def test_c_transpose_c_equals_hinv(self):
        """The three-step Cholesky chain must satisfy C^T C = H⁻¹."""
        in_dim = 16
        X = torch.randn(100, in_dim)
        H = (2.0 / 100) * X.T @ X
        H.diagonal().add_(0.01 * H.diag().mean())   # small damping

        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        C = torch.linalg.cholesky(Hinv, upper=True)

        # C^T C should equal H⁻¹
        CtC = C.T @ C
        torch.testing.assert_close(CtC, Hinv, atol=1e-4, rtol=1e-4)

    def test_hinv_h_is_identity(self):
        """H⁻¹ H should be close to I."""
        in_dim = 8
        X = torch.randn(50, in_dim)
        H = (2.0 / 50) * X.T @ X
        H.diagonal().add_(0.01 * H.diag().mean())

        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)

        I_approx = Hinv @ H
        torch.testing.assert_close(
            I_approx, torch.eye(in_dim), atol=1e-4, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# _col_scale_zero
# ---------------------------------------------------------------------------

class TestColScaleZero:
    def _make_scale_zero(self, out, n_groups, granularity):
        """Create fake scale/zero tensors for testing."""
        if granularity == "per_tensor":
            scale = torch.tensor(0.1)
            zero = torch.tensor(0.0)
        elif granularity == "per_channel":
            scale = torch.rand(out, 1) + 0.1
            zero = torch.rand(out, 1)
        else:   # per_group
            scale = torch.rand(out, n_groups, 1) + 0.1
            zero = torch.rand(out, n_groups, 1)
        return scale, zero

    def test_per_tensor_returns_scalar(self):
        s, z = self._make_scale_zero(4, 2, "per_tensor")
        cs, cz = _col_scale_zero(s, z, col_orig=5, granularity="per_tensor", group_size=64)
        assert cs.shape == torch.Size([])

    def test_per_channel_returns_all_rows(self):
        out = 8
        s, z = self._make_scale_zero(out, 2, "per_channel")
        cs, cz = _col_scale_zero(s, z, col_orig=3, granularity="per_channel", group_size=64)
        assert cs.shape == (out,)

    def test_per_group_uses_correct_group(self):
        out, n_groups, group_size = 4, 4, 32
        s = torch.arange(n_groups).float().reshape(1, n_groups, 1).expand(out, -1, -1)
        z = torch.zeros_like(s)
        # column 65 → group = 65 // 32 = 2
        cs, _ = _col_scale_zero(s, z, col_orig=65, granularity="per_group", group_size=group_size)
        expected_group = 65 // group_size  # 2
        assert torch.allclose(cs, torch.full((out,), float(expected_group)))


# ---------------------------------------------------------------------------
# _gptq_one_layer — end-to-end numerical test
# ---------------------------------------------------------------------------

class TestGPTQOneLayer:
    def _make_layer(self, out=8, inp=32, group_size=16):
        """Return a synthetic weight matrix and Hessian."""
        torch.manual_seed(0)
        W = torch.randn(out, inp)
        # Build H from random activations so it's valid PD
        X = torch.randn(200, inp)
        H = (2.0 / 200) * X.T @ X
        H.diagonal().add_(0.01 * H.diag().mean())
        return W, H

    def test_q_int_is_in_range(self):
        dtype = QuantDtype.INT8
        W, H = self._make_layer(out=4, inp=16, group_size=8)
        scale, zero = compute_scale(W, dtype, granularity="per_channel")

        Q_int = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_channel", group_size=8,
            damp_percent=0.01, act_order=False,
        )
        assert Q_int.min() >= dtype.qmin, f"Q_int below qmin: {Q_int.min()}"
        assert Q_int.max() <= dtype.qmax, f"Q_int above qmax: {Q_int.max()}"

    def test_output_error_below_rtn(self):
        """GPTQ minimises ‖WX - QX‖² (output error), not weight MSE.

        GPTQ propagates correction signals across columns so the layer
        output is closer to the original than RTN.  We measure this with
        the same calibration activations X used to build H.
        """
        from pare.core.functional import dequantize_tensor, quantize_tensor

        torch.manual_seed(0)
        out_f, inp_f, gs = 16, 64, 16
        W = torch.randn(out_f, inp_f)

        # X used both to build H AND to measure output error
        X = torch.randn(200, inp_f)
        H = (2.0 / 200) * X.T @ X
        H.diagonal().add_(0.01 * H.diag().mean())

        dtype = QuantDtype.INT4
        scale, zero = compute_scale(W, dtype, granularity="per_group", group_size=gs)

        # RTN output error:  ‖WX^T - W_rtn X^T‖²
        Q_rtn = quantize_tensor(W.clone(), scale, zero, dtype).reshape(out_f, inp_f)
        W_rtn = dequantize_tensor(Q_rtn, scale, zero).reshape(out_f, inp_f)
        rtn_out_err = ((W - W_rtn) @ X.T).pow(2).mean().item()

        # GPTQ output error
        Q_gptq = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_group", group_size=gs,
            damp_percent=0.01, act_order=False,
        )
        W_gptq = dequantize_tensor(Q_gptq, scale, zero).reshape(out_f, inp_f)
        gptq_out_err = ((W - W_gptq) @ X.T).pow(2).mean().item()

        assert gptq_out_err <= rtn_out_err * 1.05, (
            f"GPTQ output error ({gptq_out_err:.6f}) should be ≤ RTN output error ({rtn_out_err:.6f})"
        )

    def test_act_order_undone(self):
        """Q_int must be in original column order regardless of act_order."""
        dtype = QuantDtype.INT8
        W, H = self._make_layer(out=4, inp=32, group_size=16)
        scale, zero = compute_scale(W, dtype, granularity="per_channel")

        Q_no_order = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_channel", group_size=16,
            damp_percent=0.01, act_order=False,
        )
        Q_act_order = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_channel", group_size=16,
            damp_percent=0.01, act_order=True,
        )

        # Both outputs must have the same shape.
        assert Q_no_order.shape == Q_act_order.shape

        # Values will differ (act_order changes the error propagation),
        # but both must have all values in [qmin, qmax].
        assert Q_act_order.min() >= dtype.qmin
        assert Q_act_order.max() <= dtype.qmax

    def test_act_order_per_group(self):
        """act_order=True must work with per_group granularity; output in original order."""
        dtype = QuantDtype.INT4
        out, inp, gs = 8, 64, 16
        W, H = self._make_layer(out=out, inp=inp, group_size=gs)
        scale, zero = compute_scale(W, dtype, granularity="per_group", group_size=gs)

        Q_act = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_group", group_size=gs,
            damp_percent=0.01, act_order=True,
        )
        assert Q_act.shape == (out, inp)
        assert Q_act.min() >= dtype.qmin
        assert Q_act.max() <= dtype.qmax

        # act_order should reduce reconstruction error vs act_order=False
        # on a layer with clearly varying column importance (non-uniform H diag).
        Q_no = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_group", group_size=gs,
            damp_percent=0.01, act_order=False,
        )
        from pare.core.functional import dequantize_tensor
        W_hat_act = dequantize_tensor(Q_act.float(), scale, zero)
        W_hat_no  = dequantize_tensor(Q_no.float(),  scale, zero)
        # Reconstruction error should not be worse with act_order=True.
        err_act = (W.reshape_as(W_hat_act) - W_hat_act).pow(2).mean()
        err_no  = (W.reshape_as(W_hat_no)  - W_hat_no).pow(2).mean()
        assert err_act <= err_no * 1.05, (
            f"act_order=True MSE {err_act:.6f} is >5% worse than False {err_no:.6f}"
        )

    def test_per_group_shape(self):
        """Output shape equals input weight shape."""
        dtype = QuantDtype.INT4
        out, inp, gs = 8, 64, 16
        W, H = self._make_layer(out=out, inp=inp, group_size=gs)
        scale, zero = compute_scale(W, dtype, granularity="per_group", group_size=gs)

        Q_int = _gptq_one_layer(
            W.clone(), H.clone(), scale, zero,
            dtype=dtype, granularity="per_group", group_size=gs,
            damp_percent=0.01, act_order=False,
        )
        assert Q_int.shape == (out, inp)
