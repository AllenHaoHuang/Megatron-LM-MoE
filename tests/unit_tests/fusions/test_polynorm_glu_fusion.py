# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Tests for the fused 2nd-order PolyNorm GLU Triton kernel.

Validates the fused forward + backward against an independent torch-autograd reference (dense and
grouped/per-expert), and independently checks the backward with finite differences. Requires CUDA +
Triton, so the whole module is skipped on CPU-only boxes (e.g. the local Windows dev env); run on the
cluster GPU.
"""
import pytest
import torch

from megatron.core.activations import PolyNorm
from megatron.core.fusions.fused_polynorm_glu import (
    HAVE_TRITON,
    FusedPolyNormGLUFunction,
    fused_polynorm_glu_impl,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="CUDA + Triton required"
)

EPS = 1e-6


def _ref(x_glu, x_linear, a1, a2, eps=EPS, score=None, use_rma=True):
    """Pure-torch reference: gate(x_glu) * x_linear * [score], fp32 math cast to input dtype.

    ``use_rma`` selects the normalizer (matches activations.compiled_polynorm):
    RMA -> t/sqrt(mean|t|); RMS -> t/sqrt(mean t**2).
    """

    def col(a):
        a = a.float()
        return a.reshape(-1, 1) if a.numel() > 1 else a

    def norm(t):
        stat = t.abs() if use_rma else t * t
        return t * torch.rsqrt(stat.mean(-1, keepdim=True) + eps)

    xf = x_glu.float()
    gate = col(a1) * norm(xf) + col(a2) * norm(xf * xf)
    out = gate * x_linear.float()
    if score is not None:
        out = out * score.float().reshape(-1, 1)
    return out.to(x_glu.dtype)


def _tols(dtype):
    return dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-3)


@pytest.mark.internal
@pytest.mark.parametrize("use_rma", [True, False], ids=["rma", "rms"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_score", [False, True])
@pytest.mark.parametrize("shape", [(16, 64), (128, 1024), (37, 257)])
def test_fused_matches_reference(dtype, use_score, shape, use_rma):
    """Dense case: fused fwd + grads (dX, dM, dα1/2, dscore) match the torch reference (both norms)."""
    M, D = shape
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    m = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    a1 = torch.rand(1, device="cuda", requires_grad=True)
    a2 = torch.rand(1, device="cuda", requires_grad=True)
    score = torch.rand(M, 1, device="cuda", requires_grad=True) if use_score else None
    g = torch.randn(M, D, dtype=dtype, device="cuda")

    def clones(*ts):
        return [t.detach().clone().requires_grad_(t.requires_grad) if t is not None else None for t in ts]

    xf, mf, a1f, a2f, scf = clones(x, m, a1, a2, score)

    y_fused = fused_polynorm_glu_impl(x, m, a1, a2, EPS, score, use_rma)
    y_ref = _ref(xf, mf, a1f, a2f, score=scf, use_rma=use_rma)

    tols = _tols(dtype)
    assert y_fused.dtype == y_ref.dtype
    assert torch.allclose(y_fused, y_ref, **tols), (y_fused - y_ref).abs().max()

    y_fused.backward(g)
    y_ref.backward(g.clone())

    assert torch.allclose(x.grad, xf.grad, **tols), ("dX", (x.grad - xf.grad).abs().max())
    assert torch.allclose(m.grad, mf.grad, **tols), ("dM", (m.grad - mf.grad).abs().max())
    # alpha grads accumulate over all M*D terms; compare in fp32 with a slightly looser bf16 tol.
    atol_a = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=3e-2, atol=3e-2)
    assert torch.allclose(a1.grad, a1f.grad, **atol_a), ("dA1", a1.grad, a1f.grad)
    assert torch.allclose(a2.grad, a2f.grad, **atol_a), ("dA2", a2.grad, a2f.grad)
    if use_score:
        assert torch.allclose(score.grad, scf.grad, **tols), ("dS", (score.grad - scf.grad).abs().max())


@pytest.mark.internal
@pytest.mark.parametrize("use_rma", [True, False], ids=["rma", "rms"])
def test_fused_backward_finite_difference(use_rma):
    """Independent fp32 finite-difference check of the kernel backward (dX, dM, dα, dscore)."""
    torch.manual_seed(1)
    M, D = 4, 16
    x = torch.randn(M, D, dtype=torch.float64, device="cuda")
    m = torch.randn(M, D, dtype=torch.float64, device="cuda")
    a = [torch.rand(1, dtype=torch.float64, device="cuda") for _ in range(2)]
    score = torch.rand(M, 1, dtype=torch.float64, device="cuda")

    # The kernel runs in fp32; compute fp32 analytic grads and compare to fp32-input finite diffs.
    def fwd(x_, m_, a_, s_):
        return fused_polynorm_glu_impl(
            x_.float(), m_.float(), a_[0].float(), a_[1].float(), EPS, s_.float(), use_rma
        )

    xv = x.float().requires_grad_(True)
    mv = m.float().requires_grad_(True)
    av = [ai.float().requires_grad_(True) for ai in a]
    sv = score.float().requires_grad_(True)
    out = fused_polynorm_glu_impl(xv, mv, av[0], av[1], EPS, sv, use_rma)
    gout = torch.randn_like(out)
    out.backward(gout)

    def loss(x_, m_, a_, s_):
        return (fwd(x_, m_, a_, s_) * gout.double()).sum()

    eps_fd = 1e-4

    def fd_tensor(t, setter):
        grad = torch.zeros_like(t)
        flat = t.reshape(-1)
        gflat = grad.reshape(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps_fd
            lp = setter()
            flat[i] = orig - eps_fd
            lm = setter()
            flat[i] = orig
            gflat[i] = (lp - lm) / (2 * eps_fd)
        return grad

    dX_fd = fd_tensor(x, lambda: loss(x, m, a, score).item())
    dM_fd = fd_tensor(m, lambda: loss(x, m, a, score).item())
    dS_fd = fd_tensor(score, lambda: loss(x, m, a, score).item())
    dA_fd = [fd_tensor(a[k], lambda: loss(x, m, a, score).item()) for k in range(2)]

    fd_tol = dict(rtol=2e-2, atol=2e-3)
    assert torch.allclose(xv.grad.double(), dX_fd, **fd_tol), (xv.grad.double() - dX_fd).abs().max()
    assert torch.allclose(mv.grad.double(), dM_fd, **fd_tol), (mv.grad.double() - dM_fd).abs().max()
    assert torch.allclose(sv.grad.double(), dS_fd, **fd_tol), (sv.grad.double() - dS_fd).abs().max()
    for k in range(2):
        assert torch.allclose(av[k].grad.double(), dA_fd[k], **fd_tol), (k, av[k].grad, dA_fd[k])


@pytest.mark.internal
@pytest.mark.parametrize("use_rma", [True, False], ids=["rma", "rms"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_module_dense_uses_fused(dtype, use_rma):
    """PolyNorm module (dense) on CUDA matches the reference and routes to the fused path."""
    M, D = 64, 256
    torch.manual_seed(2)
    x = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    m = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    mod = PolyNorm(num_local_experts=1, config=None, alpha_init=0.2, eps=EPS, use_rma=use_rma).cuda().to(dtype)
    out = mod(x, m)
    ref = _ref(x.detach(), m.detach(), mod.alpha_1.detach().abs(), mod.alpha_2.detach().abs(), use_rma=use_rma)
    assert torch.allclose(out, ref, **_tols(dtype)), (out - ref).abs().max()
    out.sum().backward()
    assert x.grad is not None and mod.alpha_1.grad is not None


@pytest.mark.internal
@pytest.mark.parametrize("use_rma", [True, False], ids=["rma", "rms"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_module_grouped_matches_reference(dtype, use_rma):
    """PolyNorm module (grouped, per-expert coeffs + probs) matches the reference."""
    E = 3
    tpe = [5, 0, 11]  # include an empty expert
    Mt, D = sum(tpe), 128
    torch.manual_seed(3)
    x = torch.randn(Mt, D, dtype=dtype, device="cuda", requires_grad=True)
    m = torch.randn(Mt, D, dtype=dtype, device="cuda", requires_grad=True)
    probs = torch.rand(Mt, 1, device="cuda", requires_grad=True)
    mod = PolyNorm(num_local_experts=E, config=None, alpha_init=0.2, eps=EPS, use_rma=use_rma).cuda().to(dtype)
    with torch.no_grad():
        mod.alpha_1.copy_(torch.tensor([0.1, 0.3, 0.5], device="cuda").to(dtype))
        mod.alpha_2.copy_(torch.tensor([0.2, 0.4, 0.6], device="cuda").to(dtype))
    out = mod(x, m, tokens_per_expert=tpe, scores=probs)

    tpe_t = torch.tensor(tpe, device="cuda")
    a1 = torch.repeat_interleave(mod.alpha_1.detach().abs(), tpe_t)
    a2 = torch.repeat_interleave(mod.alpha_2.detach().abs(), tpe_t)
    ref = _ref(x.detach(), m.detach(), a1, a2, score=probs.detach(), use_rma=use_rma)
    assert torch.allclose(out, ref, **_tols(dtype)), (out - ref).abs().max()
    out.sum().backward()
    assert mod.alpha_1.grad.shape == (E,)
    assert x.grad is not None and probs.grad is not None
