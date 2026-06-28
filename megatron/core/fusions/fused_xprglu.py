# Copyright (c) 2026, Swiss AI Institute
"""Fused Triton kernels for the XPRGLU activation.

For a token row ``x`` (the GLU gate half) and ``y`` (the GLU linear half ``x_linear``) this computes,
**element-wise** over the ffn feature dimension ``D``::

    gate_i = ap2 * x_i**2 + ap1 * x_i + b          if x_i  > 0
    gate_i = an  * softsign(x_i) + b               if x_i <= 0
    out_i  = gate_i * y_i * score                  # score optional (per token)

where ``softsign(x) = x / (1 + |x|)`` and ``ap1, ap2, an, b`` are the (already positive) XPRGLU
coefficients. Unlike the PolyNorm GLU gate, XPRGLU has **no reduction over the feature dimension** in
the forward pass — every output element depends only on its own ``(x_i, y_i)`` — so there is no
RMS-style cross-feature statistic to save for backward. The kernel still iterates one program per
token row so it is shape-agnostic in the token count (no torch.compile-style recompiles on the
variable-token MoE path), mirroring ``fused_polynorm_glu.py``.

The coefficients ``ap1/ap2/an/b`` are passed *per token* (shape ``(M,)``, contiguous); the owning
module (:class:`~megatron.core.activations.XPRGLU`) expands its per-expert coefficients to per-token
before calling, so this kernel serves both the dense and the grouped-expert paths with one code path.
``abs()`` (and the ``an = |beta| + |alpha_n|`` coupling) is applied by the module *outside* the
autograd Function, so the gradients returned here are w.r.t. the already-positive coefficients;
torch's ``abs``/``add`` backward then supplies the sign and the ``beta`` coupling, and
``repeat_interleave``/``expand`` backward reduces the per-token coefficient gradients back to the
per-expert parameters.
"""
from typing import Optional

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    # Triton requires a CUDA device; on CPU-only boxes we keep the module importable but route all
    # real work to the torch fallback in XPRGLU.
    HAVE_TRITON = torch.cuda.is_available()
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    from unittest.mock import MagicMock

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


# Largest per-shard feature dim the single-block kernel handles. Above this the module routes to the
# torch fallback (keeps register/SRAM pressure bounded). Per-shard ffn sizes are well within this.
MAX_FUSED_FEATURE_DIM = 8192


def _num_warps_for(block_size: int) -> int:
    """Heuristic warp count for a single-row pass over ``block_size`` features."""
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 2


@triton.jit
def _xprglu_fwd_kernel(
    out_ptr,  # (M, D) output
    x_ptr,  # (M, D) gate half (x_glu)
    y_ptr,  # (M, D) linear half (x_linear)
    ap1_ptr,  # (M,) per-token coefficient for the linear positive term
    ap2_ptr,  # (M,) per-token coefficient for the quadratic positive term
    an_ptr,  # (M,) per-token coefficient for the negative softsign term
    b_ptr,  # (M,) per-token additive (bias) term, present in both branches
    score_ptr,  # (M,) per-token multiplier (only read when HAS_SCORE)
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    stride_out_row,
    stride_out_col,
    D,
    HAS_SCORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_y_row + cols * stride_y_col, mask=mask, other=0.0).to(tl.float32)

    ap1 = tl.load(ap1_ptr + row)
    ap2 = tl.load(ap2_ptr + row)
    an = tl.load(an_ptr + row)
    b = tl.load(b_ptr + row)

    pos = x > 0.0
    softsign = x / (1.0 + tl.abs(x))
    gate = tl.where(pos, ap2 * x * x + ap1 * x + b, an * softsign + b)

    out = gate * y
    if HAS_SCORE:
        out = out * tl.load(score_ptr + row)

    tl.store(out_ptr + row * stride_out_row + cols * stride_out_col, out, mask=mask)


@triton.jit
def _xprglu_bwd_kernel(
    dx_ptr,  # (M, D) grad w.r.t. x_glu
    dy_ptr,  # (M, D) grad w.r.t. x_linear
    dap1_ptr,  # (M,) per-token grad for ap1
    dap2_ptr,  # (M,) per-token grad for ap2
    dan_ptr,  # (M,) per-token grad for an
    db_ptr,  # (M,) per-token grad for b
    dscore_ptr,  # (M,) per-token grad for score (only written when HAS_SCORE)
    dout_ptr,  # (M, D) incoming grad
    x_ptr,  # (M, D) gate half (saved)
    y_ptr,  # (M, D) linear half (saved)
    ap1_ptr,
    ap2_ptr,
    an_ptr,
    b_ptr,
    score_ptr,
    stride_dout_row,
    stride_dout_col,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    stride_dx_row,
    stride_dx_col,
    stride_dy_row,
    stride_dy_col,
    D,
    HAS_SCORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_y_row + cols * stride_y_col, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(
        dout_ptr + row * stride_dout_row + cols * stride_dout_col, mask=mask, other=0.0
    ).to(tl.float32)

    ap1 = tl.load(ap1_ptr + row)
    ap2 = tl.load(ap2_ptr + row)
    an = tl.load(an_ptr + row)
    b = tl.load(b_ptr + row)

    if HAS_SCORE:
        score = tl.load(score_ptr + row)
    else:
        score = 1.0

    pos = x > 0.0
    absx = tl.abs(x)
    denom = 1.0 + absx
    softsign = x / denom
    # softsign'(x) = 1 / (1 + |x|)**2 (valid for both signs).
    softsign_grad = 1.0 / (denom * denom)

    gate = tl.where(pos, ap2 * x * x + ap1 * x + b, an * softsign + b)
    gate_grad = tl.where(pos, 2.0 * ap2 * x + ap1, an * softsign_grad)

    # w = dL/dgate (per element), the multiplier shared by dx and the coefficient grads.
    w = dout * y * score

    # grad w.r.t. x_linear (the GLU "mul" operand) and x_glu (through the gate).
    dy = dout * gate * score
    dx = w * gate_grad
    tl.store(dy_ptr + row * stride_dy_row + cols * stride_dy_col, dy, mask=mask)
    tl.store(dx_ptr + row * stride_dx_row + cols * stride_dx_col, dx, mask=mask)

    # Per-token coefficient grads: sum over the feature dim of the element-wise contribution.
    # Masked (padding) elements load x=y=0 -> contribute b only via w (which is 0 when dout=0),
    # but dout is also masked to 0, so w=0 on padding and nothing leaks.
    zero = 0.0
    dap2 = tl.sum(tl.where(pos, w * x * x, zero), axis=0)
    dap1 = tl.sum(tl.where(pos, w * x, zero), axis=0)
    dan = tl.sum(tl.where(pos, zero, w * softsign), axis=0)
    db = tl.sum(w, axis=0)

    tl.store(dap1_ptr + row, dap1)
    tl.store(dap2_ptr + row, dap2)
    tl.store(dan_ptr + row, dan)
    tl.store(db_ptr + row, db)

    if HAS_SCORE:
        # dscore_r = sum_j dout_j * gate_j * y_j
        tl.store(dscore_ptr + row, tl.sum(dout * gate * y, axis=0))


def _launch_fwd(x_glu, x_linear, ap1, ap2, an, b, score):
    """Raw forward kernel launch. Returns the activation output ``(M, D)``."""
    M, D = x_glu.shape
    out = torch.empty((M, D), dtype=x_glu.dtype, device=x_glu.device)
    has_score = score is not None
    score_arg = score if has_score else x_glu  # dummy pointer when unused

    block = triton.next_power_of_2(D)
    _xprglu_fwd_kernel[(M,)](
        out,
        x_glu,
        x_linear,
        ap1,
        ap2,
        an,
        b,
        score_arg,
        x_glu.stride(0),
        x_glu.stride(1),
        x_linear.stride(0),
        x_linear.stride(1),
        out.stride(0),
        out.stride(1),
        D,
        HAS_SCORE=has_score,
        BLOCK_SIZE=block,
        num_warps=_num_warps_for(block),
    )
    return out


def _launch_bwd(grad_output, x_glu, x_linear, ap1, ap2, an, b, score):
    """Raw backward kernel launch. Returns ``(dx, dy, dap1, dap2, dan, db, dscore)``."""
    M, D = x_glu.shape
    grad_output = grad_output.contiguous()
    dx = torch.empty((M, D), dtype=x_glu.dtype, device=x_glu.device)
    dy = torch.empty((M, D), dtype=x_linear.dtype, device=x_linear.device)
    # Match each coefficient's dtype so autograd receives a same-dtype grad (the coeffs may be
    # fp32 master params or bf16 under mixed precision); the kernel casts fp32->dtype on store.
    dap1 = torch.empty((M,), dtype=ap1.dtype, device=x_glu.device)
    dap2 = torch.empty((M,), dtype=ap2.dtype, device=x_glu.device)
    dan = torch.empty((M,), dtype=an.dtype, device=x_glu.device)
    db = torch.empty((M,), dtype=b.dtype, device=x_glu.device)
    has_score = score is not None
    if has_score:
        dscore = torch.empty((M,), dtype=score.dtype, device=score.device)
        score_arg = score
    else:
        dscore = None
        score_arg = x_glu  # dummy pointer

    block = triton.next_power_of_2(D)
    _xprglu_bwd_kernel[(M,)](
        dx,
        dy,
        dap1,
        dap2,
        dan,
        db,
        dscore if has_score else x_glu,
        grad_output,
        x_glu,
        x_linear,
        ap1,
        ap2,
        an,
        b,
        score_arg,
        grad_output.stride(0),
        grad_output.stride(1),
        x_glu.stride(0),
        x_glu.stride(1),
        x_linear.stride(0),
        x_linear.stride(1),
        dx.stride(0),
        dx.stride(1),
        dy.stride(0),
        dy.stride(1),
        D,
        HAS_SCORE=has_score,
        BLOCK_SIZE=block,
        num_warps=_num_warps_for(block),
    )
    return dx, dy, dap1, dap2, dan, db, (dscore if has_score else None)


class FusedXPRGLUFunction(torch.autograd.Function):
    """Autograd wrapper around the fused XPRGLU Triton kernels.

    Inputs are 2D ``(M, D)`` (the caller flattens leading dims). ``ap1/ap2/an/b`` are the positive
    per-token coefficients of shape ``(M,)``; ``score`` is an optional per-token multiplier ``(M,)``.
    """

    # Raw (non-autograd) launches reused by the FP8 offloading path, whose enclosing autograd
    # Function manages the graph manually.
    @classmethod
    def call_forward(cls, x_glu, x_linear, ap1, ap2, an, b, score):
        out = _launch_fwd(x_glu, x_linear, ap1, ap2, an, b, score)
        # No reduction-derived state to cache (cf. PolyNorm's inv); backward recomputes the gate
        # from x. Return ``None`` state for interface symmetry with fused_polynorm_glu.
        return out, None

    @classmethod
    def call_backward(cls, grad_output, saved):
        x_glu, x_linear, ap1, ap2, an, b = saved[:6]
        score = saved[6] if len(saved) > 6 else None
        dx, dy, dap1, dap2, dan, db, dscore = _launch_bwd(
            grad_output, x_glu, x_linear, ap1, ap2, an, b, score
        )
        return dx, dy, dap1, dap2, dan, db, dscore

    @staticmethod
    def forward(ctx, x_glu, x_linear, ap1, ap2, an, b, score):
        out = _launch_fwd(x_glu, x_linear, ap1, ap2, an, b, score)
        has_score = score is not None
        saved = [x_glu, x_linear, ap1, ap2, an, b]
        if has_score:
            saved.append(score)
        ctx.save_for_backward(*saved)
        ctx.has_score = has_score
        return out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x_glu, x_linear, ap1, ap2, an, b = saved[:6]
        score = saved[6] if ctx.has_score else None
        dx, dy, dap1, dap2, dan, db, dscore = _launch_bwd(
            grad_output, x_glu, x_linear, ap1, ap2, an, b, score
        )
        # score grad (position 6) is None when score was None.
        return dx, dy, dap1, dap2, dan, db, (dscore if ctx.has_score else None)


def fused_xprglu_impl(
    x_glu: torch.Tensor,
    x_linear: torch.Tensor,
    ap1: torch.Tensor,
    ap2: torch.Tensor,
    an: torch.Tensor,
    b: torch.Tensor,
    score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused XPRGLU: ``gate(x_glu) * x_linear * [score]``.

    Args:
        x_glu, x_linear: ``(..., D)`` gate / linear halves of the fc1 output (same shape & dtype).
        ap1, ap2, an, b: positive per-token coefficients, broadcastable to ``(M,)`` where ``M`` is the
            number of tokens (``prod(x_glu.shape[:-1])``). The owning module passes either a single
            shared coefficient or one per token (grouped experts).
        score: optional per-token multiplier ``(..., 1)`` / ``(...)`` (MoE router probs / scale).

    Returns:
        Tensor of ``x_glu``'s shape and dtype.
    """
    ori_shape = x_glu.shape
    D = ori_shape[-1]
    x2 = x_glu.reshape(-1, D)
    m2 = x_linear.reshape(-1, D)
    M = x2.shape[0]

    ap1, ap2, an, b = _per_token_coeffs(ap1, ap2, an, b, M)

    score2 = score.reshape(-1) if score is not None else None

    out = FusedXPRGLUFunction.apply(x2, m2, ap1, ap2, an, b, score2)
    return out.reshape(ori_shape)


def _split_glu_halves(fc1_output: torch.Tensor):
    """Split a fused ``(..., 2D)`` fc1 output into its gate / linear ``(M, D)`` halves."""
    ori_shape = fc1_output.shape
    two_d = ori_shape[-1]
    assert two_d % 2 == 0, "fc1 output last dim must be 2*D for a gated linear unit"
    d = two_d // 2
    flat = fc1_output.reshape(-1, two_d)
    x_glu, x_linear = torch.split(flat, d, dim=-1)
    return x_glu, x_linear, ori_shape[:-1], d


def _per_token_coeffs(ap1, ap2, an, b, num_tokens: int):
    """Normalize the coefficients to per-token ``(M,)`` tensors for the row-indexed kernels.

    A single shared coefficient is expanded to one per token (the ``expand`` keeps the autograd
    link so the gradient reduces back to the shared parameter); already-per-token coefficients pass
    through unchanged.
    """
    def fix(c):
        c = c.reshape(-1)
        if c.numel() == 1:
            c = c.expand(num_tokens).contiguous()
        return c

    return fix(ap1), fix(ap2), fix(an), fix(b)


def fused_xprglu_forward(
    fc1_output: torch.Tensor,
    ap1: torch.Tensor,
    ap2: torch.Tensor,
    an: torch.Tensor,
    b: torch.Tensor,
    score: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, None]:
    """XPRGLU forward for the FP8 offloading path (raw kernel, no autograd graph).

    Returns ``(out, state)`` where ``state`` is ``None`` (XPRGLU caches no reduction state; the
    ``None`` mirrors PolyNorm's ``inv`` slot so the offloading code can thread a uniform tuple).
    """
    x_glu, x_linear, lead_shape, d = _split_glu_halves(fc1_output)
    ap1, ap2, an, b = _per_token_coeffs(ap1, ap2, an, b, x_glu.shape[0])
    score2 = score.reshape(-1) if score is not None else None
    out, _ = FusedXPRGLUFunction.call_forward(x_glu, x_linear, ap1, ap2, an, b, score2)
    return out.reshape(lead_shape + (d,)), None


def fused_xprglu_backward(
    grad_output: torch.Tensor,
    fc1_output: torch.Tensor,
    ap1: torch.Tensor,
    ap2: torch.Tensor,
    an: torch.Tensor,
    b: torch.Tensor,
    score: Optional[torch.Tensor] = None,
):
    """XPRGLU backward for the FP8 offloading path.

    Recomputes the gate from ``fc1_output`` (no saved state) then runs the fused backward kernel.
    Returns ``(grad_fc1, dap1, dap2, dan, db, grad_score)`` with the per-token coefficient grads
    reduced to the input coefficient shapes.
    """
    x_glu, x_linear, lead_shape, d = _split_glu_halves(fc1_output)
    # A single shared coefficient receives the summed per-token gradient (mirrors expand-backward in
    # fused_xprglu_impl); a per-token coefficient keeps its shape.
    shared = [c.numel() == 1 for c in (ap1, ap2, an, b)]
    ap1_c, ap2_c, an_c, b_c = _per_token_coeffs(ap1, ap2, an, b, x_glu.shape[0])
    score2 = score.reshape(-1) if score is not None else None

    saved = [x_glu, x_linear, ap1_c, ap2_c, an_c, b_c]
    if score2 is not None:
        saved.append(score2)

    grad_flat = grad_output.reshape(-1, d)
    dx, dy, dap1, dap2, dan, db, dscore = FusedXPRGLUFunction.call_backward(grad_flat, saved)
    grad_fc1 = torch.cat([dx, dy], dim=-1).reshape(lead_shape + (2 * d,))

    grads = [dap1, dap2, dan, db]
    refs = [ap1, ap2, an, b]
    out_grads = []
    for g, is_shared, ref in zip(grads, shared, refs):
        out_grads.append(g.sum(0, keepdim=True).reshape(ref.shape) if is_shared else g.reshape(ref.shape))

    grad_score = dscore.reshape(score.shape[:-1]) if score is not None else None
    return grad_fc1, out_grads[0], out_grads[1], out_grads[2], out_grads[3], grad_score
