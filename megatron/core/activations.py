# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.fusions.fused_polynorm_glu import (
    MAX_FUSED_FEATURE_DIM,
    HAVE_TRITON as HAVE_FUSED_PNGLU,
    fused_polynorm_glu_impl,
)
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def compiled_polynorm(x, alpha_1, alpha_2, eps: float = 1e-6, use_rma: bool = True):
    """Core PolyNorm GLU gate: ``a1*norm(x) + a2*norm(x**2)``.

    ``norm(t) = t / sqrt(mean_j(g(t_j)) + eps)`` reduces over the last (feature) dimension. The
    statistic ``g`` is selected by ``use_rma``:

    * ``use_rma=True`` (default) — **RMA**, ``g=|t|`` (root-mean-abs)::

          norm(x)    = x    / sqrt(mean|x|)       # degree-1/2 in x
          norm(x**2) = x**2 / sqrt(mean(x**2))    # degree-1   in x  (since |x**2| == x**2)

    * ``use_rma=False`` — **RMS**, ``g=t**2`` (the classic scale-invariant RMSNorm)::

          norm(x)    = x    / sqrt(mean(x**2))    # degree-0 in x
          norm(x**2) = x**2 / sqrt(mean(x**4))

    RMA is the default because the degree-1/2 norm is *not* exactly scale-invariant in the input,
    so the upstream fc1 weight is not left as an unconstrained ("free") direction the loss cannot
    anchor — the RMS (degree-0) gate is scale-invariant and leaves fc1 with effective LR ~
    1/||W||^2 and no restoring force, a training-stability hazard. RMS is retained for A/B.

    The math is done in fp32 and cast back to the input dtype, mirroring the
    RMSNorm/LayerNorm layers in this codebase under mixed precision.

    ``alpha_1``/``alpha_2`` broadcast against ``x``. They are either a single
    (broadcastable) coefficient of shape ``(1,)`` (dense / single-expert case) or per-token
    coefficients of shape ``(num_tokens, 1)`` (grouped-expert case, where each token already
    carries the coefficient of the expert it was routed to).

    This is the (torch.compile-fused) **gate-only** computation used by the non-Triton fallback
    paths; the CUDA fast path fuses the gate, the ``* x_linear`` and the ``* score`` multiplies in
    a single Triton kernel (see ``fused_polynorm_glu_impl``).
    """
    input_dtype = x.dtype
    x = x.float()

    def norm(t):
        # RMA: g=|t| (for the x**2 term |x**2|==x**2, i.e. divides by sqrt(mean x**2)=RMS(x)).
        # RMS: g=t**2.
        stat = t.abs() if use_rma else t * t
        return t * torch.rsqrt(stat.mean(-1, keepdim=True) + eps)

    out = alpha_1 * norm(x) + alpha_2 * norm(x * x)
    return out.to(input_dtype)


class _AllReduceSumSymmetric(torch.autograd.Function):
    """All-reduce(sum) over ``group`` in BOTH the forward and backward passes.

    Used to turn each rank's partial feature-sum into the full sum when the result is then
    consumed independently on every rank (each rank normalizes its own tokens with the shared
    statistic). Because the reduced value feeds rank-local downstream work, the gradient must
    be summed back across the group — unlike ``reduce_from_tensor_model_parallel_region``
    (forward all-reduce, backward identity), which is only correct when the reduced value feeds
    *replicated* downstream work.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        x = x.clone()
        torch.distributed.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class _SyncGradSum(torch.autograd.Function):
    """Identity in the forward pass; all-reduce(sum) the gradient over ``group`` in backward.

    Applied to the (TP-replicated) alpha coefficients so each rank's partial coefficient
    gradient — a sum over only that rank's feature shard — is completed into the full gradient,
    keeping the replicas in sync. (Same semantics as ``copy_to_tensor_model_parallel_region``,
    but over an arbitrary group so it also works for the expert-tensor-parallel group.)
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class PolyNorm(MegatronModule):
    """Learnable PolyNorm GLU activation — a drop-in replacement for the gate of a gated
    linear unit (e.g. SiLU in SwiGLU).

    In a GLU the first linear layer produces ``[x_glu, x_linear]`` and the block output is
    ``gate(x_glu) * x_linear``. Standard SwiGLU uses ``gate = SiLU``. Here the gate is the
    (2nd-order) PolyNorm::

        gate(x) = |alpha_1| * x / sqrt(mean|x|) + |alpha_2| * x**2 / sqrt(mean(x**2))

    i.e. ``a1*norm(x) + a2*norm(x**2)`` with the ``sqrt(mean|.|)`` normalizer (see
    :func:`compiled_polynorm` for why this degree-1/2 norm is used instead of RMS: it avoids
    making the upstream fc1 weight a loss-unconstrained "free" direction). ``alpha_1``/``alpha_2``
    are learnable (``abs`` keeps them positive).

    ``forward`` takes *both* GLU halves and returns the full ``gate(x_glu) * x_linear * [score]``
    product. On CUDA (and ``tp_size == 1``) the gate, the GLU multiply and the optional per-token
    ``score`` multiply (MoE router probs / per-token scale) are fused into a single Triton kernel
    (see ``megatron.core.fusions.fused_polynorm_glu``) so the op runs close to SwiGLU speed and is
    shape-agnostic over the (variable) MoE token count. Otherwise the gate is computed with the
    torch fallback (``compiled_polynorm`` or, when TP-sharded, ``_tp_forward``) and the
    multiplies are applied in eager torch.

    To support grouped MoE experts (where the activations of all local experts are
    concatenated along the token dimension and processed in a single call) this module holds
    one ``(alpha_1, alpha_2)`` pair *per local expert*: ``alpha_1``/``alpha_2`` have shape
    ``(num_local_experts,)``. When ``tokens_per_expert`` is supplied the
    per-expert coefficients are expanded to per-token coefficients, so every token is gated by the
    coefficients of the expert it was routed to. For a dense MLP (or a ``SequentialMLP``
    expert) ``num_local_experts == 1`` and the single coefficient is broadcast to all tokens.

    Tensor parallelism: the gate's normalization reduces over the ffn feature dimension, which is
    sharded across ``tp_group`` (the main TP group for dense/shared MLPs, the expert-TP group for
    MoE experts). When ``tp_group`` has size > 1, the per-token partial feature sums (sum|x| and
    sum(x**2)) are all-reduced over the group (forward and backward) so every rank uses the
    *full-feature* statistics, and the replicated ``alpha`` gradients are all-reduced over the
    group so the replicas stay in sync. The result is therefore identical to (and bitwise-
    consistent across) any TP/ETP degree.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_init: float = 0.2,
        eps: float = 1e-6,
        tp_group: "torch.distributed.ProcessGroup | None" = None,
        use_rma: "bool | None" = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_1 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_2 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.eps = eps
        # Normalizer choice: RMA (sqrt(mean|.|), default) vs RMS (sqrt(mean .**2)). Explicit
        # ``use_rma`` wins; otherwise read ``config.pnglu_norm`` ('rma'/'rms'), defaulting to RMA.
        if use_rma is None:
            use_rma = getattr(config, "pnglu_norm", "rma") == "rma"
        self.use_rma = use_rma
        # The group over which the ffn feature dimension is sharded. tp_size==1 (no sharding,
        # e.g. local CPU runs or ETP=1 experts) takes the cheap fused path with no collectives.
        self.tp_group = tp_group
        if tp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.tp_size = torch.distributed.get_world_size(group=tp_group)
        else:
            self.tp_size = 1

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        # Keep the coefficients positive.
        alpha_1 = torch.abs(self.alpha_1)  # (num_local_experts,)
        alpha_2 = torch.abs(self.alpha_2)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    "PolyNorm with num_local_experts > 1 requires `tokens_per_expert` so "
                    "the per-expert coefficients can be mapped onto the concatenated tokens."
                )
            # Single coefficient broadcast to every token: shape (1,).
            a1, a2 = alpha_1, alpha_2
        else:
            # Expand per-expert coefficients to per-token coefficients: shape (num_tokens,).
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x_glu.device)
            a1 = torch.repeat_interleave(alpha_1, tpe_tensor)
            a2 = torch.repeat_interleave(alpha_2, tpe_tensor)

        use_fused = (
            HAVE_FUSED_PNGLU
            and x_glu.is_cuda
            and self.tp_size == 1
            and x_glu.shape[-1] <= MAX_FUSED_FEATURE_DIM
            and (self.config is None or getattr(self.config, "pnglu_fusion", True))
        )
        if use_fused:
            # Single fused kernel: gate + (* x_linear) + (* scores), shape-agnostic over tokens.
            return fused_polynorm_glu_impl(x_glu, x_linear, a1, a2, self.eps, scores, self.use_rma)

        # Fallback: compute the gate in torch, then apply the multiplies in eager mode.
        a1b = a1.unsqueeze(-1) if a1.dim() == 1 and self.num_local_experts > 1 else a1
        a2b = a2.unsqueeze(-1) if a2.dim() == 1 and self.num_local_experts > 1 else a2
        if self.tp_size == 1:
            # ffn feature dim is whole on this rank: cheap fused per-token norm.
            gate = compiled_polynorm(x_glu, a1b, a2b, self.eps, self.use_rma)
        else:
            # ffn feature dim is TP-sharded: reduce the feature statistics across the group.
            gate = self._tp_forward(x_glu, a1b, a2b)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out

    def _tp_forward(self, x, alpha_1, alpha_2):
        """TP-invariant path: recover the full-feature norm statistics from the local shards.

        Honours :attr:`use_rma` (see :func:`compiled_polynorm`):
        RMA -> norm(x)/sqrt(mean|x|), norm(x**2)/sqrt(mean x**2);
        RMS -> norm(x)/sqrt(mean x**2), norm(x**2)/sqrt(mean x**4).
        """
        input_dtype = x.dtype
        xf = x.float()
        # Each ColumnParallel rank holds an equal 1/tp_size slice of the ffn features.
        n_global = xf.shape[-1] * self.tp_size
        # Per-token partial feature sums on this rank for the two norm denominators (s1 for norm(x),
        # s2 for norm(x**2)). One symmetric all-reduce completes both into full-feature sums.
        if self.use_rma:
            s1 = xf.abs().sum(-1, keepdim=True)        # mean|x|
            s2 = xf.pow(2).sum(-1, keepdim=True)       # mean x^2  (|x^2|==x^2)
        else:
            s1 = xf.pow(2).sum(-1, keepdim=True)       # mean x^2
            s2 = xf.pow(2).pow(2).sum(-1, keepdim=True)  # mean x^4
        s = _AllReduceSumSymmetric.apply(torch.cat([s1, s2], dim=-1), self.tp_group)
        inv1 = torch.rsqrt(s[..., 0:1] / n_global + self.eps)
        inv2 = torch.rsqrt(s[..., 1:2] / n_global + self.eps)
        # alpha is replicated across the group; all-reduce its gradient so the replicas stay
        # in sync (forward is identity, so the value is unchanged).
        alpha_1 = _SyncGradSum.apply(alpha_1.float(), self.tp_group)
        alpha_2 = _SyncGradSum.apply(alpha_2.float(), self.tp_group)
        out = alpha_1 * (xf * inv1) + alpha_2 * (xf * xf * inv2)
        return out.to(input_dtype)


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
