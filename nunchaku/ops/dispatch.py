"""
Backend-dispatching wrappers for the low-level ops.

Each function here forwards to either the compiled CUDA extension or the
Triton backend, chosen by :func:`nunchaku.ops.backend.get_backend`. Call sites
should import from this module rather than reaching into ``nunchaku._C``
directly, so that they keep working on hardware where the extension does not
exist.
"""

import torch
import torch.nn.functional as F

from .backend import get_backend, load_cuda_ops

__all__ = ["gemm_awq", "gemv_awq", "attention_fp16", "scaled_dot_product_attention"]


def gemv_awq(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor:
    """AWQ W4A16 GEMV. See :func:`nunchaku.ops.gemv.awq_gemv_w4a16_cuda`."""
    if get_backend() == "triton":
        from .triton.awq import awq_gemv_w4a16

        return awq_gemv_w4a16(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
    return load_cuda_ops().gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)


def gemm_awq(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    """AWQ W4A16 GEMM."""
    if get_backend() == "triton":
        from .triton.awq import awq_gemm_w4a16

        return awq_gemm_w4a16(in_feats, kernel, scaling_factors, zeros)
    return load_cuda_ops().gemm_awq(in_feats, kernel, scaling_factors, zeros)


def attention_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    scale: float,
) -> None:
    """FP16 attention with an fp16 accumulator."""
    if get_backend() == "triton":
        from .triton.attention import attention_fp16 as _attn

        _attn(q, k, v, o, scale)
        return
    load_cuda_ops().attention_fp16(q, k, v, o, scale)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Drop-in for ``F.scaled_dot_product_attention`` on ``(B, H, T, D)`` inputs (no dropout, not causal).

    On the Triton backend, unmasked fp16/bf16 attention runs on the Triton flash-attention kernel:
    torch's SDPA on ROCm can fall back to a slow math path (~8x slower and run-to-run erratic on
    Windows ROCm / RDNA3). Anything the kernel doesn't cover (a mask, fp32, a non-power-of-two head
    dim, a non-contiguous head dim) falls back to torch.
    """
    head_dim = query.shape[-1]
    if (
        get_backend() != "triton"
        or attn_mask is not None
        or query.dtype not in (torch.float16, torch.bfloat16)
        or not (key.dtype == value.dtype == query.dtype)
        or head_dim < 16
        or head_dim > 256
        or head_dim & (head_dim - 1)
        or any(t.stride(-1) != 1 for t in (query, key, value))
    ):
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    from .triton.attention import attention_fp16 as _attn

    batch_size, heads, tokens, _ = query.shape
    out = torch.empty(batch_size, tokens, heads * head_dim, dtype=query.dtype, device=query.device)
    _attn(query, key, value, out, head_dim**-0.5)
    return out.view(batch_size, tokens, heads, head_dim).transpose(1, 2)
