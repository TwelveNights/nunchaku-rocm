"""
Backend-dispatching wrappers for the low-level ops.

Each function here forwards to either the compiled CUDA extension or the
Triton backend, chosen by :func:`nunchaku.ops.backend.get_backend`. Call sites
should import from this module rather than reaching into ``nunchaku._C``
directly, so that they keep working on hardware where the extension does not
exist.
"""

import torch

from .backend import get_backend, load_cuda_ops

__all__ = ["gemm_awq", "gemv_awq", "attention_fp16"]


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
