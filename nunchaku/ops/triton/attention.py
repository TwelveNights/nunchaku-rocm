"""
Triton port of ``src/kernels/zgemm/attention.cu`` -- a FlashAttention-style
fused attention kernel.

Layout note. The CUDA kernel consumes q/k/v in an mma-fragment "packed"
layout produced by ``EpiloguePackQKV``; that layout only means anything on
NVIDIA tensor cores. This port takes the plain logical layout
``[B, H, T, D]`` instead, which is what the Triton QKV producer emits.
"""

import torch
import triton
import triton.language as tl

from ._common import num_warps_for


def _attn_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=ns, num_warps=num_warps_for(nw))
        for bm, bn, ns, nw in [
            # num_stages=1 variants: ~8% faster on RX 7800 XT (gfx1101) at FLUX's
            # T=4608, D=128 (6.4 vs 7.0 ms) -- the extra LDS stages only add pressure
            (128, 64, 1, 8),
            (64, 64, 1, 4),
            (128, 64, 3, 8),
            (128, 128, 3, 8),
            (64, 64, 4, 4),
            (64, 128, 4, 8),
            (256, 64, 2, 8),
        ]
    ]


@triton.autotune(configs=_attn_configs(), key=["N_CTX_Q", "N_CTX_KV", "HEAD_DIM"])
@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    scale,
    N_CTX_Q,
    N_CTX_KV,
    H,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_om,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + batch.to(tl.int64) * stride_qb + head.to(tl.int64) * stride_qh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=offs_m[:, None] < N_CTX_Q,
        other=0.0,
    )
    # fold the log2(e) factor into q so the inner loop can use exp2
    q = (q.to(tl.float32) * scale).to(q.dtype)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    k_base = k_ptr + batch.to(tl.int64) * stride_kb + head.to(tl.int64) * stride_kh
    v_base = v_ptr + batch.to(tl.int64) * stride_vb + head.to(tl.int64) * stride_vh

    for start_n in tl.range(0, N_CTX_KV, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < N_CTX_KV

        k = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None], mask=nmask[None, :], other=0.0)
        qk = tl.dot(q, k)
        qk = tl.where(nmask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # guard against an all-masked tile producing inf - inf
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        p = tl.where(nmask[None, :], p, 0.0)

        alpha = tl.math.exp2(tl.where(m_i == float("-inf"), 0.0, m_i) - m_ij_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=nmask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_ij

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]

    # output is [B, TokensQ, H * HEAD_DIM]
    o_base = o_ptr + batch.to(tl.int64) * stride_ob + head.to(tl.int64) * HEAD_DIM
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=offs_m[:, None] < N_CTX_Q,
    )


def attention_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    scale: float,
) -> None:
    """
    Fused scaled dot-product attention, no mask (full bidirectional attention).

    Parameters
    ----------
    q : (B, H, TokensQ, HEAD_DIM)
    k, v : (B, H, TokensKV, HEAD_DIM)
    o : (B, TokensQ, H * HEAD_DIM), written in place
    scale : softmax scale, normally ``head_dim ** -0.5``
    """
    B, H, N_CTX_Q, HEAD_DIM = q.shape
    N_CTX_KV = k.shape[2]
    assert k.shape[3] == HEAD_DIM and v.shape[3] == HEAD_DIM
    assert o.shape == (B, N_CTX_Q, H * HEAD_DIM), f"unexpected output shape {tuple(o.shape)}"

    # exp2-based softmax: fold log2(e) into the scale
    scale = scale * 1.4426950408889634

    def grid(META):
        return (triton.cdiv(N_CTX_Q, META["BLOCK_M"]), H, B)

    _attention_kernel[grid](
        q,
        k,
        v,
        o,
        scale,
        N_CTX_Q,
        N_CTX_KV,
        H,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        HEAD_DIM=HEAD_DIM,
    )
