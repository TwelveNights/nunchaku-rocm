"""
Triton ports of ``src/kernels/gemm_f16.cu`` and ``src/kernels/gemm_batched.cu``.

These replace CUTLASS ``device::Gemm`` / ``device::GemmBatched`` instantiations.
"""

import torch
import triton
import triton.language as tl

from ._common import matmul_group_m, num_warps_for


def _fp16_configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": matmul_group_m()},
            num_stages=ns,
            num_warps=num_warps_for(nw),
        )
        for bm, bn, bk, ns, nw in [
            (128, 256, 64, 3, 8),
            (256, 128, 64, 3, 8),
            (128, 128, 64, 4, 8),
            (128, 128, 32, 4, 4),
            (64, 128, 64, 4, 4),
            (64, 64, 64, 4, 4),
        ]
    ]


@triton.autotune(configs=_fp16_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_fp16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    alpha,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=tl.trans(kmask), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc *= alpha

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)[None, :]

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def gemm_fp16(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    ``out = alpha * (input @ weight.T) + bias`` in fp16/bf16.

    ``weight`` is (N, K) row-major, matching the CUTLASS column-major-B
    convention used by the CUDA kernel.
    """
    shape_in = input.shape
    input2d = input.reshape(-1, shape_in[-1])
    M, K = input2d.shape
    N = weight.shape[0]
    assert weight.shape[1] == K

    if out is None:
        out = torch.empty(M, N, dtype=weight.dtype, device=input.device)
    out2d = out.reshape(-1, N)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _gemm_fp16_kernel[grid](
        input2d,
        weight,
        out2d,
        bias if bias is not None else input2d,
        alpha,
        M,
        N,
        K,
        input2d.stride(0),
        input2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        HAS_BIAS=bias is not None,
    )
    return out.reshape(*shape_in[:-1], N)


@triton.jit
def _gemm_batched_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + batch * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + batch * stride_bb + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & tl.trans(kmask), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def gemm_batched_fp16(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """
    Batched ``out[i] = a[i] @ b[i].T``, fp16 in / fp32 out.

    ``a`` is [..., M, K] row-major and ``b`` is [..., N, K], matching the
    column-major-B convention of the CUTLASS kernel.
    """
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-2]
    assert b.shape[-1] == K

    a3 = a.reshape(-1, M, K)
    b3 = b.reshape(-1, N, K)
    batch = a3.shape[0]

    if out is None:
        out = torch.empty(*a.shape[:-1], N, dtype=torch.float32, device=a.device)
    out3 = out.reshape(-1, M, N)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    grid = (batch, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_batched_kernel[grid](
        a3,
        b3,
        out3,
        M,
        N,
        K,
        a3.stride(0),
        a3.stride(1),
        a3.stride(2),
        b3.stride(0),
        b3.stride(1),
        b3.stride(2),
        out3.stride(0),
        out3.stride(1),
        out3.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps_for(4),
        num_stages=3,
    )
    return out
