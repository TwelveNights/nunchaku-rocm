"""
Triton port of the W8A8 GEMM and activation quantizer
(``src/kernels/zgemm/gemm_w8a8.cu``).
"""

import torch
import triton
import triton.language as tl

from ._common import matmul_group_m, num_warps_for, round_to_int8, silu_impl


def _w8a8_configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": matmul_group_m()},
            num_stages=ns,
            num_warps=num_warps_for(nw),
        )
        for bm, bn, bk, ns, nw in [
            (128, 128, 64, 3, 8),
            (128, 256, 64, 3, 8),
            (256, 128, 64, 3, 8),
            (128, 128, 128, 4, 8),
            (64, 128, 64, 4, 4),
            (128, 64, 64, 4, 4),
        ]
    ]


@triton.autotune(configs=_w8a8_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_w8a8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ascale_ptr,
    wscale_ptr,
    bias_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0)
        b = tl.load(b_ptrs, mask=tl.trans(kmask), other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # dequantize: per-token activation scale x per-channel weight scale
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ascale = tl.load(ascale_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    wscale = tl.load(wscale_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * ascale[:, None] * wscale[None, :]

    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)[None, :]

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=mask)


def gemm_w8a8(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    INT8 x INT8 -> FP16/BF16 GEMM with per-token and per-channel dequant scales.

    Parameters
    ----------
    act : (M, K) int8
    wgt : (N, K) int8, row-major (i.e. B is used transposed)
    out : (M, N) float16/bfloat16, written in place
    ascales : (M,) per-token activation scales
    wscales : (N,) per-channel weight scales
    bias : (N,) optional
    """
    act = act.reshape(-1, act.shape[-1])
    M, K = act.shape
    N = wgt.shape[0]
    assert wgt.shape[1] == K

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _gemm_w8a8_kernel[grid](
        act,
        wgt,
        out.reshape(-1, N),
        ascales.reshape(-1),
        wscales.reshape(-1),
        bias if bias is not None else act,
        M,
        N,
        K,
        act.stride(0),
        act.stride(1),
        wgt.stride(0),
        wgt.stride(1),
        out.reshape(-1, N).stride(0),
        out.reshape(-1, N).stride(1),
        HAS_BIAS=bias is not None,
    )
    return out


@triton.jit
def _quantize_w8a8_act_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    K,
    K_OUT,
    FUSE_GLU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)

    amax = tl.zeros((BLOCK,), tl.float32)
    for start in tl.range(0, K_OUT, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < K_OUT
        x = tl.load(in_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        if FUSE_GLU:
            gate = tl.load(in_ptr + row * K + K_OUT + offs, mask=mask, other=0.0).to(tl.float32)
            x = silu_impl(x) * gate
        amax = tl.maximum(amax, tl.where(mask, tl.abs(x), 0.0))

    scale = tl.max(amax, axis=0) / 127.0
    scale = tl.maximum(scale, 1e-9)
    tl.store(scale_ptr + row, scale.to(scale_ptr.dtype.element_ty))

    for start in tl.range(0, K_OUT, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < K_OUT
        x = tl.load(in_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        if FUSE_GLU:
            gate = tl.load(in_ptr + row * K + K_OUT + offs, mask=mask, other=0.0).to(tl.float32)
            x = silu_impl(x) * gate
        tl.store(out_ptr + row * K_OUT + offs, round_to_int8(x / scale), mask=mask)


def quantize_w8a8_act(
    input: torch.Tensor,
    output: torch.Tensor,
    oscales: torch.Tensor,
    fuse_glu: bool = False,
) -> None:
    """
    Per-token symmetric int8 quantization of activations.

    With ``fuse_glu``, the input's last dim is split in half and
    ``silu(x) * gate`` is computed before quantizing, so ``output`` has half
    the last dim of ``input``.
    """
    input = input.reshape(-1, input.shape[-1]).contiguous()
    M, K = input.shape
    k_out = K // 2 if fuse_glu else K
    block = min(triton.next_power_of_2(k_out), 2048)
    _quantize_w8a8_act_kernel[(M,)](
        input, output, oscales.reshape(-1), K, k_out, FUSE_GLU=fuse_glu, BLOCK=block
    )
