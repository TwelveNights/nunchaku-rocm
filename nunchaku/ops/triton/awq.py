"""
Triton ports of the AWQ W4A16 kernels (``src/kernels/awq/``).

The packed weight layout is the TinyChat one produced by
:func:`nunchaku.models.text_encoders.tinychat_utils.pack_w4`. Decoding it is
done inline so the fp16 weight is never materialised.

Layout recap. Starting from a logical ``[oc, ic]`` uint4 weight, ``pack_w4``:

1. packs four nibbles that are 8 apart along ``ic`` into one int16, giving
   ``A[o, g * 8 + j]`` holding logical element ``[o, g * 32 + m * 8 + j]`` at
   bit ``4 * m``;
2. permutes ``A.view(oc // 4, 4, ic // 64, 16)`` to ``(0, 2, 1, 3)``, giving the
   stored ``B`` of shape ``[oc // 4, ic]`` (int16).

So for a logical ``(n, k)`` the nibble lives at ``B[n // 4, e]``, bit
``4 * ((k % 32) // 8)``, where ``e`` interleaves ``n % 4`` between the two
halves of the ``k`` index. :func:`_decode_nibble` implements that inverse.
"""

import torch
import triton
import triton.language as tl

from ._common import num_warps_for


@triton.jit
def _decode_nibble(b_ptr, n, k, stride_bn):
    """
    Fetch the uint4 value for logical output channel ``n``, input channel ``k``.

    ``b_ptr`` points at the int16 packed tensor of shape ``[oc // 4, ic]``.
    """
    a = n // 4
    b = n % 4
    g = k // 32
    m = (k % 32) // 8
    j = k % 8
    t = g * 8 + j
    c = t // 16
    d = t % 16
    e = c * 64 + b * 16 + d

    raw = tl.load(b_ptr + a * stride_bn + e).to(tl.int32) & 0xFFFF
    return (raw >> (4 * m)) & 0xF


@triton.jit
def _awq_dequant_kernel(
    b_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    N,
    K,
    group_size,
    stride_bn,
    stride_sg,
    stride_sn,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    nn = offs_n[:, None]
    kk = offs_k[None, :]
    mask = (nn < N) & (kk < K)

    q = _decode_nibble(b_ptr, nn, kk, stride_bn).to(tl.float32)
    grp = kk // group_size
    s = tl.load(scale_ptr + grp * stride_sg + nn * stride_sn, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(zero_ptr + grp * stride_sg + nn * stride_sn, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + nn * stride_on + kk, (q * s + z).to(out_ptr.dtype.element_ty), mask=mask)


def awq_dequantize(
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Unpack an AWQ W4 weight to a dense ``[n, k]`` tensor.

    Useful on its own, and used as the GEMM path for larger batches where a
    dense matmul beats decoding the weight once per output tile.
    """
    packed = kernel.view(torch.int16)
    out = torch.empty(n, k, dtype=dtype, device=kernel.device)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(k, BLOCK_K))
    _awq_dequant_kernel[grid](
        packed,
        scaling_factors,
        zeros,
        out,
        n,
        k,
        group_size,
        packed.stride(0),
        scaling_factors.stride(0),
        scaling_factors.stride(1),
        out.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps_for(4),
    )
    return out


@triton.jit
def _awq_gemv_g64_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    N,
    K,
    stride_xm,
    stride_bn,
    stride_sg,
    stride_om,
    BLOCK_A: tl.constexpr,
):
    """
    Fused AWQ GEMV for ``group_size == 64``, streaming the packed weight once.

    Walks the stored int16 tensor ``B[n // 4, :]`` in its own order instead of
    gathering per logical element. For packed row ``a`` and stored column
    ``e = c * 64 + b * 16 + d`` (see the module docstring), the 4 nibbles
    ``m = 0..3`` hold logical ``(n, k) = (4a + b, 64c + (d // 8) * 32 + m * 8 + d % 8)``,
    so one 64-column stored block is exactly one scale group ``c``. Per group:
    ``out[n] += s[c, n] * sum_k x[k] q[n, k] + z[c, n] * sum_k x[k]``.
    """
    pid_n = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    BLOCK_N: tl.constexpr = BLOCK_A * 4

    offs_a = pid_n * BLOCK_A + tl.arange(0, BLOCK_A)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_e = tl.arange(0, 64)
    offs_d = tl.arange(0, 16)
    xk = (offs_d // 8) * 32 + offs_d % 8  # logical k offset of nibble m = 0, within the group

    acc = tl.zeros((BLOCK_N,), tl.float32)
    x_row = x_ptr + row * stride_xm
    for c in tl.range(0, K // 64):
        raw = (
            tl.load(
                b_ptr + offs_a[:, None] * stride_bn + c * 64 + offs_e[None, :],
                mask=(offs_a[:, None] < N // 4),
                other=0,
            ).to(tl.int32)
            & 0xFFFF
        )
        raw = tl.reshape(raw, (BLOCK_N, 16))  # rows: n = 4a + b, cols: d
        base = x_row + c * 64
        x0 = tl.load(base + xk).to(tl.float32)
        x1 = tl.load(base + xk + 8).to(tl.float32)
        x2 = tl.load(base + xk + 16).to(tl.float32)
        x3 = tl.load(base + xk + 24).to(tl.float32)
        part = (raw & 0xF).to(tl.float32) * x0[None, :]
        part += ((raw >> 4) & 0xF).to(tl.float32) * x1[None, :]
        part += ((raw >> 8) & 0xF).to(tl.float32) * x2[None, :]
        part += ((raw >> 12) & 0xF).to(tl.float32) * x3[None, :]
        dotq = tl.sum(part, axis=1)
        xsum = tl.sum(x0 + x1 + x2 + x3, axis=0)
        s = tl.load(scale_ptr + c * stride_sg + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        z = tl.load(zero_ptr + c * stride_sg + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += s * dotq + z * xsum
    tl.store(out_ptr + row * stride_om + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def awq_gemv_w4a16(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """
    AWQ W4A16 GEMV for small batch sizes.

    Matches the signature of the CUDA ``gemv_awq`` op. For ``group_size == 64``
    the weight is decoded on the fly by a fused kernel that reads the packed
    int4 weight exactly once; otherwise it is dequantized to a dense tensor and
    multiplied.
    """
    x = in_feats.reshape(m, k)
    if group_size == 64 and k % 64 == 0 and n % 4 == 0:
        x = x.contiguous()
        packed = kernel.view(torch.int16)
        assert packed.shape == (n // 4, k), f"unexpected packed AWQ weight shape {tuple(packed.shape)}"
        assert scaling_factors.stride(1) == 1 and zeros.stride() == scaling_factors.stride()
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
        BLOCK_A = 16
        _awq_gemv_g64_kernel[(triton.cdiv(n // 4, BLOCK_A), m)](
            x,
            packed,
            scaling_factors,
            zeros,
            out,
            n,
            k,
            x.stride(0),
            packed.stride(0),
            scaling_factors.stride(0),
            out.stride(0),
            BLOCK_A=BLOCK_A,
            num_warps=num_warps_for(4),
        )
        return out.reshape(*in_feats.shape[:-1], n)
    w = awq_dequantize(kernel, scaling_factors, zeros, n, k, group_size, x.dtype)
    return torch.matmul(x, w.t()).reshape(*in_feats.shape[:-1], n)


def awq_gemm_w4a16(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    AWQ W4A16 GEMM.

    The CUDA version is a CUTLASS kernel specialised to ``group_size=128``.
    This port decodes the weight with a Triton kernel and then runs a dense
    matmul, which keeps the packed on-disk format while staying portable.
    """
    shape = in_feats.shape
    k = shape[-1]
    x = in_feats.reshape(-1, k)
    n = scaling_factors.shape[-1]
    w = awq_dequantize(kernel, scaling_factors, zeros, n, k, group_size, x.dtype)
    return torch.matmul(x, w.t()).reshape(*shape[:-1], n)
