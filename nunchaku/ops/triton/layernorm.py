"""
Triton ports of ``src/kernels/layernorm_kernels.cu``.
"""

import torch
import triton
import triton.language as tl

from ._common import round_to_int8


@triton.jit
def _rms_norm_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    eps,
    hidden,
    USE_QUANT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)

    var = tl.zeros((BLOCK,), tl.float32)
    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(in_ptr + row * hidden + offs, mask=offs < hidden, other=0.0).to(tl.float32)
        var += x * x
    rstd = tl.rsqrt(tl.sum(var, axis=0) / hidden + eps)

    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(in_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        if USE_QUANT:
            tl.store(out_ptr + row * hidden + offs, round_to_int8(x * rstd * w.to(tl.float32)), mask=mask)
        else:
            # the CUDA kernel narrows to the input dtype before scaling by the
            # weight, so round the same way to stay bit-comparable
            normed = (x * rstd).to(in_ptr.dtype.element_ty)
            tl.store(out_ptr + row * hidden + offs, normed * w, mask=mask)


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_quant: bool = False,
) -> None:
    """
    RMSNorm: ``out = input * rsqrt(mean(input^2) + eps) * weight``.

    With ``use_quant``, the result is quantized to int8 with a fixed 1.0 scale.
    """
    hidden = input.shape[-1]
    num_tokens = input.numel() // hidden
    block = min(triton.next_power_of_2(hidden), 2048)
    _rms_norm_kernel[(num_tokens,)](
        out, input.contiguous(), weight.contiguous(), epsilon, hidden, USE_QUANT=use_quant, BLOCK=block
    )


@triton.jit
def _rms_norm_general_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    sum_ptr,
    eps,
    hidden,
    PER_TOKEN: tl.constexpr,
    FUSE_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)

    var = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(in_ptr + row * hidden + offs, mask=offs < hidden, other=0.0).to(tl.float32)
        var += x * x
        if FUSE_SUM:
            acc += x
    rstd = tl.rsqrt(tl.sum(var, axis=0) / hidden + eps)
    if FUSE_SUM:
        tl.store(sum_ptr + row, tl.sum(acc, axis=0).to(sum_ptr.dtype.element_ty))

    if PER_TOKEN:
        amax = tl.full((BLOCK,), 1e-6, tl.float32)
        for start in tl.range(0, hidden, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hidden
            x = tl.load(in_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.where(mask, tl.abs(x * rstd * w), 0.0))
        abs_max = tl.max(amax, axis=0)
        rscale = 127.0 / abs_max
        tl.store(scale_ptr + row, (abs_max / 127.0).to(scale_ptr.dtype.element_ty))
    else:
        rscale = tl.load(scale_ptr).to(tl.float32)

    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(in_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * hidden + offs, round_to_int8(x * rstd * w * rscale), mask=mask)


def rms_norm_general(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scaling: torch.Tensor,
    epsilon: float,
    use_per_token_quant: bool = True,
    input_sum: torch.Tensor | None = None,
) -> None:
    """
    RMSNorm with int8 output and either per-token or per-tensor scaling.

    With per-token scaling, ``scaling`` is written with the chosen scale per
    row; otherwise it is read as a single per-tensor scale. Passing
    ``input_sum`` additionally accumulates the per-row sum of the input, which
    is the ``rms_norm_general_fuse_sum`` variant.
    """
    hidden = input.shape[-1]
    num_tokens = input.numel() // hidden
    block = min(triton.next_power_of_2(hidden), 2048)
    _rms_norm_general_kernel[(num_tokens,)](
        out,
        input.contiguous(),
        weight.contiguous(),
        scaling,
        input_sum if input_sum is not None else scaling,
        epsilon,
        hidden,
        PER_TOKEN=use_per_token_quant,
        FUSE_SUM=input_sum is not None,
        BLOCK=block,
    )


@triton.jit
def _layernorm_kernel(
    out_ptr, in_ptr, w_ptr, b_ptr, eps, hidden, HAS_W: tl.constexpr, HAS_B: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)

    s = tl.zeros((BLOCK,), tl.float32)
    sq = tl.zeros((BLOCK,), tl.float32)
    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(in_ptr + row * hidden + offs, mask=offs < hidden, other=0.0).to(tl.float32)
        s += x
        sq += x * x

    mean = tl.sum(s, axis=0) / hidden
    # Var[x] = E[x^2] - E[x]^2, matching USE_DIFF_OF_SQUARES=true
    var = tl.sum(sq, axis=0) / hidden - mean * mean
    rstd = tl.rsqrt(var + eps)

    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(in_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        if HAS_W:
            y *= tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_B:
            y += tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * hidden + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def layernorm_general(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    epsilon: float,
) -> None:
    """Standard LayerNorm with optional affine weight and bias."""
    hidden = input.shape[-1]
    num_tokens = input.numel() // hidden
    block = min(triton.next_power_of_2(hidden), 2048)
    _layernorm_kernel[(num_tokens,)](
        out,
        input.contiguous(),
        weight if weight is not None else input,
        bias if bias is not None else input,
        epsilon,
        hidden,
        HAS_W=weight is not None,
        HAS_B=bias is not None,
        BLOCK=block,
    )
