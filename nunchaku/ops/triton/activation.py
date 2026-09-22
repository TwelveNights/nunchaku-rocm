"""
Triton ports of ``src/kernels/activation_kernels.cu``.
"""

import torch
import triton
import triton.language as tl

from ._common import gelu_fast_impl, gelu_tanh, silu_impl


@triton.jit
def _act_kernel(out_ptr, in_ptr, d, ACT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    for start in tl.range(0, d, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < d
        x = tl.load(in_ptr + row * d + offs, mask=mask)
        if ACT == 0:
            y = silu_impl(x)
        elif ACT == 1:
            y = gelu_tanh(x)
        else:
            y = gelu_fast_impl(x)
        tl.store(out_ptr + row * d + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_act(out: torch.Tensor, input: torch.Tensor, act: int) -> None:
    d = input.shape[-1]
    num_tokens = input.numel() // d
    _act_kernel[(num_tokens,)](out, input.contiguous(), d, ACT=act, BLOCK=1024)


def silu(out: torch.Tensor, input: torch.Tensor) -> None:
    """``out = silu(input)``, elementwise."""
    _launch_act(out, input, 0)


def gelu_new(out: torch.Tensor, input: torch.Tensor) -> None:
    """``out = gelu(input)`` using the tanh approximation."""
    _launch_act(out, input, 1)


def gelu_fast(out: torch.Tensor, input: torch.Tensor) -> None:
    """``out = gelu(input)`` using the 'fast' approximation."""
    _launch_act(out, input, 2)


@triton.jit
def _silu_and_mul_kernel(out_ptr, in_ptr, d, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    for start in tl.range(0, d, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < d
        x = tl.load(in_ptr + row * 2 * d + offs, mask=mask)
        y = tl.load(in_ptr + row * 2 * d + d + offs, mask=mask)
        res = silu_impl(x) * y.to(tl.float32)
        tl.store(out_ptr + row * d + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Gated SiLU: ``out = silu(input[..., :d]) * input[..., d:]``.

    ``input`` has last dim ``2 * d``; ``out`` has last dim ``d``.
    """
    d = input.shape[-1] // 2
    num_tokens = input.numel() // input.shape[-1]
    _silu_and_mul_kernel[(num_tokens,)](out, input.contiguous(), d, BLOCK=1024)
