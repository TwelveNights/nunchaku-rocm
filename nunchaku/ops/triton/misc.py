"""
Triton ports of ``src/kernels/misc_kernels.cu``.
"""

import torch
import triton
import triton.language as tl

from ._common import ceil_div, gelu_tanh, round_to_int8


@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise ``a + b`` for contiguous same-shape tensors."""
    assert a.shape == b.shape and a.dtype == b.dtype
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    _add_kernel[(ceil_div(n, 1024),)](a, b, out, n, BLOCK=1024)
    return out


@triton.jit
def _mul_add_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    scale_shift,
    n,
    mod_scale,
    mod_bias,
    stride_x,
    stride_scale,
    stride_bias,
    HAS_SCALE: tl.constexpr,
    CLAMP_FP16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    x = tl.load(x_ptr + offs + stride_x * batch, mask=mask)
    b = tl.load(bias_ptr + (offs % mod_bias) + stride_bias * batch, mask=mask)

    if HAS_SCALE:
        s = tl.load(scale_ptr + (offs % mod_scale) + stride_scale * batch, mask=mask)
        out = x * (s + scale_shift.to(s.dtype)) + b
    else:
        out = x + b

    if CLAMP_FP16:
        out = tl.minimum(tl.maximum(out, -65504.0), 65504.0)

    tl.store(x_ptr + offs + stride_x * batch, out, mask=mask)


def mul_add(x: torch.Tensor, scale: torch.Tensor | None, bias: torch.Tensor) -> None:
    """
    In-place ``x = x * scale + bias`` with broadcasting by modular indexing.

    ``scale`` may be None, in which case the multiply is skipped.
    """
    n = x.numel()
    _mul_add_kernel[(ceil_div(n, 1024), 1)](
        x,
        scale if scale is not None else x,
        bias,
        0.0,
        n,
        scale.numel() if scale is not None else 1,
        bias.numel(),
        0,
        0,
        0,
        HAS_SCALE=scale is not None,
        CLAMP_FP16=x.dtype == torch.float16,
        BLOCK=1024,
    )


def mul_add_batch(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    batch_scale: bool,
    scale_shift: float,
    bias: torch.Tensor,
    batch_bias: bool,
) -> None:
    """
    Batched in-place ``x = x * (scale + scale_shift) + bias``.

    ``batch_scale`` / ``batch_bias`` select whether the corresponding tensor is
    indexed per batch element or broadcast across the batch.
    """
    batch_size = x.shape[0]
    numel = x.numel() // batch_size
    numel_scale = (scale.numel() // (batch_size if batch_scale else 1)) if scale is not None else 1
    numel_bias = bias.numel() // (batch_size if batch_bias else 1)

    _mul_add_kernel[(ceil_div(numel, 1024), batch_size)](
        x,
        scale if scale is not None else x,
        bias,
        scale_shift,
        numel,
        numel_scale,
        numel_bias,
        x.stride(0),
        scale.stride(0) if (scale is not None and batch_scale) else 0,
        bias.stride(0) if batch_bias else 0,
        HAS_SCALE=scale is not None,
        CLAMP_FP16=x.dtype == torch.float16,
        BLOCK=1024,
    )


@triton.jit
def _embedding_kernel(ids_ptr, out_ptr, lookup_ptr, embed_dim, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(ids_ptr + row).to(tl.int64)
    for start in tl.range(0, embed_dim, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < embed_dim
        val = tl.load(lookup_ptr + token * embed_dim + offs, mask=mask)
        tl.store(out_ptr + row.to(tl.int64) * embed_dim + offs, val, mask=mask)


def embedding(input_id: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    """Gather rows of ``lookup`` indexed by ``input_id``."""
    assert lookup.ndim == 2
    embed_dim = lookup.shape[-1]
    out = torch.empty(*input_id.shape, embed_dim, dtype=lookup.dtype, device=input_id.device)
    ids = input_id.contiguous().reshape(-1)
    _embedding_kernel[(ids.numel(),)](ids, out, lookup.contiguous(), embed_dim, BLOCK=1024)
    return out


@triton.jit
def _argmax_kernel(x_ptr, out_ptr, hidden, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    best_val = tl.full((BLOCK,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK,), tl.int32)
    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hidden
        v = tl.load(x_ptr + row.to(tl.int64) * hidden + offs, mask=mask, other=float("-inf")).to(tl.float32)
        take = v > best_val
        best_val = tl.where(take, v, best_val)
        best_idx = tl.where(take, offs.to(tl.int32), best_idx)
    m = tl.max(best_val, axis=0)
    # lowest index among ties, matching a sequential scan
    idx = tl.min(tl.where(best_val == m, best_idx, 0x7FFFFFFF), axis=0)
    tl.store(out_ptr + row, idx)


def argmax_sample(logits: torch.Tensor) -> torch.Tensor:
    """Per-row argmax over a 2-D logits tensor."""
    assert logits.ndim == 2
    out = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    _argmax_kernel[(logits.shape[0],)](logits.contiguous(), out, logits.shape[1], BLOCK=1024)
    return out


@triton.jit
def _splitqkv_kernel(qkv_ptr, q_ptr, k_ptr, v_ptr, dim_q, dim_k, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    total = dim_q + 2 * dim_k
    for start in tl.range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        data = tl.load(qkv_ptr + row * total + offs, mask=mask)

        in_q = offs < dim_q
        in_k = (offs >= dim_q) & (offs < dim_q + dim_k)
        in_v = offs >= dim_q + dim_k

        tl.store(q_ptr + row * dim_q + offs, data, mask=mask & in_q)
        tl.store(k_ptr + row * dim_k + offs - dim_q, data, mask=mask & in_k)
        tl.store(v_ptr + row * dim_k + offs - dim_q - dim_k, data, mask=mask & in_v)


def splitqkv(qkv: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Split a fused QKV tensor into separate q/k/v buffers."""
    dim_q = q.shape[-1] * q.shape[-2]
    dim_k = k.shape[-1] * k.shape[-2]
    dim_v = v.shape[-1] * v.shape[-2]
    assert dim_k == dim_v
    assert dim_q + dim_k + dim_v == qkv.shape[-1]
    num_tokens = qkv.numel() // qkv.shape[-1]
    _splitqkv_kernel[(num_tokens,)](qkv.contiguous(), q, k, v, dim_q, dim_k, BLOCK=1024)


@triton.jit
def _split_mod_kernel(in_ptr, out_ptr, n_out, N: tl.constexpr, IDX: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    val = tl.load(in_ptr + offs * N + IDX, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


def split_mod(input: torch.Tensor, n: int) -> list[torch.Tensor]:
    """
    De-interleave the last dimension into ``n`` tensors, taking every n-th element.
    """
    assert input.shape[-1] % n == 0
    shape_out = list(input.shape)
    shape_out[-1] //= n
    inp = input.contiguous()
    n_out = inp.numel() // n
    outs = [torch.empty(shape_out, dtype=input.dtype, device=input.device) for _ in range(n)]
    for k in range(n):
        _split_mod_kernel[(ceil_div(n_out, 1024),)](inp, outs[k], n_out, N=n, IDX=k, BLOCK=1024)
    return outs


@triton.jit
def _quant_static_kernel(x_ptr, out_ptr, rscale, n, FUSE_GELU: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    if FUSE_GELU:
        x = gelu_tanh(x)
    tl.store(out_ptr + offs, round_to_int8(x * rscale), mask=mask)


def quant_static(x: torch.Tensor, scale: float, fuse_gelu: bool = False) -> torch.Tensor:
    """Static (per-tensor) int8 quantization, optionally fusing GELU first."""
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    n = x.numel()
    _quant_static_kernel[(ceil_div(n, 1024),)](x, out, 1.0 / scale, n, FUSE_GELU=fuse_gelu, BLOCK=1024)
    return out


def quant_static_fuse_gelu(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Static int8 quantization with a fused GELU."""
    return quant_static(x, scale, fuse_gelu=True)


@triton.jit
def _cast_kernel(in_ptr, out_ptr, n, CLAMP_FP16: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
    if CLAMP_FP16:
        val = tl.minimum(tl.maximum(val, -65504.0), 65504.0)
    tl.store(out_ptr + offs, val.to(out_ptr.dtype.element_ty), mask=mask)


def cast(input: torch.Tensor, output: torch.Tensor) -> None:
    """Cast ``input`` into the (pre-allocated) ``output``, saturating for fp16."""
    assert input.is_contiguous() and output.is_contiguous()
    assert input.shape == output.shape
    n = input.numel()
    _cast_kernel[(ceil_div(n, 1024),)](input, output, n, CLAMP_FP16=output.dtype == torch.float16, BLOCK=1024)


def topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Indices of the top-``k`` values along the last dim, largest first.

    The CUDA version runs an insertion sort in registers; on a GPU with a fast
    sort available there is no reason to reimplement that in Triton, so this
    defers to ``torch.topk`` and just matches the int32 output dtype.
    """
    return x.topk(k, dim=-1, largest=True, sorted=True).indices.to(torch.int32)
