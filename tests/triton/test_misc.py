"""
Parity tests for the Triton elementwise / activation / norm kernels.

These compare against torch reference implementations rather than the CUDA
extension, so they run on any GPU (including AMD, where the CUDA backend
cannot be built at all).
"""

import pytest
import torch

from .conftest import load_kernels

activation = load_kernels("activation")
layernorm = load_kernels("layernorm")
misc = load_kernels("misc")

DTYPES = [torch.float16, torch.bfloat16]


def _tol(dtype):
    return (2e-2, 2e-2) if dtype is torch.bfloat16 else (1e-2, 1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_add(dtype):
    a = torch.randn(4096, 128, dtype=dtype, device="cuda")
    b = torch.randn(4096, 128, dtype=dtype, device="cuda")
    torch.testing.assert_close(misc.add(a, b), a + b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mul_add(dtype):
    x = torch.randn(256, 512, dtype=dtype, device="cuda")
    scale = torch.randn(512, dtype=dtype, device="cuda")
    bias = torch.randn(512, dtype=dtype, device="cuda")
    ref = x * scale + bias
    misc.mul_add(x, scale, bias)
    torch.testing.assert_close(x, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mul_add_no_scale(dtype):
    x = torch.randn(256, 512, dtype=dtype, device="cuda")
    bias = torch.randn(512, dtype=dtype, device="cuda")
    ref = x + bias
    misc.mul_add(x, None, bias)
    torch.testing.assert_close(x, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_scale,batch_bias", [(True, True), (False, False), (True, False)])
def test_mul_add_batch(dtype, batch_scale, batch_bias):
    B, M, N = 4, 32, 256
    x = torch.randn(B, M, N, dtype=dtype, device="cuda")
    scale = (
        torch.randn(B, 1, N, dtype=dtype, device="cuda")
        if batch_scale
        else torch.randn(1, N, dtype=dtype, device="cuda")
    )
    bias = (
        torch.randn(B, 1, N, dtype=dtype, device="cuda")
        if batch_bias
        else torch.randn(1, N, dtype=dtype, device="cuda")
    )
    shift = 1.0

    ref = x * (scale + shift) + bias
    misc.mul_add_batch(x, scale, batch_scale, shift, bias, batch_bias)
    torch.testing.assert_close(x, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_embedding(dtype):
    lookup = torch.randn(1000, 256, dtype=dtype, device="cuda")
    ids = torch.randint(0, 1000, (64,), dtype=torch.int32, device="cuda")
    torch.testing.assert_close(misc.embedding(ids, lookup), lookup[ids.long()])


def test_argmax_sample():
    logits = torch.randn(128, 3000, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(misc.argmax_sample(logits), logits.argmax(-1).to(torch.int32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_splitqkv(dtype):
    heads, head_dim, tokens = 8, 64, 100
    dim = heads * head_dim
    qkv = torch.randn(tokens, 3 * dim, dtype=dtype, device="cuda")
    q = torch.empty(tokens, heads, head_dim, dtype=dtype, device="cuda")
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    misc.splitqkv(qkv, q, k, v)
    torch.testing.assert_close(q.reshape(tokens, dim), qkv[:, :dim])
    torch.testing.assert_close(k.reshape(tokens, dim), qkv[:, dim : 2 * dim])
    torch.testing.assert_close(v.reshape(tokens, dim), qkv[:, 2 * dim :])


@pytest.mark.parametrize("n", [2, 3, 6])
def test_split_mod(n):
    x = torch.randn(16, 64 * n, dtype=torch.float16, device="cuda")
    outs = misc.split_mod(x, n)
    flat = x.reshape(-1)
    for k, out in enumerate(outs):
        torch.testing.assert_close(out.reshape(-1), flat[k::n])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fuse_gelu", [False, True])
def test_quant_static(dtype, fuse_gelu):
    x = torch.randn(64, 1024, dtype=dtype, device="cuda")
    scale = 0.05
    ref = x.float()
    if fuse_gelu:
        ref = torch.nn.functional.gelu(ref, approximate="tanh")
    ref = (ref / scale).round().clamp(-128, 127).to(torch.int8)
    out = misc.quant_static(x, scale, fuse_gelu=fuse_gelu)
    # allow off-by-one from differing rounding of exact .5 ties
    assert (out.int() - ref.int()).abs().max() <= 1


def test_cast():
    x = torch.randn(128, 128, dtype=torch.float32, device="cuda")
    out = torch.empty(128, 128, dtype=torch.float16, device="cuda")
    misc.cast(x, out)
    torch.testing.assert_close(out, x.half())


def test_topk():
    x = torch.randn(32, 512, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(misc.topk(x, 8), x.topk(8, -1).indices.to(torch.int32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu(dtype):
    x = torch.randn(64, 1024, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    activation.silu(out, x)
    torch.testing.assert_close(out, torch.nn.functional.silu(x.float()).to(dtype), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul(dtype):
    x = torch.randn(64, 2048, dtype=dtype, device="cuda")
    out = torch.empty(64, 1024, dtype=dtype, device="cuda")
    activation.silu_and_mul(out, x)
    ref = (torch.nn.functional.silu(x[:, :1024].float()) * x[:, 1024:].float()).to(dtype)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_new(dtype):
    x = torch.randn(64, 1024, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    activation.gelu_new(out, x)
    ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(dtype)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_fast(dtype):
    x = torch.randn(64, 1024, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    activation.gelu_fast(out, x)
    ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(dtype)
    # the 'fast' variant is a coarser approximation; compare loosely
    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [256, 1536, 4096])
def test_rms_norm(dtype, hidden):
    x = torch.randn(64, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    layernorm.rms_norm(out, x, w, 1e-6)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype) * w
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_quant_per_token(dtype):
    hidden = 1024
    x = torch.randn(32, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    out = torch.empty(32, hidden, dtype=torch.int8, device="cuda")
    scaling = torch.empty(32, dtype=torch.float16, device="cuda")
    layernorm.rms_norm_general(out, x, w, scaling, 1e-6, use_per_token_quant=True)

    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    amax = normed.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    ref = (normed * (127.0 / amax)).round().clamp(-128, 127).to(torch.int8)
    assert (out.int() - ref.int()).abs().max() <= 1
    torch.testing.assert_close(scaling.float(), (amax / 127.0).squeeze(-1).half().float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_layernorm_general(dtype, affine):
    hidden = 1024
    x = torch.randn(32, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda") if affine else None
    b = torch.randn(hidden, dtype=dtype, device="cuda") if affine else None
    out = torch.empty_like(x)
    layernorm.layernorm_general(out, x, w, b, 1e-6)
    ref = torch.nn.functional.layer_norm(
        x.float(), (hidden,), w.float() if affine else None, b.float() if affine else None, 1e-6
    )
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
