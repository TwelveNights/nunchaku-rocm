"""
Parity tests for the Triton GEMM kernels (W8A8, FP16, batched, AWQ).
"""

import pytest
import torch

from .conftest import load_kernels

awq = load_kernels("awq")
gemm_fp16 = load_kernels("gemm_fp16")
gemm_w8a8 = load_kernels("gemm_w8a8")

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N,K", [(256, 512, 1024), (128, 128, 256), (77, 384, 512)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_gemm_w8a8(dtype, M, N, K, has_bias):
    act = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    wgt = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
    ascales = torch.rand(M, dtype=dtype, device="cuda") * 0.01 + 0.001
    wscales = torch.rand(N, dtype=dtype, device="cuda") * 0.01 + 0.001
    bias = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None

    out = torch.empty(M, N, dtype=dtype, device="cuda")
    gemm_w8a8.gemm_w8a8(act, wgt, out, ascales, wscales, bias)

    ref = (act.float() @ wgt.float().t()) * ascales.float()[:, None] * wscales.float()[None, :]
    if has_bias:
        ref = ref + bias.float()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fuse_glu", [False, True])
def test_quantize_w8a8_act(dtype, fuse_glu):
    M, K = 64, 1024
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    k_out = K // 2 if fuse_glu else K
    out = torch.empty(M, k_out, dtype=torch.int8, device="cuda")
    scales = torch.empty(M, dtype=dtype, device="cuda")
    gemm_w8a8.quantize_w8a8_act(x, out, scales, fuse_glu=fuse_glu)

    ref_in = x.float()
    if fuse_glu:
        ref_in = torch.nn.functional.silu(ref_in[:, :k_out]) * ref_in[:, k_out:]
    amax = ref_in.abs().amax(-1, keepdim=True)
    ref_scale = (amax / 127.0).clamp(min=1e-9)
    ref = (ref_in / ref_scale).round().clamp(-128, 127).to(torch.int8)

    assert (out.int() - ref.int()).abs().max() <= 1
    torch.testing.assert_close(scales.float(), ref_scale.squeeze(-1).to(dtype).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N,K", [(256, 512, 1024), (64, 64, 128), (100, 256, 384)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_gemm_fp16(dtype, M, N, K, has_bias):
    x = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=dtype, device="cuda") * 0.1
    bias = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None

    out = gemm_fp16.gemm_fp16(x, w, bias=bias)
    ref = x.float() @ w.float().t()
    if has_bias:
        ref = ref + bias.float()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_gemm_fp16_alpha():
    M, N, K = 128, 128, 256
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1
    out = gemm_fp16.gemm_fp16(x, w, alpha=0.5)
    torch.testing.assert_close(out.float(), 0.5 * (x.float() @ w.float().t()), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("batch,M,N,K", [(4, 32, 32, 64), (2, 64, 128, 256)])
def test_gemm_batched_fp16(batch, M, N, K):
    a = torch.randn(batch, M, K, dtype=torch.float16, device="cuda") * 0.1
    b = torch.randn(batch, N, K, dtype=torch.float16, device="cuda") * 0.1
    out = gemm_fp16.gemm_batched_fp16(a, b)
    ref = torch.bmm(a.float(), b.float().transpose(1, 2))
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def _pack_w4(weight: torch.Tensor) -> torch.Tensor:
    """Reference TinyChat packing, mirroring nunchaku.models.text_encoders.tinychat_utils.pack_w4."""
    oc, ic = weight.shape
    w = weight.view(-1, 4, 8)
    w = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
    w = w.view(oc // 4, 4, ic // 64, 16).permute(0, 2, 1, 3).reshape(oc // 4, ic)
    return w.to(torch.int16)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N,K,group_size", [(256, 512, 128), (128, 256, 64)])
def test_awq_dequantize_roundtrip(dtype, N, K, group_size):
    """The decoder must exactly invert the TinyChat packing."""
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device="cuda")
    packed = _pack_w4(q)

    ng = K // group_size
    scale = torch.ones(ng, N, dtype=dtype, device="cuda")
    zero = torch.zeros(ng, N, dtype=dtype, device="cuda")

    out = awq.awq_dequantize(packed, scale, zero, N, K, group_size, dtype)
    torch.testing.assert_close(out.float(), q.float())


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_dequantize_scaled(dtype):
    N, K, group_size = 128, 256, 64
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device="cuda")
    packed = _pack_w4(q)
    ng = K // group_size
    scale = (torch.rand(ng, N, device="cuda") * 0.1 + 0.01).to(dtype)
    zero = (torch.randn(ng, N, device="cuda") * 0.1).to(dtype)

    out = awq.awq_dequantize(packed, scale, zero, N, K, group_size, dtype)
    grp = torch.arange(K, device="cuda") // group_size
    ref = q.float() * scale.float()[grp].t() + zero.float()[grp].t()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [1, 4])
def test_awq_gemv(dtype, m):
    N, K, group_size = 128, 256, 64
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device="cuda")
    packed = _pack_w4(q)
    ng = K // group_size
    scale = (torch.rand(ng, N, device="cuda") * 0.1 + 0.01).to(dtype)
    zero = (torch.randn(ng, N, device="cuda") * 0.1).to(dtype)
    x = torch.randn(m, K, dtype=dtype, device="cuda") * 0.1

    out = awq.awq_gemv_w4a16(x, packed, scale, zero, m, N, K, group_size)

    grp = torch.arange(K, device="cuda") // group_size
    w = q.float() * scale.float()[grp].t() + zero.float()[grp].t()
    torch.testing.assert_close(out.float(), x.float() @ w.t(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_gemm(dtype):
    N, K, group_size = 256, 512, 128
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device="cuda")
    packed = _pack_w4(q)
    ng = K // group_size
    scale = (torch.rand(ng, N, device="cuda") * 0.1 + 0.01).to(dtype)
    zero = (torch.randn(ng, N, device="cuda") * 0.1).to(dtype)
    x = torch.randn(32, K, dtype=dtype, device="cuda") * 0.1

    out = awq.awq_gemm_w4a16(x, packed, scale, zero, group_size)
    grp = torch.arange(K, device="cuda") // group_size
    w = q.float() * scale.float()[grp].t() + zero.float()[grp].t()
    torch.testing.assert_close(out.float(), x.float() @ w.t(), rtol=3e-2, atol=3e-2)
