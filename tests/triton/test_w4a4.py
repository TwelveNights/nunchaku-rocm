"""
Parity tests for the Triton W4A4 (SVDQuant) GEMM, quantizer, and attention.

References are computed in fp32 from the same packed tensors the kernel
reads, so these check the kernel's arithmetic and its unpacking, not just a
self-consistent round trip.
"""

import pytest
import torch

from .conftest import load_kernels

attention = load_kernels("attention")
w4a4 = load_kernels("gemm_w4a4")

DTYPES = [torch.float16, torch.bfloat16]

# e2m1 code -> value, matching _decode_e2m1
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def unpack(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack [M, K//2] uint8 to [M, K] int codes; even index = low nibble."""
    b = packed.to(torch.int32)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    out = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return out[:, :K]


def dequant(packed, scales, K, group, fp4, unsigned=False):
    """Dequantize to fp32 the same way the kernel does."""
    codes = unpack(packed, K)
    if fp4:
        vals = E2M1.to(packed.device)[codes]
    elif unsigned:
        vals = codes.float()
    else:
        vals = torch.where(codes >= 8, codes - 16, codes).float()
    grp = torch.arange(K, device=packed.device) // group
    return vals * scales.float()[grp].t()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (128, 512, 1024), (64, 128, 256)])
def test_gemm_w4a4_int4(dtype, M, N, K):
    group = 64
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.05 + 0.01).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.05 + 0.01).to(dtype)

    out = w4a4.gemm_w4a4(act, wgt, ascales=ascales, wscales=wscales, out_dtype=dtype)

    a = dequant(act, ascales, K, group, fp4=False)
    w = dequant(wgt, wscales, K, group, fp4=False)
    torch.testing.assert_close(out.float(), a @ w.t(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_w4a4_act_unsigned(dtype):
    M, N, K, group = 128, 128, 256, 64
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.05 + 0.01).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.05 + 0.01).to(dtype)

    out = w4a4.gemm_w4a4(act, wgt, ascales=ascales, wscales=wscales, act_unsigned=True, out_dtype=dtype)

    a = dequant(act, ascales, K, group, fp4=False, unsigned=True)
    w = dequant(wgt, wscales, K, group, fp4=False)
    torch.testing.assert_close(out.float(), a @ w.t(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("act_unsigned", [False, True])
def test_gemm_w4a4_int8_dot_matches_int32_reference(act_unsigned):
    """
    Directly targets the int4->int8 dot path (``USE_INT8_DOT``, gemm_w4a4.py):
    per-group scaling is applied to an exact int32 accumulation (no float
    rounding inside the K-loop), so the kernel's output should match a
    reference built the same way -- per-group int32 dot products, scaled and
    summed in fp64 -- far more tightly than the generic fp32-dequant
    references above (which only bound against ~2e-2 relative error from the
    old bf16-dot path's rounding). This is the case flagged as worth
    double-checking in isolation: unsigned int4 codes ([0, 15]) must still be
    valid *signed* int8 operands to ``tl.dot`` (Triton's AMD int8 dot path
    only accepts ``tl.int8``, not ``tl.uint8``), which holds trivially since
    15 fits well within the signed int8 range.
    """
    M, N, K, group = 128, 128, 256, 64
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.05 + 0.01).to(torch.float32)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.05 + 0.01).to(torch.float32)

    out = w4a4.gemm_w4a4(act, wgt, ascales=ascales, wscales=wscales, act_unsigned=act_unsigned, out_dtype=torch.float32)

    a_codes = unpack(act, K).to(torch.int32)
    w_codes = unpack(wgt, K).to(torch.int32)
    if not act_unsigned:
        a_codes = torch.where(a_codes >= 8, a_codes - 16, a_codes)
    w_codes = torch.where(w_codes >= 8, w_codes - 16, w_codes)

    ref = torch.zeros(M, N, dtype=torch.float64, device="cuda")
    for k0 in range(0, K, group):
        a_grp = a_codes[:, k0 : k0 + group]
        w_grp = w_codes[:, k0 : k0 + group]
        # exact int32 dot product per group, matching tl.dot(out_dtype=tl.int32)
        dot_i32 = (a_grp.unsqueeze(1) * w_grp.unsqueeze(0)).sum(dim=-1).to(torch.int32)
        grp = k0 // group
        ref += dot_i32.double() * ascales[grp].double()[:, None] * wscales[grp].double()[None, :]

    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_w4a4_fp4(dtype):
    M, N, K, group = 128, 256, 512, 16
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.05 + 0.01).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.05 + 0.01).to(dtype)

    out = w4a4.gemm_w4a4(act, wgt, ascales=ascales, wscales=wscales, fp4=True, out_dtype=dtype)

    a = dequant(act, ascales, K, group, fp4=True)
    w = dequant(wgt, wscales, K, group, fp4=True)
    torch.testing.assert_close(out.float(), a @ w.t(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_w4a4_bias_lora(dtype):
    M, N, K, R, group = 128, 256, 512, 32, 64
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.05 + 0.01).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.05 + 0.01).to(dtype)
    bias = torch.randn(N, dtype=dtype, device="cuda")
    lora_act = torch.randn(M, R, dtype=torch.float32, device="cuda") * 0.1
    lora_up = torch.randn(N, R, dtype=dtype, device="cuda") * 0.1

    out = w4a4.gemm_w4a4(
        act,
        wgt,
        ascales=ascales,
        wscales=wscales,
        bias=bias,
        lora_act_in=lora_act,
        lora_up=lora_up,
        out_dtype=dtype,
    )

    a = dequant(act, ascales, K, group, fp4=False)
    w = dequant(wgt, wscales, K, group, fp4=False)
    ref = a @ w.t() + lora_act @ lora_up.float().t() + bias.float()
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_w4a4_wcscales_alpha(dtype):
    M, N, K, group = 128, 128, 256, 16
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.05 + 0.01).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.05 + 0.01).to(dtype)
    wcscales = (torch.rand(N, device="cuda") * 0.5 + 0.5).to(dtype)
    alpha = 0.75

    out = w4a4.gemm_w4a4(
        act,
        wgt,
        ascales=ascales,
        wscales=wscales,
        wcscales=wcscales,
        alpha=alpha,
        fp4=True,
        out_dtype=dtype,
    )

    a = dequant(act, ascales, K, group, fp4=True)
    w = dequant(wgt, wscales, K, group, fp4=True)
    ref = (a @ w.t()) * wcscales.float()[None, :] * alpha
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp4", [False, True])
def test_quantize_w4a4_roundtrip(dtype, fp4):
    """Quantize then dequantize should approximate the input."""
    M, K = 64, 512
    group = 16 if fp4 else 64
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    packed, scales, _ = w4a4.quantize_w4a4_act_fuse_lora(x, fp4=fp4)
    assert packed.shape == (M, K // 2)
    assert scales.shape == (K // group, M)

    deq = dequant(packed, scales.float(), K, group, fp4=fp4)
    # 4-bit with per-group scales: check relative error per group is sane
    err = (deq - x.float()).abs().amax()
    scale_max = scales.float().amax()
    assert err <= scale_max * 1.01, f"max abs err {err} exceeds one quantization step {scale_max}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("R", [32, 288])  # 288: runtime LoRAs appended to the rank-32 branch, split over grid axis 1
def test_quantize_w4a4_fuse_lora(dtype, R):
    M, K = 64, 512
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    lora_down = torch.randn(K, R, dtype=dtype, device="cuda") * 0.1

    _, _, lora_act = w4a4.quantize_w4a4_act_fuse_lora(x, lora_down=lora_down)
    torch.testing.assert_close(lora_act, x.float() @ lora_down.float(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_w4a4_smooth(dtype):
    M, K = 32, 256
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    smooth = (torch.rand(K, device="cuda") * 0.5 + 0.75).to(dtype)

    packed, scales, _ = w4a4.quantize_w4a4_act_fuse_lora(x, smooth=smooth)
    deq = dequant(packed, scales.float(), K, 64, fp4=False)
    ref = x.float() / smooth.float()
    assert (deq - ref).abs().amax() <= scales.float().amax() * 1.01


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_w4a4_fuse_glu(dtype):
    M, K = 32, 512
    x = torch.randn(M, 2 * K, dtype=dtype, device="cuda")
    packed, scales, _ = w4a4.quantize_w4a4_act_fuse_lora(x, fuse_glu=True)
    assert packed.shape == (M, K // 2)

    ref = x[:, :K].float() * torch.sigmoid(x[:, K:].float()) * x[:, K:].float()
    deq = dequant(packed, scales.float(), K, 64, fp4=False)
    assert (deq - ref).abs().amax() <= scales.float().amax() * 1.01


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,H,TQ,TKV,D", [(2, 8, 256, 256, 64), (1, 4, 128, 512, 128), (2, 2, 100, 77, 64)])
def test_attention_fp16(dtype, B, H, TQ, TKV, D):
    q = torch.randn(B, H, TQ, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, TKV, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, TKV, D, dtype=dtype, device="cuda")
    o = torch.empty(B, TQ, H * D, dtype=dtype, device="cuda")

    scale = D**-0.5
    attention.attention_fp16(q, k, v, o, scale)

    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    ref = ref.transpose(1, 2).reshape(B, TQ, H * D)
    torch.testing.assert_close(o.float(), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,H,T", [(2, 4, 64), (1, 8, 128)])
def test_gemm_w4a4_rmsnorm_rope(dtype, B, H, T):
    """
    RMSNorm+RoPE fusion on a QKV projection's output, matching
    EpilogueRMSNormRope: per-head RMSNorm(eps=1e-6) then RoPE on Q and K
    thirds of N; V passes through unchanged.
    """
    HEAD_DIM = 128
    M = B * T
    K = 256
    N = 3 * H * HEAD_DIM
    group = 64

    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.02 + 0.005).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.02 + 0.005).to(dtype)
    norm_q = torch.rand(HEAD_DIM, dtype=dtype, device="cuda") * 0.5 + 0.75
    norm_k = torch.rand(HEAD_DIM, dtype=dtype, device="cuda") * 0.5 + 0.75

    # natural (pre-pack_rotemb) rotary tensor: [M, HEAD_DIM // 2, 1, 2], last dim (sin, cos)
    theta = torch.rand(M, HEAD_DIM // 2, device="cuda") * 6.28
    rotary_emb = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1).reshape(M, HEAD_DIM // 2, 1, 2).float()

    out = w4a4.gemm_w4a4(
        act,
        wgt,
        ascales=ascales,
        wscales=wscales,
        norm_q=norm_q,
        norm_k=norm_k,
        rotary_emb=rotary_emb,
        out_dtype=dtype,
    )

    a = dequant(act, ascales, K, group, fp4=False)
    w = dequant(wgt, wscales, K, group, fp4=False)
    raw = a @ w.t()  # fp32 reference of the un-fused GEMM output

    third = N // 3
    q, k, v = raw[:, :third], raw[:, third : 2 * third], raw[:, 2 * third :]

    def rmsnorm_rope(x, norm_w):
        x = x.reshape(M, H, HEAD_DIM)
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        x = x * rstd * norm_w.float()
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        sin = rotary_emb[:, None, :, 0, 0]
        cos = rotary_emb[:, None, :, 0, 1]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.stack([rot_even, rot_odd], dim=-1).reshape(M, H, HEAD_DIM)
        return out.reshape(M, H * HEAD_DIM)

    ref_q = rmsnorm_rope(q, norm_q)
    ref_k = rmsnorm_rope(k, norm_k)
    ref = torch.cat([ref_q, ref_k, v], dim=1)

    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("R2", [32, 288])  # 288: runtime LoRAs on the next layer, LoRA-down chunked over R2
def test_gemm_w4a4_fused_next_layer_quant(dtype, fp4, R2):
    """
    qout/oscales/smooth_factor/lora_down/lora_act_out fusion, matching
    EpilogueGelu + EpilogueQuantize + EpilogueLoraDown: GELU -> int4 unsigned
    shift (skipped for fp4) -> divide by smooth_factor -> per-group quantize
    for ``qout``; ``lora_down`` projects the GELU'd activation *without* the
    shift or smoothing -- EpilogueLoraDown runs directly on the loaded
    activation, strictly before EpilogueQuantize applies either.
    """
    M, K, N = 64, 256, 512
    group = 16 if fp4 else 64

    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    ascales = (torch.rand(K // group, M, device="cuda") * 0.02 + 0.005).to(dtype)
    wscales = (torch.rand(K // group, N, device="cuda") * 0.02 + 0.005).to(dtype)
    smooth = (torch.rand(N, device="cuda") * 0.5 + 0.75).to(dtype)
    lora_down = torch.randn(N, R2, dtype=dtype, device="cuda") * 0.1

    qout = torch.empty(M, N // 2, dtype=torch.uint8, device="cuda")
    qgroup = 16 if fp4 else 64
    oscales = torch.empty(N // qgroup, M, dtype=torch.float8_e4m3fn if fp4 else dtype, device="cuda")
    lora_act_out = torch.empty(M, R2, dtype=torch.float32, device="cuda")

    w4a4.gemm_w4a4(
        act,
        wgt,
        ascales=ascales,
        wscales=wscales,
        fp4=fp4,
        qout=qout,
        oscales=oscales,
        smooth_factor=smooth,
        lora_down=lora_down,
        lora_act_out=lora_act_out,
        out_dtype=dtype,
    )

    a = dequant(act, ascales, K, group, fp4=fp4)
    w = dequant(wgt, wscales, K, group, fp4=fp4)
    raw = a @ w.t()

    gelu = torch.nn.functional.gelu(raw, approximate="tanh")

    # The LoRA-down projection consumes the raw GELU'd activation -- neither
    # the unsigned-activation shift nor the smoothing division reach it.
    ref_lora_act = gelu @ lora_down.float()
    torch.testing.assert_close(lora_act_out, ref_lora_act, rtol=3e-2, atol=3e-2)

    # qout (the main quantized path) DOES include the shift and smoothing.
    shift = 0.0 if fp4 else 0.171875
    shifted_smoothed = (gelu + shift) / smooth.float()
    deq_qout = dequant(qout, oscales.float(), N, qgroup, fp4=fp4, unsigned=not fp4)
    err = (deq_qout - shifted_smoothed).abs().amax()
    scale_max = oscales.float().amax()
    assert err <= scale_max * 1.05, f"max abs err {err} exceeds ~one quant step {scale_max}"
