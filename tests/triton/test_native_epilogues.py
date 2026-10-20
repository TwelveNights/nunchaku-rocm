"""
Tests for the fused epilogues of the native gfx11 W4A4 GEMM (``nunchaku/ops/triton/native/``):

- ``w4a4qk``: per-head RMSNorm + RoPE on the Q/K thirds of a QKV projection;
- ``w4a4gq``: GELU + the next layer's LoRA-down + unsigned int4 quantization (MLP fc1).

Each is compared with the Triton fused kernel and with an fp64 reference built from the same packed tensors, at
FLUX.1 / FLUX.2 [klein] shapes and at small odd shapes; plus dispatch from ``gemm_w4a4()``.
"""

import importlib

import pytest
import torch

from .conftest import load_kernels

w4a4 = load_kernels("gemm_w4a4")
native = importlib.import_module(w4a4.__package__ + ".native.w4a4")

IS_HIP = torch.version.hip is not None
SHIFT = 0.171875


def _arch() -> str:
    return native.device_arch(torch.cuda.current_device()) if torch.cuda.is_available() else ""


requires_gfx11 = pytest.mark.skipif(
    not (IS_HIP and torch.cuda.is_available() and _arch().startswith("gfx11")),
    reason=f"the native W4A4 kernels are RDNA3-only (gfx11); this GPU is {_arch() or 'not AMD'}",
)


@pytest.fixture
def native_env(monkeypatch):
    monkeypatch.delenv("NUNCHAKU_NATIVE_GEMM", raising=False)
    native._reset()
    yield monkeypatch
    native._reset()


def _unpack(p: torch.Tensor, signed: bool) -> torch.Tensor:
    b = p.view(torch.uint8).to(torch.int32)
    x = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(p.shape[0], -1)
    return torch.where(x >= 8, x - 16, x) if signed else x


def _lin64(act, wgt, asc, wsc, bias, la, lu, rows, unsigned=False):
    """fp64 GEMM + LoRA-up (operand rounded to 16 bits, as the kernels) + bias, for the given rows."""
    a = _unpack(act[rows], not unsigned).double()
    w = _unpack(wgt, True).double()
    out = torch.zeros(len(rows), wgt.shape[0], dtype=torch.float64, device=act.device)
    for gi in range(a.shape[1] // 64):
        sl = slice(gi * 64, gi * 64 + 64)
        out += (a[:, sl] @ w[:, sl].T) * asc[gi, rows].double()[:, None] * wsc[gi].double()[None, :]
    if lu is not None:
        out += la[rows].to(lu.dtype).double() @ lu.double().T
    if bias is not None:
        out += bias.double()[None, :]
    return out


def _qk64(lin, nq, nk, rot_rows):
    M, N = lin.shape
    x = lin.clone().view(M, 3, N // 384, 128)
    sc = rot_rows.double().view(M, 1, 64, 2)
    for p, w in ((0, nq), (1, nk)):
        h = x[:, p]
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w.double()
        xe, xo = h[..., 0::2], h[..., 1::2]
        x[:, p] = torch.stack([xe * sc[..., 1] - xo * sc[..., 0], xe * sc[..., 0] + xo * sc[..., 1]], -1).flatten(-2)
    return x.view(M, N)


def _problem(M, N, K, dtype, seed=0, R=32, bias=True, unit=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    if unit:
        one = lambda r: torch.ones(K // 64, r, dtype=dtype, device="cuda")  # noqa: E731
        return act, wgt, one(M), one(N), None, None, None
    asc = (torch.rand(K // 64, M, device="cuda", generator=g) * 0.02 + 0.01).to(dtype)
    wsc = (torch.rand(K // 64, N, device="cuda", generator=g) * 0.02 + 0.01).to(dtype)
    b = (torch.randn(N, device="cuda", generator=g) * 0.3).to(dtype) if bias else None
    la = torch.randn(M, R, device="cuda", generator=g)
    lu = (torch.randn(N, R, device="cuda", generator=g) * 0.05).to(dtype)
    return act, wgt, asc, wsc, b, la, lu


def _rows(M, actual_m):
    return torch.tensor(
        sorted(set(range(min(64, actual_m))) | set(range(max(0, actual_m - 64), actual_m))), device="cuda"
    )


def _ulp(x: torch.Tensor, dtype) -> torch.Tensor:
    mant = 7 if dtype == torch.bfloat16 else 10
    return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30))) - mant)


def _qk_inputs(M, actual_m, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + 100)
    nq = (torch.rand(128, device="cuda", generator=g) + 0.5).to(dtype)
    nk = (torch.rand(128, device="cuda", generator=g) + 0.5).to(dtype)
    ang = torch.rand(actual_m, 64, device="cuda", generator=g) * 300
    rot = torch.stack([torch.sin(ang), torch.cos(ang)], -1).reshape(actual_m, 64, 1, 2)
    return nq, nk, rot


# ---------------------------------------------------------------------------------------------------------------
# Q/K RMSNorm + RoPE

QK_SHAPES = [  # M, N, K, actual_m
    (4608, 9216, 3072, 4608),  # FLUX.1 / klein 4B single blocks
    (512, 12288, 4096, 500),  # klein 9B text stream (padded activations)
    (136, 384, 256, 130),  # partial M tile, one head per third
]


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,K,actual_m", QK_SHAPES)
def test_qk_int_path(native_env, dtype, M, N, K, actual_m):
    """Unit scales, no LoRA/bias: V is the exact integer GEMM rounded once; Q/K correctly rounded from fp64."""
    act, wgt, asc, wsc, *_ = _problem(M, N, K, dtype, unit=True)
    nq, nk, rot = _qk_inputs(M, actual_m, dtype)
    out = torch.empty(actual_m, N, dtype=dtype, device="cuda")
    native.gemm_qk(act, wgt, out, asc, wsc, None, None, None, nq, nk, rot)
    rows = _rows(M, actual_m)
    lin = _lin64(act, wgt, asc, wsc, None, None, None, rows)
    v = 2 * N // 3
    assert torch.equal(out[rows, v:], lin[:, v:].to(dtype))
    ref = _qk64(lin, nq, nk, rot[rows])[:, :v]
    err = (out[rows, :v].double() - ref).abs() / _ulp(torch.maximum(ref.abs(), 1e-3 * ref.abs().max()), dtype)
    assert err.max().item() <= 0.6  # rounded once from an fp32 value within a few hundredths of an output ulp


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("M,N,K,actual_m", QK_SHAPES)
def test_qk_vs_triton_and_fp64(native_env, dtype, has_bias, M, N, K, actual_m):
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, dtype, bias=has_bias)
    nq, nk, rot = _qk_inputs(M, actual_m, dtype)
    buf = torch.full((actual_m + 16, N), float("nan"), dtype=dtype, device="cuda")
    native.gemm_qk(act, wgt, buf[:actual_m], asc, wsc, bias, la, lu, nq, nk, rot)
    assert buf[actual_m:].isnan().all(), "rows past actual_m were written"
    out = buf[:actual_m]
    again = torch.empty_like(out)
    native.gemm_qk(act, wgt, again, asc, wsc, bias, la, lu, nq, nk, rot)
    assert torch.equal(out, again), "not deterministic"

    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    tri = w4a4.gemm_w4a4(
        act,
        wgt,
        out=torch.empty_like(out),
        ascales=asc,
        wscales=wsc,
        bias=bias,
        lora_act_in=la,
        lora_up=lu,
        norm_q=nq,
        norm_k=nk,
        rotary_emb=rot,
    )
    rows = _rows(M, actual_m)
    ref = _qk64(_lin64(act, wgt, asc, wsc, bias, la, lu, rows), nq, nk, rot[rows])
    scale = ref.abs().max().item()
    err_n = (out[rows].double() - ref).abs().max().item() / scale
    err_t = (tri[rows].double() - ref).abs().max().item() / scale
    assert err_n <= 1.05 * err_t + 1e-6, f"native {err_n:.3g} vs triton {err_t:.3g}"
    cr_n = (out[rows] == ref.to(dtype)).float().mean().item()
    cr_t = (tri[rows] == ref.to(dtype)).float().mean().item()
    assert cr_n >= cr_t - 1e-3, f"correctly rounded: native {cr_n:.5f} vs triton {cr_t:.5f}"


# ---------------------------------------------------------------------------------------------------------------
# GELU + LoRA-down + int4 quantize


def _gq_inputs(N, R2, dtype, seed=0, smooth=True):
    g = torch.Generator(device="cuda").manual_seed(seed + 200)
    sm = (torch.rand(N, device="cuda", generator=g) * 1.5 + 0.25).to(dtype) if smooth else None
    ld = (torch.randn(N, R2, device="cuda", generator=g) * 0.02).to(dtype) if R2 else None
    return sm, ld


def _gq_out(M, N, R2, dtype, pad=16):
    qout = torch.zeros(M + pad, N // 2, dtype=torch.uint8, device="cuda")
    osc = torch.empty(N // 64, M + pad, dtype=dtype, device="cuda")
    lao = torch.empty(M + pad, R2, device="cuda") if R2 else None
    return qout, osc, lao


GQ_SHAPES = [(4608, 12288, 3072), (512, 12288, 3072), (144, 256, 256)]  # FLUX.1 fc1 (img+txt / txt), odd


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,K", GQ_SHAPES)
def test_gelu_quant_int_path(native_env, dtype, M, N, K):
    """The optional pre-GELU output is the exact integer GEMM rounded once (unit scales, no LoRA/bias)."""
    act, wgt, asc, wsc, *_ = _problem(M, N, K, dtype, unit=True)
    sm, ld = _gq_inputs(N, 32, dtype)
    qout, osc, lao = _gq_out(M, N, 32, dtype)
    out = torch.empty(M, N, dtype=dtype, device="cuda")
    native.gemm_gelu_quant(act, wgt, asc, wsc, None, None, None, qout, osc, sm, ld, lao, out=out)
    rows = _rows(M, M)
    assert torch.equal(out[rows], _lin64(act, wgt, asc, wsc, None, None, None, rows).to(dtype))


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,K", GQ_SHAPES)
def test_gelu_quant_vs_triton_and_fp64(native_env, dtype, M, N, K):
    R2 = 32
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, dtype)
    sm, ld = _gq_inputs(N, R2, dtype)
    res = []
    for who in ("native", "native", "triton"):
        qout, osc, lao = _gq_out(M, N, R2, dtype)
        if who == "native":
            native.gemm_gelu_quant(act, wgt, asc, wsc, bias, la, lu, qout, osc, sm, ld, lao)
        else:
            native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
            w4a4.gemm_w4a4(
                act,
                wgt,
                ascales=asc,
                wscales=wsc,
                bias=bias,
                lora_act_in=la,
                lora_up=lu,
                qout=qout,
                oscales=osc,
                smooth_factor=sm,
                lora_down=ld,
                lora_act_out=lao,
            )
        res.append((qout[:M], osc[:, :M], lao[:M]))
        assert (qout[M:] == 0).all(), "qout rows past M were written"
    assert all(torch.equal(a, b) for a, b in zip(res[0], res[1])), "not deterministic"
    (qn, sn, ln), _, (qt, st, lt) = res
    cn, ct = _unpack(qn, False), _unpack(qt, False)
    # the codes / scales follow Triton's op sequence; they may differ only where an fp32 rounding difference in the
    # GEMM output moves a value across a rounding midpoint
    assert (cn - ct).abs().max().item() <= 1
    assert (cn != ct).float().mean().item() < 1e-5
    # (fp16's finer ulp turns more of the scales' fp32 differences into a 1-ulp rounding difference)
    assert (sn != st).float().mean().item() < (1e-4 if dtype == torch.bfloat16 else 1e-3)
    assert ((sn.float() - st.float()).abs() <= _ulp(st.float(), dtype)).all()

    rows = _rows(M, M)
    lin = _lin64(act, wgt, asc, wsc, bias, la, lu, rows)
    x = 0.5 * lin * (1 + torch.tanh(0.7978845608028654 * (lin + 0.044715 * lin**3)))
    lref = x @ ld.double()
    e_n = ((ln[rows].double() - lref).abs().max() / lref.abs().max()).item()
    e_t = ((lt[rows].double() - lref).abs().max() / lref.abs().max()).item()
    assert e_n <= 1.05 * e_t + 1e-6, f"lora_act_out error: native {e_n:.3g} vs triton {e_t:.3g}"
    x2 = (x + SHIFT) / sm.double()

    def deq_err(c, s):  # noqa: E306
        d = c[rows].double().view(len(rows), N // 64, 64) * s[:, rows].T.double()[..., None]
        return ((d.reshape(len(rows), N) - x2).pow(2).mean().sqrt() / x2.pow(2).mean().sqrt()).item()

    assert deq_err(cn, sn) <= 1.001 * deq_err(ct, st)


@requires_gfx11
@pytest.mark.parametrize("R2", [16, 48, 64, 0])
@pytest.mark.parametrize("smooth", [True, False])
def test_gelu_quant_ranks_and_options(native_env, R2, smooth):
    """LoRA-down ranks needing 1-2 slabs (and none), no smoothing, no bias / LoRA-up: same as Triton."""
    M, N, K, dtype = 272, 512, 384, torch.bfloat16
    act, wgt, asc, wsc, _, la, lu = _problem(M, N, K, dtype, seed=3)
    sm, ld = _gq_inputs(N, R2, dtype, smooth=smooth)
    outs = []
    for use_native in (True, False):
        native_env.setenv("NUNCHAKU_NATIVE_GEMM", "1" if use_native else "0")
        qout, osc, lao = _gq_out(M, N, R2, dtype)
        w4a4.gemm_w4a4(
            act,
            wgt,
            ascales=asc,
            wscales=wsc,
            lora_act_in=la,
            lora_up=lu,
            qout=qout,
            oscales=osc,
            smooth_factor=sm,
            lora_down=ld,
            lora_act_out=lao,
        )
        outs.append((qout, osc, lao))
    (qn, sn, ln), (qt, st, lt) = outs
    cn, ct = _unpack(qn[:M], False), _unpack(qt[:M], False)
    assert (cn - ct).abs().max().item() <= 1 and (cn != ct).float().mean().item() < 1e-4
    torch.testing.assert_close(sn[:, :M].float(), st[:, :M].float(), rtol=1e-2, atol=0)
    if R2:
        torch.testing.assert_close(ln[:M], lt[:M], rtol=1e-2, atol=1e-2 * lt[:M].abs().max().item())


# ---------------------------------------------------------------------------------------------------------------
# dispatch from gemm_w4a4()


def _spy(monkeypatch, name):
    calls = []
    orig = getattr(native, name)

    def spy(*args, **kwargs):
        r = orig(*args, **kwargs)
        calls.append(r)
        return r

    monkeypatch.setattr(native, name, spy)
    return calls


def _qk_call(M=256, N=384, K=256, dtype=torch.bfloat16, rot_dtype=torch.float32):
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, dtype)
    nq, nk, rot = _qk_inputs(M, M, dtype)
    rot = rot.to(rot_dtype)
    return lambda: w4a4.gemm_w4a4(
        act,
        wgt,
        ascales=asc,
        wscales=wsc,
        bias=bias,
        lora_act_in=la,
        lora_up=lu,
        norm_q=nq,
        norm_k=nk,
        rotary_emb=rot,
        out_dtype=dtype,
    )


def _gq_call(M=256, N=256, K=256, dtype=torch.bfloat16):
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, dtype)
    sm, ld = _gq_inputs(N, 32, dtype)
    qout, osc, lao = _gq_out(M, N, 32, dtype, pad=0)

    def f():
        w4a4.gemm_w4a4(
            act,
            wgt,
            ascales=asc,
            wscales=wsc,
            bias=bias,
            lora_act_in=la,
            lora_up=lu,
            qout=qout,
            oscales=osc,
            smooth_factor=sm,
            lora_down=ld,
            lora_act_out=lao,
        )
        return qout.clone(), osc.clone(), lao.clone()

    return f


@requires_gfx11
def test_dispatch_fused_calls_go_native(native_env):
    qk_calls = _spy(native_env, "try_gemm_qk")
    gq_calls = _spy(native_env, "try_gemm_gelu_quant")
    plain_calls = _spy(native_env, "try_gemm")
    f_qk, f_gq = _qk_call(), _gq_call()
    o_qk, o_gq = f_qk(), f_gq()
    assert qk_calls == [True] and gq_calls == [True] and plain_calls == []
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    r_qk, r_gq = f_qk(), f_gq()
    assert qk_calls == [True, False] and gq_calls == [True, False]
    torch.testing.assert_close(o_qk.float(), r_qk.float(), rtol=1e-2, atol=1e-2 * r_qk.float().abs().max().item())
    assert (_unpack(o_gq[0], False) - _unpack(r_gq[0], False)).abs().max().item() <= 1
    torch.testing.assert_close(o_gq[2], r_gq[2], rtol=1e-2, atol=1e-2 * r_gq[2].abs().max().item())


def test_dispatch_fused_opt_out(native_env):
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    qk_calls = _spy(native_env, "try_gemm_qk")
    gq_calls = _spy(native_env, "try_gemm_gelu_quant")
    _qk_call()()
    _gq_call()()
    assert qk_calls == [False] and gq_calls == [False]
    assert native._state == {}


def test_dispatch_fused_not_on_other_archs(native_env):
    if _arch().startswith("gfx11"):
        native_env.setattr(native, "device_arch", lambda dev: "gfx942")
    qk_calls = _spy(native_env, "try_gemm_qk")
    gq_calls = _spy(native_env, "try_gemm_gelu_quant")
    _qk_call()()
    _gq_call()()
    assert qk_calls == [False] and gq_calls == [False]
    assert native._state == {}


@pytest.mark.parametrize(
    "case",
    ["qk_k128", "qk_rot_dtype", "gq_m16"],
)
def test_dispatch_fused_unsupported_falls_back(native_env, case):
    """Calls the native epilogues cannot take run on Triton (and give the Triton result)."""
    if case == "qk_k128":  # K % 128 != 0
        f, name = _qk_call(K=192), "try_gemm_qk"
    elif case == "qk_rot_dtype":
        f, name = _qk_call(rot_dtype=torch.float16), "try_gemm_qk"
    else:  # M % 16 != 0
        f, name = _gq_call(M=264), "try_gemm_gelu_quant"
    calls = _spy(native_env, name)
    f()
    assert calls == [False]


def test_unsupported_reason_fused_checks():
    M, N, K, dtype = 64, 384, 256, torch.bfloat16
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, dtype)
    nq, nk, rot = _qk_inputs(M, M, dtype)
    rot2 = rot.reshape(M, 128)
    out = torch.empty(M, N, dtype=dtype, device="cuda")
    ok = (act, wgt, out, asc, wsc, bias, la, lu)
    assert native.unsupported_reason_qk(*ok, nq, nk, rot2) is None
    assert native.unsupported_reason_qk(*ok, nq.half(), nk, rot2) is not None
    assert native.unsupported_reason_qk(*ok, nq, nk, rot2.half()) is not None
    assert native.unsupported_reason_qk(*ok, nq[:64], nk, rot2) is not None
    assert native.unsupported_reason_qk(*ok, nq, nk, rot2[:32]) is not None

    sm, ld = _gq_inputs(N, 32, dtype)
    qout, osc, lao = _gq_out(M, N, 32, dtype, pad=0)
    gq = (act, wgt, None, asc, wsc, bias, la, lu)
    assert native.unsupported_reason_gelu_quant(*gq, qout, osc, sm, ld, lao) is None
    assert native.unsupported_reason_gelu_quant(*gq, qout, osc, sm, ld, None) is not None  # half a LoRA-down
    assert native.unsupported_reason_gelu_quant(*gq, qout, osc.half(), sm, ld, lao) is not None  # scale dtype
    assert native.unsupported_reason_gelu_quant(*gq, qout[:, :8], osc, sm, ld, lao) is not None
    assert native.unsupported_reason_gelu_quant(*gq, qout, osc, sm, ld[:, :8], lao[:, :8]) is not None  # R2 % 16
    assert native.unsupported_reason_gelu_quant(*gq, qout, osc, sm.half(), ld, lao) is not None


# ---------------------------------------------------------------------------------------------------------------
# SwiGLU + signed int4 quantize (FLUX.2 [klein] MLP input projections), gemm_w4a4(..., fuse_swiglu=True)


def _swiglu_problem(M, N, K, dtype, R2=32, seed=0, bias=False):
    """[gate | up] problem plus its interleaved copy; quantization parameters are for the (M, N/2) result."""
    act, wgt, asc, wsc, b, la, lu = _problem(M, N, K, dtype, seed=seed, bias=bias)
    idx = native.interleave_swiglu(N, "cuda")
    inter = (wgt[idx].contiguous(), wsc[:, idx].contiguous(), b[idx] if b is not None else None, lu[idx].contiguous())
    sm, ld = _gq_inputs(N // 2, R2, dtype, seed=seed)
    return act, (wgt, wsc, b, lu), inter, asc, la, sm, ld


def _swiglu_call(act, inter, asc, la, sm, ld, M, H, dtype, R2):
    wgt_i, wsc_i, b_i, lu_i = inter
    lao = torch.empty(M + 16, R2, device="cuda")
    qout = torch.zeros(M + 16, H // 2, dtype=torch.uint8, device="cuda")
    osc = torch.empty(H // 64, M + 16, dtype=dtype, device="cuda")
    r = w4a4.gemm_w4a4(
        act,
        wgt_i,
        ascales=asc,
        wscales=wsc_i,
        bias=b_i,
        lora_act_in=la,
        lora_up=lu_i,
        qout=qout,
        oscales=osc,
        smooth_factor=sm,
        lora_down=ld,
        lora_act_out=lao,
        out_dtype=dtype,
        fuse_swiglu=True,
    )
    assert r is None
    return qout, osc, lao


def test_interleave_swiglu_roundtrip():
    N = 192
    idx = native.interleave_swiglu(N)
    assert sorted(idx.tolist()) == list(range(N))
    x = torch.arange(N)[idx].view(1, N // 32, 2, 16)
    assert torch.equal(x[:, :, 0].reshape(-1), torch.arange(N // 2))  # gate
    assert torch.equal(x[:, :, 1].reshape(-1), torch.arange(N // 2, N))  # up


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_swiglu_fallback_matches_unfused(native_env, dtype):
    """Any GPU: with the native path off, fuse_swiglu is bit-identical to the unfused model path (GEMM on the
    [gate | up] weights, Flux2SwiGLU, then the activation quantizer)."""
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    M, N, K, R2 = 264, 512, 384, 32
    act, (wgt, wsc, b, lu), inter, asc, la, sm, ld = _swiglu_problem(M, N, K, dtype, bias=True)
    qf, sf, lf = _swiglu_call(act, inter, asc, la, sm, ld, M, N // 2, dtype, R2)
    x = w4a4.gemm_w4a4(act, wgt, ascales=asc, wscales=wsc, bias=b, lora_act_in=la, lora_up=lu, out_dtype=dtype)
    y = torch.nn.functional.silu(x[:, : N // 2]) * x[:, N // 2 :]
    qu, su, lu_out = w4a4.quantize_w4a4_act_fuse_lora(y, lora_down=ld, smooth=sm)
    assert torch.equal(qf[:M], qu[:M]) and torch.equal(sf[:, :M], su[:, :M]) and torch.equal(lf[:M], lu_out[:M])


SQ_SHAPES = [(4608, 18432, 3072), (512, 18432, 3072), (4096, 24576, 4096), (144, 256, 256)]  # klein 4B / 9B, odd


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,K", SQ_SHAPES)
def test_swiglu_quant_vs_unfused_and_fp64(native_env, dtype, M, N, K):
    """Native fused vs the unfused path (which rounds the GEMM output and the SwiGLU result to 16 bits first) and an
    fp64 reference: the native error is no larger; codes agree up to those roundings."""
    H, R2 = N // 2, 32
    act, (wgt, wsc, b, lu), inter, asc, la, sm, ld = _swiglu_problem(M, N, K, dtype)
    qn, sn, ln = _swiglu_call(act, inter, asc, la, sm, ld, M, H, dtype, R2)
    qn2, sn2, ln2 = _swiglu_call(act, inter, asc, la, sm, ld, M, H, dtype, R2)
    assert (
        torch.equal(qn, qn2) and torch.equal(sn[:, :M], sn2[:, :M]) and torch.equal(ln[:M], ln2[:M])
    ), "not deterministic"
    assert (qn[M:] == 0).all(), "qout rows past M were written"
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    qt, st, lt = _swiglu_call(act, inter, asc, la, sm, ld, M, H, dtype, R2)

    rows = _rows(M, M)
    lin = _lin64(act, wgt, asc, wsc, b, la, lu, rows)
    y = torch.nn.functional.silu(lin[:, :H]) * lin[:, H:]
    lref = y @ ld.double()
    v = y / sm.double()

    def deq_err(q, s):  # noqa: E306
        c = _unpack(q[rows], True).double().view(len(rows), H // 64, 64) * s[:, rows].T.double()[..., None]
        return ((c.reshape(len(rows), H) - v).pow(2).mean().sqrt() / v.pow(2).mean().sqrt()).item()

    def lerr(l):  # noqa: E306, E741
        return ((l[rows].double() - lref).abs().max() / lref.abs().max()).item()

    d_n, d_t = deq_err(qn, sn), deq_err(qt, st)
    assert d_n <= 1.001 * d_t, f"dequantization error: native {d_n:.5f} vs unfused {d_t:.5f}"
    assert lerr(ln) <= 1.05 * lerr(lt) + 1e-6, f"lora_act_out error: native {lerr(ln):.3g} vs unfused {lerr(lt):.3g}"
    cn, ct = _unpack(qn[:M], True), _unpack(qt[:M], True)
    assert (cn - ct).abs().max().item() <= 1
    assert (cn != ct).float().mean().item() < 0.02  # the unfused path's 16-bit roundings move values across midpoints


@requires_gfx11
def test_dispatch_swiglu_goes_native(native_env):
    calls = _spy(native_env, "try_gemm_swiglu_quant")
    M, N, K = 256, 512, 256
    act, _, inter, asc, la, sm, ld = _swiglu_problem(M, N, K, torch.bfloat16)
    _swiglu_call(act, inter, asc, la, sm, ld, M, N // 2, torch.bfloat16, 32)
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    _swiglu_call(act, inter, asc, la, sm, ld, M, N // 2, torch.bfloat16, 32)
    assert calls == [True, False]
