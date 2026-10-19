"""
Tests for the native (hand-written HIP) kernels under nunchaku/ops/triton/native/.

- The runtime-compilation infrastructure (HIP runtime binding, hipRTC, code
  object cache) is exercised on any AMD GPU with a tiny fixture kernel.
- The gfx11 int4 W4A4 GEMM is checked against an fp64 reference and the
  Triton kernel (gfx11 only).
- Dispatch from ``gemm_w4a4()``: which calls go native, opt-out, and
  fallback to Triton on unsupported calls or build failures.
"""

import ctypes
import importlib
import shutil

import pytest
import torch

from .conftest import load_kernels

w4a4 = load_kernels("gemm_w4a4")  # also imports the native package as _nunchaku_triton.native
native = importlib.import_module(w4a4.__package__ + ".native.w4a4")
build = importlib.import_module(w4a4.__package__ + ".native.build")

IS_HIP = torch.version.hip is not None


def _arch() -> str:
    return native.device_arch(torch.cuda.current_device()) if torch.cuda.is_available() else ""


requires_hip = pytest.mark.skipif(not IS_HIP, reason="needs a ROCm build of torch (HIP runtime + hipRTC)")
requires_gfx11 = pytest.mark.skipif(
    not (IS_HIP and torch.cuda.is_available() and _arch().startswith("gfx11")),
    reason=f"the native W4A4 GEMM is RDNA3-only (gfx11); this GPU is {_arch() or 'not AMD'}",
)


@pytest.fixture
def native_env(monkeypatch):
    """Default dispatch mode and a clean native-kernel state, restored afterwards."""
    monkeypatch.delenv("NUNCHAKU_NATIVE_GEMM", raising=False)
    native._reset()
    yield monkeypatch
    native._reset()


# ---------------------------------------------------------------------------------------------------------------
# runtime compilation infrastructure (any AMD GPU)

_FIXTURE_KERNEL = r"""
#include "hip_prelude.h"
struct AxpyArgs { float* y; const float* x; float alpha; int n; };
EXTC __global__ void __launch_bounds__(256) axpy(AxpyArgs a) {
  const int i = bid_x() * 256 + tid_x();
  if (i < a.n) a.y[i] += a.alpha * a.x[i];
}
"""


class _AxpyArgs(ctypes.Structure):
    _fields_ = [("y", ctypes.c_void_p), ("x", ctypes.c_void_p), ("alpha", ctypes.c_float), ("n", ctypes.c_int)]


@requires_hip
def test_hiprtc_compile_load_launch_and_cache(tmp_path, monkeypatch):
    """Compile a fixture kernel for this GPU with hipRTC, launch it via the HIP runtime, then hit the cache."""
    from importlib import import_module

    hip = import_module(w4a4.__package__ + ".native.hip")
    src = tmp_path / "src"
    src.mkdir()
    (src / "axpy.hip").write_text(_FIXTURE_KERNEL)
    shutil.copy(build.SOURCE_DIR / "hip_prelude.h", src / "hip_prelude.h")
    monkeypatch.setenv("NUNCHAKU_CACHE_DIR", str(tmp_path / "cache"))
    dev = torch.cuda.current_device()
    arch = native.device_arch(dev)

    code, cached = build.code_object("axpy.hip", ["hip_prelude.h"], arch, ["-O3"], "test kernel", source_dir=src)
    assert not cached and len(code) > 0
    entries = list((tmp_path / "cache" / "native").glob(f"axpy-{arch}-*.co"))
    assert len(entries) == 1 and entries[0].read_bytes() == code
    assert not list((tmp_path / "cache" / "native").glob("*.tmp"))

    code2, cached2 = build.code_object("axpy.hip", ["hip_prelude.h"], arch, ["-O3"], "test kernel", source_dir=src)
    assert cached2 and code2 == code
    # different options -> different cache entry
    _, cached3 = build.code_object("axpy.hip", ["hip_prelude.h"], arch, ["-O2"], "test kernel", source_dir=src)
    assert not cached3
    assert len(list((tmp_path / "cache" / "native").glob(f"axpy-{arch}-*.co"))) == 2

    mod = hip.Module(code, dev)
    n = 1000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    expect = y + 0.5 * x
    mod.launch("axpy", (n + 255) // 256, 256, (_AxpyArgs(y.data_ptr(), x.data_ptr(), 0.5, n),))
    torch.testing.assert_close(y, expect)

    with pytest.raises(hip.HipError):
        mod.function("no_such_kernel")


@requires_hip
def test_hiprtc_compile_error_is_reported(tmp_path, monkeypatch):
    hiprtc = importlib.import_module(w4a4.__package__ + ".native.hiprtc")
    (tmp_path / "bad.hip").write_text("this is not C++;")
    monkeypatch.setenv("NUNCHAKU_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(hiprtc.HipRtcError, match="compile"):
        build.code_object(
            "bad.hip", [], native.device_arch(torch.cuda.current_device()), [], "bad", source_dir=tmp_path
        )


def test_cache_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NUNCHAKU_CACHE_DIR", str(tmp_path))
    assert build.cache_dir() == tmp_path / "native"


# ---------------------------------------------------------------------------------------------------------------
# the gfx11 W4A4 GEMM


def _unpack(p: torch.Tensor, signed: bool) -> torch.Tensor:
    b = p.to(torch.int32)
    x = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(p.shape[0], -1)
    return torch.where(x >= 8, x - 16, x) if signed else x


def _ref64(act, wgt, asc, wsc, bias, la, lu, unsigned, rows):
    """fp64 reference for the given output rows (LoRA operand rounded to 16 bits, as the kernels do)."""
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


def _problem(M, N, K, R, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    asc = (torch.rand(K // 64, M, device="cuda", generator=g) * 0.02 + 0.01).to(dtype)
    wsc = (torch.rand(K // 64, N, device="cuda", generator=g) * 0.02 + 0.01).to(dtype)
    bias = (torch.randn(N, device="cuda", generator=g) * 0.1).to(dtype)
    la = torch.randn(M, max(R, 16), device="cuda", generator=g)[:, :R]
    lu = (torch.randn(N, R, device="cuda", generator=g) * 0.02).to(dtype)
    return act, wgt, asc, wsc, bias, la, lu


def _sample_rows(M, actual_m):
    rows = sorted(set(range(min(64, actual_m))) | set(range(max(0, actual_m - 64), actual_m)))
    return torch.tensor(rows, device="cuda")


FLUX_SHAPES = [(4608, 3072, 3072), (512, 3072, 12288), (4096, 12288, 3072)]
ODD_SHAPES = [(8, 8, 128), (40, 200, 128), (264, 72, 384), (136, 264, 256)]


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("M,N,K", FLUX_SHAPES + ODD_SHAPES)
def test_native_int_path_bit_exact(native_env, dtype, unsigned, M, N, K):
    """Unit scales, no bias/LoRA: the output must be the exact integer GEMM, rounded once."""
    act, wgt, *_ = _problem(M, N, K, 0, dtype)
    one_a = torch.ones(K // 64, M, dtype=dtype, device="cuda")
    one_w = torch.ones(K // 64, N, dtype=dtype, device="cuda")
    out = torch.empty(M, N, dtype=dtype, device="cuda")
    native.gemm(act, wgt, out, one_a, one_w, act_unsigned=unsigned)
    rows = _sample_rows(M, M)
    exact = (_unpack(act[rows], not unsigned).double() @ _unpack(wgt, True).double().T).to(dtype)
    assert torch.equal(out[rows], exact)


@requires_gfx11
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("R", [0, 32])
@pytest.mark.parametrize("M,N,K,actual_m", [(4608, 3072, 3072, 4600), (512, 3072, 12288, 512), (264, 200, 384, 250)])
def test_native_vs_triton_and_fp64(native_env, dtype, unsigned, has_bias, R, M, N, K, actual_m):
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, R, dtype)
    bias = bias if has_bias else None
    la, lu = (la, lu) if R else (None, None)
    buf = torch.full((actual_m + 16, N), float("nan"), dtype=dtype, device="cuda")
    out = buf[:actual_m]
    native.gemm(act, wgt, out, asc, wsc, bias, la, lu, unsigned)
    assert buf[actual_m:].isnan().all(), "rows past actual_m were written"

    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    ref_t = w4a4.gemm_w4a4(
        act,
        wgt,
        out=torch.empty(actual_m, N, dtype=dtype, device="cuda"),
        ascales=asc,
        wscales=wsc,
        bias=bias,
        lora_act_in=la,
        lora_up=lu,
        act_unsigned=unsigned,
    )
    rows = _sample_rows(M, actual_m)
    ref = _ref64(act, wgt, asc, wsc, bias, la, lu, unsigned, rows)
    scale = ref.abs().max().item()
    err_n = (out[rows].double() - ref).abs().max().item() / scale
    err_t = (ref_t[rows].double() - ref).abs().max().item() / scale
    # both are fp32-accumulated and rounded once to 16 bits: within a couple of output ulps of fp64
    ulp = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-11
    assert err_n <= 2 * ulp, f"native error {err_n:.3g} (triton {err_t:.3g})"
    assert err_n <= 2 * err_t + 1e-6
    torch.testing.assert_close(out.float(), ref_t.float(), rtol=2 * ulp, atol=2 * ulp * scale)


@requires_gfx11
@pytest.mark.parametrize("tile", native.TILES)
def test_native_every_tile(native_env, tile):
    """Every compiled tile shape (not just the ones pick_tile chooses), on a shape with partial tiles."""
    M, N, K, R = 392, 456, 640, 48
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, R, torch.bfloat16, seed=1)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    native.gemm(act, wgt, out, asc, wsc, bias, la, lu, False, tile=tile)
    rows = _sample_rows(M, M)
    ref = _ref64(act, wgt, asc, wsc, bias, la, lu, False, rows)
    assert (out[rows].double() - ref).abs().max().item() <= 2 * 2.0**-8 * ref.abs().max().item()


@requires_gfx11
def test_native_strided_views(native_env):
    """Row strides larger than the row (views into wider buffers) are fine when 16-byte aligned."""
    M, N, K, R = 256, 256, 512, 16
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, R, torch.bfloat16)
    act_w = torch.zeros(M, K // 2 + 32, dtype=torch.uint8, device="cuda")
    act_w[:, : K // 2] = act
    out_w = torch.zeros(M, N + 24, dtype=torch.bfloat16, device="cuda")
    la_w = torch.zeros(M, R + 4, device="cuda")
    la_w[:, :R] = la
    native.gemm(act_w[:, : K // 2], wgt, out_w[:, :N], asc, wsc, bias, la_w[:, :R], lu)
    expect = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    native.gemm(act, wgt, expect, asc, wsc, bias, la, lu)
    assert torch.equal(out_w[:, :N], expect)
    assert (out_w[:, N:] == 0).all()


# ---------------------------------------------------------------------------------------------------------------
# dispatch from gemm_w4a4()


def _spy(monkeypatch):
    calls = []
    orig = native.try_gemm

    def spy(*args, **kwargs):
        r = orig(*args, **kwargs)
        calls.append(r)
        return r

    monkeypatch.setattr(native, "try_gemm", spy)
    return calls


def _plain_call(M=512, N=384, K=512, R=32, dtype=torch.bfloat16):
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, R, dtype)
    return lambda: w4a4.gemm_w4a4(
        act, wgt, ascales=asc, wscales=wsc, bias=bias, lora_act_in=la, lora_up=lu, out_dtype=dtype
    )


@requires_gfx11
def test_dispatch_uses_native_on_gfx11(native_env):
    calls = _spy(native_env)
    f = _plain_call()
    out = f()
    assert calls == [True]
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    ref = f()
    assert calls == [True, False]
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2 * ref.float().abs().max().item())


def test_dispatch_opt_out(native_env):
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "0")
    calls = _spy(native_env)
    _plain_call()()
    assert calls == [False]
    assert native._state == {}  # nothing was compiled or loaded


@pytest.mark.parametrize("M,N,K", [(512, 384, 192), (500, 384, 512), (512, 380, 512)])
def test_dispatch_unsupported_shape_falls_back(native_env, M, N, K):
    """K % 128, M % 8, N % 8 violations run on Triton (and still give the right answer)."""
    calls = _spy(native_env)
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, 32, torch.bfloat16)
    out = w4a4.gemm_w4a4(act, wgt, ascales=asc, wscales=wsc, bias=bias, lora_act_in=la, lora_up=lu)
    assert calls == [False]
    rows = _sample_rows(M, M)
    ref = _ref64(act, wgt, asc, wsc, bias, la, lu, False, rows)
    assert (out[rows].double() - ref).abs().max().item() <= 2 * 2.0**-8 * ref.abs().max().item()


def test_dispatch_skips_fused_epilogues(native_env):
    """Calls with a fused epilogue never reach the native kernel."""
    calls = _spy(native_env)
    M, N, K = 256, 3 * 128, 256
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, 32, torch.bfloat16)
    nq = torch.rand(128, device="cuda", dtype=torch.bfloat16)
    rot = torch.randn(M, 64, 1, 2, device="cuda")
    w4a4.gemm_w4a4(
        act, wgt, ascales=asc, wscales=wsc, bias=bias, lora_act_in=la, lora_up=lu, norm_q=nq, norm_k=nq, rotary_emb=rot
    )
    w4a4.gemm_w4a4(act, wgt, ascales=asc, wscales=wsc, fuse_silu=True)
    assert calls == []


def test_dispatch_not_on_other_archs(native_env):
    """In auto mode, non-gfx11 GPUs never compile anything."""
    if _arch().startswith("gfx11"):
        native_env.setattr(native, "device_arch", lambda dev: "gfx942")
    calls = _spy(native_env)
    _plain_call()()
    assert calls == [False]
    assert native._state == {}


def test_build_failure_falls_back_with_one_warning(native_env):
    """A compile/load failure disables the native path with a single warning; results come from Triton."""
    native_env.setattr(native, "device_arch", lambda dev: "gfx1101")

    def broken(*args, **kwargs):
        raise RuntimeError("simulated hipRTC failure")

    native_env.setattr(build, "code_object", broken)
    calls = _spy(native_env)
    f = _plain_call()
    with pytest.warns(RuntimeWarning, match="simulated hipRTC failure"):
        out = f()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f()  # no second warning
    assert calls == [False, False]
    assert "disabled" in native.status()[torch.cuda.current_device()]
    assert out.shape == (512, 384)

    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "1")
    with pytest.raises(RuntimeError, match="unavailable"):
        f()


@requires_hip
def test_force_mode_raises_on_non_gfx11(native_env):
    """NUNCHAKU_NATIVE_GEMM=1 on a GPU without int4 WMMA raises instead of silently falling back."""
    if _arch().startswith("gfx11"):
        pytest.skip("gfx11 supports the kernel")
    native_env.setenv("NUNCHAKU_NATIVE_GEMM", "1")
    with pytest.raises(RuntimeError):
        _plain_call()()


def test_unsupported_reason_checks():
    M, N, K = 64, 64, 256
    act, wgt, asc, wsc, bias, la, lu = _problem(M, N, K, 16, torch.bfloat16)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    ok = (act, wgt, out, asc, wsc, bias, la, lu)
    assert native.unsupported_reason(*ok) is None
    assert native.unsupported_reason(act, wgt, out, asc, wsc, bias, la, None) is not None  # half a LoRA
    assert native.unsupported_reason(act, wgt, out.float(), asc, wsc, bias, la, lu) is not None
    assert native.unsupported_reason(act, wgt, out, asc.half(), wsc, bias, la, lu) is not None
    assert native.unsupported_reason(act, wgt, out, asc, wsc, bias, la, lu[:, :8]) is not None  # R % 16
    assert native.unsupported_reason(act[:, 1:121], wgt[:, :120], out, asc, wsc, None, None, None) is not None
    big = torch.empty(M + 8, N, dtype=torch.bfloat16, device="cuda")
    assert native.unsupported_reason(act, wgt, big, asc, wsc, bias, la, lu) is not None  # out rows > M
