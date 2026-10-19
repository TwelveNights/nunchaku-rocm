"""
Host side of the native RDNA3 W4A4 GEMM (``w4a4_gfx11.hip``).

:func:`try_gemm` is called by :func:`nunchaku.ops.triton.gemm_w4a4.gemm_w4a4`
for every plain (no fused epilogue) int4 GEMM. It launches the native kernel
and returns ``True``, or returns ``False`` to let the Triton kernel run:

- ``NUNCHAKU_NATIVE_GEMM=0`` disables the native kernel;
- unset (or ``auto``) uses it on gfx11 GPUs only;
- ``NUNCHAKU_NATIVE_GEMM=1`` tries it on any AMD GPU and *raises* when it
  cannot be built, loaded or launched, instead of falling back (for tests
  and debugging).

Calls the kernel does not support (:func:`unsupported_reason`) always go to
Triton. Otherwise, the first supported call on a device compiles (or loads
from the cache, see :mod:`.build`) every kernel variant and runs a small
self-test of each against a torch reference; any failure there, or a launch
error later, disables the native path on that device for the rest of the
process with a single warning.
"""

import ctypes
import os
import threading
import time
import warnings

import torch

from . import build

__all__ = ["TILES", "gemm", "mode", "pick_tile", "status", "try_gemm", "unsupported_reason"]

_SOURCE = "w4a4_gfx11.hip"
_HEADERS = ["hip_prelude.h"]
_OPTIONS = ["-O3", "-std=c++17"]

# (BM, BN, WM, WN): workgroup tile and per-wave tile. Must match the W4A4_TILE list in w4a4_gfx11.hip.
TILES = [
    (64, 128, 32, 32),
    (128, 128, 32, 64),
]

_DTYPES = {torch.bfloat16: "bf16", torch.float16: "fp16"}


class _GemmArgs(ctypes.Structure):
    """Mirror of ``struct GemmArgs`` in w4a4_gfx11.hip (same field order, natural alignment)."""

    _fields_ = [
        (name, ctypes.c_void_p) for name in ("act", "wgt", "out", "ascales", "wscales", "bias", "lora_act", "lora_up")
    ] + [
        (name, ctypes.c_int)
        for name in (
            "M",
            "N",
            "K",
            "actual_m",
            "R",
            "stride_am",
            "stride_wn",
            "stride_asg",
            "stride_wsg",
            "stride_om",
            "stride_lam",
            "stride_lun",
        )
    ]


def mode() -> str:
    """``"off"``, ``"auto"`` or ``"force"``, from ``NUNCHAKU_NATIVE_GEMM`` (read on every call)."""
    v = os.environ.get("NUNCHAKU_NATIVE_GEMM", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


_arch_cache: dict[int, str] = {}


def device_arch(device: int) -> str:
    """The GPU's gfx arch, e.g. ``gfx1101`` (empty on non-AMD builds)."""
    arch = _arch_cache.get(device)
    if arch is None:
        arch = ""
        if torch.version.hip is not None:
            arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[0]
        _arch_cache[device] = arch
    return arch


def pick_tile(M: int, N: int) -> tuple[int, int, int, int]:
    """
    Tile for an M x N output, measured on an RX 7800 XT at FLUX.1 and FLUX.2
    [klein] shapes: the small tile only pays off when the big one would leave
    too few workgroups for the GPU's 60 CUs.
    """
    if M <= 1024 and N <= 4096:
        return (64, 128, 32, 32)
    return (128, 128, 32, 64)


def _aligned(t: torch.Tensor, row_elems: int) -> bool:
    """Unit inner stride, and rows (and the base) 16-byte aligned for vector loads."""
    return t.stride(-1) == 1 and t.stride(0) % row_elems == 0 and t.data_ptr() % 16 == 0


def unsupported_reason(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None,
    lora_act: torch.Tensor | None,
    lora_up: torch.Tensor | None,
) -> str | None:
    """
    Why the native kernel cannot compute this GEMM, or ``None`` if it can.
    ``act`` (M, K/2), ``wgt`` (N, K/2), ``out`` (actual_m <= M, N) are 2-D;
    pass ``lora_act``/``lora_up`` both or neither.
    """
    T = out.dtype
    if T not in _DTYPES:
        return f"output dtype {T}"
    if act.dim() != 2 or wgt.dim() != 2 or out.dim() != 2:
        return "act/wgt/out must be 2-D"
    if act.element_size() != 1 or wgt.element_size() != 1:
        return "act/wgt must be packed byte tensors"
    M, K, N = act.shape[0], act.shape[1] * 2, wgt.shape[0]
    if wgt.shape[1] * 2 != K or out.shape[1] != N or out.shape[0] > M:
        return "shape mismatch"
    if M == 0 or N == 0 or K == 0:
        return "empty GEMM"
    # the main loop consumes K in stages of 128 (two scale groups); the scale copy reads 8-row chunks
    if K % 128 or M % 8 or N % 8:
        return f"M={M}, N={N}, K={K} needs K % 128 == M % 8 == N % 8 == 0"
    if max(M, N, K) >= 2**31 or max(act.stride(0), wgt.stride(0), out.stride(0)) >= 2**31:
        return "too large for 32-bit indices"
    if not (_aligned(act, 16) and _aligned(wgt, 16)):
        return "act/wgt rows not 16-byte aligned"
    if out.stride(-1) != 1:
        return "out not row-major"
    devs = {act.device, wgt.device, out.device, ascales.device, wscales.device}
    for name, s, rows in (("ascales", ascales, M), ("wscales", wscales, N)):
        if s.dtype != T:
            return f"{name} dtype {s.dtype} != output dtype {T}"
        if s.dim() != 2 or s.shape[0] < K // 64 or s.shape[1] < rows:
            return f"{name} shape {tuple(s.shape)}"
        if not _aligned(s, 8):
            return f"{name} rows not 16-byte aligned"
    if bias is not None:
        if bias.dtype != T or bias.dim() != 1 or bias.shape[0] < N or bias.stride(0) != 1:
            return "bias must be a contiguous (N,) tensor of the output dtype"
        devs.add(bias.device)
    if (lora_act is None) != (lora_up is None):
        return "lora_act and lora_up must be given together"
    if lora_up is not None:
        R = lora_up.shape[1]
        if lora_act.dtype != torch.float32 or lora_up.dtype != T:
            return f"LoRA dtypes {lora_act.dtype}/{lora_up.dtype}"
        if R % 16 or lora_up.shape[0] < N or lora_act.shape[0] < M or lora_act.shape[1] < R:
            return f"LoRA rank {R} not a multiple of 16, or LoRA shapes"
        if R and not (_aligned(lora_act, 4) and _aligned(lora_up, 8)):
            return "LoRA rows not 16-byte aligned"
        devs |= {lora_act.device, lora_up.device}
    if len(devs) != 1:
        return "tensors on different devices"
    return None


def _kernel_name(dtype: torch.dtype, act_unsigned: bool, tile: tuple[int, int, int, int]) -> str:
    bm, bn, wm, wn = tile
    return f"w4a4_{_DTYPES[dtype]}_{'u' if act_unsigned else 's'}_{bm}x{bn}_{wm}x{wn}"


class _Kernels:
    """The compiled kernels, loaded and self-tested on one device."""

    def __init__(self, device: int, arch: str):
        from .hip import Module

        t0 = time.perf_counter()
        code, self.from_cache = build.code_object(_SOURCE, _HEADERS, arch, _OPTIONS, "the native W4A4 GEMM")
        try:
            self.module = Module(code, device)
        except Exception:
            if not self.from_cache:
                raise
            # a stale or damaged cache entry: rebuild it once
            code, self.from_cache = build.code_object(
                _SOURCE, _HEADERS, arch, _OPTIONS, "the native W4A4 GEMM", refresh=True
            )
            self.module = Module(code, device)
        self.device = device
        self.arch = arch
        self.load_seconds = time.perf_counter() - t0
        self._self_test()

    def launch(self, act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned, tile=None) -> None:
        M, N, K = act.shape[0], wgt.shape[0], act.shape[1] * 2
        R = lora_up.shape[1] if lora_up is not None else 0
        tile = tile or pick_tile(M, N)
        bm, bn, wm, wn = tile
        args = _GemmArgs(
            act.data_ptr(),
            wgt.data_ptr(),
            out.data_ptr(),
            ascales.data_ptr(),
            wscales.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            lora_act.data_ptr() if R else None,
            lora_up.data_ptr() if R else None,
            M,
            N,
            K,
            out.shape[0],
            R,
            act.stride(0) // 4,
            wgt.stride(0) // 4,
            ascales.stride(0),
            wscales.stride(0),
            out.stride(0),
            lora_act.stride(0) if R else 0,
            lora_up.stride(0) if R else 0,
        )
        grid = -(-M // bm) * -(-N // bn)
        self.module.launch(_kernel_name(out.dtype, act_unsigned, tile), grid, (bm // wm) * (bn // wn) * 32, (args,))

    def _self_test(self) -> None:
        """Run every kernel once on a small odd-shaped problem and compare with a torch reference."""
        dev = torch.device("cuda", self.device)
        g = torch.Generator(device=dev).manual_seed(0)
        M, N, K, R, actual_m = 136, 264, 256, 16, 130  # partial tiles in M and N, two K stages
        act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev, generator=g)
        wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g)
        lora_act = torch.randn(M, R, device=dev, generator=g)
        for dtype in _DTYPES:
            asc = (torch.rand(K // 64, M, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            wsc = (torch.rand(K // 64, N, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            bias = torch.randn(N, device=dev, generator=g).to(dtype)
            lora_up = (torch.randn(N, R, device=dev, generator=g) * 0.1).to(dtype)
            for unsigned in (False, True):
                ref = reference(act, wgt, asc, wsc, bias, lora_act, lora_up, unsigned)[:actual_m]
                tol = 1e-2 * ref.abs().max().item()
                for tile in TILES:
                    buf = torch.full((actual_m + 8, N), float("nan"), dtype=dtype, device=dev)
                    self.launch(act, wgt, buf[:actual_m], asc, wsc, bias, lora_act, lora_up, unsigned, tile)
                    err = (buf[:actual_m].float() - ref).abs().max().item()
                    if not err <= tol or not bool(buf[actual_m:].isnan().all()):
                        raise RuntimeError(
                            f"self-test failed for {_kernel_name(dtype, unsigned, tile)}: max error {err:.3g} "
                            f"(tolerance {tol:.3g}), rows past actual_m written: {not buf[actual_m:].isnan().all()}"
                        )


def _unpack(packed: torch.Tensor, signed: bool) -> torch.Tensor:
    b = packed.view(torch.uint8).to(torch.int32)
    x = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(packed.shape[0], -1)
    return torch.where(x >= 8, x - 16, x) if signed else x


def reference(act, wgt, ascales, wscales, bias, lora_act, lora_up, act_unsigned) -> torch.Tensor:
    """fp32 reference of the native GEMM (per-group integer dots are exact in fp32)."""
    a = _unpack(act, not act_unsigned).float()
    w = _unpack(wgt, True).float()
    out = torch.zeros(a.shape[0], w.shape[0], device=a.device)
    for gi in range(a.shape[1] // 64):
        sl = slice(gi * 64, gi * 64 + 64)
        out += (a[:, sl] @ w[:, sl].T) * ascales[gi, : a.shape[0]].float()[:, None] * wscales[gi, : w.shape[0]].float()
    if lora_up is not None and lora_up.shape[1]:
        R = lora_up.shape[1]
        out += lora_act[: a.shape[0], :R].to(lora_up.dtype).float() @ lora_up[: w.shape[0]].float().T
    if bias is not None:
        out += bias[: w.shape[0]].float()
    return out


_lock = threading.Lock()
_state: dict[int, "_Kernels | BaseException"] = {}


def _kernels(device: int, force: bool) -> "_Kernels | None":
    st = _state.get(device)
    if st is None:
        with _lock:
            st = _state.get(device)
            if st is None:
                try:
                    st = _Kernels(device, device_arch(device))
                except Exception as e:  # noqa: BLE001 -- any failure means "use Triton"
                    st = e
                    if not force:
                        _warn(device, e)
                _state[device] = st
    if isinstance(st, BaseException):
        if force:
            raise RuntimeError(f"native W4A4 GEMM unavailable on cuda:{device}") from st
        return None
    return st


def _warn(device: int, err: BaseException) -> None:
    warnings.warn(
        f"Native gfx11 W4A4 GEMM disabled on cuda:{device}, using the Triton kernel instead "
        f"({type(err).__name__}: {err}). Set NUNCHAKU_NATIVE_GEMM=0 to silence this.",
        RuntimeWarning,
        stacklevel=4,
    )


def status() -> dict[int, str]:
    """Per device: how the native kernels were obtained, or why they are disabled (for diagnostics)."""
    out = {}
    for dev, st in _state.items():
        if isinstance(st, BaseException):
            out[dev] = f"disabled: {type(st).__name__}: {st}"
        else:
            out[dev] = f"ok ({st.arch}, {'cached' if st.from_cache else 'compiled'}, {st.load_seconds:.2f} s)"
    return out


def _reset() -> None:
    """Forget loaded kernels and failures (tests)."""
    with _lock:
        _state.clear()


def gemm(act, wgt, out, ascales, wscales, bias=None, lora_act=None, lora_up=None, act_unsigned=False, tile=None):
    """
    Run the native kernel directly (no fallback): raises if the call is
    unsupported or the kernels are unavailable. ``out`` may have fewer rows
    than ``act`` (only those are written). Returns ``out``.
    """
    reason = unsupported_reason(act, wgt, out, ascales, wscales, bias, lora_act, lora_up)
    if reason is not None:
        raise ValueError(f"native W4A4 GEMM does not support this call: {reason}")
    if tile is not None and tuple(tile) not in TILES:
        raise ValueError(f"unknown tile {tile}")
    device = act.device.index if act.device.index is not None else torch.cuda.current_device()
    _kernels(device, force=True).launch(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned, tile)
    return out


def try_gemm(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned) -> bool:
    """
    Compute the GEMM natively if enabled and supported; ``False`` means the
    caller must run the Triton kernel. Arguments as :func:`gemm`.
    """
    m = mode()
    if m == "off" or act.device.type != "cuda":
        return False
    device = act.device.index if act.device.index is not None else torch.cuda.current_device()
    if m == "auto" and not device_arch(device).startswith("gfx11"):
        return False
    if unsupported_reason(act, wgt, out, ascales, wscales, bias, lora_act, lora_up) is not None:
        return False
    k = _kernels(device, force=m == "force")
    if k is None:
        return False
    try:
        k.launch(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned)
    except Exception as e:  # noqa: BLE001
        if m == "force":
            raise
        with _lock:
            _state[device] = e
        _warn(device, e)
        return False
    return True
