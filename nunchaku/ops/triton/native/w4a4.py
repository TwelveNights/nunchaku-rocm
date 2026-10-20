"""
Host side of the native RDNA3 W4A4 GEMM (``w4a4_gfx11.hip``).

:func:`nunchaku.ops.triton.gemm_w4a4.gemm_w4a4` calls :func:`try_gemm` for
plain int4 GEMMs (bias / LoRA-up only), :func:`try_gemm_qk` for QKV
projections with the fused per-head RMSNorm + RoPE epilogue, and
:func:`try_gemm_gelu_quant` for GELU MLP fc1 layers with the fused GELU +
next-layer LoRA-down + int4 quantization epilogue. Each launches the native
kernel and returns ``True``, or returns ``False`` to let the Triton kernel run:

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

__all__ = [
    "GQ_TILES",
    "QK_TILES",
    "TILES",
    "gemm",
    "gemm_gelu_quant",
    "gemm_qk",
    "gemm_swiglu_quant",
    "interleave_swiglu",
    "mode",
    "pick_tile",
    "status",
    "try_gemm",
    "try_gemm_gelu_quant",
    "try_gemm_qk",
    "try_gemm_swiglu_quant",
    "unsupported_reason",
    "unsupported_reason_gelu_quant",
    "unsupported_reason_qk",
    "unsupported_reason_swiglu_quant",
]

_SOURCE = "w4a4_gfx11.hip"
_HEADERS = ["hip_prelude.h"]
_OPTIONS = ["-O3", "-std=c++17"]

# (BM, BN, WM, WN): workgroup tile and per-wave tile. Must match the W4A4_TILE list in w4a4_gfx11.hip.
TILES = [
    (64, 128, 32, 32),
    (128, 128, 32, 64),
]

# Fused-epilogue kernels (w4a4qk_*, w4a4gq_*): BN must be 128 for QK (one head per workgroup); the GELU-quant
# epilogue needs WN % 64 == 0. Must match the W4A4_VARIANTS lists in w4a4_gfx11.hip.
QK_TILES = [(128, 128, 32, 64)]
GQ_TILES = [(128, 128, 32, 64)]
# SwiGLU-quant (w4a4sq_*): BN = 128 = 2 waves of WN = 64 -> 64 output columns, one quantization group per row
SQ_TILES = [(128, 128, 32, 64)]
SWIGLU_BLOCK = 16  # linear_in rows interleaved in blocks of 16 (gate, up), see interleave_swiglu
HEAD_DIM = 128
GELU_SHIFT = 0.171875  # FLUX's unsigned-activation shift after GELU (as the Triton / CUDA kernels)

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


class _QKArgs(ctypes.Structure):
    """Mirror of ``QKNormRopeEpilogue::Args``."""

    _fields_ = [
        ("norm_q", ctypes.c_void_p),
        ("norm_k", ctypes.c_void_p),
        ("rot", ctypes.c_void_p),
        ("stride_rm", ctypes.c_int),
        ("eps", ctypes.c_float),
    ]


class _GQArgs(ctypes.Structure):
    """Mirror of ``QuantArgs`` (GeluQuantEpilogue / SwigluQuantEpilogue)."""

    _fields_ = [
        (name, ctypes.c_void_p) for name in ("qout", "oscales", "smooth", "lora_down", "lora_part")
    ] + [("stride_lop", ctypes.c_longlong)] + [
        (name, ctypes.c_int) for name in ("R2", "stride_qm", "stride_osg", "stride_ld", "stride_lom")
    ] + [("shift", ctypes.c_float)]


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
    out: torch.Tensor | None,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None,
    lora_act: torch.Tensor | None,
    lora_up: torch.Tensor | None,
    dtype: torch.dtype | None = None,
) -> str | None:
    """
    Why the native kernel cannot compute this GEMM, or ``None`` if it can.
    ``act`` (M, K/2), ``wgt`` (N, K/2), ``out`` (actual_m <= M, N) are 2-D;
    pass ``lora_act``/``lora_up`` both or neither. ``out`` may be ``None``
    (GELU-quant epilogue without the 16-bit output) when ``dtype`` names T.
    """
    T = out.dtype if out is not None else dtype
    if T not in _DTYPES:
        return f"output dtype {T}"
    if act.dim() != 2 or wgt.dim() != 2 or (out is not None and out.dim() != 2):
        return "act/wgt/out must be 2-D"
    if act.element_size() != 1 or wgt.element_size() != 1:
        return "act/wgt must be packed byte tensors"
    M, K, N = act.shape[0], act.shape[1] * 2, wgt.shape[0]
    if wgt.shape[1] * 2 != K or (out is not None and (out.shape[1] != N or out.shape[0] > M)):
        return "shape mismatch"
    if M == 0 or N == 0 or K == 0:
        return "empty GEMM"
    # the main loop consumes K in stages of 128 (two scale groups); the scale copy reads 8-row chunks
    if K % 128 or M % 8 or N % 8:
        return f"M={M}, N={N}, K={K} needs K % 128 == M % 8 == N % 8 == 0"
    if max(M, N, K) >= 2**31 or max(act.stride(0), wgt.stride(0), out.stride(0) if out is not None else 0) >= 2**31:
        return "too large for 32-bit indices"
    if not (_aligned(act, 16) and _aligned(wgt, 16)):
        return "act/wgt rows not 16-byte aligned"
    if out is not None and out.stride(-1) != 1:
        return "out not row-major"
    devs = {act.device, wgt.device, ascales.device, wscales.device}
    if out is not None:
        devs.add(out.device)
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


def unsupported_reason_qk(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rot) -> str | None:
    """
    As :func:`unsupported_reason`, for the fused Q/K RMSNorm + RoPE epilogue:
    ``norm_q``/``norm_k`` (128,) of the output dtype, ``rot`` the rotary table
    as (actual_m, 128) fp32 rows of (sin, cos) pairs.
    """
    reason = unsupported_reason(act, wgt, out, ascales, wscales, bias, lora_act, lora_up)
    if reason is not None:
        return reason
    N, T = wgt.shape[0], out.dtype
    if N % 3 or (N // 3) % HEAD_DIM:
        return f"N={N} is not three whole sets of {HEAD_DIM}-wide heads"
    for name, w in (("norm_q", norm_q), ("norm_k", norm_k)):
        if w is None or w.dtype != T or w.numel() != HEAD_DIM or not w.is_contiguous() or w.device != out.device:
            return f"{name} must be a contiguous ({HEAD_DIM},) tensor of the output dtype"
    if rot is None or rot.dtype != torch.float32 or rot.dim() != 2 or rot.shape != (out.shape[0], HEAD_DIM):
        return "rotary_emb must be (actual_m, 128) fp32"
    if rot.stride(-1) != 1 or rot.stride(0) % 2 or rot.data_ptr() % 8 or rot.device != out.device:
        return "rotary_emb rows not 8-byte aligned"
    return None


def unsupported_reason_gelu_quant(
    act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out
) -> str | None:
    """
    As :func:`unsupported_reason`, for the fused GELU + next-layer LoRA-down +
    unsigned int4 quantization epilogue. ``out`` (the pre-GELU output) is
    optional; ``qout`` (>= M, N/2) uint8, ``oscales`` (N/64, >= M), ``smooth``
    (N,), ``lora_down`` (N, R2) of the output dtype, ``lora_act_out`` (>= M, R2)
    fp32; ``lora_down``/``lora_act_out`` both or neither.
    """
    T = oscales.dtype
    reason = unsupported_reason(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, dtype=T)
    if reason is not None:
        return reason
    if out is not None and out.dtype != T:
        return "out / oscales dtypes differ"
    M, N = act.shape[0], wgt.shape[0]
    bn = GQ_TILES[0][1]
    if M % 16 or N % bn:
        return f"M={M}, N={N} needs M % 16 == 0 and N % {bn} == 0"
    dev = act.device
    if qout.dtype != torch.uint8 or qout.dim() != 2 or qout.shape[0] < M or qout.shape[1] != N // 2:
        return f"qout shape/dtype {tuple(qout.shape)} {qout.dtype}"
    if qout.stride(-1) != 1 or qout.stride(0) % 4 or qout.data_ptr() % 4 or qout.device != dev:
        return "qout rows not 4-byte aligned"
    if oscales.dim() != 2 or oscales.shape[0] != N // 64 or oscales.shape[1] < M or oscales.stride(-1) != 1:
        return f"oscales shape {tuple(oscales.shape)}"
    if oscales.device != dev:
        return "tensors on different devices"
    if smooth is not None and (smooth.dtype != T or smooth.numel() != N or not smooth.is_contiguous() or smooth.device != dev):
        return "smooth_factor must be a contiguous (N,) tensor of the output dtype"
    if (lora_down is None) != (lora_act_out is None):
        return "lora_down and lora_act_out must be given together"
    if lora_down is not None:
        R2 = lora_down.shape[1]
        if lora_down.dtype != T or lora_down.dim() != 2 or lora_down.shape[0] != N or R2 % 16 or R2 == 0:
            return f"lora_down shape/dtype {tuple(lora_down.shape)} {lora_down.dtype}"
        if not _aligned(lora_down, 8) or lora_down.device != dev:
            return "lora_down rows not 16-byte aligned"
        if lora_act_out.dtype != torch.float32 or lora_act_out.dim() != 2 or lora_act_out.shape[0] < M:
            return "lora_act_out must be (>= M, R2) fp32"
        if lora_act_out.shape[1] != R2 or lora_act_out.device != dev:
            return "lora_act_out shape"
    return None


def unsupported_reason_swiglu_quant(
    act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out
) -> str | None:
    """
    As :func:`unsupported_reason_gelu_quant`, for the fused SwiGLU + signed int4
    quantization epilogue: ``wgt`` (2H, K/2) with interleaved gate/up rows
    (:func:`interleave_swiglu`); ``qout`` (>= M, H/2), ``oscales`` (H/64, >= M),
    ``smooth`` (H,), ``lora_down`` (H, R2) describe the quantized (M, H) output.
    """
    T = oscales.dtype
    reason = unsupported_reason(act, wgt, None, ascales, wscales, bias, lora_act, lora_up, dtype=T)
    if reason is not None:
        return reason
    M, N = act.shape[0], wgt.shape[0]
    bn = SQ_TILES[0][1]
    if M % 16 or N % bn:
        return f"M={M}, N={N} needs M % 16 == 0 and N % {bn} == 0"
    H, dev = N // 2, act.device
    if qout.dtype != torch.uint8 or qout.dim() != 2 or qout.shape[0] < M or qout.shape[1] != H // 2:
        return f"qout shape/dtype {tuple(qout.shape)} {qout.dtype}"
    if qout.stride(-1) != 1 or qout.stride(0) % 4 or qout.data_ptr() % 4 or qout.device != dev:
        return "qout rows not 4-byte aligned"
    if oscales.dim() != 2 or oscales.shape[0] != H // 64 or oscales.shape[1] < M or oscales.stride(-1) != 1:
        return f"oscales shape {tuple(oscales.shape)}"
    if oscales.device != dev:
        return "tensors on different devices"
    if smooth is not None and (smooth.dtype != T or smooth.numel() != H or not smooth.is_contiguous() or smooth.device != dev):
        return "smooth_factor must be a contiguous (H,) tensor of the output dtype"
    if (lora_down is None) != (lora_act_out is None):
        return "lora_down and lora_act_out must be given together"
    if lora_down is not None:
        R2 = lora_down.shape[1]
        if lora_down.dtype != T or lora_down.dim() != 2 or lora_down.shape[0] != H or R2 % 16 or R2 == 0:
            return f"lora_down shape/dtype {tuple(lora_down.shape)} {lora_down.dtype}"
        if not _aligned(lora_down, 8) or lora_down.device != dev:
            return "lora_down rows not 16-byte aligned"
        if lora_act_out.dtype != torch.float32 or lora_act_out.dim() != 2 or lora_act_out.shape[0] < M:
            return "lora_act_out must be (>= M, R2) fp32"
        if lora_act_out.shape[1] != R2 or lora_act_out.device != dev:
            return "lora_act_out shape"
    return None


def interleave_swiglu(n: int, device=None) -> torch.Tensor:
    """
    Row order for a SwiGLU projection whose output is ``[gate (n/2) | up (n/2)]``:
    interleaved row ``32 b + w`` is gate row ``16 b + w`` for ``w < 16``, else up
    row ``16 b + w - 16``. ``weight[idx]`` gives the layout the SwiGLU epilogue
    reads; ``out.view(M, n // 32, 2, 16)`` un-interleaves an output (``[:, :, 0]``
    gate, ``[:, :, 1]`` up).
    """
    h, b = n // 2, SWIGLU_BLOCK
    g = torch.arange(h, device=device).view(h // b, b)
    return torch.stack([g, g + h], dim=1).reshape(-1)


def _kernel_name(dtype: torch.dtype, act_unsigned: bool, tile: tuple[int, int, int, int], name: str = "w4a4") -> str:
    bm, bn, wm, wn = tile
    return f"{name}_{_DTYPES[dtype]}_{'u' if act_unsigned else 's'}_{bm}x{bn}_{wm}x{wn}"


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

    @staticmethod
    def _gemm_args(act, wgt, out, ascales, wscales, bias, lora_act, lora_up) -> "_GemmArgs":
        M, N, K = act.shape[0], wgt.shape[0], act.shape[1] * 2
        R = lora_up.shape[1] if lora_up is not None else 0
        return _GemmArgs(
            act.data_ptr(),
            wgt.data_ptr(),
            out.data_ptr() if out is not None else None,
            ascales.data_ptr(),
            wscales.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            lora_act.data_ptr() if R else None,
            lora_up.data_ptr() if R else None,
            M,
            N,
            K,
            out.shape[0] if out is not None else M,
            R,
            act.stride(0) // 4,
            wgt.stride(0) // 4,
            ascales.stride(0),
            wscales.stride(0),
            out.stride(0) if out is not None else 0,
            lora_act.stride(0) if R else 0,
            lora_up.stride(0) if R else 0,
        )

    def _run(self, name, dtype, act_unsigned, tile, M, N, args) -> None:
        bm, bn, wm, wn = tile
        grid = -(-M // bm) * -(-N // bn)
        self.module.launch(_kernel_name(dtype, act_unsigned, tile, name), grid, (bm // wm) * (bn // wn) * 32, args)

    def launch(self, act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned, tile=None) -> None:
        M, N = act.shape[0], wgt.shape[0]
        g = self._gemm_args(act, wgt, out, ascales, wscales, bias, lora_act, lora_up)
        self._run("w4a4", out.dtype, act_unsigned, tile or pick_tile(M, N), M, N, (g,))

    def launch_qk(self, act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rot, act_unsigned,
                  eps=1e-6, tile=None) -> None:
        M, N = act.shape[0], wgt.shape[0]
        g = self._gemm_args(act, wgt, out, ascales, wscales, bias, lora_act, lora_up)
        e = _QKArgs(norm_q.data_ptr(), norm_k.data_ptr(), rot.data_ptr(), rot.stride(0), eps)
        self._run("w4a4qk", out.dtype, act_unsigned, tile or QK_TILES[0], M, N, (g, e))

    def launch_gelu_quant(self, act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth,
                          lora_down, lora_act_out, act_unsigned, shift=GELU_SHIFT, tile=None, name="w4a4gq") -> None:
        M, N = act.shape[0], wgt.shape[0]
        tile = tile or GQ_TILES[0]
        bn = tile[1]
        g = self._gemm_args(act, wgt, out, ascales, wscales, bias, lora_act, lora_up)
        R2 = lora_down.shape[1] if lora_down is not None else 0
        # the next layer's LoRA-down: one partial per N-tile, summed below in a fixed order (deterministic; fp32
        # atomics made same-seed images differ)
        part = torch.empty(N // bn, M, R2, dtype=torch.float32, device=act.device) if R2 else None
        e = _GQArgs(
            qout.data_ptr(),
            oscales.data_ptr(),
            smooth.data_ptr() if smooth is not None else None,
            lora_down.data_ptr() if R2 else None,
            part.data_ptr() if R2 else None,
            part.stride(0) if R2 else 0,
            R2,
            qout.stride(0),
            oscales.stride(0),
            lora_down.stride(0) if R2 else 0,
            part.stride(1) if R2 else 0,
            shift,
        )
        self._run(name, oscales.dtype, act_unsigned, tile, M, N, (g, e))
        if R2:
            torch.sum(part, dim=0, out=lora_act_out[:M])

    def launch_swiglu_quant(self, act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth,
                            lora_down, lora_act_out, act_unsigned, tile=None) -> None:
        self.launch_gelu_quant(act, wgt, None, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth,
                               lora_down, lora_act_out, act_unsigned, 0.0, tile or SQ_TILES[0], "w4a4sq")

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
        self._self_test_qk(g)
        self._self_test_gelu_quant(g)
        self._self_test_swiglu_quant(g)

    def _self_test_swiglu_quant(self, g) -> None:
        dev = torch.device("cuda", self.device)
        M, N, K, R, R2 = 144, 256, 256, 16, 48  # a partial M tile, two N tiles, two LoRA-down slabs
        H = N // 2
        act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev, generator=g)
        wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g)
        lora_act = torch.randn(M, R, device=dev, generator=g)
        idx = interleave_swiglu(N, dev)
        for dtype in _DTYPES:
            asc = (torch.rand(K // 64, M, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            wsc = (torch.rand(K // 64, N, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            bias = torch.randn(N, device=dev, generator=g).to(dtype)
            lora_up = (torch.randn(N, R, device=dev, generator=g) * 0.1).to(dtype)
            smooth = (torch.rand(H, device=dev, generator=g) + 0.5).to(dtype)
            lora_down = (torch.randn(H, R2, device=dev, generator=g) * 0.1).to(dtype)
            for unsigned in (False, True):
                lin = reference(act, wgt, asc, wsc, bias, lora_act, lora_up, unsigned)  # [gate | up] order
                q_ref, s_ref, l_ref = reference_swiglu_quant(lin, smooth, lora_down, dtype)
                for tile in SQ_TILES:
                    name = _kernel_name(dtype, unsigned, tile, "w4a4sq")
                    qout = torch.zeros(M + 16, H // 2, dtype=torch.uint8, device=dev)
                    osc = torch.empty(H // 64, M, dtype=dtype, device=dev)
                    lao = torch.empty(M, R2, device=dev)
                    self.launch_swiglu_quant(act, wgt[idx], asc, wsc[:, idx].contiguous(), bias[idx], lora_act,
                                             lora_up[idx], qout, osc, smooth, lora_down, lao, unsigned, tile=tile)
                    codes = _unpack(qout[:M], True)
                    bad = [
                        what
                        for what, ok in (
                            ("oscales", ((osc.float() - s_ref).abs() <= 1e-2 * s_ref.abs() + 1e-6).all().item()),
                            ("codes", (codes - q_ref).abs().max().item() <= 1 and (codes != q_ref).float().mean() < 1e-2),
                            ("lora_act_out", (lao - l_ref).abs().max().item() <= 1e-2 * l_ref.abs().max().item()),
                            ("rows past M", (qout[M:] == 0).all().item()),
                        )
                        if not ok
                    ]
                    if bad:
                        raise RuntimeError(f"self-test failed for {name}: {', '.join(bad)}")

    def _self_test_qk(self, g) -> None:
        dev = torch.device("cuda", self.device)
        M, N, K, R, actual_m = 136, 3 * HEAD_DIM, 256, 16, 130  # a partial M tile; Q, K and V workgroups
        act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev, generator=g)
        wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g)
        lora_act = torch.randn(M, R, device=dev, generator=g)
        rot = torch.randn(actual_m, HEAD_DIM, device=dev, generator=g)
        for dtype in _DTYPES:
            asc = (torch.rand(K // 64, M, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            wsc = (torch.rand(K // 64, N, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            bias = torch.randn(N, device=dev, generator=g).to(dtype)
            lora_up = (torch.randn(N, R, device=dev, generator=g) * 0.1).to(dtype)
            nq = (torch.rand(HEAD_DIM, device=dev, generator=g) + 0.5).to(dtype)
            nk = (torch.rand(HEAD_DIM, device=dev, generator=g) + 0.5).to(dtype)
            for unsigned in (False, True):
                ref = reference_qk(reference(act, wgt, asc, wsc, bias, lora_act, lora_up, unsigned)[:actual_m], nq, nk, rot)
                for tile in QK_TILES:
                    buf = torch.full((actual_m + 8, N), float("nan"), dtype=dtype, device=dev)
                    self.launch_qk(act, wgt, buf[:actual_m], asc, wsc, bias, lora_act, lora_up, nq, nk, rot, unsigned,
                                   tile=tile)
                    err = (buf[:actual_m].float() - ref).abs().max().item()
                    tol = 1e-2 * ref.abs().max().item()
                    if not err <= tol or not bool(buf[actual_m:].isnan().all()):
                        raise RuntimeError(
                            f"self-test failed for {_kernel_name(dtype, unsigned, tile, 'w4a4qk')}: max error "
                            f"{err:.3g} (tolerance {tol:.3g})"
                        )

    def _self_test_gelu_quant(self, g) -> None:
        dev = torch.device("cuda", self.device)
        M, N, K, R, R2 = 144, 256, 256, 16, 48  # a partial M tile, two N tiles, two LoRA-down slabs
        act = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev, generator=g)
        wgt = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g)
        lora_act = torch.randn(M, R, device=dev, generator=g)
        for dtype in _DTYPES:
            asc = (torch.rand(K // 64, M, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            wsc = (torch.rand(K // 64, N, device=dev, generator=g) * 0.02 + 0.01).to(dtype)
            bias = torch.randn(N, device=dev, generator=g).to(dtype)
            lora_up = (torch.randn(N, R, device=dev, generator=g) * 0.1).to(dtype)
            smooth = (torch.rand(N, device=dev, generator=g) + 0.5).to(dtype)
            lora_down = (torch.randn(N, R2, device=dev, generator=g) * 0.1).to(dtype)
            for unsigned in (False, True):
                lin = reference(act, wgt, asc, wsc, bias, lora_act, lora_up, unsigned)
                q_ref, s_ref, l_ref = reference_gelu_quant(lin, smooth, lora_down, dtype)
                for tile in GQ_TILES:
                    name = _kernel_name(dtype, unsigned, tile, "w4a4gq")
                    out = torch.empty(M, N, dtype=dtype, device=dev)
                    qout = torch.zeros(M + 16, N // 2, dtype=torch.uint8, device=dev)
                    osc = torch.empty(N // 64, M, dtype=dtype, device=dev)
                    lao = torch.empty(M, R2, device=dev)
                    self.launch_gelu_quant(act, wgt, out, asc, wsc, bias, lora_act, lora_up, qout, osc, smooth,
                                           lora_down, lao, unsigned, tile=tile)
                    codes = _unpack(qout[:M], False)
                    bad = [
                        what
                        for what, ok in (
                            ("out", (out.float() - lin).abs().max().item() <= 1e-2 * lin.abs().max().item()),
                            ("oscales", ((osc.float() - s_ref).abs() <= 1e-2 * s_ref.abs() + 1e-6).all().item()),
                            ("codes", (codes - q_ref).abs().max().item() <= 1 and (codes != q_ref).float().mean() < 1e-2),
                            ("lora_act_out", (lao - l_ref).abs().max().item() <= 1e-2 * l_ref.abs().max().item()),
                            ("rows past M", (qout[M:] == 0).all().item()),
                        )
                        if not ok
                    ]
                    if bad:
                        raise RuntimeError(f"self-test failed for {name}: {', '.join(bad)}")


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


def reference_qk(lin: torch.Tensor, norm_q, norm_k, rot) -> torch.Tensor:
    """fp32 per-head RMSNorm + RoPE on the Q/K thirds of ``lin`` (actual_m, N); ``rot`` (actual_m, 128)."""
    M, N = lin.shape
    x = lin.float().clone().view(M, 3, N // 3 // HEAD_DIM, HEAD_DIM)
    sc = rot.float().view(M, 1, HEAD_DIM // 2, 2)
    for p, w in ((0, norm_q), (1, norm_k)):
        h = x[:, p]
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w.float()
        xe, xo = h[..., 0::2], h[..., 1::2]
        sin, cos = sc[..., 0], sc[..., 1]
        x[:, p] = torch.stack([xe * cos - xo * sin, xe * sin + xo * cos], -1).flatten(-2)
    return x.view(M, N)


def reference_gelu_quant(lin: torch.Tensor, smooth, lora_down, dtype, shift: float = GELU_SHIFT):
    """fp32 GELU + next-layer LoRA-down + unsigned int4 quantization of ``lin`` (M, N): (codes, scales, lora_act)."""
    M, N = lin.shape
    x = lin.float()
    x = 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    lora = x.to(dtype).float() @ lora_down.float() if lora_down is not None else None
    v = (x + shift) / (smooth.float() if smooth is not None else 1.0)
    scale = v.view(M, N // 64, 64).amax(-1).clamp_min(0) / 15
    q = v.view(M, N // 64, 64) / torch.where(scale == 0, 1.0, scale)[..., None]
    codes = torch.round(q).clamp(0, 15).to(torch.int32).view(M, N)
    return codes, scale.T, lora


def reference_swiglu_quant(lin: torch.Tensor, smooth, lora_down, dtype):
    """fp32 SwiGLU (``lin`` = [gate | up]) + LoRA-down + signed int4 quantization: (codes, scales, lora_act)."""
    M, N = lin.shape
    H = N // 2
    y = torch.nn.functional.silu(lin[:, :H].float()) * lin[:, H:].float()
    lora = y.to(dtype).float() @ lora_down.float() if lora_down is not None else None
    v = y / (smooth.float() if smooth is not None else 1.0)
    scale = v.view(M, H // 64, 64).abs().amax(-1) / 7
    q = v.view(M, H // 64, 64) / torch.where(scale == 0, 1.0, scale)[..., None]
    codes = torch.round(q).clamp(-8, 7).to(torch.int32).view(M, H)
    return codes, scale.T, lora


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


def gemm_qk(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rotary_emb,
            act_unsigned=False, tile=None):
    """
    The native GEMM with the fused Q/K RMSNorm + RoPE epilogue, directly (raises
    if unsupported). ``rotary_emb``: (actual_m, 64, 1, 2) or (actual_m, 128) fp32.
    """
    rot = _rot2d(rotary_emb, out)
    reason = unsupported_reason_qk(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rot)
    if reason is not None:
        raise ValueError(f"native W4A4 QK-norm GEMM does not support this call: {reason}")
    _kernels(_device(act), force=True).launch_qk(
        act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rot, act_unsigned, tile=tile
    )
    return out


def gemm_gelu_quant(act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down,
                    lora_act_out, act_unsigned=False, out=None, tile=None):
    """The native GEMM with the fused GELU + LoRA-down + int4 quantization epilogue, directly (raises if unsupported)."""
    reason = unsupported_reason_gelu_quant(
        act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out
    )
    if reason is not None:
        raise ValueError(f"native W4A4 GELU-quant GEMM does not support this call: {reason}")
    _kernels(_device(act), force=True).launch_gelu_quant(
        act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out,
        act_unsigned, tile=tile,
    )


def gemm_swiglu_quant(act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down,
                      lora_act_out, act_unsigned=False, tile=None):
    """The native GEMM with the fused SwiGLU + LoRA-down + signed int4 quantization epilogue, directly."""
    reason = unsupported_reason_swiglu_quant(
        act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out
    )
    if reason is not None:
        raise ValueError(f"native W4A4 SwiGLU-quant GEMM does not support this call: {reason}")
    _kernels(_device(act), force=True).launch_swiglu_quant(
        act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out,
        act_unsigned, tile=tile,
    )


def _device(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _rot2d(rotary_emb, out):
    if rotary_emb is None or out is None or rotary_emb.numel() != out.shape[0] * HEAD_DIM:
        return None
    return rotary_emb.reshape(out.shape[0], HEAD_DIM)


def _try(act, reason, run) -> bool:
    """Shared dispatch policy of the try_* functions: ``reason()`` gates, ``run(kernels)`` launches."""
    m = mode()
    if m == "off" or act.device.type != "cuda":
        return False
    device = _device(act)
    if m == "auto" and not device_arch(device).startswith("gfx11"):
        return False
    if reason() is not None:
        return False
    k = _kernels(device, force=m == "force")
    if k is None:
        return False
    try:
        run(k)
    except Exception as e:  # noqa: BLE001
        if m == "force":
            raise
        with _lock:
            _state[device] = e
        _warn(device, e)
        return False
    return True


def try_gemm(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned) -> bool:
    """
    Compute the GEMM natively if enabled and supported; ``False`` means the
    caller must run the Triton kernel. Arguments as :func:`gemm`.
    """
    return _try(
        act,
        lambda: unsupported_reason(act, wgt, out, ascales, wscales, bias, lora_act, lora_up),
        lambda k: k.launch(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, act_unsigned),
    )


def try_gemm_qk(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rotary_emb,
                act_unsigned) -> bool:
    """:func:`gemm_qk` if enabled and supported, else ``False`` (run Triton)."""
    rot = _rot2d(rotary_emb, out)
    return _try(
        act,
        lambda: unsupported_reason_qk(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rot),
        lambda k: k.launch_qk(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, norm_q, norm_k, rot,
                              act_unsigned),
    )


def try_gemm_gelu_quant(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down,
                        lora_act_out, act_unsigned) -> bool:
    """:func:`gemm_gelu_quant` if enabled and supported, else ``False`` (run Triton)."""
    return _try(
        act,
        lambda: unsupported_reason_gelu_quant(
            act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out
        ),
        lambda k: k.launch_gelu_quant(act, wgt, out, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth,
                                      lora_down, lora_act_out, act_unsigned),
    )


def try_gemm_swiglu_quant(act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down,
                          lora_act_out, act_unsigned) -> bool:
    """:func:`gemm_swiglu_quant` if enabled and supported, else ``False`` (run Triton)."""
    return _try(
        act,
        lambda: unsupported_reason_swiglu_quant(
            act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth, lora_down, lora_act_out
        ),
        lambda k: k.launch_swiglu_quant(act, wgt, ascales, wscales, bias, lora_act, lora_up, qout, oscales, smooth,
                                        lora_down, lora_act_out, act_unsigned),
    )
