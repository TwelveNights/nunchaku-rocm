"""
Shared helpers for the Triton kernel backend.
"""

import functools

import torch
import triton
import triton.language as tl


def is_hip() -> bool:
    """True when running on an AMD ROCm build of Triton."""
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    return str(triton.runtime.driver.active.get_current_target().arch)


def is_cdna() -> bool:
    """True on AMD CDNA (MI2xx/MI3xx) parts, which have 64-wide wavefronts."""
    return is_hip() and _arch() in ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")


def num_warps_for(n: int) -> int:
    """
    Pick a warp count.

    A CDNA wavefront is 64 lanes versus 32 on NVIDIA, so the same tile needs
    half as many "warps" to cover it. Triton still counts in units of the
    target's native wavefront, so we halve the NVIDIA-tuned number.
    """
    if is_cdna():
        return max(1, n // 2)
    return n


def matmul_group_m() -> int:
    """
    Row-group size for L2-friendly tile swizzling.

    MI300X has 304 CUs and a 4 MB L2 per XCD; a larger group amortizes weight
    re-reads better than the usual NVIDIA value of 8.
    """
    return 16 if is_cdna() else 8


def torch_dtype_to_tl(dtype: torch.dtype):
    """Map a torch dtype to the matching ``triton.language`` dtype."""
    return {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.int8: tl.int8,
        torch.int32: tl.int32,
    }[dtype]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def contiguous_2d(x: torch.Tensor) -> torch.Tensor:
    """Flatten all leading dims into rows, returning a contiguous 2-D view."""
    return x.reshape(-1, x.shape[-1]).contiguous()


@triton.jit
def gelu_tanh(x):
    """GELU, tanh approximation. Matches ``vllm::gelu_new_kernel``."""
    x = x.to(tl.float32)
    inner = 0.79788456 * (x + 0.044715 * x * x * x)
    # tanh(z) via the numerically-stable exp form; Triton lowers tl.math.tanh
    # inconsistently across backends, so spell it out.
    t = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    return 0.5 * x * (1.0 + t)


@triton.jit
def gelu_fast_impl(x):
    """GELU, 'fast' variant. Matches ``vllm::gelu_fast_kernel``."""
    x = x.to(tl.float32)
    inner = x * 0.79788456 * (1.0 + 0.044715 * x * x)
    t = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    return 0.5 * x * (1.0 + t)


@triton.jit
def silu_impl(x):
    """SiLU / swish: ``x * sigmoid(x)``."""
    x = x.to(tl.float32)
    return x / (1.0 + tl.exp(-x))


@triton.jit
def round_to_int8(x):
    """
    Round-to-nearest-even then saturate to int8.

    Mirrors ``float_to_int8_rn``, which uses the ``cvt.rni.s8.f32`` PTX
    instruction (saturating). Triton has no saturating cast, so clamp first.
    """
    x = tl.minimum(tl.maximum(_rne(x), -128.0), 127.0)
    return x.to(tl.int8)


@triton.jit
def _rne(x):
    """Round half to even, matching IEEE ``cvt.rni``."""
    floor = tl.floor(x)
    diff = x - floor
    # exact .5 ties go to the even neighbour
    is_tie = diff == 0.5
    even = (tl.floor(floor * 0.5) * 2.0) == floor
    up = tl.where(is_tie, tl.where(even, 0.0, 1.0), tl.where(diff > 0.5, 1.0, 0.0))
    return floor + up
