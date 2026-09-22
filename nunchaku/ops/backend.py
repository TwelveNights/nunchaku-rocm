"""
Backend selection for Nunchaku's kernels.

Two backends are available:

``cuda``
    The hand-written CUDA/PTX kernels in ``src/kernels``, exposed through the
    compiled ``nunchaku._C`` extension. This is the default and remains the
    fastest option on NVIDIA hardware.

``triton``
    Portable Triton kernels under :mod:`nunchaku.ops.triton`. These run on AMD
    GPUs as well, where ``nunchaku._C`` cannot be built (the CUDA kernels use
    NVIDIA-only ``mma`` / ``cp.async`` PTX).

Select a backend with the ``NUNCHAKU_BACKEND`` environment variable or
:func:`set_backend`. The default is ``cuda`` when the extension is importable
and ``triton`` otherwise.
"""

import functools
import os
import warnings

__all__ = ["get_backend", "set_backend", "has_cuda_extension", "load_cuda_ops"]

_VALID = ("cuda", "triton")
_override: str | None = None


@functools.lru_cache(maxsize=1)
def has_cuda_extension() -> bool:
    """True when the compiled ``nunchaku._C`` extension can be imported."""
    try:
        import nunchaku._C  # noqa: F401
    except Exception:
        return False
    return True


def load_cuda_ops():
    """
    Import and return ``nunchaku._C.ops``.

    Raises a clear error when the extension is unavailable, which is the normal
    situation on AMD GPUs.
    """
    try:
        from nunchaku._C import ops
    except Exception as exc:  # pragma: no cover - depends on build
        raise RuntimeError(
            "The compiled CUDA extension (nunchaku._C) is unavailable. "
            "It is NVIDIA-only; on other hardware set NUNCHAKU_BACKEND=triton."
        ) from exc
    return ops


def get_backend() -> str:
    """
    Return the active backend name, ``"cuda"`` or ``"triton"``.

    Resolution order: :func:`set_backend` override, then ``NUNCHAKU_BACKEND``,
    then ``cuda`` if the extension is importable, else ``triton``.
    """
    if _override is not None:
        return _override

    env = os.environ.get("NUNCHAKU_BACKEND")
    if env is not None:
        env = env.strip().lower()
        if env not in _VALID:
            raise ValueError(f"NUNCHAKU_BACKEND must be one of {_VALID}, got {env!r}")
        if env == "cuda" and not has_cuda_extension():
            warnings.warn(
                "NUNCHAKU_BACKEND=cuda was requested but nunchaku._C is not available; "
                "falling back to the Triton backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "triton"
        return env

    return "cuda" if has_cuda_extension() else "triton"


def set_backend(name: str | None) -> None:
    """
    Override the backend for the current process.

    Pass ``None`` to clear the override and fall back to environment-based
    resolution.
    """
    global _override
    if name is not None:
        name = name.strip().lower()
        if name not in _VALID:
            raise ValueError(f"backend must be one of {_VALID}, got {name!r}")
    _override = name
