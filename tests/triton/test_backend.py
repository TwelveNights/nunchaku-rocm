"""
Tests for backend selection and the dispatching op wrappers.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def backend():
    mod = _load("_nunchaku_backend", _ROOT / "nunchaku" / "ops" / "backend.py")
    yield mod
    mod.set_backend(None)


def test_default_without_extension(backend, monkeypatch):
    """With no compiled extension present, the Triton backend is selected."""
    monkeypatch.delenv("NUNCHAKU_BACKEND", raising=False)
    backend.has_cuda_extension.cache_clear()
    monkeypatch.setattr(backend, "has_cuda_extension", lambda: False)
    assert backend.get_backend() == "triton"


def test_env_var_selects_triton(backend, monkeypatch):
    monkeypatch.setenv("NUNCHAKU_BACKEND", "triton")
    assert backend.get_backend() == "triton"


def test_env_var_rejects_unknown(backend, monkeypatch):
    monkeypatch.setenv("NUNCHAKU_BACKEND", "opencl")
    with pytest.raises(ValueError):
        backend.get_backend()


def test_cuda_request_falls_back_with_warning(backend, monkeypatch):
    """Asking for CUDA where it cannot exist warns rather than hard-failing."""
    monkeypatch.setenv("NUNCHAKU_BACKEND", "cuda")
    backend.has_cuda_extension.cache_clear()
    monkeypatch.setattr(backend, "has_cuda_extension", lambda: False)
    with pytest.warns(RuntimeWarning):
        assert backend.get_backend() == "triton"


def test_set_backend_overrides_env(backend, monkeypatch):
    monkeypatch.setenv("NUNCHAKU_BACKEND", "cuda")
    backend.set_backend("triton")
    assert backend.get_backend() == "triton"
    backend.set_backend(None)


def test_set_backend_rejects_unknown(backend):
    with pytest.raises(ValueError):
        backend.set_backend("metal")


def test_load_cuda_ops_raises_clearly(backend, monkeypatch):
    monkeypatch.setattr(backend, "has_cuda_extension", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA-only"):
        backend.load_cuda_ops()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("layout", ["bhtd", "bthd_view"])
def test_scaled_dot_product_attention_wrapper(dtype, layout):
    """The SDPA drop-in matches torch for both contiguous (B, H, T, D) inputs and (B, T, H, D) transposed views."""
    import torch

    from nunchaku.ops.backend import set_backend
    from nunchaku.ops.dispatch import scaled_dot_product_attention

    if not torch.cuda.is_available():
        pytest.skip("no GPU available")
    set_backend("triton")
    try:
        dtype = getattr(torch, dtype)
        B, H, T, D = 2, 4, 300, 128
        if layout == "bhtd":
            q, k, v = (torch.randn(B, H, T, D, dtype=dtype, device="cuda") for _ in range(3))
        else:
            q, k, v = (torch.randn(B, T, H, D, dtype=dtype, device="cuda").transpose(1, 2) for _ in range(3))
        out = scaled_dot_product_attention(q, k, v)
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        assert out.shape == ref.shape
        torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)

        # a mask is outside the kernel's coverage and must fall back to torch, not be ignored
        mask = torch.rand(B, 1, T, T, device="cuda") > 0.5
        mask[..., 0] = True
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask)
        torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
    finally:
        set_backend(None)
