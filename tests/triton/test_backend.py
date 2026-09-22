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
