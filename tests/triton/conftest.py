"""
Test configuration for the Triton kernel suite.

The kernel modules are loaded directly from their files rather than via
``import nunchaku.ops.triton``. Importing through the package would execute
``nunchaku/__init__.py``, which pulls in the model layer and hence the
compiled ``nunchaku._C`` extension. That extension is NVIDIA-only, and the
whole point of the Triton backend is to work where it does not exist.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_TRITON_DIR = Path(__file__).resolve().parents[2] / "nunchaku" / "ops" / "triton"
_PKG = "_nunchaku_triton"


def load_kernels(name: str):
    """Load ``nunchaku/ops/triton/<name>.py`` standalone, resolving its relative imports."""
    if _PKG not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(_PKG, loader=None, is_package=True)
        )
        pkg.__path__ = [str(_TRITON_DIR)]
        sys.modules[_PKG] = pkg

    full = f"{_PKG}.{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, _TRITON_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda_ext: requires the compiled nunchaku._C extension")


@pytest.fixture(autouse=True)
def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")
