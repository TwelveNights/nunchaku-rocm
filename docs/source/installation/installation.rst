.. _installation-installation:

Installation
============

This is the AMD ROCm/Triton fork of Nunchaku -- there are no prebuilt wheels
and no C++/CUDA extension to compile. Installation is a plain editable
pip install of a pure-Python package.

Prerequisites
-------------

- A ROCm-capable AMD GPU (verified on Instinct MI300X/CDNA3, Radeon PRO
  W7900 and Radeon RX 7800 XT/RDNA3) with a ROCm build of PyTorch installed
  *before* this package. On Linux see `PyTorch's ROCm install instructions
  <pytorch_home_>`_; on Windows 11 install AMD's ROCm 7.2.1 PyTorch wheels and
  ``triton-windows`` as described in the repository's ``README.md``. E.g. on
  Linux:

  .. code-block:: shell

      pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

- Triton (installed automatically as a PyTorch dependency on ROCm builds;
  verify with ``python -c "import triton; print(triton.__version__)"``).

Installing Nunchaku
--------------------

.. code-block:: shell

    git clone https://github.com/TwelveNights/nunchaku-rocm.git
    cd nunchaku-rocm
    pip install -e ".[dev,docs]"

This installs the package in editable mode along with development
dependencies, useful for contributing or running tests. If you only need to
use Nunchaku, ``pip install -e "."`` is sufficient.

Selecting the Triton Backend
-----------------------------

The Triton backend is used automatically when the (NVIDIA-only) compiled
``nunchaku._C`` extension is not importable -- the normal case on AMD. You can
also select it explicitly:

.. code-block:: shell

    NUNCHAKU_BACKEND=triton python your_script.py

See `nunchaku/ops/triton/README.md <https://github.com/TwelveNights/nunchaku-rocm/blob/main/nunchaku/ops/triton/README.md>`__
in the repository for details on what is implemented, benchmarks, and known gaps.
