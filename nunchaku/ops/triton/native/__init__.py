"""
Hand-written HIP kernels for GPUs where Triton leaves performance behind.

Currently one: an int4 W4A4 GEMM for RDNA3 (gfx11) built on the
``v_wmma_i32_16x16x16_iu4`` instruction, which Triton cannot emit (it lowers
the int4 GEMM to int8 WMMA at half the rate). See ``w4a4_gfx11.hip`` and
:mod:`.w4a4`; ``nunchaku.ops.triton.gemm_w4a4.gemm_w4a4`` dispatches to it.

- :mod:`.hip`: ctypes bindings to the HIP runtime torch already loaded;
  loads code objects and launches on torch's current stream.
- :mod:`.hiprtc`: ctypes bindings to hipRTC (runtime compilation, no host
  toolchain needed).
- :mod:`.build`: compiles shipped ``.hip`` sources for the GPU's arch on
  first use and caches the code objects on disk.

Importing this package touches neither library; that happens on the first
native call.
"""
