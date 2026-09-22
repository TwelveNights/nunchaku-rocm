"""
Triton implementations of Nunchaku's kernels.

This backend is a portable alternative to the hand-written CUDA kernels in
``src/kernels``. It runs on both NVIDIA and AMD (CDNA/RDNA) GPUs, whereas the
CUDA backend depends on NVIDIA-only ``mma``/``cp.async`` PTX.

The CUDA backend remains the default. Select this one with::

    NUNCHAKU_BACKEND=triton

or programmatically via :func:`nunchaku.ops.backend.set_backend`.
"""

from .activation import gelu_fast, gelu_new, silu, silu_and_mul
from .attention import attention_fp16
from .awq import awq_dequantize, awq_gemm_w4a16, awq_gemv_w4a16
from .gemm_fp16 import gemm_batched_fp16, gemm_fp16
from .gemm_w4a4 import gemm_w4a4, quantize_w4a4_act, quantize_w4a4_act_fuse_lora, quantize_w4a4_wgt
from .gemm_w8a8 import gemm_w8a8, quantize_w8a8_act
from .layernorm import layernorm_general, rms_norm, rms_norm_general
from .misc import (
    add,
    argmax_sample,
    cast,
    embedding,
    mul_add,
    mul_add_batch,
    quant_static,
    quant_static_fuse_gelu,
    split_mod,
    splitqkv,
    topk,
)

__all__ = [
    "add",
    "mul_add",
    "mul_add_batch",
    "embedding",
    "argmax_sample",
    "splitqkv",
    "split_mod",
    "quant_static",
    "quant_static_fuse_gelu",
    "cast",
    "topk",
    "silu",
    "silu_and_mul",
    "gelu_new",
    "gelu_fast",
    "rms_norm",
    "rms_norm_general",
    "layernorm_general",
    "gemm_w8a8",
    "quantize_w8a8_act",
    "gemm_fp16",
    "gemm_batched_fp16",
    "awq_dequantize",
    "awq_gemv_w4a16",
    "awq_gemm_w4a16",
    "gemm_w4a4",
    "quantize_w4a4_act",
    "quantize_w4a4_act_fuse_lora",
    "quantize_w4a4_wgt",
    "attention_fp16",
]
