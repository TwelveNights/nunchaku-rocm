"""
Run Diffusers' native "Nunchaku Lite" checkpoints (``quantization_config.quant_method == "nunchaku_lite"``, e.g.
``lite-infer/LTX-2.3-Distilled-v1.1-Diffusers-nunchaku-lite-int4``) on the Triton backend.

Diffusers loads those checkpoints with its own ``SVDQW4A4Linear`` / ``AWQW4A16Linear`` and fetches compiled CUDA kernels
(``rootonchair/nunchaku-lite-kernels``) through the Hugging Face ``kernels`` package. :func:`enable` answers that
request with this package's Triton kernels instead -- nothing is downloaded or executed from the Hub -- lets the
quantizer accept ROCm GPUs, and converts every W4A4 layer to the natural layout the Triton GEMM reads once its weights
are loaded::

    import nunchaku.diffusers_lite

    nunchaku.diffusers_lite.enable()  # before loading the pipeline
    pipe = LTX2Pipeline.from_pretrained("lite-infer/LTX-2.3-Distilled-v1.1-Diffusers-nunchaku-lite-int4", ...)
"""

import os
import sys
import types

import torch

_KERNEL_REPO = "rootonchair/nunchaku-lite-kernels"


def _quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4):
    from .ops.triton.gemm_w4a4 import quantize_w4a4_act_fuse_lora

    quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)


def _gemm_w4a4(*args):
    from .ops.gemm import svdq_gemm_w4a4_cuda

    args = list(args)
    if isinstance(args[23], torch.Tensor):  # alpha: Diffusers passes the NVFP4 wtscale parameter
        args[23] = float(args[23])
    svdq_gemm_w4a4_cuda(*args)


def _awq_gemm_w4a16_g64_int32(x, qweight, wscales, wzeros):
    from .ops.triton.awq import awq_gemm_w4a16

    return awq_gemm_w4a16(x, qweight, wscales, wzeros, group_size=64)


def _gemv_awq(x, qweight, wscales, wzeros, m, n, k, group_size):
    from .ops.triton.awq import awq_gemv_w4a16

    return awq_gemv_w4a16(x, qweight, wscales, wzeros, m, n, k, group_size)


ops = types.SimpleNamespace(
    quantize_w4a4_act_fuse_lora=_quantize_w4a4_act_fuse_lora,
    gemm_w4a4=_gemm_w4a4,
    awq_gemm_w4a16_g64_int32=_awq_gemm_w4a16_g64_int32,
    gemv_awq=_gemv_awq,
)


def _get_kernel(original):
    def get_kernel(repo_id, *args, **kwargs):
        if repo_id == _KERNEL_REPO:
            return types.SimpleNamespace(ops=ops)
        if original is None:
            raise ImportError(f"the `kernels` package is not installed (requested {repo_id})")
        return original(repo_id, *args, **kwargs)

    return get_kernel


def enable() -> None:
    """Route Diffusers' Nunchaku Lite layers to the Triton kernels. Call before loading a Nunchaku Lite pipeline."""
    os.environ.setdefault("NUNCHAKU_BACKEND", "triton")
    if "diffusers.quantizers.nunchaku.utils" in sys.modules:
        raise RuntimeError("call nunchaku.diffusers_lite.enable() before loading a Nunchaku Lite model")

    try:
        import kernels
    except ImportError:
        kernels = types.ModuleType("kernels")
        kernels.get_kernel = None
        sys.modules["kernels"] = kernels
    if not getattr(kernels.get_kernel, "_nunchaku", False):
        kernels.get_kernel = _get_kernel(kernels.get_kernel)
        kernels.get_kernel._nunchaku = True

    import diffusers.utils as dutils
    import diffusers.utils.constants as constants
    from diffusers.quantizers.nunchaku import nunchaku_quantizer

    # nothing remote runs: the kernel request is answered locally
    constants.DIFFUSERS_TRUST_REMOTE_KERNELS = True
    dutils.is_kernels_available = lambda: True
    dutils.is_kernels_version = lambda op, version: True
    nunchaku_quantizer.is_kernels_available = lambda: True

    from diffusers.quantizers.nunchaku import utils as lite

    quantizer = nunchaku_quantizer.NunchakuLiteQuantizer
    if getattr(quantizer, "_nunchaku_triton", False):
        return
    validate = quantizer.validate_environment

    def validate_environment(self, *args, **kwargs):
        if torch.version.hip is None:
            return validate(self, *args, **kwargs)
        if not torch.cuda.is_available():
            raise ValueError("Loading Nunchaku checkpoints requires a GPU.")

    def process_model_after_weight_loading(self, model, **kwargs):
        from .ops.triton.layout import convert_svdq_linear_to_natural_layout

        for module in model.modules():
            if isinstance(module, lite.SVDQW4A4Linear) and not getattr(module, "_natural_layout", False):
                convert_svdq_linear_to_natural_layout(module)
                module._natural_layout = True
        return model

    quantizer.validate_environment = validate_environment
    quantizer._process_model_after_weight_loading = process_model_after_weight_loading
    quantizer._nunchaku_triton = True
