"""
End-to-end FLUX.1-schnell image generation on AMD GPUs via the Triton backend.

Mirrors examples/v1/flux.1-dev.py, but forces NUNCHAKU_BACKEND=triton so the
model runs entirely on the portable Triton kernels (no nunchaku._C
extension), and uses FLUX.1-schnell since it needs no CFG / few steps.
"""

import os

os.environ.setdefault("NUNCHAKU_BACKEND", "triton")

import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2

transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    "nunchaku-ai/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors",
    torch_dtype=torch.bfloat16,
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline(
    "A cat holding a sign that says hello world",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
image.save("flux.1-schnell-triton.png")
print("saved flux.1-schnell-triton.png")
