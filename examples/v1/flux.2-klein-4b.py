"""
FLUX.2 [klein] 4B text-to-image with a 4-bit SVDQuant transformer on the Triton backend (AMD GPUs).

Uses the community INT4 checkpoint from tonera/FLUX.2-klein-4B-Nunchaku. INT4 is the right choice on AMD:
the NVFP4 checkpoint also loads, but no AMD GPU has FP4 matrix units, so it runs on an emulated path.
"""

import os

os.environ.setdefault("NUNCHAKU_BACKEND", "triton")

import torch
from diffusers import Flux2KleinPipeline

from nunchaku import NunchakuFlux2Transformer2DModel

transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
    "tonera/FLUX.2-klein-4B-Nunchaku/svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors", torch_dtype=torch.bfloat16
)
pipeline = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", transformer=transformer, torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()  # fits 16 GB cards; or pipeline.to("cuda") with more VRAM
image = pipeline(
    prompt="A cat holding a sign that says hello world",
    num_inference_steps=4,
    height=1024,
    width=1024,
    generator=torch.Generator().manual_seed(0),
).images[0]
image.save("flux.2-klein-4b-int4.png")
