"""
Check Nunchaku LoRA support against ComfyUI's own LoRA loading on its full-precision model.

  python check_lora.py --comfy <ComfyUI dir> --kind flux2 --nunchaku <svdq .safetensors> \
      --ref <ComfyUI bf16 .safetensors> --repo black-forest-labs/FLUX.2-klein-4B --vae <vae> \
      --lora <lora .safetensors> [--strength 1.0] [--prompt ...] [--out dir]
  (--kind flux1: --repo black-forest-labs/FLUX.1-schnell, no --ref; ComfyUI converts the diffusers transformer)

The reference is ComfyUI's bf16 model with the LoRA applied by ``comfy.sd.load_lora_for_models``; ours is
the Nunchaku model with ``Nunchaku LoRA Loader`` (adapter + nunchaku.lora.runtime).

1. Per call, on the same latent and real text embeddings: the LoRA's effect on the output
   (``with LoRA - without``) for both models -- cosine similarity and relative L2 between the two effects --
   and the overall error of ours-with-LoRA against bf16-with-LoRA.
2. End to end through ComfyUI's sampler: images with and without the LoRA for both models, PSNR of ours vs
   bf16 with the LoRA, and (with --out) a montage [bf16 | bf16 + LoRA | ours | ours + LoRA].
"""

import argparse
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True)
ap.add_argument("--kind", choices=["flux1", "flux2"], required=True)
ap.add_argument("--nunchaku", required=True)
ap.add_argument("--ref")
ap.add_argument("--repo", required=True)
ap.add_argument("--vae", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--strength", type=float, default=1.0)
ap.add_argument("--prompt", default="A cat holding a sign that says hello world")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--out")
args = ap.parse_args()

sys.path[:0] = [args.comfy, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_nodes")]
os.environ.setdefault("NUNCHAKU_BACKEND", "triton")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import comfy.model_management as mm  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from nunchaku_rocm.nodes import add_lora, load_model  # noqa: E402

dev, dt = mm.get_torch_device(), torch.bfloat16
ours = load_model(args.nunchaku)
if args.kind == "flux2":
    from diffusers import Flux2KleinPipeline

    ref = comfy.sd.load_diffusion_model(args.ref)
    pipe = Flux2KleinPipeline.from_pretrained(args.repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
    ctx, _ = pipe.encode_prompt(prompt=args.prompt, device=dev)
    pooled, latent_shape = None, (1, 128, 64, 64)
else:
    from diffusers import FluxPipeline

    snap = snapshot_download(args.repo, allow_patterns=["transformer/*"])
    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "transformer", "*.safetensors"))):
        sd.update(comfy.utils.load_torch_file(f))
    ref = comfy.sd.load_diffusion_model_state_dict(sd)
    del sd
    pipe = FluxPipeline.from_pretrained(args.repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
    ctx, pooled, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=None, device=dev, max_sequence_length=256)
    latent_shape = (1, 16, 128, 128)
del pipe

lora_sd = comfy.utils.load_torch_file(args.lora, safe_load=True)
ref_l, _ = comfy.sd.load_lora_for_models(ref, None, lora_sd, args.strength, 0)
ours_l = add_lora(ours, args.lora, args.strength)

ctx_dm = torch.nn.functional.pad(ctx, (0, 0, 512 - ctx.shape[1], 0)) if args.kind == "flux2" else ctx
x = torch.randn(latent_shape, generator=torch.Generator().manual_seed(0)).to(dev, dt)
guid = dict(guidance=torch.tensor([3.5], device=dev)) if args.kind == "flux1" else {}


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def run_calls(patcher, topts):
    mm.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    with torch.no_grad():
        return [
            dm(x, torch.tensor([s], device=dev), ctx_dm, y=pooled, transformer_options=topts, **guid)
            for s in (0.6, 0.3)
        ]


r_base, r_lora = run_calls(ref, {}), run_calls(ref_l, {})
q_base = run_calls(ours, {})
q_lora = run_calls(ours_l, ours_l.model_options["transformer_options"])
print(
    f"{'sigma':>6} {'effect cos':>11} {'effect rel-L2':>14} {'effect size (bf16)':>19} {'ours+LoRA vs bf16+LoRA':>23}"
)
for s, rb, rl, qb, ql in zip((0.6, 0.3), r_base, r_lora, q_base, q_lora):
    print(
        f"{s:>6} {cos(ql - qb, rl - rb):>11.4f} {rel(ql - qb, rl - rb):>14.4f} {rel(rl, rb):>19.4f} "
        f"{rel(ql, rl):>23.4f}",
        flush=True,
    )

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
extra = {"pooled_output": pooled} if pooled is not None else {}
if args.kind == "flux1":
    extra["guidance"] = 3.5
positive, negative = [[ctx, extra]], [[torch.zeros_like(ctx), dict(extra)]]
latent = torch.zeros(latent_shape)
noise = comfy.sample.prepare_noise(latent, 0)
images = {}
for name, model in (("bf16", ref), ("bf16+lora", ref_l), ("ours", ours), ("ours+lora", ours_l)):
    samples = comfy.sample.sample(
        model, noise, args.steps, 1.0, "euler", "simple", positive, negative, latent, disable_pbar=True, seed=0
    )
    with torch.no_grad():
        images[name] = (vae.decode(samples)[0].detach().float().clamp(0, 1).cpu().numpy() * 255).round()
psnr = lambda a, b: 10 * np.log10(255**2 / ((a - b) ** 2).mean())  # noqa: E731
print(
    f"end to end: PSNR ours+LoRA vs bf16+LoRA {psnr(images['ours+lora'], images['bf16+lora']):.2f} dB "
    f"(without LoRA: {psnr(images['ours'], images['bf16']):.2f} dB; LoRA vs no LoRA, bf16: "
    f"{psnr(images['bf16+lora'], images['bf16']):.2f} dB)",
    flush=True,
)
if args.out:
    from PIL import Image

    tiles = [Image.fromarray(images[k].astype(np.uint8)).resize((512, 512)) for k in images]
    montage = Image.new("RGB", (512 * len(tiles), 512))
    for i, t in enumerate(tiles):
        montage.paste(t, (512 * i, 0))
    montage.save(os.path.join(args.out, f"lora-{args.kind}-{os.path.splitext(os.path.basename(args.lora))[0]}.png"))
os._exit(0)
