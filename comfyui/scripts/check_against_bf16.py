"""
Check the Nunchaku ComfyUI adapter against ComfyUI's own full-precision model.

  python check_against_bf16.py --comfy <ComfyUI dir> --kind flux2 --nunchaku <svdq .safetensors> \
      --ref <ComfyUI bf16 .safetensors> --repo black-forest-labs/FLUX.2-klein-4B --vae <flux2-vae.safetensors> [--out dir]
  python check_against_bf16.py --comfy <ComfyUI dir> --kind flux1 --nunchaku <svdq .safetensors> \
      --repo black-forest-labs/FLUX.1-schnell --vae <ae.safetensors> [--out dir]

(FLUX.1's reference is built from the diffusers transformer in --repo, which ComfyUI converts.)

1. Per call, at a few noise levels, on the same latent and real text embeddings (from the diffusers
   pipeline's encoder): relative L2 of the Nunchaku output against ComfyUI's bf16 model and against the
   diffusers bf16 transformer run on the exact tokens / position ids ComfyUI built (captured from the
   adapter). The diffusers-vs-ComfyUI bf16 column is the convention check: it must be at bf16-noise level.
2. End to end: ComfyUI's sampler (euler / simple, cfg 1) with each model from the same noise and
   conditioning, decoded with ComfyUI's VAE; image PSNR of Nunchaku vs bf16.
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

from nunchaku_rocm.nodes import load_model  # noqa: E402

PROMPT = "A cat holding a sign that says hello world, photo"
dev, dt = mm.get_torch_device(), torch.bfloat16

ours = load_model(args.nunchaku)
if args.kind == "flux2":
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

    ref = comfy.sd.load_diffusion_model(args.ref)
    pipe = Flux2KleinPipeline.from_pretrained(args.repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
    ctx, _ = pipe.encode_prompt(prompt=PROMPT, device=dev)
    pooled = None
    latent_shape = (1, 128, 64, 64)  # 1024 x 1024
    diff_cls = Flux2Transformer2DModel
else:
    from diffusers import FluxPipeline, FluxTransformer2DModel

    snap = snapshot_download(args.repo, allow_patterns=["transformer/*"])
    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "transformer", "*.safetensors"))):
        sd.update(comfy.utils.load_torch_file(f))
    ref = comfy.sd.load_diffusion_model_state_dict(sd)
    del sd
    pipe = FluxPipeline.from_pretrained(args.repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
    ctx, pooled, _ = pipe.encode_prompt(prompt=PROMPT, prompt_2=None, device=dev, max_sequence_length=256)
    latent_shape = (1, 16, 128, 128)
    diff_cls = FluxTransformer2DModel
del pipe
mm.load_models_gpu([ref, ours])
dm_ref, dm_q = ref.model.diffusion_model, ours.model.diffusion_model


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


# ---- 1. per call
captured = {}
orig_forward_orig = dm_q.forward_orig


def capture(img, img_ids, txt, txt_ids, timesteps, y, guidance=None, control=None, **kw):
    captured.update(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=timesteps, y=y, guidance=guidance)
    return orig_forward_orig(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, **kw)


dm_q.forward_orig = capture
# local snapshot path: diffusers queries the Hub for sharded checkpoints given a repo id, even offline
diff_dir = snapshot_download(args.repo, allow_patterns=["transformer/*"])
diff = diff_cls.from_pretrained(diff_dir, subfolder="transformer", torch_dtype=dt).to(dev)
# what ComfyUI's model_base.Flux2.extra_conds does to the text before it reaches the diffusion model
ctx_dm = torch.nn.functional.pad(ctx, (0, 0, 512 - ctx.shape[1], 0)) if args.kind == "flux2" else ctx
x = torch.randn(latent_shape, generator=torch.Generator().manual_seed(0)).to(dev, dt)
guid = dict(guidance=torch.tensor([3.5], device=dev)) if args.kind == "flux1" else {}

print(f"{'sigma':>6} {'nunchaku vs comfy bf16':>23} {'nunchaku vs diffusers bf16':>27} {'diffusers vs comfy bf16':>24}")
with torch.no_grad():
    for sigma in (0.95, 0.6, 0.3):
        t = torch.tensor([sigma], device=dev)
        out_ref = dm_ref(x, t, ctx_dm, y=pooled, **guid)
        out_q = dm_q(x, t, ctx_dm, y=pooled, **guid)
        c = captured
        dkw = dict(hidden_states=c["img"], encoder_hidden_states=c["txt"], timestep=c["timesteps"],
                   img_ids=c["img_ids"][0], txt_ids=c["txt_ids"][0], return_dict=False)
        if args.kind == "flux1":
            dkw["pooled_projections"] = c["y"]
            if diff.config.guidance_embeds:
                dkw["guidance"] = c["guidance"]
        tok_d = diff(**dkw)[0]
        tok_ref, tok_q = dm_ref.process_img(out_ref)[0], dm_q.process_img(out_q)[0]
        print(f"{sigma:>6} {rel(out_q, out_ref):>23.4f} {rel(tok_q, tok_d):>27.4f} {rel(tok_d, tok_ref):>24.4f}",
              flush=True)
dm_q.forward_orig = orig_forward_orig
del diff
torch.cuda.empty_cache()

# ---- 2. end to end through ComfyUI's sampler
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
extra = {"pooled_output": pooled} if pooled is not None else {}
if args.kind == "flux1":
    extra["guidance"] = 3.5
positive = [[ctx, extra]]
negative = [[torch.zeros_like(ctx), dict(extra)]]
latent = torch.zeros(latent_shape)
noise = comfy.sample.prepare_noise(latent, 0)
images = {}
for name, model in (("bf16", ref), ("nunchaku", ours)):
    samples = comfy.sample.sample(model, noise, args.steps, 1.0, "euler", "simple", positive, negative, latent,
                                  disable_pbar=True, seed=0)
    with torch.no_grad():
        decoded = vae.decode(samples)
    images[name] = (decoded[0].detach().float().clamp(0, 1).cpu().numpy() * 255).round()
    if args.out:
        from PIL import Image

        Image.fromarray(images[name].astype(np.uint8)).save(os.path.join(args.out, f"comfy-{args.kind}-{name}.png"))
a, b = images["nunchaku"], images["bf16"]
print(f"end to end ({args.steps} steps): image PSNR nunchaku vs bf16 {10 * np.log10(255**2 / ((a - b) ** 2).mean()):.2f} dB",
      flush=True)
os._exit(0)
