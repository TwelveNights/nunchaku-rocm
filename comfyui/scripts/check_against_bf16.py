"""
Check the Nunchaku ComfyUI adapter against ComfyUI's own full-precision model.

  python check_against_bf16.py --comfy <ComfyUI dir> --kind flux2 --nunchaku <svdq .safetensors> \
      --ref <ComfyUI bf16 .safetensors> --repo black-forest-labs/FLUX.2-klein-4B --vae <flux2-vae.safetensors> [--out dir]
  python check_against_bf16.py --comfy <ComfyUI dir> --kind flux1 --nunchaku <svdq .safetensors> \
      --repo black-forest-labs/FLUX.1-schnell --vae <ae.safetensors> [--out dir]

  python check_against_bf16.py --comfy <ComfyUI dir> --kind {qwen-image,z-image,sdxl} --nunchaku <svdq .safetensors> \
      --repo <diffusers repo: Qwen/Qwen-Image, Tongyi-MAI/Z-Image-Turbo, stabilityai/stable-diffusion-xl-base-1.0> \
      [--cpu-offload enable] [--out dir]

(The FLUX.1, Qwen-Image, Z-Image and SDXL references are built from the diffusers weights in --repo, which ComfyUI
converts.)

1. Per call, at a few noise levels, on the same latent and real text embeddings (from the diffusers
   pipeline's encoder): relative L2 of the Nunchaku output against ComfyUI's bf16 model and against the
   diffusers bf16 transformer run on the exact tokens / position ids ComfyUI built (captured from the
   adapter). The diffusers-vs-ComfyUI bf16 column is the convention check: it must be at bf16-noise level.
2. End to end: ComfyUI's sampler (euler / simple, cfg 1) with each model from the same noise and
   conditioning, decoded with ComfyUI's VAE; image PSNR of Nunchaku vs bf16.

For Qwen-Image, Z-Image and SDXL the convention check runs the diffusers bf16 model through the same adapter
class as the Nunchaku model (the adapters only need the diffusers interface), and the end-to-end images are
decoded with the diffusers VAE.
"""

import argparse
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True)
ap.add_argument("--kind", choices=["flux1", "flux2", "qwen-image", "z-image", "sdxl"], required=True)
ap.add_argument("--nunchaku", required=True)
ap.add_argument("--ref")
ap.add_argument("--repo", required=True)
ap.add_argument("--vae", help="ComfyUI VAE file (FLUX kinds)")
ap.add_argument("--steps", type=int, help="end-to-end sampling steps (default: 4 for FLUX, see KINDS otherwise)")
ap.add_argument("--cpu-offload", default="disable", help="Qwen-Image: auto / enable / disable")
ap.add_argument("--out")
args = ap.parse_args()

sys.path[:0] = [args.comfy, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_nodes")]
os.environ.setdefault("NUNCHAKU_BACKEND", "triton")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from nunchaku_rocm.nodes import load_model  # noqa: E402

PROMPT = "A cat holding a sign that says hello world, photo"
dev, dt = mm.get_torch_device(), torch.bfloat16


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def psnr(a, b):
    return 10 * np.log10(255**2 / ((a - b) ** 2).mean())


# per kind: diffusers pipeline / model classes, per-call noise levels (sigma, or the discrete SDXL timestep),
# end-to-end steps, cfg, sampler, scheduler
KINDS = {
    "qwen-image": dict(sub="transformer", t=(0.95, 0.6, 0.3), steps=20, cfg=4.0, sampler=("euler", "simple")),
    "z-image": dict(sub="transformer", t=(0.95, 0.6, 0.3), steps=8, cfg=1.0, sampler=("euler", "simple")),
    "sdxl": dict(sub="unet", t=(999.0, 600.0, 300.0), steps=20, cfg=5.0, sampler=("euler", "normal")),
}


def check_other_kind():
    import diffusers
    from nunchaku_rocm.qwen_image import NunchakuQwenImageAdapter
    from nunchaku_rocm.sdxl import NunchakuSDXLAdapter
    from nunchaku_rocm.z_image import NunchakuZImageAdapter

    kind, k = args.kind, KINDS[args.kind]
    steps = args.steps or k["steps"]
    weights = "*.fp16.safetensors" if kind == "sdxl" else "*.safetensors"
    patterns = [
        "*.json",
        "*.txt",
        "tokenizer*/*",
        f"text_encoder*/{weights}",
        f"{k['sub']}/{weights}",
        f"vae/{weights}",
    ]
    repo = snapshot_download(args.repo, allow_patterns=patterns)
    ours = load_model(args.nunchaku, cpu_offload=args.cpu_offload)

    # ComfyUI's bf16 model, converted by ComfyUI from the diffusers weights
    sd = {}
    for f in sorted(glob.glob(os.path.join(repo, k["sub"], "*.safetensors"))):
        sd.update(comfy.utils.load_torch_file(f))
    ref = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"dtype": dt})
    del sd

    # text conditioning from the diffusers pipeline's encoders; the diffusers bf16 model in the same adapter
    if kind == "qwen-image":
        pipe = diffusers.QwenImagePipeline.from_pretrained(repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
        ctx, _ = pipe.encode_prompt(PROMPT, device=dev)
        neg, _ = pipe.encode_prompt(" ", device=dev)
        diff = diffusers.QwenImageTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dt)
        diff_dm = NunchakuQwenImageAdapter(diff.to(dev), diff.config, dt)
        vae = diffusers.AutoencoderKLQwenImage.from_pretrained(repo, subfolder="vae", torch_dtype=dt).to(dev)
        latent_shape, pos_extra, neg_extra, call_extra = (1, 16, 1, 128, 128), {}, {}, {}
    elif kind == "z-image":
        pipe = diffusers.ZImagePipeline.from_pretrained(repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
        ctx = pipe.encode_prompt(PROMPT, device=dev, do_classifier_free_guidance=False)[0][0].unsqueeze(0)
        neg = torch.zeros_like(ctx)
        diff = diffusers.ZImageTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dt)
        diff_dm = NunchakuZImageAdapter(diff.to(dev), diff.config, dt)
        vae = diffusers.AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dt).to(dev)
        latent_shape, pos_extra, neg_extra = (1, 16, 128, 128), {}, {}
        call_extra = dict(num_tokens=ctx.shape[1])
    else:
        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
            repo, unet=None, vae=None, variant="fp16", torch_dtype=dt
        ).to(dev)
        ctx, neg, pooled, neg_pooled = pipe.encode_prompt(PROMPT, device=dev, do_classifier_free_guidance=True)
        diff = diffusers.UNet2DConditionModel.from_pretrained(repo, subfolder="unet", variant="fp16", torch_dtype=dt)
        diff_dm = NunchakuSDXLAdapter(diff.to(dev), dt)
        vae = diffusers.AutoencoderKL.from_pretrained(repo, subfolder="vae", variant="fp16", torch_dtype=torch.float32)
        vae = vae.to(dev)
        size = dict(width=1024, height=1024)
        latent_shape, pos_extra, neg_extra = (
            (1, 4, 128, 128),
            dict(pooled_output=pooled, **size),
            dict(pooled_output=neg_pooled, **size),
        )
        call_extra = dict(y=ref.model.encode_adm(**pos_extra).to(dev, dt))
    del pipe
    mm.load_models_gpu([ref, ours])
    dm_ref, dm_q = ref.model.diffusion_model, ours.model.diffusion_model

    # ---- 1. per call
    x = torch.randn(latent_shape, generator=torch.Generator().manual_seed(0)).to(dev, dt)
    print(f"{'t':>6} {'nunchaku vs comfy bf16':>23} {'nunchaku vs diffusers bf16':>27} {'diffusers vs comfy bf16':>24}")
    with torch.no_grad():
        for t in k["t"]:
            t = torch.tensor([t], device=dev)
            out_ref, out_q, out_d = (dm(x, t, ctx, **call_extra) for dm in (dm_ref, dm_q, diff_dm))
            print(
                f"{t.item():>6} {rel(out_q, out_ref):>23.4f} {rel(out_q, out_d):>27.4f} {rel(out_d, out_ref):>24.4f}",
                flush=True,
            )

    # ---- 2. end to end through ComfyUI's sampler; the diffusers bf16 model runs as a third MODEL
    cfg = type(ref.model.model_config)(dict(ref.model.model_config.unet_config, disable_unet_model_creation=True))
    cfg.set_inference_dtype(dt, None)
    diff_model = cfg.get_model({})
    diff_model.diffusion_model = diff_dm
    diff_patcher = comfy.model_patcher.ModelPatcher(
        diff_model, load_device=dev, offload_device=mm.unet_offload_device()
    )
    positive, negative = [[ctx, pos_extra]], [[neg, neg_extra]]
    latent = torch.zeros(latent_shape)
    noise = comfy.sample.prepare_noise(latent, 0)
    images = {}
    for name, model in (("bf16", ref), ("nunchaku", ours), ("diffusers-bf16", diff_patcher)):
        samples = comfy.sample.sample(
            model, noise, steps, k["cfg"], *k["sampler"], positive, negative, latent, disable_pbar=True, seed=0
        )
        with torch.no_grad():
            img = vae.decode(samples.to(dev, vae.dtype)).sample
        img = img[:, :, 0] if img.ndim == 5 else img
        images[name] = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().permute(1, 2, 0).cpu().numpy()
        if args.out:
            from PIL import Image

            Image.fromarray(images[name].astype(np.uint8)).save(os.path.join(args.out, f"comfy-{kind}-{name}.png"))
    print(
        f"end to end ({steps} steps, cfg {k['cfg']}): image PSNR vs ComfyUI bf16: nunchaku "
        f"{psnr(images['nunchaku'], images['bf16']):.2f} dB, diffusers bf16 {psnr(images['diffusers-bf16'], images['bf16']):.2f} dB",
        flush=True,
    )


if args.kind in KINDS:
    check_other_kind()
    os._exit(0)
if args.vae is None:
    ap.error("--vae is required for the FLUX kinds")
args.steps = args.steps or 4

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
        dkw = dict(
            hidden_states=c["img"],
            encoder_hidden_states=c["txt"],
            timestep=c["timesteps"],
            img_ids=c["img_ids"][0],
            txt_ids=c["txt_ids"][0],
            return_dict=False,
        )
        if args.kind == "flux1":
            dkw["pooled_projections"] = c["y"]
            if diff.config.guidance_embeds:
                dkw["guidance"] = c["guidance"]
        tok_d = diff(**dkw)[0]
        tok_ref, tok_q = dm_ref.process_img(out_ref)[0], dm_q.process_img(out_q)[0]
        print(
            f"{sigma:>6} {rel(out_q, out_ref):>23.4f} {rel(tok_q, tok_d):>27.4f} {rel(tok_d, tok_ref):>24.4f}",
            flush=True,
        )
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
    samples = comfy.sample.sample(
        model, noise, args.steps, 1.0, "euler", "simple", positive, negative, latent, disable_pbar=True, seed=0
    )
    with torch.no_grad():
        decoded = vae.decode(samples)
    images[name] = (decoded[0].detach().float().clamp(0, 1).cpu().numpy() * 255).round()
    if args.out:
        from PIL import Image

        Image.fromarray(images[name].astype(np.uint8)).save(os.path.join(args.out, f"comfy-{args.kind}-{name}.png"))
a, b = images["nunchaku"], images["bf16"]
print(
    f"end to end ({args.steps} steps): image PSNR nunchaku vs bf16 {10 * np.log10(255**2 / ((a - b) ** 2).mean()):.2f} dB",
    flush=True,
)
os._exit(0)
