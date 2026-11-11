# ComfyUI nodes for nunchaku-rocm

Run Nunchaku 4-bit SVDQuant models in [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on AMD GPUs, on this
repo's Triton backend (and its native int4 kernels on RDNA3 / gfx11): FLUX.1, FLUX.2 [klein], Qwen-Image,
Z-Image-Turbo and SDXL.

Everything except the diffusion model is stock ComfyUI -- its text-encoder, VAE, sampler, scheduler, model-sampling
and reference-latent nodes -- so the usual workflows apply with one node swapped.

## Nodes

Category *Nunchaku (ROCm)*:

- **Nunchaku DiT Loader (ROCm)** (`NunchakuRocmDiTLoader`) loads a Nunchaku checkpoint from `models/diffusion_models`
  as a ComfyUI `MODEL`. The model type (FLUX.1, FLUX.2, Qwen-Image, Z-Image, SDXL) is read from the checkpoint's
  metadata. Optional inputs, used by Qwen-Image only:
  - `cpu_offload` (`auto` / `enable` / `disable`): keep the 60 transformer blocks in CPU memory and stream them
    through the GPU one at a time (overlapped with compute). `auto` turns it on for GPUs with less than 20 GiB.
  - `num_blocks_on_gpu` (default 1): blocks that stay resident with offload on. More blocks use more VRAM and less
    PCIe traffic.
- **Nunchaku LoRA Loader (ROCm)** (`NunchakuRocmLoraLoader`) adds a LoRA to a model from the loader
  (chain several for multiple LoRAs). See [LoRA](#lora).

## Supported models

| Model                                   | Checkpoint (`models/diffusion_models`)                                        | Text encoder(s) (`models/text_encoders`)                    | VAE (`models/vae`)           | Status                                      |
| --------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------- | ------------------------------------------- |
| FLUX.2 [klein] 4B                       | `tonera/FLUX.2-klein-4B-Nunchaku`: `svdq-int4_r32-…`                          | `qwen_3_4b.safetensors` (type `flux2`)                      | `flux2-vae.safetensors`      | verified (text-to-image, reference editing) |
| FLUX.2 [klein] 9B                       | `tonera/FLUX.2-klein-9B-Nunchaku`: `svdq-int4_r32-…`                          | `qwen_3_8b_fp8mixed.safetensors` (type `flux2`)             | `flux2-vae.safetensors`      | verified                                    |
| FLUX.1-schnell                          | `nunchaku-ai/nunchaku-flux.1-schnell`: `svdq-int4_r32-…`                      | `clip_l` + `t5xxl_fp16` (DualCLIPLoader, type `flux`)       | `ae.safetensors`             | verified                                    |
| FLUX.1-dev / Krea / Kontext, fine-tunes | Nunchaku FLUX.1 `svdq-int4_*`                                                 | `clip_l` + `t5xxl_fp16`                                     | `ae.safetensors`             | same code path, not tested                  |
| Qwen-Image, Qwen-Image Lightning        | `nunchaku-ai/nunchaku-qwen-image`: `svdq-int4_r32-qwen-image…`                | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (type `qwen_image`) | `qwen_image_vae.safetensors` | verified (text-to-image)                    |
| Z-Image-Turbo                           | `nunchaku-ai/nunchaku-z-image-turbo`: `svdq-int4_r32-z-image-turbo` (or r128) | `qwen_3_4b.safetensors` (type `lumina2`)                    | `ae.safetensors`             | verified                                    |
| SDXL, SDXL-Turbo                        | `nunchaku-ai/nunchaku-sdxl`, `nunchaku-ai/nunchaku-sdxl-turbo`                | `clip_l` + `clip_g` (DualCLIPLoader, type `sdxl`)           | `sdxl_vae.safetensors`       | verified (text-to-image)                    |

Where the encoder and VAE files come from:

- `qwen_3_4b.safetensors`, `flux2-vae.safetensors`: `Comfy-Org/vae-text-encorder-for-flux-klein-4b`
  (`split_files/…`); the same `qwen_3_4b.safetensors` (identical file) and `ae.safetensors` are also in
  `Comfy-Org/z_image_turbo`. `qwen_3_8b_fp8mixed.safetensors`: `Comfy-Org/vae-text-encorder-for-flux-klein-9b`.
- `clip_l.safetensors`, `t5xxl_fp16.safetensors`: `comfyanonymous/flux_text_encoders`; `ae.safetensors`:
  `black-forest-labs/FLUX.1-schnell`.
- `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors`: `Comfy-Org/Qwen-Image_ComfyUI`
  (`split_files/text_encoders`, `split_files/vae`).
- SDXL: `clip_l.safetensors` is the same CLIP-L as above; `clip_g.safetensors` is
  `text_encoder_2/model.fp16.safetensors` and `sdxl_vae.safetensors` is `vae/diffusion_pytorch_model.fp16.safetensors`
  from `stabilityai/stable-diffusion-xl-base-1.0`, renamed (ComfyUI reads the diffusers format).

Use INT4 checkpoints: no AMD GPU has FP4 matrix units, so NVFP4 files run on an emulated path.

Not supported yet: ControlNet, Qwen-Image-Edit / reference latents for Qwen-Image and Z-Image's omni inputs (the
nodes raise an error), model-patching nodes that replace transformer blocks or
attention (they are ignored with a warning), FLUX.2's `index_timestep_zero` reference method.

## Install

The nodes need this repo installed (`pip install -e`) into **ComfyUI's Python environment** with a ROCm PyTorch, and
ComfyUI pointed at `comfyui/custom_nodes`. Never let ComfyUI's `requirements.txt` replace the ROCm PyTorch.

### Windows 11 (Radeon RX 7000 / PRO W7000): ComfyUI from git, in the nunchaku-rocm environment (tested)

1. Install nunchaku-rocm as in the [main README](../README.md#install) (Python 3.12 venv with the ROCm 7.2.1 PyTorch
   wheels and `triton-windows`), and in that venv:

   ```powershell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   python -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ['torch', 'torchvision', 'triton-windows']))" | Out-File -Encoding ascii constraints.txt
   pip install -r ComfyUI\requirements.txt -c constraints.txt
   ```

   (Validated with ComfyUI commit `b1a247c`.)

1. Create `ComfyUI\extra_model_paths.yaml` (paths may point into a Hugging Face cache snapshot directory):

   ```yaml
   nunchaku_rocm:
       base_path: C:/path/to/nunchaku-rocm
       custom_nodes: comfyui/custom_nodes
       # diffusion_models: C:/path/to/folder/with/svdq-int4_*.safetensors
   ```

   (Copying `comfyui\custom_nodes\nunchaku_rocm` into `ComfyUI\custom_nodes\` works too.)

1. Start ComfyUI from a shell with the ROCm SDK on `PATH` (see the main README), e.g.
   `python ComfyUI\main.py --highvram`, and load a workflow from [`workflows/`](workflows).

### Windows: ComfyUI portable for AMD (not tested)

The portable AMD build ships its own Python and ROCm PyTorch in `python_embeded`. The same steps should apply there:
`python_embeded\python.exe -m pip install triton-windows==<matching version> -e C:\path\to\nunchaku-rocm` (check
first that this does not replace its PyTorch), then the `extra_model_paths.yaml` entry above. This combination has
not been run; the tested setup is the one above.

### Linux

Install nunchaku-rocm with a ROCm PyTorch ([main README](../README.md#install)), then ComfyUI into the same
environment with its requirements constrained to the installed PyTorch:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
pip freeze | grep -E '^(torch|torchvision|triton|pytorch-triton-rocm|triton-rocm)==' > constraints.txt
pip install -r ComfyUI/requirements.txt -c constraints.txt
```

and the `extra_model_paths.yaml` entry with Linux paths. Validated on MI300X (ROCm 7.2, PyTorch 2.10 ROCm 7.0 build, ComfyUI `b1a247c`).

### Caches

On first use each model compiles and autotunes its Triton kernels and, on RDNA3, builds the native kernels (cached
under `TRITON_CACHE_DIR` and `NUNCHAKU_CACHE_DIR`); the first image takes up to a minute or two longer. On Windows,
set `NUNCHAKU_CACHE_DIR` and `AMD_COMGR_CACHE_DIR` to keep these caches out of `%LOCALAPPDATA%` if you like (see
[`nunchaku/ops/triton/README.md`](../nunchaku/ops/triton/README.md)).

## Example workflows

API-format workflows in [`workflows/`](workflows) (drag into ComfyUI, or submit to `/prompt`; all 1024×1024 except
SDXL-Turbo):

| Workflow                      | Model                       | Sampling                                                 |
| ----------------------------- | --------------------------- | -------------------------------------------------------- |
| `flux.2-klein-4b.json`, `-9b` | FLUX.2 [klein] 4B / 9B      | 4 steps, euler, cfg 1                                    |
| `flux.2-klein-4b-edit.json`   | Klein 4B reference edit     | as above, image from ComfyUI's `input/`                  |
| `flux.2-klein-4b-lora.json`   | Klein 4B + LoRA             | LoRA in `models/loras`                                   |
| `flux.1-schnell.json`         | FLUX.1-schnell              | 4 steps, euler, cfg 1                                    |
| `qwen-image.json`             | Qwen-Image                  | shift 3.1, 20 steps, euler, cfg 2.5; `cpu_offload` auto  |
| `qwen-image-lightning.json`   | Qwen-Image Lightning 8-step | shift 3.1, 8 steps, euler, cfg 1; `cpu_offload` auto     |
| `z-image-turbo.json`          | Z-Image-Turbo               | shift 3, 8 steps, res_multistep, cfg 1                   |
| `sdxl.json`                   | SDXL                        | CLIPTextEncodeSDXL 1024, 30 steps, euler / normal, cfg 5 |
| `sdxl-turbo.json`             | SDXL-Turbo, 512×512         | SDTurboScheduler 4 steps, euler_ancestral, cfg 1         |

## VRAM on 16 GB cards

On Windows the driver does not report out-of-memory past the card's VRAM: it silently spills into shared system
memory and everything gets several times slower. Measured peaks below stay under 16 GB with these settings:

- **Qwen-Image** (int4 transformer 11.5 GB, text encoder 9.4 GB): leave `cpu_offload` on `auto` (on below 20 GiB);
  0.65 GB of the transformer stays on the GPU (embeddings, output layers, one resident block and two streaming
  buffers; each further `num_blocks_on_gpu` adds 0.19 GB) and the rest needs ~11 GB of (pinned) system RAM. With
  the transformer resident, the diffusers pipeline peaked at 19.8 GiB, more than a 16 GB card has.
- **FLUX.2 [klein] 9B**, **Z-Image**, **SDXL**, **FLUX.1**, **Klein 4B**: load whole; ComfyUI moves the text encoder
  out of the way when sampling starts.
- `--highvram` keeps the models on the GPU between runs and is fastest when everything fits; without it ComfyUI may
  partially unload the transformer before the VAE decode and reload it for the next image (Klein 9B: 9.0-9.2 s per
  image instead of 7.9-8.4 s). The Nunchaku layers cannot run from CPU memory the way ComfyUI's `--lowvram` mode runs
  its own layers, so the loader always loads (the GPU-resident part of) the model whole.

## LoRA

Use **Nunchaku LoRA Loader (ROCm)** -- ComfyUI's own LoraLoader patches full-precision weights by name and cannot reach
the 4-bit layers. FLUX.1 and FLUX.2 [klein] LoRAs in ComfyUI / BFL (`diffusion_model.double_blocks...`), kohya
(`lora_unet_...`) and diffusers / PEFT (`transformer.transformer_blocks...`) format all work, and so do Z-Image,
Qwen-Image (also with `cpu_offload`) and SDXL LoRAs in ComfyUI, diffusers or kohya format, including SDXL convolution
(LoCon) LoRAs such as LCM-LoRA; text-encoder weights in a LoRA file are ignored. The loader is implemented by `nunchaku.lora.runtime.set_loras` (usable from Python too): on the
4-bit SVDQuant layers the LoRA is appended to the layer's existing 16-bit low-rank branch (no extra kernels; ~0.3 s
per Klein 4B image on the RX 7800 XT), other layers get a small hook. Switching LoRAs or strengths between runs
re-applies them in a fraction of a second; MODEL branches with and without LoRAs can share one loaded model.

Checked against ComfyUI's own LoRA loading on its bf16 model (`scripts/check_lora.py`, MI300X):

| LoRA (format)                              | Model          | Image PSNR, ours+LoRA vs bf16+LoRA | bf16+LoRA vs bf16 without |
| ------------------------------------------ | -------------- | ---------------------------------- | ------------------------- |
| Norod78 old-gods (ComfyUI/BFL, r32)        | Klein 4B       | 17.4 dB                            | 13.0 dB                   |
| Shakker-Labs Logo-Design (kohya, r4)       | FLUX.1-schnell | 19.8 dB                            | 16.4 dB                   |
| alvdansen softserve_anime (diffusers, r64) | FLUX.1-schnell | 15.0 dB                            | 7.3 dB                    |

In every case the LoRA's effect (style / subject change) is reproduced; the per-image gap is the usual 4-bit one.

Z-Image, Qwen-Image and SDXL, against ComfyUI's LoRA loading on the bf16 models (Radeon PRO W7900, ComfyUI 0.45, same
seed and sampler):

| LoRA (format)                                         | Model                     | Image PSNR, ours+LoRA vs bf16+LoRA | bf16+LoRA vs bf16 without |
| ----------------------------------------------------- | ------------------------- | ---------------------------------- | ------------------------- |
| renderartist Coloring-Book (ComfyUI, r32)             | Z-Image-Turbo, 8 steps    | 14.2 dB                            | 13.5 dB                   |
| ostris watercolor (kohya, attention)                  | SDXL, 20 steps            | 18.9 dB                            | 13.0 dB                   |
| latent-consistency LCM-LoRA (kohya, incl. conv)       | SDXL, 4 steps LCM         | 22.4 dB                            | 15.7 dB                   |
| lightx2v Qwen-Image-Lightning 8-step V1.1 (diffusers) | Qwen-Image, 8 steps cfg 1 | 18.4 dB (same with `cpu_offload`)  | 17.7 dB                   |

The 4-bit model's activations are quantized too, so a LoRA's effect on it is close to, not identical with, its effect
on the bf16 model; per layer the added LoRA term is exact.

## Performance (RX 7800 XT, 16 GB, Windows 11, ROCm 7.2.1)

Headless through ComfyUI's API (`scripts/run_workflows.py`), `--highvram`, the workflows above, warm runs (models
loaded, kernels compiled; the seed changes every run, so sampling and decoding always run). "Peak" is the most device
memory in use during the run, from ComfyUI's `/system_stats` (it includes the text encoder while it is resident).

| Workflow                                         | Time / image (warm) | Peak GPU memory, warm (first run) | First run |
| ------------------------------------------------ | ------------------- | --------------------------------- | --------- |
| FLUX.2 [klein] 4B, 4 steps                       | 2.9 s               | 4.3 GiB (10.7 GiB)                | 45 s      |
| FLUX.2 [klein] 4B, reference edit (1 MP ref.) \* | 7.9-8.6 s           |                                   |           |
| FLUX.2 [klein] 9B, 4 steps \*                    | 7.9-8.4 s           |                                   |           |
| FLUX.1-schnell, 4 steps                          | 5.5-5.7 s           | 7.7 GiB (12.9 GiB)                | 70 s      |
| Z-Image-Turbo, 8 steps                           | 8.4-8.5 s           | 5.3 GiB (11.8 GiB)                | 51 s      |
| Qwen-Image Lightning, 8 steps, offload `auto`    | 11.6 s              | 10.1 GiB (9.7 GiB)                | 102 s     |
| Qwen-Image, 20 steps, cfg 2.5, offload `auto`    | 55.0 s              | 9.4 GiB (11.1 GiB)                | 123 s     |
| SDXL, 30 steps, cfg 5                            | 10.6-10.7 s         | 4.3 GiB (4.8 GiB)                 | 33 s      |
| SDXL-Turbo, 512², 4 steps                        | 0.52-0.55 s         | 4.6 GiB (4.6 GiB)                 | 18 s      |

One ComfyUI session per workflow, 3 runs each; "first run" is the first image of the session (model loading from
disk and text encoding; the Triton / native kernel caches were already populated -- a cold cache adds up to a minute
or two). * Measured in an earlier session of the same setup, not re-run. Running all workflows in one session (models
of earlier workflows kept in RAM) made the later ones slower (Klein 4B 5.0 s, FLUX.1-schnell 7.7-8.1 s); keep only the
models you use loaded. Without `--highvram`, ComfyUI partially unloads the transformer before every VAE decode and
reloads it for the next image (Klein 9B: 9.0-9.2 s per image instead of 7.9-8.4 s).

For comparison, the diffusers pipelines on the same card (main README) take 10.8 s (Z-Image, model CPU offload),
15.8 s (Qwen-Image Lightning, per-block + sequential offload) and 11.6 s (SDXL): ComfyUI keeps the text encoder off
the GPU while sampling and the transformer resident instead.

## Validation

**Against ComfyUI's own bf16 models** (`scripts/check_against_bf16.py`, MI300X). Per call: the same latent and real
text embeddings into the Nunchaku model and ComfyUI's bf16 model (and, as a convention check, the diffusers bf16 model
through the same adapter class); end to end: ComfyUI's sampler from the same noise and conditioning.

| Model             | Per call, rel. L2 vs ComfyUI bf16 (vs diffusers bf16) | diffusers bf16 vs ComfyUI bf16 | End to end, PSNR vs ComfyUI bf16 (diffusers bf16 vs ComfyUI bf16) |
| ----------------- | ----------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------- |
| FLUX.2 [klein] 4B | σ 0.6 / 0.3: 0.371 / 0.479 (0.371 / 0.479)            | 0.015 / 0.014                  | 15.7 dB, 4 steps                                                  |
| FLUX.1-schnell    | σ 0.6 / 0.3: 0.205 / 0.142 (0.205 / 0.142)            | 0.018 / 0.011                  | 16.4 dB, 4 steps                                                  |
| Qwen-Image        | σ 0.6 / 0.3: 0.038 / 0.062 (0.038 / 0.044)            | 0.006 / 0.024                  | 21.9 dB (30.7 dB), 20 steps, cfg 4                                |
| Z-Image-Turbo     | σ 0.6 / 0.3: 0.302 / 0.271 (0.302 / 0.271)            | 0.014 / 0.010                  | 12.8 dB (35.0 dB), 8 steps                                        |
| SDXL              | t 600 / 300: 0.008 / 0.046 (0.008 / 0.046)            | 0.002 / 0.004                  | 23.9 dB (32.1 dB), 20 steps, cfg 5                                |

The 4-bit model's error against ComfyUI's bf16 model equals its error against the diffusers bf16 model (up to the
two bf16 implementations' own difference), and the diffusers bf16 model run through the adapter matches ComfyUI's
bf16 model to within bf16 rounding (30-35 dB end to end), so the adapters reproduce
ComfyUI's conventions (latent packing, timesteps, positions, text handling) and add no error; what remains is the
4-bit quantization, as large as in diffusers. Qwen-Image with `cpu_offload` on gives bit-identical images.

**The example workflows, through ComfyUI's API** (`scripts/run_workflows.py --out / --reference`), same seeds, image
PSNR:

| Workflow (seed 0 / seed 1) | RX 7800 XT int4 vs MI300X bf16 | MI300X int4 vs MI300X bf16 | RX 7800 XT int4 vs MI300X int4 |
| -------------------------- | ------------------------------ | -------------------------- | ------------------------------ |
| Z-Image-Turbo              | 12.9 / 11.1 dB                 | 12.7 / 11.0 dB             | 15.9 / 15.4 dB                 |
| Qwen-Image                 | 20.1 / 24.5 dB                 | 18.0 / 23.6 dB             | 18.2 / 27.5 dB                 |
| Qwen-Image Lightning       | (no bf16 reference)            |                            | 21.2 / 19.8 dB                 |
| SDXL                       | 19.8 / 17.2 dB                 | 19.7 / 18.6 dB             | 27.3 / 21.0 dB                 |
| SDXL-Turbo                 | 12.1 / 15.4 dB                 | 14.6 / 16.7 dB             | 14.1 / 16.0 dB                 |

The RX 7800 XT (native gfx11 kernels, Windows) is as close to the bf16 model as the MI300X (Triton kernels, Linux);
the two int4 runs differ from each other about as much as each differs from bf16, the usual sensitivity of a
4-bit model to rounding differences. All images keep the bf16 composition and the sign's text.

(The bf16 references are the same workflows with ComfyUI's `UNETLoader` on the original weights, run on MI300X.
The ancestral sampler of the SDXL-Turbo workflow injects fresh noise every step, which amplifies small differences.
Qwen-Image Lightning has no bf16 counterpart: it is a distilled LoRA merged into the base weights.)

`tests/comfyui/test_adapters.py` checks the adapters without checkpoints (small random models: adapter vs ComfyUI's
own implementation, and the Qwen-Image offload path bit-exact against the resident one, with back-to-back calls).
Run it with `COMFYUI_PATH=<ComfyUI checkout> pytest tests/comfyui`.

## How it works

The loader builds ComfyUI's own model configuration for the checkpoint (FLUX / FLUX.2, Qwen-Image, Lumina 2
(Z-Image) or SDXL), so ComfyUI's model sampling, latent format and conditioning are untouched, and replaces only its
`diffusion_model` with a thin adapter (`custom_nodes/nunchaku_rocm/`) around the Nunchaku model:

- **FLUX.1 / FLUX.2** (`adapter.py`): inherits ComfyUI's `Flux` and replaces `forward_orig`, so latent packing,
  reference latents and RoPE position ids are ComfyUI's.
- **Qwen-Image** (`qwen_image.py`): inherits ComfyUI's patchifying and position ids and converts its RoPE to the
  Nunchaku blocks' complex form; runs the Nunchaku embeddings, blocks and output layers. Per-block offload keeps the
  offloaded blocks outside the module tree, so ComfyUI's memory management sees (and moves) only the resident part.
- **Z-Image** (`z_image.py`): converts ComfyUI's `NextDiT` call (latent batch, sigma) to diffusers' list-based
  `ZImageTransformer2DModel` call (`t = 1 - sigma`) and negates the output as ComfyUI does.
- **SDXL** (`sdxl.py`): feeds ComfyUI's `y` (pooled CLIP-G + size embeddings) straight into the UNet's
  `add_embedding`.

`scripts/run_workflows.py` runs workflows headless through ComfyUI's HTTP API (it starts and stops the server
itself), times them, records peak GPU memory and optionally saves / compares the images.
