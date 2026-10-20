# ComfyUI nodes for nunchaku-rocm

Run Nunchaku 4-bit SVDQuant models in [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on AMD GPUs,
with this repo's Triton backend (and its native int4 kernels on RDNA3 / gfx11).

Two nodes (category *Nunchaku (ROCm)*): **Nunchaku DiT Loader (ROCm)** (`NunchakuRocmDiTLoader`) loads a
Nunchaku checkpoint as a ComfyUI `MODEL`, and **Nunchaku LoRA Loader (ROCm)** (`NunchakuRocmLoraLoader`)
adds a LoRA to it (chain several for multiple LoRAs). Everything else is stock ComfyUI: its own text-encoder, VAE,
sampler, scheduler and reference-latent nodes. Under the hood the loader builds ComfyUI's FLUX.1 /
FLUX.2 model config and swaps in a thin adapter (`custom_nodes/nunchaku_rocm/adapter.py`) that reuses
ComfyUI's latent packing, reference-latent handling and RoPE position ids and only replaces the
transformer itself, so ComfyUI's samplers see exactly the model they expect.

## Supported models

| Model                                          | Checkpoint (`models/diffusion_models`)                  | Text encoder (`models/text_encoders`) | VAE (`models/vae`)      | Status                                      |
| ---------------------------------------------- | ------------------------------------------------------- | ------------------------------------- | ----------------------- | ------------------------------------------- |
| FLUX.2 [klein] 4B                              | `tonera/FLUX.2-klein-4B-Nunchaku` `svdq-int4_r32-…`     | `qwen_3_4b.safetensors`               | `flux2-vae.safetensors` | verified (text-to-image, reference editing) |
| FLUX.2 [klein] 9B                              | `tonera/FLUX.2-klein-9B-Nunchaku` `svdq-int4_r32-…`     | `qwen_3_8b_fp8mixed.safetensors`      | `flux2-vae.safetensors` | verified                                    |
| FLUX.1-schnell                                 | `nunchaku-ai/nunchaku-flux.1-schnell` `svdq-int4_r32-…` | `clip_l` + `t5xxl_fp16`               | `ae.safetensors`        | verified                                    |
| FLUX.1-dev / Krea / Kontext, FLUX.1 fine-tunes | Nunchaku FLUX.1 `svdq-int4_*`                           | `clip_l` + `t5xxl_fp16`               | `ae.safetensors`        | same code path, not yet tested              |

Text encoders and VAEs are ComfyUI's repackaged single files: `Comfy-Org/vae-text-encorder-for-flux-klein-4b`
and `-9b` (`split_files/…`), `comfyanonymous/flux_text_encoders`, and `ae.safetensors` from
`black-forest-labs/FLUX.1-schnell`. Use INT4 checkpoints: no AMD GPU has FP4 matrix units, so NVFP4 files
run on an emulated path.

Not supported yet: ControlNet, model-patching nodes that replace transformer blocks (they are
ignored with a warning), the `index_timestep_zero` reference method, Qwen-Image / Z-Image / SDXL
Nunchaku models.

## LoRA

Use **Nunchaku LoRA Loader (ROCm)** -- ComfyUI's own LoraLoader patches full-precision weights by name and
cannot reach the 4-bit layers. FLUX.1 and FLUX.2 [klein] LoRAs in ComfyUI / BFL (`diffusion_model.double_blocks...`),
kohya (`lora_unet_...`) and diffusers / PEFT (`transformer.transformer_blocks...`) format all work; text-encoder
weights in a LoRA file are ignored. The loader is implemented by `nunchaku.lora.runtime.set_loras` (usable
from Python too): on the 4-bit SVDQuant layers the LoRA is appended to the layer's existing 16-bit low-rank
branch (no extra kernels; ~0.3 s per Klein 4B image on the RX 7800 XT), other layers get a small hook. Switching
LoRAs or strengths between runs re-applies them in a fraction of a second; MODEL branches with and without
LoRAs can share one loaded model.

Checked against ComfyUI's own LoRA loading on its bf16 model (`scripts/check_lora.py`, MI300X):

| LoRA (format)                              | Model          | Image PSNR, ours+LoRA vs bf16+LoRA | bf16+LoRA vs bf16 without |
| ------------------------------------------ | -------------- | ---------------------------------- | ------------------------- |
| Norod78 old-gods (ComfyUI/BFL, r32)        | Klein 4B       | 17.4 dB                            | 13.0 dB                   |
| Shakker-Labs Logo-Design (kohya, r4)       | FLUX.1-schnell | 19.8 dB                            | 16.4 dB                   |
| alvdansen softserve_anime (diffusers, r64) | FLUX.1-schnell | 15.0 dB                            | 7.3 dB                    |

In every case the LoRA's effect (style / subject change) is reproduced; the per-image gap is the usual 4-bit
one. `workflows/flux.2-klein-4b-lora.json` is a Klein example (LoRA in `models/loras`).

## Install

1. Install this repo (`pip install -e .` with a ROCm PyTorch + Triton) into ComfyUI's Python environment,
   without letting ComfyUI's requirements replace your ROCm PyTorch:

   ```bash
   python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > constraints.txt
   pip install -r ComfyUI/requirements.txt -c constraints.txt
   ```

1. Point ComfyUI at the nodes (and, optionally, at checkpoint folders) in `ComfyUI/extra_model_paths.yaml`:

   ```yaml
   nunchaku_rocm:
       base_path: /path/to/nunchaku-rocm
       custom_nodes: comfyui/custom_nodes
       # diffusion_models: /path/to/folder/with/svdq-int4_*.safetensors
   ```

   (Copying or linking `comfyui/custom_nodes/nunchaku_rocm` into `ComfyUI/custom_nodes/` works too.)

1. Load a workflow from [`workflows/`](workflows) (API format; drag it into ComfyUI or submit it to
   `/prompt`): `flux.2-klein-4b.json`, `flux.2-klein-9b.json`, `flux.2-klein-4b-edit.json` (reference
   image editing; put the image in ComfyUI's `input/`), `flux.2-klein-4b-lora.json`, `flux.1-schnell.json`.

On Windows, keep the native kernels' compile caches out of `%LOCALAPPDATA%` if you like by setting
`NUNCHAKU_CACHE_DIR` and `AMD_COMGR_CACHE_DIR` (see `nunchaku/ops/triton/README.md`).

## Performance (RX 7800 XT, 16 GB, Windows 11, ROCm 7.2.1)

1024×1024, 4 steps, warm (prompt encoding cached, seed changed per run), `--highvram`; range over the
warm runs of one session:

| Workflow                                           | Time per image |
| -------------------------------------------------- | -------------- |
| FLUX.2 [klein] 4B                                  | 2.9–3.4 s      |
| FLUX.2 [klein] 4B, reference edit (1 MP reference) | 7.9–8.6 s      |
| FLUX.2 [klein] 9B                                  | 7.9–8.4 s      |
| FLUX.1-schnell                                     | 7.5–8.4 s      |

Without `--highvram`, ComfyUI partially unloads the transformer before every VAE decode and reloads it
for the next image (Klein 9B: 9.0–9.2 s per image instead). The first run of a session also includes model
loading, Triton autotuning and the one-time native-kernel compile.

## Validation

`scripts/check_against_bf16.py` compares the adapter with ComfyUI's own bf16 model on identical inputs
(real text embeddings), per call and end to end through ComfyUI's sampler (MI300X):

| Model             | Per call, σ = 0.6 / 0.3: rel. L2 vs ComfyUI bf16 (vs diffusers bf16) | diffusers bf16 vs ComfyUI bf16 | End to end, PSNR vs bf16 |
| ----------------- | -------------------------------------------------------------------- | ------------------------------ | ------------------------ |
| FLUX.2 [klein] 4B | 0.371 / 0.479 (0.371 / 0.479)                                        | 0.015 / 0.014                  | 15.7 dB                  |
| FLUX.1-schnell    | 0.205 / 0.142 (0.205 / 0.142)                                        | 0.018 / 0.011                  | 16.4 dB                  |

The error against ComfyUI's bf16 model equals the error against the diffusers bf16 model fed the tokens and
position ids ComfyUI built, so it is the 4-bit quantization alone (the same as in diffusers); the adapter
adds none. End-to-end images keep the bf16 composition.

`scripts/run_workflows.py` runs workflows headless through ComfyUI's HTTP API and times them (it starts
and stops the server itself).
