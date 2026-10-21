# nunchaku-rocm

**4-bit [SVDQuant](http://arxiv.org/abs/2411.05007) diffusion models on AMD GPUs** -- an AMD ROCm / Triton
derivative of [Nunchaku](https://github.com/nunchaku-ai/nunchaku). It is not an official Nunchaku release.

Upstream Nunchaku runs its 4-bit (W4A4) models through a CUDA extension built on NVIDIA-only PTX, so it cannot run on
AMD hardware at all. nunchaku-rocm replaces that extension with

- a portable **Triton kernel backend** (`nunchaku/ops/triton/`): the W4A4 GEMM with its fused epilogues (Q/K
  RMSNorm + RoPE, GELU / SwiGLU + next-layer quantization), the activation quantizer with fused LoRA-down, AWQ W4A16,
  flash attention and the small kernels, plus a converter from the CUDA kernels' checkpoint layout, so the **existing
  Nunchaku int4 checkpoints load unchanged**;
- **hand-written int4 kernels for RDNA3** (`nunchaku/ops/triton/native/`): Triton cannot emit RDNA3's int4 matrix
  instruction (`v_wmma_i32_16x16x16_iu4`), so the GEMMs run on a HIP kernel compiled at runtime with hipRTC (no
  compiler toolchain needed, also on Windows), 2.2x faster than the Triton kernel on an RX 7800 XT;
- **ComfyUI nodes** ([`comfyui/`](comfyui/README.md)) and **runtime LoRA** for FLUX (`nunchaku.lora.runtime`).

The diffusers-style model classes and the Python API are upstream's (`from nunchaku import ...`), so upstream's
model docs and examples largely apply.

## Supported hardware

| GPU                     | Arch    | OS           | Software validated                                            | Kernels              |
| ----------------------- | ------- | ------------ | ------------------------------------------------------------- | -------------------- |
| Radeon RX 7800 XT 16 GB | gfx1101 | Windows 11   | ROCm 7.2.1, PyTorch 2.9.1, triton-windows 3.6, diffusers 0.41 | native int4 + Triton |
| Radeon PRO W7900 48 GB  | gfx1100 | Ubuntu 24.04 | ROCm 7.2, PyTorch 2.10 (ROCm 7.0), Triton 3.6                 | Triton \*            |
| Instinct MI300X         | gfx942  | Linux        | ROCm 7.2, PyTorch 2.10 (ROCm 7.0), Triton 3.6, diffusers 0.41 | Triton               |

\* The W7900 validated the first Triton backend (FLUX.1-schnell end to end, kernel tests); the native int4 kernels
and the later models were validated on the RX 7800 XT (the same RDNA3 instruction set) and MI300X.

Other GPUs that ROCm PyTorch and Triton support should work on the Triton kernels; the native int4 kernels are used
on RDNA3 / RDNA3.5 (gfx11) only, and everything else falls back to Triton automatically. On NVIDIA GPUs use
upstream Nunchaku.

## Supported models

All text-to-image, with the INT4 checkpoints (no AMD GPU has FP4 matrix units; NVFP4 checkpoints load but run an
emulated path):

| Model                       | Class                                                              | INT4 checkpoints                                               | diffusers                  | ComfyUI                        |
| --------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- | -------------------------- | ------------------------------ |
| FLUX.1-schnell              | `NunchakuFluxTransformer2DModelV2`                                 | `nunchaku-ai/nunchaku-flux.1-schnell`                          | yes                        | yes                            |
| FLUX.1-dev / Krea / Kontext | same                                                               | `nunchaku-ai/nunchaku-flux.1-*`                                | same code path, not tested | same                           |
| FLUX.2 [klein] 4B / 9B      | `NunchakuFlux2Transformer2DModel`                                  | `tonera/FLUX.2-klein-4B-Nunchaku`, `...-9B-Nunchaku`           | yes, + LoRA                | yes, + LoRA, reference editing |
| Qwen-Image (+ Lightning)    | `NunchakuQwenImageTransformer2DModel`                              | `nunchaku-ai/nunchaku-qwen-image`                              | yes, per-block CPU offload | yes, per-block CPU offload     |
| Z-Image-Turbo               | `NunchakuZImageTransformer2DModel`                                 | `nunchaku-ai/nunchaku-z-image-turbo`                           | yes                        | yes                            |
| SDXL, SDXL-Turbo            | `nunchaku.models.unets.unet_sdxl.NunchakuSDXLUNet2DConditionModel` | `nunchaku-ai/nunchaku-sdxl`, `nunchaku-ai/nunchaku-sdxl-turbo` | yes                        | yes                            |

Not supported: Qwen-Image-Edit and other image-conditioned variants (untested; reference latents are rejected in
ComfyUI), ControlNet, LoRA on the non-FLUX models, and everything upstream ran only through its CUDA extension
(SANA, TeaCache, PuLID, IP-Adapter, the 4-bit T5 encoder -- see [`NOTICE`](NOTICE)).

## Install

Install a ROCm build of PyTorch **first**, then this package (it depends on `torchvision`, which would otherwise pull
in a CUDA build of PyTorch).

**Linux** (ROCm installed, Python 3.10+):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0   # brings ROCm Triton along
git clone https://github.com/TwelveNights/nunchaku-rocm.git
pip install -e ./nunchaku-rocm
```

**Windows 11** (Radeon RX 7000 / PRO W7000 series, a current AMD driver; no ROCm or Visual Studio install needed --
the ROCm runtime comes as Python wheels). Python 3.12, in PowerShell:

```powershell
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
$R = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
pip install "$R/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" "$R/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
    "$R/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$R/rocm-7.2.1.tar.gz"
pip install "$R/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$R/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
pip install "triton-windows==3.6.0.post26"
git clone https://github.com/TwelveNights/nunchaku-rocm.git
pip install -e .\nunchaku-rocm
# for Triton's compiler: put the ROCm SDK on PATH (in every new shell)
$env:ROCM_HOME = (python -m rocm_sdk path --root); $env:HIP_PATH = $env:ROCM_HOME
$env:PATH = "$env:ROCM_HOME\bin;$env:ROCM_HOME\lib\llvm\bin;$env:PATH"
```

A "Failed to find MSVC" message from Triton is harmless. Check the install with
`python -c "import torch, nunchaku; print(torch.cuda.get_device_name(0))"`. The backend is selected automatically
(`NUNCHAKU_BACKEND=triton` forces it). The first run of a model compiles and autotunes the Triton kernels and builds
the native kernels (about a minute); both are cached on disk (`TRITON_CACHE_DIR`, `NUNCHAKU_CACHE_DIR`, see
[`nunchaku/ops/triton/README.md`](nunchaku/ops/triton/README.md)).

## Quick start: diffusers

```python
import torch
from diffusers import Flux2KleinPipeline
from nunchaku import NunchakuFlux2Transformer2DModel

transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
    "tonera/FLUX.2-klein-4B-Nunchaku/svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors",
    torch_dtype=torch.bfloat16,
)
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # fits 16 GB cards; pipe.to("cuda") with more VRAM
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=4,
    height=1024,
    width=1024,
).images[0]
image.save("klein-int4.png")
```

The other models work the same way with their pipelines (`QwenImagePipeline`, `ZImagePipeline`,
`StableDiffusionXLPipeline`); see [`examples/v1/`](examples/v1). On 16 GB cards Z-Image needs
`enable_model_cpu_offload()` and Qwen-Image the transformer's per-block offload
(`transformer.set_offload(True, num_blocks_on_gpu=1)` plus `enable_sequential_cpu_offload()` for the rest, as in
`examples/v1/qwen-image-lightning.py`).

## Quick start: ComfyUI

Install nunchaku-rocm into ComfyUI's Python environment, point ComfyUI at `comfyui/custom_nodes` and load a workflow
from [`comfyui/workflows/`](comfyui/workflows). One node, **Nunchaku DiT Loader (ROCm)**, loads every model above
(ComfyUI's own loaders handle the text encoders and VAEs), and **Nunchaku LoRA Loader (ROCm)** adds FLUX LoRAs. Setup
for Windows and Linux, the model files, VRAM guidance for 16 GB cards and validation:
**[`comfyui/README.md`](comfyui/README.md)**.

## Validated results

Accuracy is measured against the bf16 models on identical inputs; 4-bit images keep the composition and text of the
bf16 ones but are not pixel-identical (typical image PSNR 13-25 dB, depending on the model).

**diffusers, RX 7800 XT** (Windows 11, native gfx11 kernels, steady state, 1024², int4 r32):

| Model                | Steps     | Setup on 16 GB                                    | Time / image | Peak VRAM |
| -------------------- | --------- | ------------------------------------------------- | ------------ | --------- |
| FLUX.1-schnell       | 4         | model CPU offload                                 | 8.8 s        | 9.2 GiB   |
| FLUX.2 [klein] 4B    | 4         | model CPU offload                                 | 5.7 s        | 7.9 GiB   |
| FLUX.2 [klein] 9B    | 4         | prompt encoded on CPU, transformer + VAE resident | 6.7 s        | 11.3 GiB  |
| Z-Image-Turbo        | 8         | model CPU offload                                 | 10.8 s       | 7.9 GiB   |
| Qwen-Image Lightning | 8         | per-block offload (1 block on GPU)                | 15.8 s       | 9.2 GiB   |
| Qwen-Image           | 20, CFG 4 | per-block offload                                 | 60.4 s       | 9.6 GiB   |
| SDXL                 | 30, CFG 5 | resident                                          | 11.6 s       | 10.1 GiB  |
| SDXL-Turbo (512²)    | 4         | resident                                          | 0.43 s       | 5.7 GiB   |

Baselines: the same FLUX runs on the Triton kernels alone take 12.2 s (FLUX.1-schnell), 7.4 s (Klein 4B) and 9.3 s
(Klein 9B); the native int4 kernels and fused epilogues account for the difference. There is no bf16 comparison on
this card: the bf16 transformers (8-40 GB) do not fit in 16 GB without heavy offloading, which was not benchmarked.

**ComfyUI, RX 7800 XT** (same card, warm runs, `--highvram`; details in [`comfyui/README.md`](comfyui/README.md)):

| Workflow                      | Time / image (warm) | Peak GPU memory |
| ----------------------------- | ------------------- | --------------- |
| FLUX.2 [klein] 4B, 4 steps    | 2.9 s               | 4.3 GiB         |
| FLUX.2 [klein] 9B, 4 steps    | 7.9-8.4 s           |                 |
| FLUX.1-schnell, 4 steps       | 5.5-5.7 s           | 7.7 GiB         |
| Z-Image-Turbo, 8 steps        | 8.4 s               | 5.3 GiB         |
| Qwen-Image Lightning, 8 steps | 11.6 s              | 10.1 GiB        |
| Qwen-Image, 20 steps, cfg 2.5 | 55.0 s              | 9.4 GiB         |
| SDXL, 30 steps, cfg 5         | 10.6 s              | 4.3 GiB         |
| SDXL-Turbo, 512², 4 steps     | 0.53 s              | 4.6 GiB         |

(Peak while sampling with the prompt already encoded; the first image of a session also loads the models and peaks
at up to 12.9 GiB with the text encoder on the GPU. Qwen-Image runs with the loader's per-block CPU offload.)

**Accuracy** (MI300X; image PSNR of the int4 model vs the bf16 model from the same noise, in diffusers and through
ComfyUI's own bf16 models):

| Model                       | diffusers, int4 vs bf16 | ComfyUI, int4 vs ComfyUI bf16 |
| --------------------------- | ----------------------- | ----------------------------- |
| FLUX.1-schnell (4 steps)    | 18.1 dB                 | 16.4 dB                       |
| FLUX.2 [klein] 4B (4 steps) | 15.7 dB                 | 15.7 dB                       |
| Z-Image-Turbo (8 steps)     | 13.5 / 12.6 dB          | 12.8 dB                       |
| Qwen-Image (20 steps, CFG)  | 17.1 / 16.8 dB          | 21.9 dB                       |
| SDXL (20-30 steps, CFG 5)   | 24.6 / 17.1-20.6 dB     | 23.9 dB                       |
| SDXL-Turbo (4 steps)        | 18.7 / 21.0 dB          | 14.6 / 16.7 dB \*             |

\* The ComfyUI SDXL-Turbo workflow (ancestral sampler, which amplifies differences); the other ComfyUI numbers are
from `comfyui/scripts/check_against_bf16.py`. The RX 7800 XT's ComfyUI images are as close to the MI300X bf16 images
as the MI300X int4 ones ([`comfyui/README.md`](comfyui/README.md#validation)).

Z-Image's error is the checkpoint's, not the kernels': its int4 GEMMs match a PyTorch fake-quant emulation of the
same checkpoint to 0.2-0.35%. The ComfyUI adapters add no error of their own: the int4 model's error against
ComfyUI's bf16 model equals its error against the diffusers bf16 model (up to the two bf16 implementations' own
difference), and the diffusers bf16 model run through the same adapter matches ComfyUI's bf16 model to 30-35 dB. Kernel-level details, per-kernel benchmarks and the native
GEMM design: [`nunchaku/ops/triton/README.md`](nunchaku/ops/triton/README.md).

**Tests:** `pytest tests/triton` (kernel and model-level checks against references; 320 passed / 1 skipped on the
RX 7800 XT, 193 passed on MI300X with the gfx11-only cases skipped) and `COMFYUI_PATH=<ComfyUI> pytest tests/comfyui`
(ComfyUI adapter conventions and offload on small random models, no checkpoints needed; 4 passed on both).

## Limitations

- **Windows VRAM:** the driver does not report out-of-memory past 16 GB; it silently spills into shared system memory,
  which is several times slower. Keep peak usage under the card's VRAM (the setups above do).
- **First run** of each model and resolution compiles and autotunes kernels (tens of seconds to a minute; cached
  afterwards).
- Only the model paths listed above are tested. Attention masks (e.g. batched prompts of different lengths) fall back
  to PyTorch SDPA, which is slow on Windows ROCm; Qwen-Image ignores its text mask, as upstream does.
- Removed with the CUDA extension: SANA, TeaCache, PuLID, IP-Adapter, the 4-bit T5 encoder, upstream's wheels and
  build system. [`NOTICE`](NOTICE) has the full list.
- No FP4 hardware on AMD GPUs: use INT4 checkpoints.

## Related projects

Other, independent efforts to run Nunchaku models on AMD GPUs:

- [ivandobskygithub/nunchaku_rocm_poc](https://github.com/ivandobskygithub/nunchaku_rocm_poc): a proof of concept of
  HIP kernels for Nunchaku on ROCm (October 2025).
- [BoomerCyb/Nunchaku-AMD](https://github.com/BoomerCyb/Nunchaku-AMD): ComfyUI loaders with Triton kernels for the
  Radeon RX 9070 XT (gfx1201) on Windows (October 2026).

nunchaku-rocm targets RDNA3 (gfx11) and CDNA3 (MI300X) GPUs on Linux and Windows, adds a native RDNA3 int4 WMMA GEMM
compiled at runtime (no compiler installation needed, also on Windows), and supports both diffusers and ComfyUI for
FLUX.1, FLUX.2 [klein], Qwen-Image, Z-Image and SDXL.

## Credits and license

nunchaku-rocm is a derivative of [Nunchaku](https://github.com/nunchaku-ai/nunchaku) by MIT HAN Lab and contributors
(SVDQuant: Li\*, Lin\*, Zhang\* et al., ICLR 2025), licensed under the Apache License 2.0 ([`LICENCE.txt`](LICENCE.txt));
[`NOTICE`](NOTICE) lists the modifications. FLUX.2 support is adapted from the community
[vitoom-nunchaku](https://huggingface.co/tonera/vitoom-nunchaku) fork (Apache 2.0). The quantized checkpoints are
published by their authors under the base models' licenses. The ComfyUI nodes run inside
[ComfyUI](https://github.com/comfyanonymous/ComfyUI), which is not included. The upstream README is kept as
[`README_upstream.md`](README_upstream.md).

If you use SVDQuant, please cite the paper:

```bibtex
@inproceedings{li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
