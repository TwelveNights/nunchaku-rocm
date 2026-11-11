"""
ComfyUI nodes for Nunchaku SVDQuant models on the Triton / native-gfx11 backend.
"""

import json
import logging
import os

import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths
import torch
from safetensors import safe_open

from .adapter import NunchakuFluxAdapter
from .lora import LoraMixin

logger = logging.getLogger(__name__)

# Qwen-Image's per-block CPU offload is switched on automatically on GPUs with less memory than this
QWEN_IMAGE_AUTO_OFFLOAD_GIB = 20


class NunchakuModelPatcher(comfy.model_patcher.ModelPatcher):
    """
    ComfyUI's lowvram mode leaves the layers that do not fit in CPU memory and casts their weights to the GPU
    per call; the Nunchaku layers cannot run that way, so the (resident part of the) model is always loaded whole.
    """

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        return super().load(device_to, lowvram_model_memory, force_patch_weights, full_load=True)


def comfy_unet_config(config: dict) -> tuple[dict, type]:
    """
    ComfyUI unet config and model-config class for a Nunchaku checkpoint, from the
    diffusers transformer config stored in its safetensors metadata. Mirrors what
    ``comfy.model_detection`` derives from a full-precision ComfyUI checkpoint.
    """
    name = config["_class_name"]
    if name == "QwenImageTransformer2DModel":
        return dict(image_model="qwen_image", disable_unet_model_creation=True), comfy.supported_models.QwenImage
    if name == "ZImageTransformer2DModel":
        # what comfy.model_detection derives from a ComfyUI Z-Image checkpoint
        unet_config = dict(
            image_model="lumina2",
            patch_size=2,
            in_channels=config.get("in_channels", 16),
            cap_feat_dim=config.get("cap_feat_dim", 2560),
            n_layers=config.get("n_layers", 30),
            qk_norm=config.get("qk_norm", True),
            dim=config.get("dim", 3840),
            n_heads=config.get("n_heads", 30),
            n_kv_heads=config.get("n_kv_heads", 30),
            axes_dims=list(config.get("axes_dims", [32, 48, 48])),
            axes_lens=list(config.get("axes_lens", [1536, 512, 512])),
            rope_theta=config.get("rope_theta", 256.0),
            ffn_dim_multiplier=8.0 / 3.0,
            z_image_modulation=True,
            time_scale=config.get("t_scale", 1000.0),
            pad_tokens_multiple=32,
            disable_unet_model_creation=True,
        )
        return unet_config, comfy.supported_models.ZImage
    if name == "UNet2DConditionModel":
        if config.get("cross_attention_dim") != 2048 or config.get("projection_class_embeddings_input_dim") != 2816:
            raise ValueError("Nunchaku DiT Loader: only SDXL UNets are supported")
        unet_config = dict(
            comfy.supported_models.SDXL.unet_config,
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 4),
            disable_unet_model_creation=True,
        )
        return unet_config, comfy.supported_models.SDXL
    heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
    common = dict(
        hidden_size=heads * head_dim,
        num_heads=heads,
        depth=config["num_layers"],
        depth_single_blocks=config["num_single_layers"],
        context_in_dim=config["joint_attention_dim"],
        guidance_embed=bool(config.get("guidance_embeds", False)),
        disable_unet_model_creation=True,
    )
    if name == "Flux2Transformer2DModel":
        patch = config.get("patch_size", 1)
        unet_config = dict(
            common,
            image_model="flux2",
            in_channels=config["in_channels"] // patch**2,
            out_channels=(config.get("out_channels") or config["in_channels"]) // patch**2,
            patch_size=patch,
            vec_in_dim=None,
            mlp_ratio=config["mlp_ratio"],
            axes_dim=list(config["axes_dims_rope"]),
            theta=config.get("rope_theta", 2000),
            qkv_bias=False,
            ops_bias=False,
            global_modulation=True,
            mlp_silu_act=True,
            default_ref_method="index",
            ref_index_scale=10.0,
            txt_ids_dims=[3],
        )
        return unet_config, comfy.supported_models.Flux2
    if name == "FluxTransformer2DModel":
        patch = 2  # diffusers FLUX.1 configs store the packed (2x2) channel count
        in_channels = config["in_channels"] // patch**2
        unet_config = dict(
            common,
            image_model="flux",
            in_channels=in_channels,
            out_channels=16,
            patch_size=patch,
            vec_in_dim=config.get("pooled_projection_dim", 768),
            mlp_ratio=4.0,
            axes_dim=list(config.get("axes_dims_rope", [16, 56, 56])),
            theta=10000,
            qkv_bias=True,
            txt_ids_dims=[],
        )
        if in_channels == 96:  # FLUX.1 Fill
            cls = comfy.supported_models.FluxInpaint
        elif not unet_config["guidance_embed"]:
            cls = comfy.supported_models.FluxSchnell
        else:
            cls = comfy.supported_models.Flux
        return unet_config, cls
    raise ValueError(
        f"Nunchaku DiT Loader: unsupported model class {name!r} "
        "(FLUX.1, FLUX.2 [klein], Qwen-Image, Z-Image and SDXL are supported)"
    )


def load_nunchaku_transformer(path: str, config: dict, dtype: torch.dtype):
    from nunchaku import (
        NunchakuFlux2Transformer2DModel,
        NunchakuFluxTransformer2DModelV2,
        NunchakuQwenImageTransformer2DModel,
        NunchakuZImageTransformer2DModel,
    )
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

    cls = {
        "Flux2Transformer2DModel": NunchakuFlux2Transformer2DModel,
        "FluxTransformer2DModel": NunchakuFluxTransformer2DModelV2,
        "QwenImageTransformer2DModel": NunchakuQwenImageTransformer2DModel,
        "ZImageTransformer2DModel": NunchakuZImageTransformer2DModel,
        "UNet2DConditionModel": NunchakuSDXLUNet2DConditionModel,
    }[config["_class_name"]]
    return cls.from_pretrained(path, torch_dtype=dtype)


def resolve_offload(cpu_offload: str, device: torch.device) -> bool:
    """``cpu_offload`` = "enable" / "disable" / "auto" (on when the GPU has < QWEN_IMAGE_AUTO_OFFLOAD_GIB)."""
    if cpu_offload == "auto":
        total = comfy.model_management.get_total_memory(device) / 1024**3
        logger.info(
            f"Nunchaku: {total:.1f} GiB GPU memory, per-block CPU offload {'on' if total < QWEN_IMAGE_AUTO_OFFLOAD_GIB else 'off'}"
        )
        return total < QWEN_IMAGE_AUTO_OFFLOAD_GIB
    if cpu_offload not in ("enable", "disable"):
        raise ValueError(f"cpu_offload must be auto, enable or disable, not {cpu_offload!r}")
    return cpu_offload == "enable"


def load_model(
    path: str, dtype: torch.dtype = torch.bfloat16, cpu_offload: str = "auto", num_blocks_on_gpu: int = 1
) -> comfy.model_patcher.ModelPatcher:
    """
    Load a Nunchaku safetensors checkpoint as a ComfyUI ``MODEL``. ``cpu_offload`` / ``num_blocks_on_gpu`` apply to
    Qwen-Image (per-block CPU offload, see qwen_image.py); the other models load whole.
    """
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    if "config" not in metadata or "quantization_config" not in metadata:
        raise ValueError(f"{os.path.basename(path)} is not a Nunchaku SVDQuant checkpoint (no config metadata)")
    config = json.loads(metadata["config"])
    unet_config, model_config_cls = comfy_unet_config(config)

    transformer = load_nunchaku_transformer(path, config, dtype)
    transformer.requires_grad_(False).eval()

    model_config = model_config_cls(unet_config)
    model_config.set_inference_dtype(dtype, None)
    model_config.custom_operations = None
    model = model_config.get_model({})
    load_device = comfy.model_management.get_torch_device()
    name = config["_class_name"]
    if name == "QwenImageTransformer2DModel":
        from .qwen_image import NunchakuQwenImageAdapter

        offload = resolve_offload(cpu_offload, load_device)
        model.diffusion_model = NunchakuQwenImageAdapter(
            transformer, config, dtype, offload=offload, num_blocks_on_gpu=num_blocks_on_gpu
        )
    elif name == "ZImageTransformer2DModel":
        from .z_image import NunchakuZImageAdapter

        model.diffusion_model = NunchakuZImageAdapter(transformer, config, dtype)
    elif name == "UNet2DConditionModel":
        from .sdxl import NunchakuSDXLAdapter

        model.diffusion_model = NunchakuSDXLAdapter(transformer, dtype)
    else:
        model.diffusion_model = NunchakuFluxAdapter(transformer, model_config.unet_config, dtype)
    return NunchakuModelPatcher(
        model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device()
    )


class NunchakuDiTLoader:
    """
    Load a Nunchaku SVDQuant diffusion model (``svdq-int4_r*-*.safetensors``): FLUX.1, FLUX.2 [klein], Qwen-Image,
    Z-Image or SDXL, detected from the checkpoint's metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "A Nunchaku SVDQuant checkpoint (INT4), e.g. svdq-int4_r32-FLUX.2-klein-4B-Nunchaku"},
                ),
            },
            "optional": {
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Qwen-Image only: keep the transformer blocks in CPU memory and stream them to the "
                        f"GPU one at a time. 'auto' enables it on GPUs with less than {QWEN_IMAGE_AUTO_OFFLOAD_GIB} GiB.",
                    },
                ),
                "num_blocks_on_gpu": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 60,
                        "tooltip": "Qwen-Image with CPU offload: how many transformer blocks stay on the GPU.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Nunchaku (ROCm)"
    DESCRIPTION = (
        "Loads a Nunchaku 4-bit SVDQuant FLUX.1, FLUX.2 [klein], Qwen-Image, Z-Image or SDXL model for AMD GPUs "
        "(Triton backend, native int4 kernels on RDNA3). Use ComfyUI's own CLIP and VAE loaders for the text "
        "encoders and VAE."
    )

    def load(self, model_name: str, cpu_offload: str = "auto", num_blocks_on_gpu: int = 1):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (load_model(path, cpu_offload=cpu_offload, num_blocks_on_gpu=num_blocks_on_gpu),)


class NunchakuLoraLoader:
    """
    Add a LoRA to a model from :class:`NunchakuDiTLoader` (chain several for multiple LoRAs). ComfyUI's own
    LoraLoader cannot patch the 4-bit layers; this one appends the LoRA to them at sampling time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A model from Nunchaku DiT Loader (ROCm)"}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Nunchaku (ROCm)"
    DESCRIPTION = "Applies a LoRA (ComfyUI, kohya or diffusers format) to a Nunchaku FLUX.1, FLUX.2 [klein], Z-Image, Qwen-Image or SDXL model."

    def load(self, model: comfy.model_patcher.ModelPatcher, lora_name: str, strength: float):
        if not isinstance(model.model.diffusion_model, LoraMixin):
            raise ValueError("Nunchaku LoRA Loader needs a model from Nunchaku DiT Loader (ROCm)")
        return (add_lora(model, folder_paths.get_full_path_or_raise("loras", lora_name), strength),)


def add_lora(model: comfy.model_patcher.ModelPatcher, path: str, strength: float) -> comfy.model_patcher.ModelPatcher:
    """A clone of a Nunchaku ``MODEL`` that also applies the LoRA at ``path`` (see adapter._apply_loras)."""
    out = model.clone()
    topts = dict(out.model_options.get("transformer_options", {}))
    topts["nunchaku_loras"] = [*topts.get("nunchaku_loras", []), (path, float(strength))]
    out.model_options = {**out.model_options, "transformer_options": topts}
    return out


NODE_CLASS_MAPPINGS = {"NunchakuRocmDiTLoader": NunchakuDiTLoader, "NunchakuRocmLoraLoader": NunchakuLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuRocmDiTLoader": "Nunchaku DiT Loader (ROCm)",
    "NunchakuRocmLoraLoader": "Nunchaku LoRA Loader (ROCm)",
}
