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

logger = logging.getLogger(__name__)


def comfy_unet_config(config: dict) -> tuple[dict, type]:
    """
    ComfyUI unet config and model-config class for a Nunchaku checkpoint, from the
    diffusers transformer config stored in its safetensors metadata. Mirrors what
    ``comfy.model_detection`` derives from a full-precision ComfyUI checkpoint.
    """
    name = config["_class_name"]
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
    raise ValueError(f"Nunchaku DiT Loader: unsupported model class {name!r} (FLUX.1 and FLUX.2 are supported)")


def load_nunchaku_transformer(path: str, config: dict, dtype: torch.dtype):
    from nunchaku import NunchakuFlux2Transformer2DModel, NunchakuFluxTransformer2DModelV2

    cls = NunchakuFlux2Transformer2DModel if config["_class_name"] == "Flux2Transformer2DModel" else (
        NunchakuFluxTransformer2DModelV2
    )
    return cls.from_pretrained(path, torch_dtype=dtype)


def load_model(path: str, dtype: torch.dtype = torch.bfloat16) -> comfy.model_patcher.ModelPatcher:
    """Load a Nunchaku safetensors checkpoint as a ComfyUI ``MODEL``."""
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
    model.diffusion_model = NunchakuFluxAdapter(transformer, model_config.unet_config, dtype)
    return comfy.model_patcher.ModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )


class NunchakuDiTLoader:
    """Load a Nunchaku SVDQuant FLUX.1 / FLUX.2 [klein] diffusion model (``svdq-int4_r*-*.safetensors``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "A Nunchaku SVDQuant checkpoint (INT4), e.g. svdq-int4_r32-FLUX.2-klein-4B-Nunchaku"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Nunchaku (ROCm)"
    DESCRIPTION = (
        "Loads a Nunchaku 4-bit SVDQuant FLUX.1 or FLUX.2 [klein] model for AMD GPUs (Triton backend, native "
        "int4 kernels on RDNA3). Use ComfyUI's own CLIP and VAE loaders for the text encoder and VAE."
    )

    def load(self, model_name: str):
        return (load_model(folder_paths.get_full_path_or_raise("diffusion_models", model_name)),)


NODE_CLASS_MAPPINGS = {"NunchakuRocmDiTLoader": NunchakuDiTLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"NunchakuRocmDiTLoader": "Nunchaku DiT Loader (ROCm)"}
