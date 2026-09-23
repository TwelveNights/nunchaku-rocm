import torch
from diffusers import FluxPipeline
from peft.tuners import lora
from vars import LORA_PATHS

from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def get_pipeline(
    model_name: str,
    precision: str,
    use_qencoder: bool = False,
    lora_name: str = "None",
    lora_weight: float = 1,
    device: str | torch.device = "cuda",
    pipeline_init_kwargs: dict = {},
) -> FluxPipeline:
    # NOTE (triton-only fork): the legacy quantized "schnell"/"dev" branches
    # (NunchakuFluxTransformer2dModel, CUDA-only) were removed. "schnell" now
    # routes to NunchakuFluxTransformer2DModelV2 (backend-dispatched
    # CUDA-or-Triton) -- it's the only quantized schnell path left, so the
    # CLI/gradio app's existing "schnell" argument/branches were kept as-is
    # rather than renamed. A bf16 "dev" fallback remains since it never
    # depended on the legacy class; there is no quantized "dev" path here.
    if model_name in ("schnell", "schnell_v2"):
        transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors"
        )
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            **pipeline_init_kwargs,
        )
    elif model_name == "dev":
        assert precision == "bf16", "quantized 'dev' is not implemented in this fork; see NOTE above"
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
        if lora_name == "All":
            # Pre-load all the LoRA weights for demo use
            for name, path in LORA_PATHS.items():
                pipeline.load_lora_weights(path["name_or_path"], weight_name=path["weight_name"], adapter_name=name)
            for m in pipeline.transformer.modules():
                if isinstance(m, lora.LoraLayer):
                    m.set_adapter(m.scaling.keys())
                    for name in m.scaling.keys():
                        m.scaling[name] = 0
        elif lora_name != "None":
            path = LORA_PATHS[lora_name]
            pipeline.load_lora_weights(path["name_or_path"], weight_name=path["weight_name"], adapter_name=lora_name)
            for m in pipeline.transformer.modules():
                if isinstance(m, lora.LoraLayer):
                    for name in m.scaling.keys():
                        m.scaling[name] = lora_weight
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    if precision == "bf16":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline
