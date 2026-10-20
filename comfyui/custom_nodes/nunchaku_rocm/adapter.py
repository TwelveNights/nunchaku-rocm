"""
Run a Nunchaku SVDQuant transformer as ComfyUI's FLUX.1 / FLUX.2 diffusion model.

ComfyUI's ``comfy.ldm.flux.model.Flux._forward`` patchifies the latent, appends
reference latents (Kontext / FLUX.2 editing), builds the position ids, and then
calls ``forward_orig(img_tokens, img_ids, txt, txt_ids, timesteps, y, guidance)``
-- the same inputs a diffusers ``FluxTransformer2DModel`` /
``Flux2Transformer2DModel`` takes. :class:`NunchakuFluxAdapter` inherits
everything up to that point and only replaces ``forward_orig`` with a call into
the Nunchaku model, so latent packing, reference handling and RoPE positions are
exactly ComfyUI's.
"""

import logging

import torch
from comfy.ldm.flux.model import Flux, FluxParams
from torch import nn

logger = logging.getLogger(__name__)

_PARAM_FIELDS = set(FluxParams.__dataclass_fields__)


class NunchakuFluxAdapter(Flux):
    """
    ComfyUI ``diffusion_model`` wrapping a Nunchaku FLUX.1 or FLUX.2 transformer.

    Parameters
    ----------
    transformer : NunchakuFluxTransformer2DModelV2 or NunchakuFlux2Transformer2DModel
        The quantized model (diffusers-style forward).
    unet_config : dict
        ComfyUI's unet config for the model (see ``nodes.comfy_unet_config``).
    dtype : torch.dtype
        Compute dtype of the transformer.
    """

    def __init__(self, transformer: nn.Module, unet_config: dict, dtype: torch.dtype):
        # Flux.__init__ would allocate the full-precision layers; only the attributes
        # that _forward / process_img read are needed
        nn.Module.__init__(self)
        self.params = FluxParams(**{k: v for k, v in unet_config.items() if k in _PARAM_FIELDS})
        self.patch_size = self.params.patch_size
        self.in_channels = self.params.in_channels * self.patch_size**2
        self.out_channels = self.params.out_channels * self.patch_size**2
        self.hidden_size = self.params.hidden_size
        self.num_heads = self.params.num_heads
        self.dtype = dtype
        self.is_flux2 = unet_config["image_model"] == "flux2"
        self.transformer = transformer
        self._warned = set()
        # LoRAs are requested per sampling call through transformer_options (see nodes.NunchakuLoraLoader), so
        # MODEL branches with and without LoRAs share this transformer; applied lazily when the request changes
        self._applied_loras: tuple = ()
        self._lora_cache: dict[str, dict] = {}

    def _apply_loras(self, requested: tuple) -> None:
        import comfy.utils

        from nunchaku.lora.runtime import load_lora, lora_model_info, set_loras

        info = lora_model_info(self.transformer)
        loras = []
        for path, strength in requested:
            if path not in self._lora_cache:
                self._lora_cache[path] = load_lora(comfy.utils.load_torch_file(path, safe_load=True), info)
            loras.append((self._lora_cache[path], strength))
        set_loras(self.transformer, loras)
        self._applied_loras = requested
        # drop cached LoRAs no longer requested
        keep = {p for p, _ in requested}
        self._lora_cache = {p: w for p, w in self._lora_cache.items() if p in keep}

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
        timestep_zero_index=None,
        transformer_options={},
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if control is not None:
            raise NotImplementedError("Nunchaku models do not support ControlNet in ComfyUI yet")
        if timestep_zero_index is not None:
            raise NotImplementedError(
                "reference_latents_method 'index_timestep_zero' (per-token timesteps) is not supported by Nunchaku models"
            )
        if attn_mask is not None:
            raise NotImplementedError("attention masks are not supported by Nunchaku models")
        if transformer_options.get("patches") or transformer_options.get("patches_replace", {}).get("dit"):
            self._warn_once(
                "patches", "Nunchaku: transformer patches (e.g. from model-patching nodes) are ignored by this model"
            )

        requested = tuple(transformer_options.get("nunchaku_loras", ()))
        if requested != self._applied_loras:
            self._apply_loras(requested)

        kwargs = dict(
            hidden_states=img,
            encoder_hidden_states=txt,
            timestep=timesteps,
            # every batch entry shares the same positions
            img_ids=img_ids[0],
            txt_ids=txt_ids[0],
            guidance=guidance if self.params.guidance_embed else None,
            return_dict=False,
        )
        if not self.is_flux2:
            if y is None:
                y = torch.zeros(img.shape[0], self.params.vec_in_dim, device=img.device, dtype=img.dtype)
            kwargs["pooled_projections"] = y[:, : self.params.vec_in_dim]
        return self.transformer(**kwargs)[0]
