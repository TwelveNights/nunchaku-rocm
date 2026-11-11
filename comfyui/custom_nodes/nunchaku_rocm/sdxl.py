"""
Run a Nunchaku SVDQuant SDXL UNet as ComfyUI's SDXL diffusion model.

ComfyUI's ``UNetModel`` and diffusers' ``UNet2DConditionModel`` take the same inputs -- the noisy latent, the
discrete timestep, the ``[CLIP-L | CLIP-G]`` text states -- except for the added conditioning: ComfyUI's
``model_base.SDXL`` already concatenates the pooled CLIP-G embedding with the Fourier features of the size / crop
conditioning into ``y`` (2816 channels), which is exactly the vector diffusers' ``get_aug_embed`` builds before
``add_embedding``. :class:`NunchakuSDXLAdapter` feeds ``y`` straight to ``add_embedding`` and otherwise runs the
Nunchaku UNet (``NunchakuSDXLUNet2DConditionModel``: 4-bit transformer blocks, Triton flash attention).
"""

import logging
import types

import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from torch import nn

from .lora import LoraMixin

logger = logging.getLogger(__name__)


def _aug_embed_from_y(self, emb, encoder_hidden_states, added_cond_kwargs):
    # replaces UNet2DConditionModel.get_aug_embed: ComfyUI's y is [pooled text embeds | time-id Fourier features]
    return self.add_embedding(added_cond_kwargs["comfy_y"].to(emb.dtype))


class NunchakuSDXLAdapter(LoraMixin, UNetModel):
    """
    ComfyUI ``diffusion_model`` wrapping a Nunchaku SDXL UNet.

    Parameters
    ----------
    unet : NunchakuSDXLUNet2DConditionModel
        The quantized model.
    dtype : torch.dtype
        Compute dtype of the UNet.
    """

    def __init__(self, unet: nn.Module, dtype: torch.dtype):
        # UNetModel.__init__ would allocate the full-precision model
        nn.Module.__init__(self)
        self.dtype = dtype
        unet.get_aug_embed = types.MethodType(_aug_embed_from_y, unet)
        self.unet = unet
        self._warned = set()
        self._init_loras()

    def _lora_model(self) -> nn.Module:
        return self.unet

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def _forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        if control is not None:
            raise NotImplementedError("Nunchaku SDXL models do not support ControlNet in ComfyUI yet")
        if y is None:
            raise ValueError("Nunchaku SDXL: missing the SDXL added conditioning (y)")
        if transformer_options.get("patches") or transformer_options.get("patches_replace"):
            self._warn_once(
                "patches", "Nunchaku: UNet patches (e.g. from model-patching nodes) are ignored by this model"
            )
        self._sync_loras(transformer_options)
        return self.unet(
            x.to(self.dtype),
            timesteps,
            encoder_hidden_states=context.to(self.dtype),
            added_cond_kwargs={"comfy_y": y},
            return_dict=False,
        )[0]
