"""
Run a Nunchaku SVDQuant Z-Image transformer as ComfyUI's Z-Image (Lumina 2 ``NextDiT``) diffusion model.

ComfyUI's ``NextDiT`` and diffusers' ``ZImageTransformer2DModel`` implement the same network: both pad the caption
and the image tokens to multiples of 32 with learned pad tokens, number caption positions from 1 and put the image
on the next temporal position, and run the context / noise refiners and the main layers over ``[caption, image]``
(diffusers orders the joint sequence ``[image, caption]``, which attention does not see). They differ in the call
convention: ComfyUI passes a latent batch, ``timesteps`` = sigma and expects ``-velocity``; diffusers takes lists
of per-image ``(C, F, H, W)`` latents and captions and ``t = 1 - sigma``. :class:`NunchakuZImageAdapter` converts
between the two and runs the Nunchaku model (``NunchakuZImageTransformer2DModel``: fused QKV + RMSNorm + RoPE and
fused SwiGLU GEMMs, Triton flash attention).
"""

import logging

import comfy.ldm.common_dit
import torch
from comfy.ldm.lumina.model import NextDiT
from torch import nn

logger = logging.getLogger(__name__)


class NunchakuZImageAdapter(NextDiT):
    """
    ComfyUI ``diffusion_model`` wrapping a Nunchaku Z-Image transformer.

    Parameters
    ----------
    transformer : NunchakuZImageTransformer2DModel
        The quantized model.
    config : dict
        Its diffusers config (from the checkpoint metadata).
    dtype : torch.dtype
        Compute dtype of the transformer.
    """

    def __init__(self, transformer: nn.Module, config: dict, dtype: torch.dtype):
        # NextDiT.__init__ would allocate the full-precision model
        nn.Module.__init__(self)
        self.dtype = dtype
        self.patch_size = 2
        self.in_channels = config.get("in_channels", 16)
        self.out_channels = self.in_channels
        self.transformer = transformer
        self._warned = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def _forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        ref_latents=[],
        ref_contexts=[],
        siglip_feats=[],
        direct_context=None,
        ref_frames=[],
        transformer_options={},
        control=None,
        **kwargs,
    ):
        if ref_latents or ref_contexts or siglip_feats or ref_frames or direct_context is not None:
            raise NotImplementedError("Z-Image omni / reference inputs are not supported by Nunchaku models")
        if control is not None:
            raise NotImplementedError("Nunchaku Z-Image models do not support ControlNet in ComfyUI yet")
        if x.ndim != 4:
            raise NotImplementedError("Nunchaku Z-Image: only image latents (B, C, H, W) are supported")
        if transformer_options.get("patches") or transformer_options.get("patches_replace", {}).get("dit"):
            self._warn_once(
                "patches", "Nunchaku: transformer patches (e.g. from model-patching nodes) are ignored by this model"
            )

        h, w = x.shape[-2:]
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        # ComfyUI's NextDiT runs the full (possibly padded) context and ignores num_tokens / attention_mask; so do we
        out = self.transformer(
            [xi.unsqueeze(1) for xi in x.to(self.dtype)],
            1.0 - timesteps.float(),
            list(context.to(self.dtype).unbind(0)),
            return_dict=False,
        )[0]
        out = torch.stack([o.squeeze(1) for o in out])[..., :h, :w]
        return -out
