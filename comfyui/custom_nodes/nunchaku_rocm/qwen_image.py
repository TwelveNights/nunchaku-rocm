"""
Run a Nunchaku SVDQuant Qwen-Image transformer as ComfyUI's Qwen-Image diffusion model.

:class:`NunchakuQwenImageAdapter` inherits ComfyUI's ``comfy.ldm.qwen_image.model.QwenImageTransformer2DModel``
for the parts around the transformer blocks -- latent patchifying, the text / image position ids and their RoPE
(``EmbedND``), unpatchifying -- and runs the embeddings, the 60 quantized blocks and the output layers of the
Nunchaku model (``NunchakuQwenImageTransformer2DModel``, whose attention runs on the Triton flash-attention kernel).
The rotary embedding ComfyUI builds is converted to the complex ``(cos + i sin)`` form the Nunchaku blocks take, so
positions are exactly ComfyUI's.

Per-block CPU offload (for 16 GB cards): all blocks but the first ``num_blocks_on_gpu`` stay in (pinned) CPU memory
and are streamed through two GPU buffer blocks on a side stream while the previous block computes. The offloaded
blocks are kept out of the module tree, so ComfyUI's model management only sees -- and moves, and counts -- the
resident part (embeddings, output layers, resident blocks and the two buffers).
"""

import copy
import logging

import torch
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
from torch import nn

logger = logging.getLogger(__name__)


class NunchakuQwenImageAdapter(QwenImageTransformer2DModel):
    """
    ComfyUI ``diffusion_model`` wrapping a Nunchaku Qwen-Image transformer.

    Parameters
    ----------
    transformer : NunchakuQwenImageTransformer2DModel
        The quantized model, loaded without offload.
    config : dict
        Its diffusers config (from the checkpoint metadata).
    dtype : torch.dtype
        Compute dtype of the transformer.
    offload : bool
        Stream the transformer blocks from CPU memory (see the module docstring).
    num_blocks_on_gpu : int
        With ``offload``, how many leading blocks stay resident on the GPU.
    use_pin_memory : bool
        With ``offload``, pin the offloaded blocks' CPU memory (faster, asynchronous uploads).
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: dict,
        dtype: torch.dtype,
        offload: bool = False,
        num_blocks_on_gpu: int = 1,
        use_pin_memory: bool = True,
    ):
        # QwenImageTransformer2DModel.__init__ would allocate the full-precision model
        nn.Module.__init__(self)
        self.dtype = dtype
        self.patch_size = config.get("patch_size", 2)
        self.in_channels = config.get("in_channels", 64)
        self.out_channels = config.get("out_channels") or 16
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.default_ref_method = "index"
        self.pe_embedder = EmbedND(
            dim=config["attention_head_dim"], theta=10000, axes_dim=list(config.get("axes_dims_rope", (16, 56, 56)))
        )
        self.transformer = transformer
        self._warned = set()

        blocks = list(transformer.transformer_blocks)
        self.offload = offload and num_blocks_on_gpu < len(blocks)
        if not self.offload:
            return
        num_blocks_on_gpu = max(1, num_blocks_on_gpu)
        # resident blocks stay registered (under transformer.transformer_blocks); the rest live in a plain list,
        # invisible to nn.Module.to() and to ComfyUI's model patcher
        transformer.transformer_blocks = nn.ModuleList(blocks[:num_blocks_on_gpu])
        self.offload_buffers = nn.ModuleList([copy.deepcopy(blocks[0]) for _ in range(2)])
        offloaded = blocks[num_blocks_on_gpu:]
        if use_pin_memory:
            with torch.no_grad():
                for block in offloaded:
                    for t in list(block.parameters()) + list(block.buffers()):
                        t.data = t.data.pin_memory()
        self.__dict__["_offloaded_blocks"] = offloaded
        self.__dict__["_memory_stream"] = None

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def _blocks(self):
        resident = list(self.transformer.transformer_blocks)
        return resident + (self.__dict__["_offloaded_blocks"] if self.offload else [])

    def _run_blocks(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        from nunchaku.utils import copy_params_into

        blocks = self._blocks()
        num_resident = len(self.transformer.transformer_blocks)
        compute = torch.cuda.current_stream(hidden_states.device)
        if self.offload:
            memory = self.__dict__["_memory_stream"]
            if memory is None or memory.device != hidden_states.device:
                memory = torch.cuda.Stream(device=hidden_states.device)
                self.__dict__["_memory_stream"] = memory
            # the previous call's last blocks may still be computing from the buffers
            memory.wait_stream(compute)
            ready = [None, None]  # buffer i holds its block once this event (memory stream) has fired
            free = [None, None]  # buffer i may be overwritten once this event (compute stream) has fired

            def prefetch(j: int) -> None:
                slot = j % 2
                with torch.cuda.stream(memory):
                    if free[slot] is not None:
                        memory.wait_event(free[slot])
                    copy_params_into(blocks[j], self.offload_buffers[slot], non_blocking=True)
                    ready[slot] = torch.cuda.Event()
                    ready[slot].record(memory)

            if num_resident < len(blocks):
                prefetch(num_resident)

        for i, block in enumerate(blocks):
            if i >= num_resident:
                if i + 1 < len(blocks):
                    prefetch(i + 1)
                compute.wait_event(ready[i % 2])
                block = self.offload_buffers[i % 2]
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            if i >= num_resident:
                free[i % 2] = torch.cuda.Event()
                free[i % 2].record(compute)
        return hidden_states

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        ref_latents=None,
        additional_t_cond=None,
        transformer_options={},
        control=None,
        **kwargs,
    ):
        if control is not None:
            raise NotImplementedError("Nunchaku Qwen-Image models do not support ControlNet in ComfyUI yet")
        if ref_latents is not None:
            raise NotImplementedError("reference latents (Qwen-Image-Edit) are not supported by Nunchaku models yet")
        if additional_t_cond is not None:
            raise NotImplementedError("additional timestep conditioning is not supported by this Nunchaku model")
        if transformer_options.get("patches") or transformer_options.get("patches_replace", {}).get("dit"):
            self._warn_once(
                "patches", "Nunchaku: transformer patches (e.g. from model-patching nodes) are ignored by this model"
            )
        if attention_mask is not None:
            masked = (
                (attention_mask < 0).any() if torch.is_floating_point(attention_mask) else (attention_mask == 0).any()
            )
            if masked:
                # as in the Nunchaku diffusers model: text padding is attended to
                self._warn_once("mask", "Nunchaku Qwen-Image: the text attention mask is ignored")

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]
        txt_start = round(
            max(
                ((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2,
            )
        )
        txt_len = context.shape[1]
        txt_ids = torch.arange(txt_start, txt_start + txt_len, device=x.device).reshape(1, -1, 1).repeat(1, 1, 3)
        # every batch entry shares the same positions; EmbedND -> (1, 1, S, D/2, 2, 2) float32 rotation matrices
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids[:1]), dim=1))[0, 0]
        freqs = torch.complex(pe[..., 0, 0], pe[..., 1, 0])  # cos + i sin, (S, D/2)
        image_rotary_emb = (freqs[txt_len:], freqs[:txt_len])  # (image, text) as the Nunchaku blocks take them

        t = self.transformer
        hidden_states = t.img_in(hidden_states)
        encoder_hidden_states = t.txt_in(t.txt_norm(context))
        temb = t.time_text_embed(timesteps.to(hidden_states.dtype), hidden_states)

        hidden_states = self._run_blocks(hidden_states, encoder_hidden_states, temb, image_rotary_emb)

        hidden_states = t.proj_out(t.norm_out(hidden_states, temb))
        hidden_states = hidden_states[:, :num_embeds].view(
            orig_shape[0], orig_shape[-3], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2
        )
        hidden_states = hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
        return hidden_states.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]
