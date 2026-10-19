"""
This module provides Nunchaku Flux2Transformer2DModel (FLUX.2, e.g. FLUX.2 [klein] 4B/9B) and its building blocks in Python.

Ported from the community vitoom-nunchaku fork (https://huggingface.co/tonera/vitoom-nunchaku, Apache-2.0), which
loads the same ``svdq-<precision>_r32-FLUX.2-*.safetensors`` checkpoints. Its packed-QKV fast path depends on the
CUDA-only ``EpiloguePackQKV`` layout and is not carried over; attention goes through
:func:`~nunchaku.ops.dispatch.scaled_dot_product_attention` instead.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
)
from huggingface_hub import utils

from ...ops.backend import get_backend
from ...ops.dispatch import scaled_dot_product_attention
from ...ops.fused import fused_qkv_norm_rottary
from ...utils import get_precision_from_quantization_config, pad_tensor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """(B, T, heads * head_dim) Q/K/V -> (B, T, heads * head_dim) attention output."""
    batch_size = query.shape[0]
    query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    hidden_states = scaled_dot_product_attention(query, key, value)
    return hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)


def _pack_rotary_emb(freqs_cis: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Convert FLUX.2's ``(cos, sin)`` rotary embeddings to the layout the fused QKV RMSNorm+RoPE epilogue consumes.

    Parameters
    ----------
    freqs_cis : tuple of torch.Tensor
        ``(cos, sin)``, each of shape (S, D). FLUX.2 uses ``repeat_interleave_real=True``, so each adjacent channel
        pair shares one value.

    Returns
    -------
    torch.Tensor
        Natural layout (S, D // 2, 1, 2) float32, last dim (sin, cos), on the Triton backend;
        the CUDA kernel's packed, 256-padded layout otherwise.
    """
    cos, sin = freqs_cis
    rotemb = torch.stack([sin[:, 0::2], cos[:, 0::2]], dim=-1).unsqueeze(-2).float().contiguous()
    if get_backend() == "triton":
        return rotemb
    return pack_rotemb(pad_tensor(rotemb.unsqueeze(0), 256, 1))


class NunchakuFlux2Attention(Flux2Attention):
    """
    Nunchaku-optimized joint (double-stream) Flux2Attention with quantized, fused QKV projections.

    Parameters
    ----------
    other : Flux2Attention
        The original attention module to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: Flux2Attention, **kwargs):
        super(Flux2Attention, self).__init__()
        for attr in ("head_dim", "inner_dim", "query_dim", "out_dim", "heads", "use_bias", "dropout"):
            setattr(self, attr, getattr(other, attr))
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias
        assert self.added_kv_proj_dim is not None, "FLUX.2 double-stream attention always has added KV projections"

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.norm_added_q = other.norm_added_q
        self.norm_added_k = other.norm_added_k

        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
            to_added_qkv = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_added_qkv = SVDQW4A4Linear.from_linear(to_added_qkv, **kwargs)

        self.to_out = other.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)
        self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for joint image/text attention.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, T_img, C)
            Image stream.
        encoder_hidden_states : torch.Tensor, shape (B, T_txt, C)
            Text stream.
        attention_mask : torch.Tensor, optional
            Not supported. Must be None.
        image_rotary_emb : tuple of torch.Tensor
            (image, text) rotary embeddings from :func:`_pack_rotary_emb`.
        **kwargs
            Unused.

        Returns
        -------
        tuple of torch.Tensor
            (image output, text output).
        """
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")
        rotary_emb_img, rotary_emb_txt = image_rotary_emb

        qkv = fused_qkv_norm_rottary(hidden_states, self.to_qkv, self.norm_q, self.norm_k, rotary_emb_img)
        qkv_context = fused_qkv_norm_rottary(
            encoder_hidden_states, self.to_added_qkv, self.norm_added_q, self.norm_added_k, rotary_emb_txt
        )
        # joint sequence is [text, image], as in diffusers
        qkv = torch.cat([qkv_context, qkv], dim=1)
        hidden_states = _attention(*qkv.chunk(3, dim=-1), self.heads, self.head_dim)

        num_txt_tokens = encoder_hidden_states.shape[1]
        encoder_hidden_states, hidden_states = hidden_states[:, :num_txt_tokens], hidden_states[:, num_txt_tokens:]
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class NunchakuFlux2FeedForward(Flux2FeedForward):
    """
    Quantized FLUX.2 SwiGLU feed-forward.

    Parameters
    ----------
    other : Flux2FeedForward
        The original feed-forward module to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: Flux2FeedForward, **kwargs):
        super(Flux2FeedForward, self).__init__()
        self.linear_in = SVDQW4A4Linear.from_linear(other.linear_in, **kwargs)
        self.act_fn = other.act_fn
        # SwiGLU outputs are signed, and FLUX.2 checkpoints are calibrated without the
        # unsigned-activation shift FLUX.1 uses after GELU, so linear_out keeps the signed path.
        self.linear_out = SVDQW4A4Linear.from_linear(other.linear_out, **kwargs)


class NunchakuFlux2ParallelSelfAttention(Flux2ParallelSelfAttention):
    """
    Quantized FLUX.2 single-stream "parallel" attention + MLP.

    The diffusers module fuses QKV with the MLP input projection (``to_qkv_mlp_proj``) and the attention output with
    the MLP output projection (``to_out``). The checkpoints split them back apart (``qkv_proj``/``mlp_fc1`` and
    ``out_proj``/``mlp_fc2``) so the QKV part can use the fused RMSNorm+RoPE epilogue.

    Parameters
    ----------
    other : Flux2ParallelSelfAttention
        The original module to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: Flux2ParallelSelfAttention, **kwargs):
        super(Flux2ParallelSelfAttention, self).__init__()
        for attr in ("head_dim", "inner_dim", "query_dim", "out_dim", "heads", "use_bias", "dropout"):
            setattr(self, attr, getattr(other, attr))
        self.mlp_ratio = other.mlp_ratio
        self.mlp_hidden_dim = other.mlp_hidden_dim
        self.mlp_mult_factor = other.mlp_mult_factor
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.mlp_act_fn = other.mlp_act_fn

        out_bias = other.to_out.bias is not None
        with torch.device("meta"):
            qkv_proj = torch.nn.Linear(self.query_dim, self.inner_dim * 3, bias=self.use_bias)
            mlp_fc1 = torch.nn.Linear(self.query_dim, self.mlp_hidden_dim * self.mlp_mult_factor, bias=self.use_bias)
            out_proj = torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
            mlp_fc2 = torch.nn.Linear(self.mlp_hidden_dim, self.out_dim, bias=out_bias)
        self.qkv_proj = SVDQW4A4Linear.from_linear(qkv_proj, **kwargs)
        self.mlp_fc1 = SVDQW4A4Linear.from_linear(mlp_fc1, **kwargs)
        self.out_proj = SVDQW4A4Linear.from_linear(out_proj, **kwargs)
        # signed activations, as in NunchakuFlux2FeedForward
        self.mlp_fc2 = SVDQW4A4Linear.from_linear(mlp_fc2, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, T_txt + T_img, C)
            Concatenated [text, image] stream.
        attention_mask : torch.Tensor, optional
            Not supported. Must be None.
        image_rotary_emb : torch.Tensor
            Rotary embedding for the concatenated stream, from :func:`_pack_rotary_emb`.
        **kwargs
            Unused.

        Returns
        -------
        torch.Tensor, shape (B, T_txt + T_img, C)
        """
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")
        qkv = fused_qkv_norm_rottary(hidden_states, self.qkv_proj, self.norm_q, self.norm_k, image_rotary_emb)
        attn_output = _attention(*qkv.chunk(3, dim=-1), self.heads, self.head_dim)
        mlp_hidden_states = self.mlp_act_fn(self.mlp_fc1(hidden_states))
        return self.out_proj(attn_output) + self.mlp_fc2(mlp_hidden_states)


class NunchakuFlux2TransformerBlock(Flux2TransformerBlock):
    """
    Nunchaku-optimized FLUX.2 double-stream block. The forward pass is inherited from diffusers.

    Parameters
    ----------
    block : Flux2TransformerBlock
        The original block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: Flux2TransformerBlock, **kwargs):
        super(Flux2TransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context
        self.attn = NunchakuFlux2Attention(block.attn, **kwargs)
        self.norm2 = block.norm2
        self.ff = NunchakuFlux2FeedForward(block.ff, **kwargs)
        self.norm2_context = block.norm2_context
        self.ff_context = NunchakuFlux2FeedForward(block.ff_context, **kwargs)


class NunchakuFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):
    """
    Nunchaku-optimized FLUX.2 single-stream block. The forward pass is inherited from diffusers.

    Parameters
    ----------
    block : Flux2SingleTransformerBlock
        The original block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: Flux2SingleTransformerBlock, **kwargs):
        super(Flux2SingleTransformerBlock, self).__init__()
        self.norm = block.norm
        self.attn = NunchakuFlux2ParallelSelfAttention(block.attn, **kwargs)


class NunchakuFlux2Transformer2DModel(Flux2Transformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized Flux2Transformer2DModel (FLUX.2 [klein] 4B/9B, ...).

    Supports the standard forward path used by ``Flux2KleinPipeline`` (text-to-image and reference-image editing,
    where reference latents are concatenated into ``hidden_states``). The reference-KV-cache modes used by
    ``Flux2KleinKVPipeline`` are not supported.
    """

    def _patch_model(self, **kwargs):
        """
        Patch the model with quantized double- and single-stream blocks.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuFlux2Transformer2DModel
            The patched model.
        """
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFlux2TransformerBlock(block, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFlux2SingleTransformerBlock(block, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuFlux2Transformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuFlux2Transformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        if kwargs.get("offload", False):
            raise NotImplementedError("Offload is not supported for NunchakuFlux2Transformer2DModel")
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"

        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata["quantization_config"])
        precision = get_precision_from_quantization_config(quantization_config)
        rank = quantization_config.get("rank", 32)

        transformer = transformer.to(torch_dtype)
        transformer._patch_model(precision=precision, rank=rank, torch_dtype=torch_dtype)
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        transformer.load_state_dict(model_state_dict)

        if get_backend() == "triton":
            from ...ops.triton.layout import convert_svdq_linear_to_natural_layout

            for module in transformer.modules():
                if isinstance(module, SVDQW4A4Linear):
                    convert_svdq_linear_to_natural_layout(module)

        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        kv_cache=None,
        kv_cache_mode: Optional[str] = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass, mirroring ``Flux2Transformer2DModel.forward`` with rotary embeddings converted once for the
        fused QKV epilogue.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image (and concatenated reference-image) latents of shape (B, T_img, in_channels).
        encoder_hidden_states : torch.Tensor
            Text embeddings of shape (B, T_txt, joint_attention_dim).
        timestep : torch.LongTensor
            Denoising step.
        img_ids : torch.Tensor
            Image position ids.
        txt_ids : torch.Tensor
            Text position ids.
        guidance : torch.Tensor, optional
            Guidance embedding for guidance-distilled variants.
        joint_attention_kwargs : dict, optional
            Not supported.
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).
        kv_cache, kv_cache_mode, num_ref_tokens, ref_fixed_timestep
            Reference-KV-cache arguments (``Flux2KleinKVPipeline``). Not supported.

        Returns
        -------
        Transformer2DModelOutput or tuple
            Output sample tensor or output tuple.
        """
        if kv_cache is not None or kv_cache_mode is not None or num_ref_tokens:
            raise NotImplementedError("Reference KV caching (Flux2KleinKVPipeline) is not supported")
        if joint_attention_kwargs:
            raise NotImplementedError("joint_attention_kwargs is not supported")

        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        rotary_emb_img = _pack_rotary_emb(image_rotary_emb)
        rotary_emb_txt = _pack_rotary_emb(text_rotary_emb)
        rotary_emb_single = _pack_rotary_emb(
            (
                torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )
        )

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=rotary_emb_single,
            )
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
