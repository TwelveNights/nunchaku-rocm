"""
Apply LoRAs to a loaded Nunchaku FLUX.1 / FLUX.2 transformer at runtime.

:func:`set_loras` attaches any number of LoRAs (with strengths) to a
:class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2` or
:class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`, replacing the
previous set; ``set_loras(model, [])`` restores the base model exactly.

- 4-bit SVDQuant layers already compute ``x @ proj_down @ proj_up.T`` (their low-rank branch, in 16
  bits, on the unquantized activation), so a LoRA ``B @ A`` is appended to that branch as extra rank:
  ``proj_down = [base | A.T]``, ``proj_up = [base | strength * scale * B]`` (padded to a multiple of 16).
  This costs no extra kernels, and works with every fused epilogue.
- Other layers (bf16 ``nn.Linear``, the 4-bit AWQ adaLN projections of FLUX.1) get a forward hook adding
  ``strength * scale * (x @ A.T) @ B.T``.

Accepted LoRA formats (FLUX.1 and FLUX.2 [klein]): ComfyUI / BFL keys (``diffusion_model.double_blocks.0.
img_attn.qkv.lora_A.weight``), kohya (``lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight`` with
``.alpha``), and diffusers / PEFT (``transformer.transformer_blocks.0.attn.to_q.lora_A.weight``). Everything
is first normalized to BFL module names, then mapped onto this repo's (diffusers-derived, fused) layers.
Text-encoder LoRA weights are ignored.
"""

import logging
import re
from dataclasses import dataclass
from os import PathLike

import torch
from torch import nn

__all__ = ["LoraWeight", "attach_lora", "detach_lora", "load_lora", "lora_model_info", "set_loras"]

logger = logging.getLogger(__name__)


@dataclass
class LoraWeight:
    """One LoRA target: ``delta_W = scale * up @ down``; down (r, in), up (out, r), both float32 on CPU."""

    down: torch.Tensor
    up: torch.Tensor
    scale: float = 1.0


# ---------------------------------------------------------------------------------------------------------------
# format normalization -> {BFL module name: LoraWeight}


def _kohya_to_bfl(name: str) -> str | None:
    """``lora_unet_double_blocks_0_img_attn_qkv`` -> ``double_blocks.0.img_attn.qkv`` (None if not a known module)."""
    rules = [
        (r"double_blocks_(\d+)_(img|txt)_(attn)_(qkv|proj)", r"double_blocks.\1.\2_\3.\4"),
        (r"double_blocks_(\d+)_(img|txt)_mlp_(\d+)", r"double_blocks.\1.\2_mlp.\3"),
        (r"double_blocks_(\d+)_(img|txt)_mod_lin", r"double_blocks.\1.\2_mod.lin"),
        (r"single_blocks_(\d+)_(linear1|linear2)", r"single_blocks.\1.\2"),
        (r"single_blocks_(\d+)_modulation_lin", r"single_blocks.\1.modulation.lin"),
        (r"final_layer_linear", r"final_layer.linear"),
        (r"final_layer_adaLN_modulation_1", r"final_layer.adaLN_modulation.1"),
        (r"(time_in|vector_in|guidance_in)_(in_layer|out_layer)", r"\1.\2"),
        (r"(img_in|txt_in)", r"\1"),
        (r"(double_stream_modulation_img|double_stream_modulation_txt|single_stream_modulation)_lin", r"\1.lin"),
    ]
    for pat, rep in rules:
        if re.fullmatch(pat, name):
            return re.sub(pat, rep, name)
    return None


def _split_pairs(sd: dict[str, torch.Tensor]) -> tuple[str, dict[str, LoraWeight]]:
    """Group raw keys into (module -> LoraWeight) and detect the naming scheme."""
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in sd.items():
        for suffix, slot in (
            (".lora_A.weight", "down"),
            (".lora_B.weight", "up"),
            (".lora_down.weight", "down"),
            (".lora_up.weight", "up"),
            (".alpha", "alpha"),
        ):
            if key.endswith(suffix):
                groups.setdefault(key[: -len(suffix)], {})[slot] = value
                break
    out = {}
    for module, g in groups.items():
        if "down" not in g or "up" not in g:
            continue
        down, up = g["down"].float().flatten(1), g["up"].float().flatten(1)
        scale = float(g["alpha"]) / down.shape[0] if "alpha" in g else 1.0
        out[module] = LoraWeight(down, up, scale)
    names = list(out)
    if any(n.startswith("lora_unet_") for n in names):
        scheme = "kohya"
    elif any(".transformer_blocks." in n or n.startswith(("transformer.", "transformer_blocks.")) for n in names):
        scheme = "diffusers"
    else:
        scheme = "bfl"
    return scheme, out


def _block_diag_cat(parts: list[LoraWeight | None], out_sizes: list[int]) -> LoraWeight | None:
    """
    Combine LoRAs of consecutive row blocks of one fused layer (e.g. separate q / k / v LoRAs for a fused
    qkv projection): ``down = cat(downs)``, ``up = block_diag(ups)`` (missing blocks stay zero), with each
    part's scale folded into its ``up``.
    """
    if all(p is None for p in parts):
        return None
    downs, blocks = [], []
    for p, rows in zip(parts, out_sizes):
        if p is not None:
            downs.append(p.down)
            blocks.append(p.up * p.scale)
        else:
            blocks.append(torch.zeros(rows, 0))
    return LoraWeight(torch.cat(downs), torch.block_diag(*blocks), 1.0)


def _diffusers_to_bfl(lora: dict[str, LoraWeight], dims: dict) -> dict[str, LoraWeight]:
    """Diffusers / PEFT names (FLUX.1 or FLUX.2) -> BFL names; fuses q/k/v (and FLUX.1 single-block q/k/v/mlp)."""
    D, mlp1 = dims["inner"], dims["mlp_in_out"]
    lora = {re.sub(r"^(transformer|base_model\.model)\.", "", k): v for k, v in lora.items()}
    out: dict[str, LoraWeight] = {}
    simple = [
        (r"transformer_blocks\.(\d+)\.attn\.to_out\.0", r"double_blocks.\1.img_attn.proj"),
        (r"transformer_blocks\.(\d+)\.attn\.to_add_out", r"double_blocks.\1.txt_attn.proj"),
        (r"transformer_blocks\.(\d+)\.ff\.net\.0\.proj", r"double_blocks.\1.img_mlp.0"),
        (r"transformer_blocks\.(\d+)\.ff\.net\.2", r"double_blocks.\1.img_mlp.2"),
        (r"transformer_blocks\.(\d+)\.ff_context\.net\.0\.proj", r"double_blocks.\1.txt_mlp.0"),
        (r"transformer_blocks\.(\d+)\.ff_context\.net\.2", r"double_blocks.\1.txt_mlp.2"),
        (r"transformer_blocks\.(\d+)\.ff\.linear_in", r"double_blocks.\1.img_mlp.0"),
        (r"transformer_blocks\.(\d+)\.ff\.linear_out", r"double_blocks.\1.img_mlp.2"),
        (r"transformer_blocks\.(\d+)\.ff_context\.linear_in", r"double_blocks.\1.txt_mlp.0"),
        (r"transformer_blocks\.(\d+)\.ff_context\.linear_out", r"double_blocks.\1.txt_mlp.2"),
        (r"transformer_blocks\.(\d+)\.norm1\.linear", r"double_blocks.\1.img_mod.lin"),
        (r"transformer_blocks\.(\d+)\.norm1_context\.linear", r"double_blocks.\1.txt_mod.lin"),
        (r"single_transformer_blocks\.(\d+)\.proj_out", r"single_blocks.\1.linear2"),
        (r"single_transformer_blocks\.(\d+)\.attn\.to_out", r"single_blocks.\1.linear2"),
        (r"single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj", r"single_blocks.\1.linear1"),
        (r"single_transformer_blocks\.(\d+)\.norm\.linear", r"single_blocks.\1.modulation.lin"),
        (r"x_embedder", "img_in"),
        (r"context_embedder", "txt_in"),
        (r"proj_out", "final_layer.linear"),
        (r"(time_text_embed|time_guidance_embed)\.timestep_embedder\.linear_1", "time_in.in_layer"),
        (r"(time_text_embed|time_guidance_embed)\.timestep_embedder\.linear_2", "time_in.out_layer"),
        (r"time_text_embed\.text_embedder\.linear_1", "vector_in.in_layer"),
        (r"time_text_embed\.text_embedder\.linear_2", "vector_in.out_layer"),
        (r"(time_text_embed|time_guidance_embed)\.guidance_embedder\.linear_1", "guidance_in.in_layer"),
        (r"(time_text_embed|time_guidance_embed)\.guidance_embedder\.linear_2", "guidance_in.out_layer"),
        (r"(double_stream_modulation_img|double_stream_modulation_txt|single_stream_modulation)\.linear", r"\1.lin"),
    ]
    for name, w in lora.items():
        for pat, rep in simple:
            if re.fullmatch(pat, name):
                out[re.sub(pat, rep, name)] = w
                break
        else:
            if name == "norm_out.linear":  # diffusers (scale, shift) -> BFL (shift, scale)
                half = w.up.shape[0] // 2
                out["final_layer.adaLN_modulation.1"] = LoraWeight(
                    w.down, torch.cat([w.up[half:], w.up[:half]]), w.scale
                )

    blocks = {int(m.group(1)) for k in lora if (m := re.match(r"(?:single_)?transformer_blocks\.(\d+)\.", k))}
    for i in sorted(blocks):
        for src, dst in (("to", "img"), ("add", "txt")):
            names = ["to_q", "to_k", "to_v"] if src == "to" else ["add_q_proj", "add_k_proj", "add_v_proj"]
            parts = [lora.get(f"transformer_blocks.{i}.attn.{n}") for n in names]
            fused = _block_diag_cat(parts, [D] * 3)
            if fused is not None:
                out[f"double_blocks.{i}.{dst}_attn.qkv"] = fused
        # FLUX.1 single blocks: separate q/k/v + proj_mlp -> linear1 rows [q | k | v | mlp]
        parts = [lora.get(f"single_transformer_blocks.{i}.attn.{n}") for n in ("to_q", "to_k", "to_v")]
        parts.append(lora.get(f"single_transformer_blocks.{i}.proj_mlp"))
        fused = _block_diag_cat(parts, [D, D, D, mlp1])
        if fused is not None:
            out[f"single_blocks.{i}.linear1"] = fused
    return out


def load_lora(lora: str | PathLike | dict[str, torch.Tensor], dims: dict) -> dict[str, LoraWeight]:
    """
    Read a LoRA (path or state dict) into ``{BFL module name: LoraWeight}``. ``dims`` describes the model
    (see :func:`_model_info`); it is needed to fuse diffusers-format q/k/v LoRAs.
    """
    if not isinstance(lora, dict):
        from safetensors.torch import load_file

        lora = load_file(str(lora))
    scheme, pairs = _split_pairs(lora)
    if not pairs:
        raise ValueError("no LoRA weights found (expected lora_A/lora_B or lora_down/lora_up keys)")
    if scheme == "kohya":
        out, skipped = {}, 0
        for name, w in pairs.items():
            if not name.startswith("lora_unet_"):
                skipped += 1  # text encoder (lora_te*)
                continue
            bfl = _kohya_to_bfl(name[len("lora_unet_") :])
            if bfl is None:
                raise ValueError(f"unsupported kohya LoRA module {name!r}")
            out[bfl] = w
        if skipped:
            logger.warning("LoRA: ignoring %d text-encoder weights", skipped)
        return out
    if scheme == "diffusers":
        out = _diffusers_to_bfl(pairs, dims)
        unused = len(pairs) - _count_diffusers_used(pairs)
        if unused:
            logger.warning("LoRA: ignoring %d diffusers weights with unrecognized names", unused)
        return out
    return {re.sub(r"^(diffusion_model|model\.diffusion_model)\.", "", k): v for k, v in pairs.items()}


def _count_diffusers_used(pairs: dict[str, LoraWeight]) -> int:
    known = re.compile(
        r"(transformer\.)?((single_)?transformer_blocks\.\d+\.|x_embedder|context_embedder|proj_out|norm_out\.linear"
        r"|time_text_embed\.|time_guidance_embed\.|double_stream_modulation_|single_stream_modulation)"
    )
    return sum(1 for k in pairs if known.match(k))


# ---------------------------------------------------------------------------------------------------------------
# BFL names -> this repo's layers


def lora_model_info(model: nn.Module) -> dict:
    """Model kind and the layer sizes LoRA conversion needs (pass to :func:`load_lora`)."""
    from ..models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel
    from ..models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2

    if isinstance(model, NunchakuFlux2Transformer2DModel):
        attn = model.single_transformer_blocks[0].attn
        return dict(kind="flux2", inner=attn.inner_dim, mlp_in_out=attn.mlp_hidden_dim * attn.mlp_mult_factor)
    if isinstance(model, NunchakuFluxTransformer2DModelV2):
        blk = model.single_transformer_blocks[0]
        return dict(kind="flux1", inner=blk.attn.inner_dim, mlp_in_out=blk.mlp_hidden_dim)
    raise TypeError(f"LoRA: unsupported model {type(model).__name__}")


def _adaln_rows(n_chunks: int, dim: int) -> torch.Tensor:
    """Row order of Nunchaku's AWQ adaLN projection: output row d * n + j holds diffusers row j * dim + d."""
    return torch.arange(n_chunks * dim).view(n_chunks, dim).t().reshape(-1)


def _targets(info: dict, name: str, w: LoraWeight):
    """Yield (module path, down, up) for a BFL-named LoRA weight; up has strength-independent scale folded in."""
    D = info["inner"]
    up = w.up * w.scale
    flux2 = info["kind"] == "flux2"
    if m := re.fullmatch(r"double_blocks\.(\d+)\.(img|txt)_(attn\.qkv|attn\.proj|mlp\.0|mlp\.2|mod\.lin)", name):
        i, s, part = m.groups()
        b = f"transformer_blocks.{i}"
        if flux2:
            table = {
                ("img", "attn.qkv"): "attn.to_qkv", ("txt", "attn.qkv"): "attn.to_added_qkv",
                ("img", "attn.proj"): "attn.to_out.0", ("txt", "attn.proj"): "attn.to_add_out",
                ("img", "mlp.0"): "ff.linear_in", ("txt", "mlp.0"): "ff_context.linear_in",
                ("img", "mlp.2"): "ff.linear_out", ("txt", "mlp.2"): "ff_context.linear_out",
            }  # fmt: skip
        else:
            table = {
                ("img", "attn.qkv"): "attn.to_qkv", ("txt", "attn.qkv"): "attn.add_qkv_proj",
                ("img", "attn.proj"): "attn.to_out.0", ("txt", "attn.proj"): "attn.to_add_out",
                ("img", "mlp.0"): "ff.net.0.proj", ("txt", "mlp.0"): "ff_context.net.0.proj",
                ("img", "mlp.2"): "ff.net.2", ("txt", "mlp.2"): "ff_context.net.2",
                ("img", "mod.lin"): "norm1.linear", ("txt", "mod.lin"): "norm1_context.linear",
            }  # fmt: skip
        target = table.get((s, part))
        if target is None:
            raise ValueError(f"LoRA module {name} has no counterpart in this model")
        if part == "mod.lin":
            up = up[_adaln_rows(6, D)]
        yield f"{b}.{target}", w.down, up
        return
    if m := re.fullmatch(r"single_blocks\.(\d+)\.(linear1|linear2|modulation\.lin)", name):
        i, part = m.groups()
        b = f"single_transformer_blocks.{i}"
        if part == "linear1":  # rows [q | k | v | mlp]
            qkv, mlp = (f"{b}.attn.qkv_proj", f"{b}.attn.mlp_fc1") if flux2 else (f"{b}.attn.to_qkv", f"{b}.mlp_fc1")
            yield qkv, w.down, up[: 3 * D]
            yield mlp, w.down, up[3 * D :]
        elif part == "linear2":  # input columns [attention | mlp]
            out, fc2 = (f"{b}.attn.out_proj", f"{b}.attn.mlp_fc2") if flux2 else (f"{b}.attn.to_out", f"{b}.mlp_fc2")
            yield out, w.down[:, :D], up
            yield fc2, w.down[:, D:], up
        else:
            if flux2:
                raise ValueError(f"LoRA module {name} has no counterpart in FLUX.2")
            yield f"{b}.norm.linear", w.down, up[_adaln_rows(3, D)]
        return
    if name == "final_layer.adaLN_modulation.1":  # BFL (shift, scale) -> diffusers (scale, shift)
        half = up.shape[0] // 2
        yield "norm_out.linear", w.down, torch.cat([up[half:], up[:half]])
        return
    emb = "time_guidance_embed" if flux2 else "time_text_embed"
    simple = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "final_layer.linear": "proj_out",
        "time_in.in_layer": f"{emb}.timestep_embedder.linear_1",
        "time_in.out_layer": f"{emb}.timestep_embedder.linear_2",
        "guidance_in.in_layer": f"{emb}.guidance_embedder.linear_1",
        "guidance_in.out_layer": f"{emb}.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
    }
    if name in simple:
        yield simple[name], w.down, up
        return
    raise ValueError(f"unsupported LoRA module {name!r}")


class _LoraHook:
    """Forward hook adding ``sum_i (x @ down_i.T) @ up_i.T`` to a non-SVDQuant layer."""

    def __init__(self):
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, module, args, output):
        x = args[0]
        if self.pairs and self.pairs[0][0].device != x.device:
            self.pairs = [(d.to(x.device), u.to(x.device)) for d, u in self.pairs]
        for down, up in self.pairs:
            output = output + ((x.to(down.dtype) @ down.t()) @ up.t()).to(output.dtype)
        return output


def attach_lora(module: nn.Module, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """
    Add ``sum_i (x @ down_i.T) @ up_i.T`` to one layer's output: appended to the low-rank branch of an
    :class:`~nunchaku.models.linear.SVDQW4A4Linear` (rank zero-padded to a multiple of 16 for the native
    kernels), otherwise through a forward hook. ``down`` (r, in), ``up`` (out, r), strength already folded
    into ``up``. Undo with :func:`detach_lora`.
    """
    from ..models.linear import SVDQW4A4Linear

    detach_lora(module)
    if isinstance(module, SVDQW4A4Linear):
        base_down, base_up = module.proj_down.data, module.proj_up.data
        downs = [d.t() for d, _ in pairs]
        ups = [u for _, u in pairs]
        pad = -(base_down.shape[1] + sum(d.shape[1] for d in downs)) % 16
        if pad:
            downs.append(torch.zeros(module.in_features, pad))
            ups.append(torch.zeros(module.out_features, pad))
        dev, dt = base_down.device, base_down.dtype
        module._lora_base = (base_down, base_up)
        module.proj_down.data = torch.cat([base_down, torch.cat(downs, 1).to(dev, dt)], 1).contiguous()
        module.proj_up.data = torch.cat([base_up, torch.cat(ups, 1).to(dev, dt)], 1).contiguous()
    else:
        # the compute dtype: AWQ layers also hold packed integer weights
        p = next(p for p in module.parameters() if p.is_floating_point())
        hook = _LoraHook()
        hook.pairs = [(d.to(p.device, p.dtype), u.to(p.device, p.dtype)) for d, u in pairs]
        module._lora_hook = hook
        module._lora_hook_handle = module.register_forward_hook(hook)


def detach_lora(module: nn.Module) -> None:
    """Remove what :func:`attach_lora` added (no-op if nothing is attached)."""
    if hasattr(module, "_lora_base"):
        dev = module.proj_down.device  # the model may have moved since the LoRAs were attached
        module.proj_down.data, module.proj_up.data = (t.to(dev) for t in module._lora_base)
        del module._lora_base
    if hasattr(module, "_lora_hook"):
        module._lora_hook_handle.remove()
        del module._lora_hook, module._lora_hook_handle


def set_loras(model: nn.Module, loras: list[tuple[str | PathLike | dict, float]]) -> None:
    """
    Apply ``loras`` -- ``[(path, state dict or already-loaded {BFL name: LoraWeight}, strength), ...]`` -- to
    a Nunchaku FLUX.1 / FLUX.2 transformer, replacing any LoRAs applied before. ``[]`` restores the base
    model. Strength 0 entries are skipped.
    """
    from ..ops.triton.native.w4a4 import interleave_swiglu

    info = lora_model_info(model)
    for module in model.modules():
        detach_lora(module)
    per_module: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for lora, strength in loras:
        if strength == 0:
            continue
        if not (isinstance(lora, dict) and lora and isinstance(next(iter(lora.values())), LoraWeight)):
            lora = load_lora(lora, info)
        for name, w in lora.items():
            for path, down, up in _targets(info, name, w):
                per_module.setdefault(path, []).append((down, up * strength))

    modules = dict(model.named_modules())
    for path, pairs in per_module.items():
        module = modules.get(path)
        if module is None:
            raise ValueError(f"LoRA targets {path!r}, which this model does not have")
        if path.endswith(("linear_in", "mlp_fc1")) and getattr(_parent(modules, path), "swiglu_fused", False):
            # FLUX.2 SwiGLU input projections had their rows interleaved at load time
            idx = interleave_swiglu(module.out_features, torch.device("cpu"))
            pairs = [(d, u[idx]) for d, u in pairs]
        attach_lora(module, pairs)


def _parent(modules: dict[str, nn.Module], path: str) -> nn.Module | None:
    return modules.get(path.rsplit(".", 1)[0]) if "." in path else None
