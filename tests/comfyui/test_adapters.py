"""
ComfyUI adapter conventions (``comfyui/custom_nodes/nunchaku_rocm``), checked without checkpoints.

Each adapter only needs the diffusers interface of the model it wraps, so a small randomly initialised diffusers
model run through the adapter must match ComfyUI's own implementation loaded with the same weights (bf16 noise):
that pins down the latent / timestep / text / RoPE conventions. The Qwen-Image per-block CPU offload must also be
bit-identical to the resident model, across back-to-back calls with no host synchronisation in between.

Needs a ComfyUI checkout: ``COMFYUI_PATH=/path/to/ComfyUI pytest tests/comfyui``; skipped otherwise.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
pytestmark = [
    pytest.mark.skipif(not COMFYUI_PATH, reason="set COMFYUI_PATH to a ComfyUI checkout"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available"),
]
if COMFYUI_PATH:
    sys.path[:0] = [COMFYUI_PATH, str(Path(__file__).resolve().parents[2] / "comfyui" / "custom_nodes")]

DEV, DT = "cuda", torch.bfloat16


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _randomize(module: torch.nn.Module, seed: int = 0) -> None:
    # diffusers zero-initialises some output layers; give every weight a non-trivial value
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.02 + (1.0 if p.ndim == 1 else 0.0)).to(p.dtype))


def _qwen_models(heads: int = 2):
    import comfy.ops
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel as ComfyQwen
    from diffusers import QwenImageTransformer2DModel

    cfg = dict(num_layers=4, attention_head_dim=128, num_attention_heads=heads, joint_attention_dim=64)
    diff = QwenImageTransformer2DModel(in_channels=64, out_channels=16, axes_dims_rope=(16, 56, 56), **cfg)
    _randomize(diff)
    diff = diff.to(DEV, DT).eval()
    comfy_model = ComfyQwen(
        patch_size=2, in_channels=64, out_channels=16, axes_dims_rope=(16, 56, 56), dtype=DT, device=DEV,
        operations=comfy.ops.disable_weight_init, **cfg,
    )  # fmt: skip
    missing, unexpected = comfy_model.load_state_dict(diff.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return diff, comfy_model.eval()


@pytest.mark.parametrize("hw", [(64, 64), (48, 80)])
def test_qwen_image_adapter_matches_comfy(hw):
    from nunchaku_rocm.qwen_image import NunchakuQwenImageAdapter

    diff, comfy_model = _qwen_models()
    adapter = NunchakuQwenImageAdapter(diff, dict(diff.config), DT)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 1, *hw, generator=g).to(DEV, DT)
    ctx = torch.randn(1, 23, 64, generator=g).to(DEV, DT)
    with torch.no_grad():
        for sigma in (0.9, 0.4):
            t = torch.tensor([sigma], device=DEV)
            out, ref = adapter(x, t, ctx), comfy_model(x, t, ctx)
            assert out.shape == ref.shape
            assert _rel(out, ref) < 0.02, (sigma, _rel(out, ref))


def test_qwen_image_offload_is_exact():
    import copy

    from nunchaku_rocm.qwen_image import NunchakuQwenImageAdapter

    # big enough that the GPU runs behind the host, so a missing stream dependency shows up as a race
    diff, _ = _qwen_models(heads=8)
    resident = NunchakuQwenImageAdapter(diff, dict(diff.config), DT)
    offloaded = NunchakuQwenImageAdapter(
        copy.deepcopy(diff).cpu(), dict(diff.config), DT, offload=True, num_blocks_on_gpu=1
    ).to(DEV)
    assert len(offloaded.transformer.transformer_blocks) == 1 and len(offloaded._offloaded_blocks) == 3
    assert all(p.device.type == "cpu" for b in offloaded._offloaded_blocks for p in b.parameters())

    g = torch.Generator().manual_seed(2)
    xs = [torch.randn(2, 16, 1, 192, 192, generator=g).to(DEV, DT) for _ in range(4)]
    ctx = torch.randn(2, 17, 64, generator=g).to(DEV, DT)
    t = torch.tensor([0.7, 0.7], device=DEV)
    with torch.no_grad():
        # queue the calls back to back: the side-stream uploads must order themselves against the compute stream
        outs = [offloaded(x, t, ctx) for x in xs]
        refs = [resident(x, t, ctx) for x in xs]
    for out, ref in zip(outs, refs):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def _diffusers_to_comfy_zimage(sd: dict, n_layers: int, dim: int) -> dict:
    import comfy.utils

    key_map = comfy.utils.z_image_to_diffusers({"n_layers": n_layers, "dim": dim})
    out = {}
    for k, v in sd.items():
        target = key_map[k]
        if isinstance(target, tuple):  # q / k / v rows of the fused qkv weight
            name, (axis, start, size) = target
            if name not in out:
                out[name] = torch.empty(3 * dim, *v.shape[1:], dtype=v.dtype)
            out[name].narrow(axis, start, size).copy_(v)
        else:
            out[target] = v
    return out


def test_z_image_adapter_matches_comfy():
    import comfy.ops
    from comfy.ldm.lumina.model import NextDiT
    from diffusers import ZImageTransformer2DModel
    from nunchaku_rocm.z_image import NunchakuZImageAdapter

    # dim 1536: diffusers' and ComfyUI's FFN widths / timestep MLP agree only for dims like the real 3840
    dim, heads, n_layers, cap_dim = 1536, 12, 2, 64
    diff = ZImageTransformer2DModel(
        in_channels=16, dim=dim, n_layers=n_layers, n_refiner_layers=2, n_heads=heads, n_kv_heads=heads,
        cap_feat_dim=cap_dim, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512], rope_theta=256.0, t_scale=1000.0,
    )  # fmt: skip
    _randomize(diff)
    comfy_model = NextDiT(
        patch_size=2, in_channels=16, dim=dim, n_layers=n_layers, n_refiner_layers=2, n_heads=heads,
        n_kv_heads=heads, multiple_of=256, ffn_dim_multiplier=8.0 / 3.0, norm_eps=1e-5, qk_norm=True,
        cap_feat_dim=cap_dim, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512], rope_theta=256.0,
        z_image_modulation=True, time_scale=1000.0, pad_tokens_multiple=32, dtype=DT, device=DEV,
        operations=comfy.ops.disable_weight_init,
    )  # fmt: skip
    missing, unexpected = comfy_model.load_state_dict(
        _diffusers_to_comfy_zimage(diff.state_dict(), n_layers, dim), strict=False
    )
    assert not missing and not unexpected, (missing, unexpected)
    diff = diff.to(DEV, DT).eval()
    adapter = NunchakuZImageAdapter(diff, dict(diff.config), DT)

    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 16, 40, 56, generator=g).to(DEV, DT)
    ctx = torch.randn(1, 19, cap_dim, generator=g).to(DEV, DT)  # not a multiple of the 32-token padding
    with torch.no_grad():
        for sigma in (0.9, 0.4):
            t = torch.tensor([sigma], device=DEV)
            out, ref = adapter(x, t, ctx, num_tokens=19), comfy_model(x, t, ctx, num_tokens=19)
            assert out.shape == ref.shape
            assert _rel(out, ref) < 0.03, (sigma, _rel(out, ref))
