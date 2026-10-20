"""
Runtime LoRA support (nunchaku.lora.runtime).

- Format conversion: LoRAs in diffusers / PEFT, kohya and ComfyUI / BFL formats must produce, for every
  layer of this repo's fused FLUX models, the weight delta implied by the LoRA's own definition (computed
  here independently from the conversion code).
- Attaching to a real 4-bit SVDQuant layer (Triton or native kernels) adds exactly ``x @ A.T @ B.T`` and
  detaching restores the layer bit for bit.
"""

import pytest
import torch

from nunchaku.lora.runtime import _targets, attach_lora, detach_lora, load_lora

D, H = 32, 64  # toy inner dim and MLP hidden dim (FLUX.1: 3072 / 12288)
FLUX1 = dict(kind="flux1", inner=D, mlp_in_out=H)
FLUX2 = dict(kind="flux2", inner=D, mlp_in_out=2 * H)
g = torch.Generator().manual_seed(0)


def lora(out_f, in_f, r=4):
    return torch.randn(r, in_f, generator=g), torch.randn(out_f, r, generator=g)


def deltas(info, lora_sd):
    """Weight delta per target layer, through load_lora + _targets."""
    out = {}
    for name, w in load_lora(lora_sd, info).items():
        for path, down, up in _targets(info, name, w):
            out[path] = out.get(path, 0) + up @ down
    return out


def adaln_rows(n, dim):
    return torch.arange(n * dim).view(n, dim).t().reshape(-1)


def check(got, expected):
    assert set(got) == set(expected), set(got) ^ set(expected)
    for k in expected:
        torch.testing.assert_close(got[k], expected[k], rtol=1e-5, atol=1e-5, msg=k)


def test_flux1_diffusers_format():
    q, k, v, add_q = lora(D, D), lora(D, D), lora(D, D), lora(D, D)
    mod, sq, sk, sv, smlp, sout, snorm = (
        lora(6 * D, D),
        lora(D, D),
        lora(D, D),
        lora(D, D),
        lora(H, D),
        lora(D, D + H),
        lora(3 * D, D),
    )
    norm_out, ff = lora(2 * D, D), lora(4 * D, D)
    sd = {}
    for name, (a, b) in {
        "transformer_blocks.0.attn.to_q": q, "transformer_blocks.0.attn.to_k": k, "transformer_blocks.0.attn.to_v": v,
        "transformer_blocks.0.attn.add_q_proj": add_q, "transformer_blocks.0.norm1.linear": mod,
        "transformer_blocks.0.ff.net.0.proj": ff,
        "single_transformer_blocks.0.attn.to_q": sq, "single_transformer_blocks.0.attn.to_k": sk,
        "single_transformer_blocks.0.attn.to_v": sv, "single_transformer_blocks.0.proj_mlp": smlp,
        "single_transformer_blocks.0.proj_out": sout, "single_transformer_blocks.0.norm.linear": snorm,
        "norm_out.linear": norm_out,
    }.items():  # fmt: skip
        sd[f"transformer.{name}.lora_A.weight"], sd[f"transformer.{name}.lora_B.weight"] = a, b
    z = torch.zeros(D, D)
    expected = {
        "transformer_blocks.0.attn.to_qkv": torch.cat([q[1] @ q[0], k[1] @ k[0], v[1] @ v[0]]),
        "transformer_blocks.0.attn.add_qkv_proj": torch.cat([add_q[1] @ add_q[0], z, z]),
        "transformer_blocks.0.norm1.linear": (mod[1] @ mod[0])[adaln_rows(6, D)],
        "transformer_blocks.0.ff.net.0.proj": ff[1] @ ff[0],
        "single_transformer_blocks.0.attn.to_qkv": torch.cat([sq[1] @ sq[0], sk[1] @ sk[0], sv[1] @ sv[0]]),
        "single_transformer_blocks.0.mlp_fc1": smlp[1] @ smlp[0],
        "single_transformer_blocks.0.attn.to_out": sout[1] @ sout[0][:, :D],
        "single_transformer_blocks.0.mlp_fc2": sout[1] @ sout[0][:, D:],
        "single_transformer_blocks.0.norm.linear": (snorm[1] @ snorm[0])[adaln_rows(3, D)],
        "norm_out.linear": norm_out[1] @ norm_out[0],
    }
    check(deltas(FLUX1, sd), expected)


def test_flux1_kohya_matches_bfl():
    """kohya (with alpha) and ComfyUI/BFL names of the same weights give the same deltas."""
    weights = {
        "double_blocks.1.txt_attn.qkv": lora(3 * D, D), "double_blocks.1.img_mlp.2": lora(D, 4 * D),
        "single_blocks.2.linear1": lora(3 * D + H, D), "single_blocks.2.linear2": lora(D, D + H),
        "final_layer.adaLN_modulation.1": lora(2 * D, D), "img_in": lora(D, 16),
    }  # fmt: skip
    alpha, r = 2.0, 4
    kohya, bfl = {
        "lora_te1_text_model_x.lora_down.weight": torch.zeros(1, 1),
        "lora_te1_text_model_x.lora_up.weight": torch.zeros(1, 1),
    }, {}
    for name, (a, b) in weights.items():
        kname = "lora_unet_" + name.replace(".", "_")
        kohya[f"{kname}.lora_down.weight"], kohya[f"{kname}.lora_up.weight"] = a, b
        kohya[f"{kname}.alpha"] = torch.tensor(alpha)
        bfl[f"diffusion_model.{name}.lora_A.weight"], bfl[f"diffusion_model.{name}.lora_B.weight"] = a, b * alpha / r
    got_kohya, got_bfl = deltas(FLUX1, kohya), deltas(FLUX1, bfl)
    check(got_kohya, got_bfl)
    a, b = weights["final_layer.adaLN_modulation.1"]
    full = b @ a * alpha / r  # BFL rows (shift, scale) -> diffusers (scale, shift)
    torch.testing.assert_close(got_bfl["norm_out.linear"], torch.cat([full[D:], full[:D]]))
    a, b = weights["single_blocks.2.linear1"]
    torch.testing.assert_close(got_bfl["single_transformer_blocks.2.mlp_fc1"], (b @ a * alpha / r)[3 * D :])


def test_flux2_bfl_and_diffusers_agree():
    """FLUX.2 [klein]: a diffusers-format LoRA and its ComfyUI/BFL equivalent map to the same layer deltas."""
    q, k, v = lora(D, D), lora(D, D), lora(D, D)
    lin1, lin2, mlp0, mod = lora(3 * D + 2 * H, D), lora(D, D + H), lora(2 * H, D), lora(6 * D, D)
    dif = {}
    for name, (a, b) in {
        "transformer_blocks.0.attn.to_q": q, "transformer_blocks.0.attn.to_k": k, "transformer_blocks.0.attn.to_v": v,
        "transformer_blocks.0.ff.linear_in": mlp0, "single_transformer_blocks.0.attn.to_qkv_mlp_proj": lin1,
        "single_transformer_blocks.0.attn.to_out": lin2, "double_stream_modulation_img.linear": mod,
    }.items():  # fmt: skip
        dif[f"transformer.{name}.lora_A.weight"], dif[f"transformer.{name}.lora_B.weight"] = a, b
    qkv = (torch.cat([q[0], k[0], v[0]]), torch.block_diag(q[1], k[1], v[1]))
    bfl = {}
    for name, (a, b) in {
        "double_blocks.0.img_attn.qkv": qkv, "double_blocks.0.img_mlp.0": mlp0, "single_blocks.0.linear1": lin1,
        "single_blocks.0.linear2": lin2, "double_stream_modulation_img.lin": mod,
    }.items():  # fmt: skip
        bfl[f"diffusion_model.{name}.lora_A.weight"], bfl[f"diffusion_model.{name}.lora_B.weight"] = a, b
    got = deltas(FLUX2, bfl)
    check(deltas(FLUX2, dif), got)
    torch.testing.assert_close(got["single_transformer_blocks.0.attn.qkv_proj"], (lin1[1] @ lin1[0])[: 3 * D])
    torch.testing.assert_close(got["single_transformer_blocks.0.attn.mlp_fc2"], lin2[1] @ lin2[0][:, D:])
    torch.testing.assert_close(got["double_stream_modulation_img.linear"], mod[1] @ mod[0])


def test_unknown_module_rejected():
    with pytest.raises(ValueError):
        deltas(
            FLUX2,
            {
                "diffusion_model.double_blocks.0.img_mod.lin.lora_A.weight": torch.zeros(4, D),
                "diffusion_model.double_blocks.0.img_mod.lin.lora_B.weight": torch.zeros(6 * D, 4),
            },
        )


def _svdq_linear(in_f, out_f, rank, device):
    """A random but valid natural-layout int4 SVDQuant layer."""
    from nunchaku.models.linear import SVDQW4A4Linear

    lin = SVDQW4A4Linear(in_f, out_f, rank=rank, bias=True, precision="int4", torch_dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        lin.qweight.copy_(torch.randint(-128, 128, lin.qweight.shape, dtype=torch.int8, device=device))
        lin.wscales.copy_(torch.rand(lin.wscales.shape, device=device) * 0.01 + 0.002)
        lin.bias.copy_(torch.randn(out_f, device=device) * 0.1)
        lin.smooth_factor.copy_(torch.rand(in_f, device=device) * 0.5 + 0.75)
        lin.smooth_factor_orig.copy_(lin.smooth_factor)
        lin.proj_down.copy_(torch.randn(in_f, rank, device=device) * 0.05)
        lin.proj_up.copy_(torch.randn(out_f, rank, device=device) * 0.05)
    return lin


@pytest.mark.parametrize("lora_ranks", [[8], [8, 24]])
def test_attach_to_svdq_layer(lora_ranks):
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")
    from nunchaku.ops.backend import set_backend

    set_backend("triton")
    try:
        dev = torch.device("cuda")
        lin = _svdq_linear(1024, 512, 32, dev)
        x = torch.randn(1, 300, 1024, device=dev, dtype=torch.bfloat16)
        y0 = lin(x)
        pairs = [(torch.randn(r, 1024) * 0.03, torch.randn(512, r) * 0.03) for r in lora_ranks]
        attach_lora(lin, pairs)
        assert lin.proj_down.shape[1] % 16 == 0 and lin.proj_up.shape[1] == lin.proj_down.shape[1]
        y1 = lin(x)
        ref = sum((x.float() @ d.to(dev).t()) @ u.to(dev).t() for d, u in pairs)
        delta = (y1 - y0).float()
        # bf16 output rounding on each of y0 / y1
        torch.testing.assert_close(delta, ref, rtol=0.05, atol=2 * y0.abs().max().item() * 2**-8)
        detach_lora(lin)
        assert lin.proj_down.shape[1] == 32
        torch.testing.assert_close(lin(x), y0, rtol=0, atol=0)
    finally:
        set_backend(None)


def test_attach_hook_to_linear():
    lin = torch.nn.Linear(64, 48, dtype=torch.float32)
    x = torch.randn(5, 64)
    y0 = lin(x)
    d, u = torch.randn(4, 64), torch.randn(48, 4)
    attach_lora(lin, [(d, u)])
    torch.testing.assert_close(lin(x) - y0, x @ d.t() @ u.t(), rtol=1e-4, atol=1e-4)
    detach_lora(lin)
    torch.testing.assert_close(lin(x), y0)
