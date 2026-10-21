"""
Z-Image model-level checks for the Triton backend.

- ``transformer_zimage._pack_freqs_cis`` must turn diffusers' complex ``freqs_cis`` into the natural (sin, cos)
  layout the fused QKV RMSNorm+RoPE epilogue reads; checked against diffusers' own Z-Image rotary embedding.
- The Z-Image loader folds ``to_qkv`` (an ``SVDQW4A4Linear``) into ``NunchakuZImageFusedModule`` while the checkpoint
  is loaded, so the fused module needs its own layout conversion; checked against a ``NunchakuWeightPacker``-packed
  synthetic checkpoint (tensors and a forward pass through the fused QKV + RMSNorm + RoPE path).
"""

import pytest
import torch

from .conftest import load_kernels

w4a4 = load_kernels("gemm_w4a4")


def _freqs_cis(seq_len: int, device="cuda"):
    pytest.importorskip("diffusers.models.transformers.transformer_z_image")
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder

    # Z-Image-Turbo's RoPE: axes (32, 48, 48), head_dim 128; ids = (frame/caption index, h, w)
    rope = RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])
    g = torch.arange(seq_len)
    ids = torch.stack([g % 7 + 1, g // 5, g % 5], dim=-1).to(device)
    return rope(ids)  # (seq_len, 64) complex64


def _zimage_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """diffusers' ZSingleStreamAttnProcessor.apply_rotary_emb; x (B, S, H, D), freqs_cis (B, S, D // 2)."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(2)).flatten(3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zimage_rotary_matches_diffusers(dtype):
    from nunchaku.ops.backend import set_backend

    set_backend("triton")
    try:
        from nunchaku.models.transformers.transformer_zimage import _pack_freqs_cis

        batch, seq, heads, head_dim = 2, 37, 3, 128
        freqs = torch.stack([_freqs_cis(seq), _freqs_cis(seq).roll(5, dims=0)])  # (B, S, 64)
        rot = _pack_freqs_cis(freqs)
        assert rot.shape == (batch * seq, head_dim // 2, 1, 2) and rot.dtype == torch.float32

        qkv = torch.randn(batch * seq, 3 * heads * head_dim, dtype=dtype, device="cuda")
        norm_q = torch.rand(head_dim, dtype=dtype, device="cuda") + 0.5
        norm_k = torch.rand(head_dim, dtype=dtype, device="cuda") + 0.5
        out = qkv.clone()
        w4a4._apply_rmsnorm_rope_qk(out, norm_q, norm_k, rot)

        def ref_qk(x, w):
            x = x.float().view(batch, seq, heads, head_dim)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            return _zimage_rope(x, freqs).reshape(batch * seq, -1)

        q, k, v = qkv.chunk(3, dim=-1)
        oq, ok, ov = out.chunk(3, dim=-1)
        torch.testing.assert_close(oq.float(), ref_qk(q, norm_q), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(ok.float(), ref_qk(k, norm_k), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(ov, v)
    finally:
        set_backend(None)


def _dequant_packed_nibbles(qx: torch.Tensor, k: int) -> torch.Tensor:
    """Signed int4 codes of a natural-layout packed-nibble tensor (see test_layout.py)."""
    b = qx.view(torch.uint8).to(torch.int32)
    codes = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(qx.shape[0], -1)[:, :k]
    return torch.where(codes >= 8, codes - 16, codes).float()


def _packed_svdq_linear(in_features, out_features, rank, dtype, device="cuda"):
    """An SVDQW4A4Linear holding a random checkpoint in the CUDA (packed) layout."""
    from nunchaku.lora.flux.packer import NunchakuWeightPacker
    from nunchaku.models.linear import SVDQW4A4Linear

    lin = SVDQW4A4Linear(
        in_features, out_features, rank=rank, bias=True, precision="int4", torch_dtype=dtype, device=device
    )
    packer = NunchakuWeightPacker(bits=4)
    n, k, group = out_features, in_features, lin.group_size
    lin.qweight.data = packer.pack_weight(torch.randint(0, 16, (n, k), dtype=torch.int32, device=device))
    lin.wscales.data = packer.pack_scale(
        (torch.rand(n, k // group, device=device) * 0.02 + 0.005).to(dtype), group_size=group
    )
    lin.bias.data = packer.pack_scale(torch.randn(n, dtype=dtype, device=device) * 0.1, group_size=-1)
    lin.smooth_factor.data = packer.pack_scale((torch.rand(k, device=device) * 0.5 + 0.75).to(dtype), group_size=-1)
    lin.proj_down.data = packer.pack_lowrank_weight((torch.randn(rank, k, device=device) * 0.05).to(dtype), down=True)
    lin.proj_up.data = packer.pack_lowrank_weight((torch.randn(n, rank, device=device) * 0.05).to(dtype), down=False)
    return lin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zimage_fused_module_natural_layout(dtype):
    import copy

    from diffusers.models.normalization import RMSNorm

    from nunchaku.ops.backend import set_backend

    set_backend("triton")
    try:
        from nunchaku.models.transformers.transformer_zimage import NunchakuZImageFusedModule, _pack_freqs_cis

        torch.manual_seed(0)
        heads, head_dim, dim, rank = 2, 128, 256, 32
        lin = _packed_svdq_linear(dim, 3 * heads * head_dim, rank, dtype)
        norm_q = RMSNorm(head_dim, eps=1e-6).to("cuda", dtype)
        norm_k = RMSNorm(head_dim, eps=1e-6).to("cuda", dtype)
        with torch.no_grad():
            norm_q.weight.uniform_(0.5, 1.5)
            norm_k.weight.uniform_(0.5, 1.5)

        ref_lin = copy.deepcopy(lin)
        layout = load_kernels("layout")
        layout.convert_svdq_linear_to_natural_layout(ref_lin)

        fused = NunchakuZImageFusedModule(lin, norm_q, norm_k)
        fused.convert_to_natural_layout()
        for name in ("qweight", "wscales", "bias", "smooth_factor", "proj_down", "proj_up"):
            torch.testing.assert_close(getattr(fused, f"qkv_{name}"), getattr(ref_lin, name), rtol=0, atol=0)

        # forward: fused QKV GEMM + RMSNorm + RoPE epilogue vs the plain converted linear + torch reference
        batch, seq = 1, 300  # not a multiple of the quantizer's 256-row padding
        x = torch.randn(batch, seq, dim, dtype=dtype, device="cuda")
        freqs = _freqs_cis(seq).unsqueeze(0)
        out = fused(x, _pack_freqs_cis(freqs))

        qkv = ref_lin(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def ref_qk(t, norm):
            t = t.float().unflatten(-1, (heads, head_dim))
            t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight.float()
            return _zimage_rope(t, freqs).flatten(2)

        oq, ok, ov = out.chunk(3, dim=-1)
        torch.testing.assert_close(oq.float(), ref_qk(q, norm_q), rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(ok.float(), ref_qk(k, norm_k), rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(ov.float(), v.float(), rtol=2e-2, atol=2e-2)
    finally:
        set_backend(None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zimage_fused_swiglu_mlp(dtype):
    """The fused SwiGLU path (rows reordered with ``up_first=True``) matches the unfused diffusers ``SwiGLU`` path."""
    import copy

    pytest.importorskip("diffusers.models.transformers.transformer_z_image")
    from diffusers.models.transformers.transformer_z_image import FeedForward as ZImageFeedForward

    from nunchaku.ops.backend import set_backend

    set_backend("triton")
    try:
        from nunchaku.models.transformers.transformer_zimage import NunchakuZImageFeedForward
        from nunchaku.ops.fused import interleave_swiglu_linear_

        torch.manual_seed(0)
        dim, hidden, rank = 256, 512, 32
        ff = NunchakuZImageFeedForward(
            ZImageFeedForward(dim, hidden).to("cuda", dtype), precision="int4", rank=rank, torch_dtype=dtype
        )
        layout = load_kernels("layout")
        ff.net[0].proj = _packed_svdq_linear(dim, 2 * hidden, rank, dtype)
        ff.net[2] = _packed_svdq_linear(hidden, dim, rank, dtype)
        for lin in (ff.net[0].proj, ff.net[2]):
            layout.convert_svdq_linear_to_natural_layout(lin)

        x = torch.randn(1, 300, dim, dtype=dtype, device="cuda")
        ref = ff(x)  # unfused: proj -> SwiGLU (up * silu(gate)) -> quantize -> fc2

        fused = copy.deepcopy(ff)
        interleave_swiglu_linear_(fused.net[0].proj, up_first=True)
        fused.swiglu_fused = True
        out = fused(x)

        # Both paths re-quantize the SwiGLU output to int4; the native gfx11 epilogue skips two 16-bit roundings, so
        # a few codes flip and the paths differ by a few % (on Triton they are bit-identical). Score both against fc2
        # applied to the unquantized fp32 SwiGLU output instead (a wrong gate/up order is off by ~100%).
        up, gate = ff.net[0].proj(x).float().chunk(2, dim=-1)
        h = up * torch.nn.functional.silu(gate)
        fc2 = ff.net[2]
        w = _dequant_packed_nibbles(fc2.qweight.data, hidden) * fc2.wscales.float().t().repeat_interleave(64, dim=1)
        truth = (h / fc2.smooth_factor.float()) @ w.t() + (h @ fc2.proj_down.float()) @ fc2.proj_up.float().t()
        truth = truth + fc2.bias.float()

        def rel(a):
            return ((a.float() - truth).norm() / truth.norm()).item()

        assert rel(out) < 1.25 * rel(ref) + 1e-3, (rel(out), rel(ref))
        assert rel(ref) < 0.3, rel(ref)  # int4 re-quantization of random activations: ~0.17
    finally:
        set_backend(None)
