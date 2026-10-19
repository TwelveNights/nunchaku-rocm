"""
FLUX.2 model-level checks for the Triton backend.

The QKV RMSNorm+RoPE epilogue reads rotary embeddings in its own layout; FLUX.2's
``Flux2PosEmbed`` produces ``(cos, sin)`` with ``repeat_interleave_real=True``.
This checks the conversion in ``transformer_flux2._pack_rotary_emb`` against
diffusers' own ``apply_rotary_emb`` on the same positions.
"""

import pytest
import torch

from .conftest import load_kernels

w4a4 = load_kernels("gemm_w4a4")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flux2_rotary_matches_diffusers(dtype):
    pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

    from nunchaku.ops.backend import set_backend

    set_backend("triton")
    try:
        from nunchaku.models.transformers.transformer_flux2 import _pack_rotary_emb

        heads, head_dim = 4, 128
        # FLUX.2 ids are (T, 4); use a small image grid plus a text-like run
        grid = torch.stack(torch.meshgrid(torch.arange(6), torch.arange(5), indexing="ij"), -1).reshape(-1, 2)
        ids = torch.cat([torch.zeros(len(grid), 2), grid.float()], dim=1)
        ids = torch.cat([torch.stack([torch.zeros(7), torch.zeros(7), torch.zeros(7), torch.arange(7.0)], 1), ids])
        ids = ids.cuda()
        cos, sin = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])(ids)
        T = ids.shape[0]

        qkv = torch.randn(T, 3 * heads * head_dim, dtype=dtype, device="cuda")
        norm_q = torch.rand(head_dim, dtype=dtype, device="cuda") + 0.5
        norm_k = torch.rand(head_dim, dtype=dtype, device="cuda") + 0.5

        out = qkv.clone()
        w4a4._apply_rmsnorm_rope_qk(out, norm_q, norm_k, _pack_rotary_emb((cos, sin)))

        def ref_qk(x, w):
            x = x.float().view(1, T, heads, head_dim)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            return apply_rotary_emb(x, (cos, sin), sequence_dim=1).reshape(T, -1)

        q, k, v = qkv.chunk(3, dim=-1)
        oq, ok, ov = out.chunk(3, dim=-1)
        torch.testing.assert_close(oq.float(), ref_qk(q, norm_q), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(ok.float(), ref_qk(k, norm_k), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(ov, v)  # V is untouched
    finally:
        set_backend(None)
