"""
Round-trip tests for the checkpoint layout inverter (`nunchaku/ops/triton/layout.py`).

These validate `unpack_weight`/`unpack_scale` against
`nunchaku.lora.flux.packer.NunchakuWeightPacker` -- the same packer already
used in production (`nunchaku/lora/flux/nunchaku_converter.py`) to merge LoRA
deltas into real checkpoints that run on the CUDA kernel. Since its output is
proven correct there, round-tripping through it is a complete correctness
check for the inverse with no GPU-specific arithmetic involved.
"""

import pytest
import torch

from .conftest import load_kernels

layout = load_kernels("layout")


@pytest.mark.parametrize("n,k", [(128, 128), (256, 256), (128, 512), (384, 128)])
def test_unpack_weight_int4(n, k):
    from nunchaku.lora.flux.packer import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    w = torch.randint(0, 16, (n, k), dtype=torch.int32, device="cuda")
    packed = packer.pack_weight(w)
    recovered = layout.unpack_weight(packed, n, k, bits=4)
    torch.testing.assert_close(recovered, w)


@pytest.mark.parametrize("n", [128, 256])
def test_unpack_scale_per_channel(n):
    from nunchaku.lora.flux.packer import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    s = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    packed = packer.pack_scale(s, group_size=-1)
    recovered = layout.unpack_scale(packed, n, group_size=-1, bits=4)
    torch.testing.assert_close(recovered.float(), s.float())


@pytest.mark.parametrize("n,k,group_size", [(128, 256, 64), (256, 512, 64)])
def test_unpack_scale_int4_grouped(n, k, group_size):
    from nunchaku.lora.flux.packer import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    ng = k // group_size
    s = torch.randn(n, ng, dtype=torch.bfloat16, device="cuda")
    packed = packer.pack_scale(s, group_size=group_size)
    recovered = layout.unpack_scale(packed, n, group_size=group_size, bits=4)
    # recovered is (k // group_size, n), matching SVDQW4A4Linear.wscales
    torch.testing.assert_close(recovered.float(), s.transpose(0, 1).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("n,k", [(128, 512), (256, 256)])
def test_unpack_micro_scale_nvfp4(n, k):
    from nunchaku.lora.flux.packer import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    group_size = 16
    ng = k // group_size
    s = (torch.rand(n, ng, device="cuda") * 400 - 200).to(torch.float16)
    packed = packer.pack_micro_scale(s, group_size=group_size)
    recovered = layout.unpack_scale(packed, n, group_size=group_size, bits=4)
    ref = s.to(torch.float8_e4m3fn).to(torch.float32).transpose(0, 1)
    torch.testing.assert_close(recovered.float(), ref)


def _dequant_packed_nibbles(qx, k, group_size, unsigned=False):
    """Dequantize a natural-layout packed-nibble int4 tensor, for test references."""
    b = qx.to(torch.int32) & 0xFF
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).reshape(qx.shape[0], -1)[:, :k]
    if unsigned:
        return codes.float()
    return torch.where(codes >= 8, codes - 16, codes).float()


def test_convert_svdq_linear_to_natural_layout():
    """
    Convert a synthetic checkpoint-format SVDQW4A4Linear and verify the
    converted tensors decode to the original logical values, and that a real
    forward pass through the converted module (quantize + Triton gemm_w4a4)
    matches a from-scratch reference computed from those logical values.

    ``qweight``/``wscales``/``bias``/``smooth_factor``/``proj_down``/``proj_up``
    are all fragment-packed on disk (confirmed against a real checkpoint --
    see ``convert_svdq_linear_to_natural_layout``'s docstring).
    """
    from nunchaku.lora.flux.packer import NunchakuWeightPacker
    from nunchaku.models.linear import SVDQW4A4Linear

    torch.manual_seed(0)
    device = "cuda"
    in_features, out_features, rank = 256, 128, 32

    lin = SVDQW4A4Linear(
        in_features,
        out_features,
        rank=rank,
        bias=True,
        precision="int4",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    packer = NunchakuWeightPacker(bits=4)
    n, k, group = out_features, in_features, lin.group_size

    q_logical = torch.randint(0, 16, (n, k), dtype=torch.int32, device=device)
    scale_logical = (torch.rand(n, k // group, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    bias_logical = torch.randn(n, dtype=torch.bfloat16, device=device)
    smooth_logical = (torch.rand(k, device=device) * 0.5 + 0.75).to(torch.bfloat16)
    proj_down_logical = torch.randn(k, rank, dtype=torch.bfloat16, device=device) * 0.1
    proj_up_logical = torch.randn(n, rank, dtype=torch.bfloat16, device=device) * 0.1

    lin.qweight.data = packer.pack_weight(q_logical)
    lin.wscales.data = packer.pack_scale(scale_logical, group_size=group)
    lin.bias.data = packer.pack_scale(bias_logical, group_size=-1)
    lin.smooth_factor.data = packer.pack_scale(smooth_logical, group_size=-1)
    # The real on-disk lora_down is packed from the *transposed* logical
    # tensor (verified against a real checkpoint's qkv_proj.lora_down); undo
    # that with a matching transpose in convert_svdq_linear_to_natural_layout.
    lin.proj_down.data = packer.pack_lowrank_weight(proj_down_logical.t().contiguous(), down=True)
    lin.proj_up.data = packer.pack_lowrank_weight(proj_up_logical, down=False)

    layout.convert_svdq_linear_to_natural_layout(lin)

    torch.testing.assert_close(lin.bias.data.float(), bias_logical.float())
    torch.testing.assert_close(lin.smooth_factor.data.float(), smooth_logical.float())
    torch.testing.assert_close(lin.proj_down.data.float(), proj_down_logical.float())
    torch.testing.assert_close(lin.proj_up.data.float(), proj_up_logical.float())
    torch.testing.assert_close(lin.wscales.data.float(), scale_logical.transpose(0, 1).float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(_dequant_packed_nibbles(lin.qweight.data, k, group, unsigned=True), q_logical.float())

    x = torch.randn(8, in_features, dtype=torch.bfloat16, device=device)
    quantized_x, ascales, lora_act = lin.quantize(x)
    out = lin.forward_quant(quantized_x, ascales, lora_act)

    a_deq = _dequant_packed_nibbles(quantized_x, in_features, group)
    grp = torch.arange(in_features, device=device) // group
    a_deq = a_deq * ascales.float()[grp].t()
    w_deq = _dequant_packed_nibbles(lin.qweight.data, k, group)
    w_deq = w_deq * scale_logical.float().repeat_interleave(group, dim=1)
    ref = a_deq @ w_deq.t() + lora_act.float() @ proj_up_logical.float().t() + bias_logical.float()
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)
