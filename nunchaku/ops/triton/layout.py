"""
Converts checkpoint tensors from the CUDA kernel's mma-fragment layout to the
natural row-major layout the Triton kernels use.

Every SVDQuant checkpoint stores ``qweight``/``wscales`` already packed for the
CUDA kernel's ``mma.m16n8k32`` register layout -- there is no C++-side
repacking step at load time (``src/Module.h``'s ``loadParam`` is a raw
``copy_``), so the bytes on disk are exactly what
:class:`nunchaku.lora.flux.packer.NunchakuWeightPacker` produces. This module
implements the exact inverse of that packer, so a checkpoint can be loaded
once and converted to the natural layout
:func:`nunchaku.ops.triton.gemm_w4a4.gemm_w4a4` expects.

The forward packing is entirely a composition of ``reshape``/``permute`` (bit
packing aside), so the inverse is the corresponding un-permute followed by
unpacking nibbles/bytes -- there is no NVIDIA-specific arithmetic involved and
no GPU is required to validate it: :func:`unpack_weight` /
:func:`pack_weight_reference` round-trip on synthetic data is a complete
correctness check.
"""

import torch

from nunchaku.lora.flux.packer import MmaWeightPackerBase, NunchakuWeightPacker

__all__ = ["unpack_weight", "unpack_weight_packed_nibbles", "unpack_scale", "convert_svdq_linear_to_natural_layout"]


def _inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def unpack_weight(packed: torch.Tensor, n: int, k: int, bits: int = 4) -> torch.Tensor:
    """
    Invert :meth:`NunchakuWeightPacker.pack_weight`.

    Parameters
    ----------
    packed : torch.Tensor, shape (n, k * bits // 8), dtype int8
        Packed weight as stored in the checkpoint.
    n, k : int
        Logical output/input feature counts (post-padding shape used when the
        checkpoint was produced -- i.e. ``n``/``k`` must be the *padded*
        sizes, divisible by ``mem_n``/``mem_k * num_k_unrolls``).
    bits : int, default 4
        Quantization bit width; only 4 (int4/nvfp4) is exercised by SVDQuant.

    Returns
    -------
    torch.Tensor, shape (n, k), dtype int32
        Logical quantized weight, values in ``[0, 2**bits)`` (raw nibble/byte
        codes, not yet sign-interpreted or dequantized).
    """
    packer = NunchakuWeightPacker(bits=bits)
    assert n % packer.mem_n == 0, f"n ({n}) must be divisible by mem_n ({packer.mem_n})"
    assert (
        k % (packer.mem_k * packer.num_k_unrolls) == 0
    ), f"k ({k}) must be divisible by mem_k * num_k_unrolls ({packer.mem_k * packer.num_k_unrolls})"
    n_tiles, k_tiles = n // packer.mem_n, k // packer.mem_k

    # packed is (n, k * bits // 8) int8; the packer's last step was
    # `.view(dtype=torch.int8).view(n, -1)` from an int32 tensor of shape
    # (n_tiles, k_tiles, num_k_packs, num_n_packs, num_n_lanes, num_k_lanes,
    #  n_pack_size, k_pack_size) -- i.e. 4 int8 bytes per int32 "reg_k" word.
    words = packed.view(torch.int32).view(
        n_tiles,
        k_tiles,
        packer.num_k_packs,
        packer.num_n_packs,
        packer.num_n_lanes,
        packer.num_k_lanes,
        packer.n_pack_size,
        packer.k_pack_size,
    )

    # undo the bit-packing: each int32 word holds `reg_k` nibbles/bytes,
    # little-endian, at `4 * bits`-bit-wide lanes.
    reg_k = packer.reg_k
    shift = torch.arange(0, 32, bits, dtype=torch.int32, device=packed.device)
    mask = (1 << bits) - 1
    unpacked = ((words.unsqueeze(-1) >> shift) & mask).to(torch.int32)
    # unpacked: (..., n_pack_size, k_pack_size, reg_k)
    assert unpacked.shape[-1] == reg_k

    # undo `.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9)`, i.e. the forward mapping
    # from (n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n, k_tiles,
    #       num_k_packs, k_pack_size, num_k_lanes, reg_k)
    # to   (n_tiles, k_tiles, num_k_packs, num_n_packs, num_n_lanes,
    #       num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k).
    # reg_n has size 1 and was squeezed out implicitly by the view; reinstate
    # it as a unit axis so the permutation is exact.
    unpacked = unpacked.unsqueeze(-2)  # insert reg_n=1 axis before reg_k
    forward_perm = (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)
    inverse_perm = _inverse_permutation(forward_perm)
    weight = unpacked.permute(*inverse_perm).contiguous()
    # weight is now (n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n,
    #                k_tiles, num_k_packs, k_pack_size, num_k_lanes, reg_k)
    weight = weight.reshape(n, k)
    return weight


def unpack_weight_packed_nibbles(packed: torch.Tensor, n: int, k: int, bits: int = 4) -> torch.Tensor:
    """
    Like :func:`unpack_weight`, but returns the natural nibble-packed byte
    layout :func:`nunchaku.ops.triton.gemm_w4a4.gemm_w4a4` reads directly:
    shape ``(n, k // 2)`` uint8, with logical column ``2j`` in the low nibble
    of byte ``j`` and column ``2j + 1`` in the high nibble (see
    ``_load_nibbles`` in ``gemm_w4a4.py``).

    Only ``bits=4`` is supported, since that packed-nibble layout is specific
    to 4-bit codes.
    """
    assert bits == 4, "packed-nibble output is only defined for 4-bit codes"
    codes = unpack_weight(packed, n, k, bits=bits)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_scale(packed: torch.Tensor, n: int, group_size: int, bits: int = 4, warp_n: int = 128) -> torch.Tensor:
    """
    Invert :meth:`NunchakuWeightPacker.pack_scale` / ``pack_micro_scale``.

    Parameters
    ----------
    packed : torch.Tensor
        Packed scale tensor as stored in the checkpoint, shape
        ``(k // group_size, n)`` (or ``(n,)`` flattened for ``group_size=-1``).
    n : int
        Output feature count (padded to a multiple of ``warp_n``).
    group_size : int
        Quantization group size (16 for NVFP4 micro-scale, 64 for INT4, or -1
        for a single per-channel scale).
    bits : int, default 4
        Quantization bit width, used only to construct the reference packer
        (determines whether the micro-scale branch applies).
    warp_n : int, default 128
        Must match the value used when the checkpoint was packed.

    Returns
    -------
    torch.Tensor
        Logical scale tensor, shape ``(n,)`` for ``group_size=-1``, else
        ``(k // group_size, n)`` matching :attr:`SVDQW4A4Linear.wscales`.
    """
    packer = NunchakuWeightPacker(bits=bits, warp_n=warp_n)
    if packer.check_if_micro_scale(group_size=group_size):
        return _unpack_micro_scale(packed, n, group_size, packer)
    return _unpack_scale_standard(packed, n, group_size, packer)


def _unpack_scale_standard(packed: torch.Tensor, n: int, group_size: int, packer: MmaWeightPackerBase) -> torch.Tensor:
    s_pack_size = min(max(packer.warp_n // packer.num_lanes, 2), 8)
    num_s_lanes = min(packer.num_lanes, packer.warp_n // s_pack_size)
    num_s_packs = packer.warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == packer.warp_n

    n_groups = n // warp_s
    k_groups = 1 if group_size == -1 else packed.numel() // n

    # forward: scale.reshape(n_groups, num_s_packs, num_s_lanes//4, s_pack_size//2, 4, 2, k_groups)
    #                .permute(0, 6, 1, 2, 4, 3, 5)
    forward_perm = (0, 6, 1, 2, 4, 3, 5)
    inverse_perm = _inverse_permutation(forward_perm)

    stored = packed.reshape(
        n_groups,
        k_groups,
        num_s_packs,
        num_s_lanes // 4,
        4,
        s_pack_size // 2,
        2,
    )
    scale = stored.permute(*inverse_perm).contiguous()
    # scale is now (n_groups, num_s_packs, num_s_lanes//4, s_pack_size//2, 4, 2, k_groups)
    scale = scale.reshape(n, k_groups)
    if group_size == -1:
        return scale.reshape(n)
    return scale.transpose(0, 1).contiguous()  # (k_groups, n), matching wscales' documented shape


def _unpack_micro_scale(packed: torch.Tensor, n: int, group_size: int, packer: MmaWeightPackerBase) -> torch.Tensor:
    assert group_size == 16
    assert packer.insn_k == 64
    s_pack_size = min(max(packer.warp_n // packer.num_lanes, 1), 4)
    num_s_lanes = 4 * 8
    num_s_packs = -(-packer.warp_n // (s_pack_size * num_s_lanes))  # ceil_divide
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == packer.warp_n

    n_groups = n // warp_s
    insn_k_groups = packer.insn_k // group_size
    # `X` is the packer's own `-1` dim: total k-groups (ng) split into
    # (X, insn_k_groups) before the forward permute.
    x = packed.numel() // n // insn_k_groups

    # forward: scale.view(n_groups, num_s_packs, s_pack_size, 4, 8, X, insn_k_groups)
    #                .permute(0, 5, 1, 4, 3, 2, 6)
    # `.view(-1, n)` afterwards is a pure reinterpretation of the same
    # contiguous buffer (the packer's own comment: "shape is just used for
    # validation"), so `packed` already holds the POST-permute memory layout
    # below -- reshape directly into it, then undo the permutation.
    forward_perm = (0, 5, 1, 4, 3, 2, 6)
    inverse_perm = _inverse_permutation(forward_perm)

    stored = packed.reshape(
        n_groups,
        x,
        num_s_packs,
        8,
        4,
        s_pack_size,
        insn_k_groups,
    )
    scale = stored.permute(*inverse_perm).contiguous()
    # scale is now (n_groups, num_s_packs, s_pack_size, 4, 8, X, insn_k_groups)
    scale = scale.reshape(n, x * insn_k_groups)
    return scale.transpose(0, 1).contiguous()


def convert_svdq_linear_to_natural_layout(linear) -> None:
    """
        Convert one :class:`~nunchaku.models.linear.SVDQW4A4Linear`'s parameters
        in place from the CUDA kernel's mma-fragment checkpoint layout to the
        natural layout the Triton ``gemm_w4a4`` kernel reads.

    Converts ``qweight``, ``wscales``, ``bias``, ``smooth_factor``, and
        ``wcscales`` -- confirmed fragment-packed on disk by direct comparison
        against a real checkpoint (``nunchaku-ai/nunchaku-flux.1-schnell``): the
        raw ``qkv_proj.bias`` differs element-for-element from a full-precision
        reference model's equivalent bias only by a fixed permutation, and
        :func:`unpack_scale` (``group_size=-1``) inverts that permutation exactly
        (zero error). ``bias``/``smooth_factor``/``wcscales`` are read through the
        same ``packed_wscale_t`` accessor as ``wscales`` in the CUDA kernel
        (``src/kernels/zgemm/gemm_w4a4_launch_impl.cuh``), which is consistent
        with them sharing its on-disk layout.

    ``proj_down``/``proj_up`` are also fragment-packed, inverted via
        :meth:`NunchakuWeightPacker.unpack_lowrank_weight` (a different, simpler
        permutation than ``pack_weight``'s). Confirmed indirectly: a real
        checkpoint's LoRA branch, added on top of the main dequantized GEMM path,
        should reduce reconstruction error against the full-precision reference
        weight (that's the LoRA branch's purpose); before this fix it increased
        the error instead, and `unpack_lowrank_weight` visibly permutes the raw
        tensor (it is not a no-op on real data).

        ``smooth_factor_orig`` is unused by the GEMM and left untouched.

        Idempotency: calling this twice on the same module would corrupt its
        weights (the inverse of an inverse is the forward pack, not a no-op), so
        callers must call this exactly once per loaded checkpoint, before any
        forward pass.
    """
    n, half_k = linear.qweight.shape
    k = half_k * 2
    bits = 4

    with torch.no_grad():
        linear.qweight.data = (
            unpack_weight_packed_nibbles(linear.qweight.data, n, k, bits=bits).to(torch.uint8).view(torch.int8)
        )
        linear.wscales.data = unpack_scale(linear.wscales.data, n, group_size=linear.group_size, bits=bits).to(
            linear.wscales.dtype
        )

        if linear.bias is not None:
            linear.bias.data = unpack_scale(linear.bias.data.reshape(1, -1), n, group_size=-1, bits=bits).to(
                linear.bias.dtype
            )

        linear.smooth_factor.data = unpack_scale(
            linear.smooth_factor.data.reshape(1, -1), k, group_size=-1, bits=bits
        ).to(linear.smooth_factor.dtype)

        if linear.wcscales is not None:
            linear.wcscales.data = unpack_scale(linear.wcscales.data.reshape(1, -1), n, group_size=-1, bits=bits).to(
                linear.wcscales.dtype
            )

        packer = NunchakuWeightPacker(bits=bits)
        if linear.proj_down.numel() > 0:
            # unpack_lowrank_weight's `down=True` branch returns (rank, in_features)
            # for this on-disk tensor, not (in_features, rank); transpose to match
            # SVDQW4A4Linear's own convention (verified against a real checkpoint).
            linear.proj_down.data = packer.unpack_lowrank_weight(linear.proj_down.data, down=True).t().contiguous()
        if linear.proj_up.numel() > 0:
            linear.proj_up.data = packer.unpack_lowrank_weight(linear.proj_up.data, down=False)
