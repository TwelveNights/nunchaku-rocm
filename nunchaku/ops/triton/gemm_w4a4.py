"""
Triton port of the SVDQuant W4A4 GEMM (``src/kernels/zgemm/gemm_w4a4.cu``).

Layout
------
The CUDA kernel stores activations, weights and scales in layouts that mirror
the register distribution of the ``mma.m16n8k32`` instruction (see the long
comments in ``gemm_base.cuh``). Those layouts are meaningless off NVIDIA
tensor cores, so this backend uses the *natural* layouts that the public
signatures already advertise:

``act``    ``[M, K // 2]`` uint8 -- element ``k`` is the low nibble of byte
           ``k // 2`` when ``k`` is even, the high nibble when odd.
``wgt``    ``[N, K // 2]`` uint8, same nibble order.
``ascales`` ``[K // group, M]``; ``wscales`` ``[K // group, N]``.

For INT4 a nibble is a two's-complement signed value in ``[-8, 7]`` (or
unsigned ``[0, 15]`` when ``act_unsigned``). For NVFP4 it is an ``e2m1``
float, decoded through :func:`_decode_e2m1`.

Existing SVDQuant checkpoints store ``qweight``/``wscales`` (and ``bias``/
``smooth_factor``/``wcscales``) in the CUDA kernel's mma-fragment layout, which
is *not* the natural layout above. Convert a loaded
:class:`~nunchaku.models.linear.SVDQW4A4Linear` once with
:func:`nunchaku.ops.triton.layout.convert_svdq_linear_to_natural_layout` before
running it through this backend.

Fused epilogues
----------------
``norm_q``/``norm_k``/``rotary_emb`` fuse a per-head RMSNorm (``eps=1e-6``,
matching FLUX's ``torch.nn.RMSNorm(dim_head, eps=1e-6)``) and RoPE rotation
into the GEMM output, applied only to the Q and K thirds of the output
channels (``N`` is split into three equal, consecutive blocks -- Q, K, V --
matching ``qkv.chunk(3, dim=-1)`` in
``nunchaku/models/attention_processors/flux.py``); V passes through
unchanged. This matches ``EpilogueRMSNormRope`` in
``src/kernels/zgemm/epilogues.cuh``. ``rotary_emb`` is consumed in its
*natural* pre-``pack_rotemb`` layout, ``[M, HEAD_DIM // 2, 1, 2]`` float32
(last dim is ``(sin, cos)`` per the CUDA epilogue's
``ROTARY_EMB_NUM_ELEMENTS == 2`` branch and ``nunchaku.models.embeddings.rope``)
-- the CUDA-only ``pack_rotemb`` fragment layout is neither produced nor
consumed by this backend.

``qout``/``oscales``/``smooth_factor``/``lora_down``/``lora_act_out`` fuse the
next layer's activation quantization (with an optional GELU and the int4
``+0.171875`` unsigned-shift trick) directly onto this GEMM's output tile,
matching ``EpilogueQuantize`` composed after ``EpilogueGelu`` in the CUDA
launch path (``gemm_w4a4_launch_impl.cuh``).
"""

import torch
import triton
import triton.language as tl

from ._common import gelu_tanh, matmul_group_m, num_warps_for, silu_impl

__all__ = [
    "gemm_w4a4",
    "quantize_w4a4_act",
    "quantize_w4a4_act_fuse_lora",
    "quantize_w4a4_wgt",
]

# e2m1 (NVFP4) decode table, indexed by the raw 4-bit pattern.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_E2M1_MAX = 6.0


@triton.jit
def _decode_e2m1(q):
    """Decode a 4-bit ``e2m1`` code to fp32."""
    sign = tl.where((q & 0x8) != 0, -1.0, 1.0)
    mag = q & 0x7
    # magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6
    val = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )
    return sign * val


@triton.jit
def _encode_e2m1(x):
    """Encode fp32 to the nearest ``e2m1`` code (round half to even)."""
    sign_bit = tl.where(x < 0, 8, 0)
    a = tl.abs(x)
    # midpoints between representable magnitudes: .25 .75 1.25 1.75 2.5 3.5 5
    code = tl.where(
        a < 0.25,
        0,
        tl.where(
            a < 0.75,
            1,
            tl.where(
                a < 1.25,
                2,
                tl.where(a < 1.75, 3, tl.where(a < 2.5, 4, tl.where(a < 3.5, 5, tl.where(a < 5.0, 6, 7)))),
            ),
        ),
    )
    return (code | sign_bit).to(tl.int32)


@triton.jit
def _load_nibbles(ptr, rows, byte_cols, stride_row, mask, NCOL: tl.constexpr):
    """
    Load a ``[len(rows), 2 * NCOL]`` tile of 4-bit codes.

    ``byte_cols`` indexes *bytes*, so the loads are contiguous along the row --
    each byte yields two logical columns, the low nibble first. Gathering with
    a ``col // 2`` index instead would issue twice the loads at half the width.
    """
    byte = tl.load(ptr + rows * stride_row + byte_cols, mask=mask, other=0).to(tl.int32) & 0xFF
    lo = byte & 0xF
    hi = (byte >> 4) & 0xF
    # interleave: [.., NCOL, 2] -> [.., 2 * NCOL] so column 2j/2j+1 = lo/hi of byte j
    pair = tl.join(lo, hi)
    return tl.reshape(pair, (pair.shape[0], 2 * NCOL))


def _w4a4_configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": matmul_group_m()},
            num_stages=ns,
            num_warps=num_warps_for(nw),
        )
        for bm, bn, ns, nw in [
            (128, 128, 3, 8),
            (128, 256, 3, 8),
            (256, 128, 3, 8),
            (64, 128, 4, 4),
            (128, 64, 4, 4),
            (64, 64, 4, 4),
        ]
    ]


@triton.autotune(configs=_w4a4_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_w4a4_kernel(
    act_ptr,
    wgt_ptr,
    out_ptr,
    ascale_ptr,
    wscale_ptr,
    bias_ptr,
    lora_act_ptr,
    lora_up_ptr,
    wcscale_ptr,
    smooth_ptr,
    M,
    N,
    K,
    R,
    actual_m,
    alpha,
    stride_am,
    stride_wn,
    stride_asg,
    stride_wsg,
    stride_om,
    stride_lam,
    stride_lun,
    ACT_UNSIGNED: tl.constexpr,
    USE_FP4: tl.constexpr,
    USE_INT8_DOT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LORA: tl.constexpr,
    HAS_WCSCALE: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    # act/ascales/lora_act_in are indexed up to M (they may be padded beyond
    # the caller's real batch size); the output store is bounded by actual_m,
    # the real number of rows the caller's `out` buffer holds.
    out_m_mask = offs_m < actual_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # One dot per scale group; the per-group scales are a rank-1 product
    # applied to the fp32 partial result, which keeps their full precision
    # instead of rounding them into the dot.
    #
    # For plain int4 (USE_INT8_DOT), decoded codes are kept as int8 and
    # ``tl.dot`` runs int8xint8->int32, which lowers to a native low-precision
    # matrix instruction (``v_mfma_i32_16x16x32_i8`` on CDNA3,
    # ``v_wmma_i32_16x16x16_iu8`` on RDNA3) -- unlike bf16xbf16, which uses a
    # slower matrix-core mode on both architectures. Int4 codes (signed
    # [-8, 7] or unsigned [0, 15]) are exactly representable in int8, so this
    # changes only speed, not numerics. NVFP4 (``USE_FP4``) decodes to
    # non-integer ``e2m1`` floats and always uses the bf16/fp16 dot below.
    for k0 in tl.range(0, K, GROUP_K):
        offs_k = k0 + tl.arange(0, GROUP_K)
        k_mask = offs_k < K
        offs_b = k0 // 2 + tl.arange(0, GROUP_K // 2)
        b_mask = offs_b < K // 2

        qa = _load_nibbles(
            act_ptr, offs_m[:, None], offs_b[None, :], stride_am,
            m_mask[:, None] & b_mask[None, :], GROUP_K // 2,
        )
        qw = _load_nibbles(
            wgt_ptr, offs_n[:, None], offs_b[None, :], stride_wn,
            n_mask[:, None] & b_mask[None, :], GROUP_K // 2,
        )

        if USE_FP4:
            a = _decode_e2m1(qa)
            w = _decode_e2m1(qw)
        else:
            if ACT_UNSIGNED:
                a = qa.to(tl.float32)
            else:
                a = tl.where(qa >= 8, qa - 16, qa).to(tl.float32)
            w = tl.where(qw >= 8, qw - 16, qw).to(tl.float32)

        if USE_INT8_DOT:
            a = tl.where(k_mask[None, :], a, 0.0).to(tl.int8)
            w = tl.where(k_mask[None, :], w, 0.0).to(tl.int8)
        else:
            a = tl.where(k_mask[None, :], a, 0.0).to(DOT_DTYPE)
            w = tl.where(k_mask[None, :], w, 0.0).to(DOT_DTYPE)

        grp = k0 // GROUP_K
        asc = tl.load(ascale_ptr + grp * stride_asg + offs_m, mask=m_mask, other=0.0).to(tl.float32)
        wsc = tl.load(wscale_ptr + grp * stride_wsg + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        if USE_INT8_DOT:
            dot = tl.dot(a, tl.trans(w), out_dtype=tl.int32).to(tl.float32)
        else:
            dot = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        acc += dot * asc[:, None] * wsc[None, :]

    if HAS_WCSCALE:
        acc *= tl.load(wcscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]

    acc *= alpha

    # low-rank correction: out += lora_act @ lora_up.T
    if HAS_LORA:
        lacc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for r0 in tl.range(0, R, 16):
            offs_r = r0 + tl.arange(0, 16)
            r_mask = offs_r < R
            la = tl.load(
                lora_act_ptr + offs_m[:, None] * stride_lam + offs_r[None, :],
                mask=m_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            lu = tl.load(
                lora_up_ptr + offs_n[:, None] * stride_lun + offs_r[None, :],
                mask=n_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            lacc = tl.dot(la, tl.trans(lu), lacc)
        acc += lacc

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]

    if FUSE_SILU:
        acc = silu_impl(acc)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_m_mask[:, None] & n_mask[None, :])


HEAD_DIM = 128


@triton.jit
def _rmsnorm_rope_qk_kernel(
    out_ptr,
    norm_q_ptr,
    norm_k_ptr,
    rotary_emb_ptr,
    M,
    N,
    epsilon,
    stride_om,
    stride_rm,
    HEAD_DIM: tl.constexpr,
):
    """
    In-place fused per-head RMSNorm + RoPE on the Q and K thirds of a QKV
    projection's output, matching ``EpilogueRMSNormRope``. One program handles
    one (row, head) pair; V (the last third of ``N``) is left untouched.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    is_k = tl.program_id(2) == 1

    third = N // 3
    base_n = (third + head * HEAD_DIM) if is_k else (head * HEAD_DIM)

    offs_d = tl.arange(0, HEAD_DIM)
    x = tl.load(out_ptr + row.to(tl.int64) * stride_om + base_n + offs_d).to(tl.float32)

    norm_w = tl.load((norm_k_ptr if is_k else norm_q_ptr) + offs_d).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + epsilon)
    x = x * rstd * norm_w

    # rotary_emb is [M, HEAD_DIM // 2, 1, 2] fp32, last dim (sin, cos);
    # applies identically to every head, indexed by (row, channel-pair)
    pair = offs_d // 2
    sin = tl.load(rotary_emb_ptr + row.to(tl.int64) * stride_rm + pair * 2 + 0)
    cos = tl.load(rotary_emb_ptr + row.to(tl.int64) * stride_rm + pair * 2 + 1)

    is_even = (offs_d % 2) == 0
    # gather the paired element: even lanes want offs_d+1, odd lanes want offs_d-1
    partner = tl.where(is_even, offs_d + 1, offs_d - 1)
    x_partner = tl.load(out_ptr + row.to(tl.int64) * stride_om + base_n + partner).to(tl.float32)
    x_partner = x_partner * rstd * tl.load((norm_k_ptr if is_k else norm_q_ptr) + partner).to(tl.float32)

    x_even = tl.where(is_even, x, x_partner)
    x_odd = tl.where(is_even, x_partner, x)
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out_val = tl.where(is_even, rot_even, rot_odd)

    tl.store(out_ptr + row.to(tl.int64) * stride_om + base_n + offs_d, out_val.to(out_ptr.dtype.element_ty))


def _apply_rmsnorm_rope_qk(
    out: torch.Tensor,
    norm_q: torch.Tensor,
    norm_k: torch.Tensor,
    rotary_emb: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """
    Apply fused per-head RMSNorm + RoPE to the Q and K thirds of ``out`` in
    place, matching ``EpilogueRMSNormRope``. ``out`` is ``(M, N)`` with ``N``
    split into three equal Q/K/V thirds; ``rotary_emb`` is the natural,
    pre-``pack_rotemb`` tensor ``[M, HEAD_DIM // 2, 1, 2]`` (last dim
    ``(sin, cos)``), reshaped here to ``[M, HEAD_DIM]``.
    """
    M, N = out.shape
    third = N // 3
    assert third % HEAD_DIM == 0, f"Q/K/V third ({third}) must be a multiple of HEAD_DIM ({HEAD_DIM})"
    num_heads = third // HEAD_DIM
    rot = rotary_emb.reshape(M, HEAD_DIM)

    grid = (M, num_heads, 2)  # axis 2: 0 = Q, 1 = K
    _rmsnorm_rope_qk_kernel[grid](
        out,
        norm_q,
        norm_k,
        rot,
        M,
        N,
        epsilon,
        out.stride(0),
        rot.stride(0),
        HEAD_DIM=HEAD_DIM,
    )


def gemm_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    wcscales: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    act_unsigned: bool = False,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    W4A4 GEMM with per-group dequantization and a fused low-rank correction.

    Computes ``out = alpha * (deq(act) @ deq(wgt).T) + lora_act @ lora_up.T + bias``.
    When ``norm_q``/``norm_k``/``rotary_emb`` are given (a QKV projection),
    additionally fuses a per-head RMSNorm + RoPE onto the Q and K thirds of
    the result; see the module docstring and ``_apply_rmsnorm_rope_qk``.

    When ``qout``/``oscales`` are given (an MLP's first layer, GELU
    activated), additionally applies a tanh-approximate GELU, the int4
    unsigned-activation shift (``+0.171875``, skipped for NVFP4), and
    per-group quantizes the result into ``qout``/``oscales`` for the next
    layer -- optionally also projecting the (GELU'd only -- not shifted or
    smoothed) activation through ``lora_down`` into ``lora_act_out``, fusing
    the next layer's LoRA down-projection. Matches ``EpilogueGelu`` +
    ``EpilogueQuantize`` + ``EpilogueLoraDown`` in the CUDA launch path.

    Parameters
    ----------
    act : (M, K // 2) uint8/int8, packed 4-bit activations
    wgt : (N, K // 2) uint8/int8, packed 4-bit weights
    ascales : (K // group, M) per-group activation scales
    wscales : (K // group, N) per-group weight scales
    lora_act_in : (M, R) float32, the low-rank hidden state
    lora_up : (N, R) up-projection
    wcscales : (N,) per-channel weight scale, NVFP4 only
    norm_q, norm_k : (HEAD_DIM,) RMSNorm weights for Q/K
    rotary_emb : (M, HEAD_DIM // 2, 1, 2) float32, natural (unpacked) layout
    qout : (M, N // 2) uint8, packed next-layer activations, written in place
    oscales : (N // group, M), next-layer per-group activation scales
    smooth_factor : (N,), next layer's smoothing factor
    lora_down : (N, R'), next layer's LoRA down-projection weight
    lora_act_out : (M, R'), next layer's LoRA hidden state, written in place
    fp4 : decode nibbles as ``e2m1`` instead of int4
    alpha : global output scale (NVFP4 weight tensor scale)
    """
    assert ascales is not None and wscales is not None, "W4A4 GEMM requires both scale tensors"

    act = act.reshape(-1, act.shape[-1])
    M = act.shape[0]
    K = act.shape[1] * 2
    N = wgt.shape[0]
    assert wgt.shape[1] * 2 == K, f"weight K mismatch: {wgt.shape[1] * 2} vs {K}"

    group_size = 16 if fp4 else 64
    assert K % group_size == 0

    if out is None:
        out = torch.empty(M, N, dtype=out_dtype, device=act.device)
    out2d = out.reshape(-1, N)
    # `act` (and ascales/lora_act_in) may be padded to more rows than the
    # caller's output buffer holds -- e.g. fused_gelu_mlp pads its quantized
    # intermediate to a multiple of 256 rows, but the final output only needs
    # the real batch size. Compute over the full M but only read/write the
    # rows that exist in every M-indexed buffer.
    actual_m = out2d.shape[0]
    assert actual_m <= M, f"out has more rows ({actual_m}) than act ({M})"

    has_lora = lora_act_in is not None and lora_up is not None
    R = lora_up.shape[1] if has_lora else 0

    # the K loop steps exactly one scale group at a time, so a group is never
    # straddled for either int4 (64) or nvfp4 (16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _gemm_w4a4_kernel[grid](
        act,
        wgt,
        out2d,
        ascales,
        wscales,
        bias if bias is not None else act,
        lora_act_in if has_lora else act,
        lora_up if has_lora else act,
        wcscales if wcscales is not None else act,
        act,
        M,
        N,
        K,
        R,
        actual_m,
        alpha if alpha is not None else 1.0,
        act.stride(0),
        wgt.stride(0),
        ascales.stride(0),
        wscales.stride(0),
        out2d.stride(0),
        lora_act_in.stride(0) if has_lora else 0,
        lora_up.stride(0) if has_lora else 0,
        ACT_UNSIGNED=act_unsigned,
        USE_FP4=fp4,
        USE_INT8_DOT=not fp4,
        HAS_BIAS=bias is not None,
        HAS_LORA=has_lora,
        HAS_WCSCALE=wcscales is not None,
        FUSE_SILU=fuse_silu,
        DOT_DTYPE=tl.float16 if out2d.dtype == torch.float16 else tl.bfloat16,
        GROUP_K=group_size,
    )

    if norm_q is not None or norm_k is not None or rotary_emb is not None:
        assert norm_q is not None and norm_k is not None and rotary_emb is not None, (
            "norm_q, norm_k, and rotary_emb must all be given together"
        )
        _apply_rmsnorm_rope_qk(out2d, norm_q, norm_k, rotary_emb)

    if qout is not None or oscales is not None:
        assert qout is not None and oscales is not None, "qout and oscales must be given together"
        shift_value = 0.0 if fp4 else 0.171875
        _launch_quantize(
            out2d,
            qout,
            oscales,
            lora_down,
            lora_act_out,
            smooth_factor,
            fuse_glu=False,
            fp4=fp4,
            fuse_gelu=True,
            shift_value=shift_value,
            unsigned=not fp4,
        )

    return out




@triton.jit
def _rint(x):
    """Round half to even, matching ``cvt.rni``."""
    f = tl.floor(x)
    d = x - f
    even = (tl.floor(f * 0.5) * 2.0) == f
    up = tl.where(d == 0.5, tl.where(even, 0.0, 1.0), tl.where(d > 0.5, 1.0, 0.0))
    return f + up


@triton.jit
def _quantize_row_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    lora_down_ptr,
    lora_act_ptr,
    smooth_ptr,
    M,
    K,
    R,
    shift_value,
    stride_im,
    stride_om,
    stride_sg,
    stride_ld,
    stride_lam,
    GROUP: tl.constexpr,
    USE_FP4: tl.constexpr,
    UNSIGNED: tl.constexpr,
    FUSE_GLU: tl.constexpr,
    FUSE_GELU: tl.constexpr,
    HAS_SMOOTH: tl.constexpr,
    HAS_LORA: tl.constexpr,
    NGROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Quantize one row of activations, one scale group per lane-block.

    Each program handles a single row so that the per-group max reduction is a
    plain 2-D reduce over a ``[NGROUP, GROUP]`` tile rather than a segmented
    scan. A row is at most a few thousand elements, which fits comfortably.
    """
    row = tl.program_id(0).to(tl.int64)

    offs_g = tl.arange(0, NGROUP)[:, None]
    offs_e = tl.arange(0, GROUP)[None, :]
    offs_k = offs_g * GROUP + offs_e
    k_mask = offs_k < K

    x = tl.load(in_ptr + row * stride_im + offs_k, mask=k_mask, other=0.0).to(tl.float32)
    if FUSE_GLU:
        gate = tl.load(in_ptr + row * stride_im + K + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x = x * (gate / (1.0 + tl.exp(-gate)))
    if FUSE_GELU:
        x = gelu_tanh(x)
    x += shift_value
    if HAS_SMOOTH:
        x = x / tl.load(smooth_ptr + offs_k, mask=k_mask, other=1.0).to(tl.float32)

    # low-rank down-projection consumes the *unquantized, unsmoothed,
    # unshifted* row (GELU/GLU applied if fused, nothing else): in the CUDA
    # kernel (quantize_w4a4_fuse_lora_kernel), EpilogueLoraDown runs directly
    # on the loaded activation fpsum, strictly before EpilogueQuantize (which
    # applies the shift and smoothing). Confirmed by diffing against a
    # full-precision reference: including the smoothing division here made
    # the LoRA contribution's magnitude wrong by a large layer-dependent
    # factor (as much as ~3.4x for an outlier-heavy layer) while its
    # direction stayed correct (corr ~0.996) -- the signature of a spurious
    # elementwise scale, not an orientation/permutation bug.
    if HAS_LORA:
        offs_r = tl.arange(0, BLOCK_R)
        acc = tl.zeros((BLOCK_R,), tl.float32)
        for k0 in tl.range(0, K, GROUP):
            kk = k0 + tl.arange(0, GROUP)
            km = kk < K
            xv = tl.load(in_ptr + row * stride_im + kk, mask=km, other=0.0).to(tl.float32)
            if FUSE_GLU:
                gv = tl.load(in_ptr + row * stride_im + K + kk, mask=km, other=0.0).to(tl.float32)
                xv = xv * (gv / (1.0 + tl.exp(-gv)))
            if FUSE_GELU:
                xv = gelu_tanh(xv)
            w = tl.load(
                lora_down_ptr + kk[:, None] * stride_ld + offs_r[None, :],
                mask=km[:, None] & (offs_r[None, :] < R),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(xv[:, None] * w, axis=0)
        tl.store(lora_act_ptr + row * stride_lam + offs_r, acc, mask=offs_r < R)

    # per-group scale: symmetric [-qmax, qmax] normally, or [0, qmax] when the
    # row is known non-negative (post-GELU-shift activations feeding an
    # unsigned-int4 next layer) -- matches EpilogueQuantize's USE_UNSIGNED.
    if USE_FP4:
        qmax = 6.0
    elif UNSIGNED:
        qmax = 15.0
    else:
        qmax = 7.0
    if UNSIGNED:
        gmax = tl.max(tl.where(k_mask, x, 0.0), axis=1)
    else:
        gmax = tl.max(tl.where(k_mask, tl.abs(x), 0.0), axis=1)
    scale = gmax / qmax
    scale_safe = tl.where(scale == 0.0, 1.0, scale)

    tl.store(
        scale_ptr + tl.arange(0, NGROUP) * stride_sg + row,
        scale.to(scale_ptr.dtype.element_ty),
        mask=tl.arange(0, NGROUP) < tl.cdiv(K, GROUP),
    )

    q = x / scale_safe[:, None]
    if USE_FP4:
        code = _encode_e2m1(q)
    elif UNSIGNED:
        code = tl.minimum(tl.maximum(_rint(q), 0.0), 15.0).to(tl.int32) & 0xF
    else:
        code = tl.minimum(tl.maximum(_rint(q), -8.0), 7.0).to(tl.int32) & 0xF

    # pack nibble pairs: lane j of the packed row holds logical 2j (low) and
    # 2j+1 (high). Reshape to [.., 2] and combine so each output byte is
    # produced by exactly one lane -- no atomics, no races.
    code = tl.reshape(code, (NGROUP, GROUP // 2, 2))
    lo = tl.reshape(tl.sum(tl.where(tl.arange(0, 2)[None, None, :] == 0, code, 0), axis=2), (NGROUP, GROUP // 2))
    hi = tl.reshape(tl.sum(tl.where(tl.arange(0, 2)[None, None, :] == 1, code, 0), axis=2), (NGROUP, GROUP // 2))
    packed = (lo | (hi << 4)).to(tl.uint8)

    offs_bg = tl.arange(0, NGROUP)[:, None]
    offs_be = tl.arange(0, GROUP // 2)[None, :]
    offs_b = offs_bg * (GROUP // 2) + offs_be
    tl.store(out_ptr + row * stride_om + offs_b, packed, mask=offs_b < K // 2)


def _launch_quantize(
    input: torch.Tensor,
    output: torch.Tensor,
    oscales: torch.Tensor,
    lora_down: torch.Tensor | None,
    lora_act_out: torch.Tensor | None,
    smooth: torch.Tensor | None,
    fuse_glu: bool,
    fp4: bool,
    fuse_gelu: bool = False,
    shift_value: float = 0.0,
    unsigned: bool = False,
) -> None:
    input = input.reshape(-1, input.shape[-1])
    M = input.shape[0]
    K = input.shape[1] // 2 if fuse_glu else input.shape[1]
    group = 16 if fp4 else 64
    assert K % group == 0, f"K={K} must be a multiple of the group size {group}"

    has_lora = lora_down is not None and lora_act_out is not None
    R = lora_down.shape[1] if has_lora else 0

    ngroup = triton.next_power_of_2(K // group)
    _quantize_row_kernel[(M,)](
        input,
        output,
        oscales,
        lora_down if has_lora else input,
        lora_act_out if has_lora else input,
        smooth if smooth is not None else input,
        M,
        K,
        R,
        shift_value,
        input.stride(0),
        output.stride(0),
        oscales.stride(0),
        lora_down.stride(0) if has_lora else 0,
        lora_act_out.stride(0) if has_lora else 0,
        GROUP=group,
        USE_FP4=fp4,
        UNSIGNED=unsigned,
        FUSE_GLU=fuse_glu,
        FUSE_GELU=fuse_gelu,
        HAS_SMOOTH=smooth is not None,
        HAS_LORA=has_lora,
        NGROUP=ngroup,
        BLOCK_R=max(16, triton.next_power_of_2(R)) if has_lora else 16,
        num_warps=num_warps_for(4),
    )


def quantize_w4a4_act_fuse_lora(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    fuse_gelu: bool = False,
    shift_value: float = 0.0,
    unsigned: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize activations to 4 bits and compute the LoRA down-projection.

    With ``fuse_gelu``, a tanh-approximate GELU is applied before quantizing
    (matching ``EpilogueGelu`` in the CUDA next-layer-quantization path);
    ``shift_value`` (``0.171875`` for int4, per FLUX's unsigned-activation
    trick) is added after that, before smoothing and quantizing. ``unsigned``
    quantizes to ``[0, 15]`` instead of the default symmetric ``[-8, 7]`` --
    set this whenever the row is known non-negative (i.e. whenever
    ``shift_value`` was used to make it so), matching ``EpilogueQuantize``'s
    ``USE_UNSIGNED = !USE_FP4`` in the CUDA next-layer-quantization path.

    Returns ``(output, oscales, lora_act_out)``; any of them are allocated if
    not supplied.
    """
    inp = input.reshape(-1, input.shape[-1])
    M = inp.shape[0]
    K = inp.shape[1] // 2 if fuse_glu else inp.shape[1]
    group = 16 if fp4 else 64

    if output is None:
        output = torch.empty(M, K // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        sdtype = torch.float8_e4m3fn if fp4 else input.dtype
        oscales = torch.empty(K // group, M, dtype=sdtype, device=input.device)
    if lora_act_out is None and lora_down is not None:
        lora_act_out = torch.empty(M, lora_down.shape[1], dtype=torch.float32, device=input.device)

    _launch_quantize(
        inp, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4, fuse_gelu, shift_value, unsigned
    )
    return output, oscales, lora_act_out


def quantize_w4a4_act(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    fp4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to 4 bits, per-group symmetric."""
    out, scales, _ = quantize_w4a4_act_fuse_lora(input, output, oscales, fp4=fp4)
    return out, scales


def quantize_w4a4_wgt(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    fp4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to 4 bits, per-group symmetric.

    Identical maths to the activation path -- the CUDA version differs only in
    the mma-fragment layout it writes, which this backend does not use.
    """
    out, scales, _ = quantize_w4a4_act_fuse_lora(input, output, oscales, fp4=fp4)
    return out, scales
