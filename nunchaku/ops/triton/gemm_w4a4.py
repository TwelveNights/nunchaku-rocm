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

Int4 fast path
--------------
For int4 (the common case) :func:`gemm_w4a4` runs ``_gemm_w4a4_int4_kernel``,
which reads the packed operands as int32 words and decodes eight nibbles per
word with bit-field extracts (see :func:`_decode_group_words`); the generic
``_gemm_w4a4_kernel`` remains for NVFP4 and unaligned inputs. On that path
the RMSNorm+RoPE epilogue and the GELU + next-layer-quantize epilogue run on
the fp32 tile in registers (the latter in ``_gemm_w4a4_int4_gelu_quant_kernel``,
which then never writes the 16-bit GEMM output) instead of as separate passes
over the stored output.

Native RDNA3 path
-----------------
On gfx11 GPUs, int4 calls run the hand-written ``v_wmma_i32_16x16x16_iu4``
kernels in ``native/`` instead (about 2x this kernel's speed) when they
support the shapes and dtypes: plain calls (bias / LoRA-up only), the fused
Q/K RMSNorm + RoPE epilogue, and the fused GELU + next-layer-quantize
epilogue; everything else (NVFP4, SiLU, ...) stays here.
``NUNCHAKU_NATIVE_GEMM=0`` disables them. See :mod:`.native.w4a4`.
"""

import torch
import triton
import triton.language as tl

from ._common import gelu_tanh, matmul_group_m, num_warps_for, silu_impl
from .native import w4a4 as _native_w4a4

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


@triton.jit
def _w4a4_epilogue(
    acc,
    offs_m,
    offs_n,
    m_mask,
    n_mask,
    bias_ptr,
    lora_act_ptr,
    lora_up_ptr,
    wcscale_ptr,
    R,
    alpha,
    stride_lam,
    stride_lun,
    HAS_BIAS: tl.constexpr,
    HAS_LORA: tl.constexpr,
    HAS_WCSCALE: tl.constexpr,
    FUSE_SILU: tl.constexpr,
):
    """wcscale/alpha, low-rank up-projection, bias, optional SiLU; returns the fp32 tile."""
    if HAS_WCSCALE:
        acc *= tl.load(wcscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]

    acc *= alpha

    # low-rank correction: out += lora_act @ lora_up.T
    if HAS_LORA:
        lacc = tl.zeros(acc.shape, tl.float32)
        for r0 in tl.range(0, R, 16):
            offs_r = r0 + tl.arange(0, 16)
            r_mask = offs_r < R
            lu = tl.load(
                lora_up_ptr + offs_n[:, None] * stride_lun + offs_r[None, :],
                mask=n_mask[:, None] & r_mask[None, :],
                other=0.0,
            )
            # the fp32 hidden state is rounded to lora_up's 16-bit dtype so the
            # dot runs on the matrix cores (an fp32 dot lowers to plain FMAs on
            # RDNA3 and cost ~10% of the whole GEMM); the CUDA EpilogueLoraUp
            # likewise feeds 16-bit operands to its mma
            la = tl.load(
                lora_act_ptr + offs_m[:, None] * stride_lam + offs_r[None, :],
                mask=m_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(lu.dtype)
            lacc = tl.dot(la, tl.trans(lu), lacc)
        acc += lacc

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]

    if FUSE_SILU:
        acc = silu_impl(acc)
    return acc


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

    acc = _w4a4_epilogue(
        acc, offs_m, offs_n, m_mask, n_mask, bias_ptr, lora_act_ptr, lora_up_ptr,
        wcscale_ptr, R, alpha, stride_lam, stride_lun, HAS_BIAS, HAS_LORA, HAS_WCSCALE, FUSE_SILU,
    )

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_m_mask[:, None] & n_mask[None, :])


HEAD_DIM = 128


def _w4a4_int4_configs():
    # int4 path (word-decoded); measured on RX 7800 XT (gfx1101) -- 64x256/8 warps
    # and 64x128/4 warps win at FLUX shapes; num_stages=2 throughout (the AMD
    # stream pipeliner gains nothing from deeper stages here).
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": matmul_group_m()},
            num_stages=ns,
            num_warps=num_warps_for(nw),
        )
        for bm, bn, ns, nw in [
            (64, 256, 2, 8),
            (64, 128, 2, 4),
            (128, 128, 2, 8),
            (128, 64, 2, 8),
            (64, 64, 2, 4),
        ]
    ]


@triton.jit
def _nibble_i8(w, j: tl.constexpr, UNSIGNED: tl.constexpr):
    """Nibble ``j`` (0 = least significant) of each int32 word, as int8."""
    if UNSIGNED:
        return ((w >> (4 * j)) & 0xF).to(tl.int8)
    else:
        # shift the field to the top, then arithmetic-shift down: sign-extends
        # in one v_bfe_i32
        return ((w << (28 - 4 * j)) >> 28).to(tl.int8)


@triton.jit
def _decode_group_words(w, UNSIGNED: tl.constexpr):
    """
    Decode ``[R, 8]`` int32 words (one 64-element scale group per row) to
    ``[R, 64]`` int8.

    The K order produced inside the group is a fixed permutation of the
    logical order (nibble-position-major rather than byte-major). A and W go
    through the identical permutation, and a dot product over K is invariant
    to it, so no shuffle back is needed. Decoding whole words this way costs
    one bit-field extract per element, versus the byte-load + mask + compare +
    select chain of :func:`_load_nibbles`, which made the GEMM VALU-bound.
    """
    x01 = tl.join(_nibble_i8(w, 0, UNSIGNED), _nibble_i8(w, 1, UNSIGNED))
    x23 = tl.join(_nibble_i8(w, 2, UNSIGNED), _nibble_i8(w, 3, UNSIGNED))
    x45 = tl.join(_nibble_i8(w, 4, UNSIGNED), _nibble_i8(w, 5, UNSIGNED))
    x67 = tl.join(_nibble_i8(w, 6, UNSIGNED), _nibble_i8(w, 7, UNSIGNED))
    x = tl.join(tl.join(x01, x23), tl.join(x45, x67))
    return tl.reshape(x, (w.shape[0], 64))


@triton.jit
def _tile_ids(M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """Grouped (L2-friendly) mapping of the 1-D program id to a (pid_m, pid_n) tile."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _int4_mainloop(
    act_ptr,
    wgt_ptr,
    ascale_ptr,
    wscale_ptr,
    rm,
    rn,
    K,
    stride_am,
    stride_wn,
    stride_asg,
    stride_wsg,
    ACT_UNSIGNED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    ``sum_g ascale[g, m] * wscale[g, n] * (A_g @ W_g.T)`` over 64-wide groups,
    for in-range row/col indices ``rm``/``rn`` (out-of-range tile rows/cols
    are wrapped by the caller rather than masked; their results land only in
    rows/cols the masked store discards). ``act``/``wgt`` are int32 words.
    """
    offs_w = tl.arange(0, 8)
    a_ptrs = act_ptr + rm[:, None] * stride_am + offs_w[None, :]
    w_ptrs = wgt_ptr + rn[:, None] * stride_wn + offs_w[None, :]
    as_ptrs = ascale_ptr + rm
    ws_ptrs = wscale_ptr + rn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    # one int8 x int8 -> int32 dot per 64-wide scale group; the per-group
    # scales are a rank-1 product applied to the fp32 partial result
    for _ in tl.range(0, K // 64):
        a = _decode_group_words(tl.load(a_ptrs), ACT_UNSIGNED)
        w = _decode_group_words(tl.load(w_ptrs), False)
        dot = tl.dot(a, tl.trans(w), out_dtype=tl.int32)
        asc = tl.load(as_ptrs).to(tl.float32)
        wsc = tl.load(ws_ptrs).to(tl.float32)
        acc += dot.to(tl.float32) * asc[:, None] * wsc[None, :]
        a_ptrs += 8
        w_ptrs += 8
        as_ptrs += stride_asg
        ws_ptrs += stride_wsg
    return acc


def _prune_int4_configs(configs, named_args, **kwargs):
    # the fused per-head RMSNorm+RoPE epilogue needs one head (128 columns) per tile
    if kwargs.get("FUSE_QK_NORM", named_args.get("FUSE_QK_NORM", False)):
        return [c for c in configs if c.kwargs["BLOCK_N"] == HEAD_DIM]
    return configs


@triton.jit
def _qk_rmsnorm_rope_tile(acc, offs_m, row_mask, part, norm_q_ptr, norm_k_ptr, rot_ptr, stride_rm, eps):
    """
    Per-head RMSNorm + RoPE on a ``[BLOCK_M, 128]`` fp32 tile that is exactly
    one head of the Q (``part == 0``) or K (``part == 1``) third of a QKV
    projection; V tiles (``part == 2``) pass through. Same maths as
    :func:`_rmsnorm_rope_qk_kernel`, applied before the output is rounded.
    """
    BM: tl.constexpr = acc.shape[0]
    D: tl.constexpr = acc.shape[1]
    offs_d = tl.arange(0, D)
    wq = tl.load(norm_q_ptr + offs_d).to(tl.float32)
    wk = tl.load(norm_k_ptr + offs_d).to(tl.float32)
    w = tl.where(part == 0, wq, wk)
    rstd = tl.rsqrt(tl.sum(acc * acc, axis=1) / D + eps)
    x = acc * rstd[:, None] * w[None, :]
    # rotary_emb rows are [D // 2 pairs, (sin, cos)] fp32
    sc = tl.load(rot_ptr + offs_m[:, None].to(tl.int64) * stride_rm + offs_d[None, :], mask=row_mask[:, None], other=0.0)
    sin, cos = tl.split(tl.reshape(sc, (BM, D // 2, 2)))
    xe, xo = tl.split(tl.reshape(x, (BM, D // 2, 2)))
    rot = tl.reshape(tl.join(xe * cos - xo * sin, xe * sin + xo * cos), (BM, D))
    return tl.where(part < 2, rot, acc)


@triton.autotune(
    configs=_w4a4_int4_configs(),
    key=["M", "N", "K", "FUSE_QK_NORM"],
    prune_configs_by={"early_config_prune": _prune_int4_configs},
)
@triton.jit
def _gemm_w4a4_int4_kernel(
    act_ptr,
    wgt_ptr,
    out_ptr,
    ascale_ptr,
    wscale_ptr,
    bias_ptr,
    lora_act_ptr,
    lora_up_ptr,
    wcscale_ptr,
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
    norm_q_ptr,
    norm_k_ptr,
    rot_ptr,
    stride_rm,
    eps,
    ACT_UNSIGNED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LORA: tl.constexpr,
    HAS_WCSCALE: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    FUSE_QK_NORM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Int4 W4A4 GEMM. ``act``/``wgt`` are the packed nibble tensors viewed as
    int32 words (``stride_am``/``stride_wn`` in words); ``K % 64 == 0``.
    Same maths as the int4 path of :func:`_gemm_w4a4_kernel`.
    """
    pid_m, pid_n = _tile_ids(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    out_m_mask = offs_m < actual_m
    acc = _int4_mainloop(
        act_ptr, wgt_ptr, ascale_ptr, wscale_ptr, offs_m % M, offs_n % N, K, stride_am, stride_wn, stride_asg, stride_wsg,
        ACT_UNSIGNED, BLOCK_M, BLOCK_N,
    )

    acc = _w4a4_epilogue(
        acc, offs_m, offs_n, m_mask, n_mask, bias_ptr, lora_act_ptr, lora_up_ptr,
        wcscale_ptr, R, alpha, stride_lam, stride_lun, HAS_BIAS, HAS_LORA, HAS_WCSCALE, FUSE_SILU,
    )

    if FUSE_QK_NORM:
        acc = _qk_rmsnorm_rope_tile(acc, offs_m, out_m_mask, (pid_n * BLOCK_N) // (N // 3), norm_q_ptr, norm_k_ptr, rot_ptr, stride_rm, eps)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_w4a4_int4_gelu_quant_kernel(
    act_ptr,
    wgt_ptr,
    out_ptr,
    ascale_ptr,
    wscale_ptr,
    bias_ptr,
    lora_act_ptr,
    lora_up_ptr,
    qout_ptr,
    oscale_ptr,
    smooth_ptr,
    lora_down_ptr,
    lora_out_ptr,
    M,
    N,
    K,
    R,
    R2,
    actual_m,
    shift_value,
    stride_am,
    stride_wn,
    stride_asg,
    stride_wsg,
    stride_om,
    stride_lam,
    stride_lun,
    stride_qm,
    stride_osg,
    stride_ld,
    stride_lop,
    stride_lom,
    ACT_UNSIGNED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LORA: tl.constexpr,
    HAS_SMOOTH: tl.constexpr,
    HAS_LORA_DOWN: tl.constexpr,
    STORE_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Int4 GEMM whose epilogue is ``EpilogueGelu`` + ``EpilogueLoraDown`` +
    ``EpilogueQuantize`` (unsigned int4, ``+shift_value``) of the CUDA
    next-layer-quantize path, applied to the fp32 tile in registers: the
    16-bit GEMM output is never written (unless ``STORE_OUT``) or re-read.

    Each tile row covers ``BLOCK_N // 64`` whole quantization groups, so the
    per-group scales are tile-local. The next layer's LoRA-down projection
    is a reduction over all of ``N``: every tile writes its ``[BLOCK_M, R2]``
    partial to ``lora_out[pid_n]`` (shape ``[N // BLOCK_N, M, R2]``) and the
    caller sums over the first dim, which keeps the result deterministic.
    """
    pid_m, pid_n = _tile_ids(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    out_m_mask = offs_m < actual_m
    acc = _int4_mainloop(
        act_ptr, wgt_ptr, ascale_ptr, wscale_ptr, offs_m % M, offs_n % N, K, stride_am, stride_wn, stride_asg, stride_wsg,
        ACT_UNSIGNED, BLOCK_M, BLOCK_N,
    )
    acc = _w4a4_epilogue(
        acc, offs_m, offs_n, m_mask, n_mask, bias_ptr, lora_act_ptr, lora_up_ptr,
        bias_ptr, R, 1.0, stride_lam, stride_lun, HAS_BIAS, HAS_LORA, False, False,
    )
    if STORE_OUT:
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=out_m_mask[:, None] & n_mask[None, :],
        )

    x = gelu_tanh(acc)
    tile_mask = m_mask[:, None] & n_mask[None, :]

    # next layer's LoRA-down consumes the GELU'd activation before shift/smooth
    if HAS_LORA_DOWN:
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < R2
        ld = tl.load(
            lora_down_ptr + offs_n[:, None] * stride_ld + offs_r[None, :],
            mask=n_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        part = tl.dot(tl.where(tile_mask, x, 0.0).to(ld.dtype), ld)
        # per-N-tile partial sums, reduced afterwards in a fixed order: fp32
        # atomics here made results (and, through the 4-bit network's
        # sensitivity, whole images) vary run to run
        tl.store(
            lora_out_ptr + pid_n * stride_lop + offs_m[:, None] * stride_lom + offs_r[None, :],
            part,
            mask=m_mask[:, None] & r_mask[None, :],
        )

    x += shift_value
    if HAS_SMOOTH:
        x = x * (1.0 / tl.load(smooth_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32))[None, :]

    NG: tl.constexpr = BLOCK_N // 64
    xg = tl.reshape(x, (BLOCK_M, NG, 64))
    gmax = tl.maximum(tl.max(xg, axis=2), 0.0)
    scale = gmax / 15.0
    scale_safe = tl.where(scale == 0.0, 1.0, scale)
    offs_g = pid_n * NG + tl.arange(0, NG)
    tl.store(
        oscale_ptr + offs_g[None, :] * stride_osg + offs_m[:, None],
        scale.to(oscale_ptr.dtype.element_ty),
        mask=m_mask[:, None] & (offs_g[None, :] < N // 64),
    )
    q = tl.reshape(xg * (1.0 / scale_safe)[:, :, None], (BLOCK_M, BLOCK_N))
    code = tl.minimum(tl.maximum(_rint_small(q), 0.0), 15.0).to(tl.int32) & 0xF
    lo, hi = tl.split(tl.reshape(code, (BLOCK_M, BLOCK_N // 2, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    offs_b = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    tl.store(
        qout_ptr + offs_m[:, None] * stride_qm + offs_b[None, :],
        packed,
        mask=m_mask[:, None] & (offs_b[None, :] < N // 2),
    )


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


def _launch_gemm_int4_gelu_quant(
    act, wgt, out2d, ascales, wscales, bias, lora_act_in, lora_up, qout, oscales, smooth, lora_down, lora_act_out,
    act_unsigned, M, N, K,
):
    act32, wgt32 = act.view(torch.int32), wgt.view(torch.int32)
    qout2d = qout.reshape(-1, qout.shape[-1])
    has_ld = lora_down is not None
    R2 = lora_down.shape[1] if has_ld else 0
    # fixed tile: 64x128 is the autotune winner for the plain int4 GEMM at
    # FLUX shapes on RX 7800 XT (and wastes no LoRA-down partial storage)
    BLOCK_M, BLOCK_N = 64, 128
    if has_ld:
        lora_part = torch.empty(triton.cdiv(N, BLOCK_N), M, R2, dtype=torch.float32, device=act.device)
    dummy = act32
    _gemm_w4a4_int4_gelu_quant_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)](
        act32,
        wgt32,
        out2d if out2d is not None else dummy,
        ascales,
        wscales,
        bias if bias is not None else dummy,
        lora_act_in if lora_act_in is not None else dummy,
        lora_up if lora_up is not None else dummy,
        qout2d,
        oscales,
        smooth if smooth is not None else dummy,
        lora_down if has_ld else dummy,
        lora_part if has_ld else dummy,
        M,
        N,
        K,
        lora_up.shape[1] if lora_up is not None else 0,
        R2,
        out2d.shape[0] if out2d is not None else M,
        0.171875,
        act32.stride(0),
        wgt32.stride(0),
        ascales.stride(0),
        wscales.stride(0),
        out2d.stride(0) if out2d is not None else 0,
        lora_act_in.stride(0) if lora_act_in is not None else 0,
        lora_up.stride(0) if lora_up is not None else 0,
        qout2d.stride(0),
        oscales.stride(0),
        lora_down.stride(0) if has_ld else 0,
        lora_part.stride(0) if has_ld else 0,
        lora_part.stride(1) if has_ld else 0,
        ACT_UNSIGNED=act_unsigned,
        HAS_BIAS=bias is not None,
        HAS_LORA=lora_act_in is not None,
        HAS_SMOOTH=smooth is not None,
        HAS_LORA_DOWN=has_ld,
        STORE_OUT=out2d is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_R=max(16, triton.next_power_of_2(R2)),
        GROUP_M=matmul_group_m(),
        num_warps=num_warps_for(4),
        num_stages=2,
    )
    if has_ld:
        torch.sum(lora_part, dim=0, out=lora_act_out[:M])


def _gemm_w4a4_swiglu_quant(
    act, wgt, ascales, wscales, bias, lora_act_in, lora_up, qout, oscales, smooth, lora_down, lora_act_out,
    act_unsigned, out, wcscales, norm_q, rotary_emb, fuse_silu, fp4, alpha, out_dtype,
) -> None:
    """``gemm_w4a4(..., fuse_swiglu=True)``: the native fused kernel on gfx11, else the unfused equivalent."""
    assert out is None and qout is not None and oscales is not None, "fuse_swiglu writes qout/oscales, not out"
    assert not (fp4 or fuse_silu or wcscales is not None or norm_q is not None or rotary_emb is not None), (
        "fuse_swiglu is int4-only and excludes the other epilogues"
    )
    assert alpha is None or alpha == 1.0
    assert (lora_down is None) == (lora_act_out is None), "lora_down and lora_act_out must be given together"
    M, N = act.shape[0], wgt.shape[0]
    has_lora = lora_act_in is not None and lora_up is not None
    la, lu = (lora_act_in, lora_up) if has_lora else (None, None)
    qout2d = qout.reshape(-1, qout.shape[-1])
    if _native_w4a4.try_gemm_swiglu_quant(
        act, wgt, ascales, wscales, bias, la, lu, qout2d, oscales, smooth, lora_down, lora_act_out, act_unsigned
    ):
        return
    # unfused: the GEMM (interleaved columns), Flux2SwiGLU on the 16-bit output, the activation quantizer
    x = gemm_w4a4(act, wgt, ascales=ascales, wscales=wscales, bias=bias, lora_act_in=la, lora_up=lu,
                  act_unsigned=act_unsigned, out_dtype=out_dtype)
    x = x.view(M, N // 32, 2, 16)
    y = (torch.nn.functional.silu(x[:, :, 0]) * x[:, :, 1]).reshape(M, N // 2)
    _launch_quantize(y, qout2d, oscales, lora_down, lora_act_out, smooth, fuse_glu=False, fp4=False)


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
    fuse_swiglu: bool = False,
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
    fuse_swiglu : SwiGLU MLP input projection (FLUX.2): ``wgt`` is the (2H, K/2)
        ``[gate | up]`` weight with rows interleaved by
        :func:`.native.w4a4.interleave_swiglu` (as are ``wscales``' columns,
        ``bias`` and ``lora_up``); instead of the GEMM output, ``silu(gate) * up``
        is quantized (signed int4, ``smooth_factor`` (H,)) into ``qout`` (M, H/2) /
        ``oscales`` (H/64, M), with the next layer's LoRA-down (``lora_down`` (H, R'))
        into ``lora_act_out`` -- what diffusers' ``Flux2SwiGLU`` followed by the next
        layer's ``quantize`` computes. ``out`` must be ``None``; returns ``None``.
    fp4 : decode nibbles as ``e2m1`` instead of int4
    alpha : global output scale (NVFP4 weight tensor scale)
    """
    assert ascales is not None and wscales is not None, "W4A4 GEMM requires both scale tensors"

    if fuse_swiglu:
        _gemm_w4a4_swiglu_quant(
            act.reshape(-1, act.shape[-1]), wgt, ascales, wscales, bias, lora_act_in, lora_up, qout, oscales,
            smooth_factor, lora_down, lora_act_out, act_unsigned, out, wcscales, norm_q, rotary_emb, fuse_silu, fp4,
            alpha, out_dtype,
        )
        return None

    act = act.reshape(-1, act.shape[-1])
    M = act.shape[0]
    K = act.shape[1] * 2
    N = wgt.shape[0]
    assert wgt.shape[1] * 2 == K, f"weight K mismatch: {wgt.shape[1] * 2} vs {K}"

    group_size = 16 if fp4 else 64
    assert K % group_size == 0

    word_ok = (
        not fp4
        and K % 64 == 0
        and act.stride(-1) == 1
        and wgt.stride(-1) == 1
        and act.stride(0) % 4 == 0
        and wgt.stride(0) % 4 == 0
        and act.data_ptr() % 4 == 0
        and wgt.data_ptr() % 4 == 0
    )
    want_quant = qout is not None or oscales is not None
    if want_quant:
        assert qout is not None and oscales is not None, "qout and oscales must be given together"
    # int4 next-layer quantization fused into the GEMM epilogue (the 16-bit
    # output then need not be materialized at all)
    fuse_quant = (
        word_ok
        and want_quant
        and norm_q is None
        and rotary_emb is None
        and not fuse_silu
        and wcscales is None
        and (alpha is None or alpha == 1.0)
        and N % 128 == 0
        and qout.stride(-1) == 1
        and (lora_down is None) == (lora_act_out is None)
    )

    if out is None and not fuse_quant:
        out = torch.empty(M, N, dtype=out_dtype, device=act.device)
    # `act` (and ascales/lora_act_in) may be padded to more rows than the
    # caller's output buffer holds -- e.g. fused_gelu_mlp pads its quantized
    # intermediate to a multiple of 256 rows, but the final output only needs
    # the real batch size. Compute over the full M but only read/write the
    # rows that exist in every M-indexed buffer.
    out2d = out.reshape(-1, N) if out is not None else None
    actual_m = out2d.shape[0] if out2d is not None else M
    assert actual_m <= M, f"out has more rows ({actual_m}) than act ({M})"

    has_lora = lora_act_in is not None and lora_up is not None
    R = lora_up.shape[1] if has_lora else 0

    # Int4 GEMMs go to the hand-written int4-WMMA kernels on RDNA3 when they
    # support the call (see native/w4a4.py): plain (bias / LoRA-up only), the
    # QKV projection's fused RMSNorm + RoPE, and the fused GELU + next-layer
    # quantization of an MLP's fc1.
    want_qk = norm_q is not None or norm_k is not None or rotary_emb is not None
    native_ok = not fp4 and wcscales is None and (alpha is None or alpha == 1.0) and not fuse_silu
    la_n, lu_n = (lora_act_in, lora_up) if has_lora else (None, None)
    if native_ok and not want_qk and not want_quant:
        if _native_w4a4.try_gemm(act, wgt, out2d, ascales, wscales, bias, la_n, lu_n, act_unsigned):
            return out
    elif native_ok and want_qk and not want_quant:
        if _native_w4a4.try_gemm_qk(
            act, wgt, out2d, ascales, wscales, bias, la_n, lu_n, norm_q, norm_k, rotary_emb, act_unsigned
        ):
            return out
    elif fuse_quant:
        if _native_w4a4.try_gemm_gelu_quant(
            act, wgt, out2d, ascales, wscales, bias, la_n, lu_n, qout.reshape(-1, qout.shape[-1]), oscales,
            smooth_factor, lora_down, lora_act_out, act_unsigned,
        ):
            return out

    if fuse_quant:
        _launch_gemm_int4_gelu_quant(
            act, wgt, out2d, ascales, wscales, bias, lora_act_in if has_lora else None, lora_up if has_lora else None,
            qout, oscales, smooth_factor, lora_down, lora_act_out, act_unsigned, M, N, K,
        )
        return out

    # the K loop steps exactly one scale group at a time, so a group is never
    # straddled for either int4 (64) or nvfp4 (16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    if want_qk:
        assert norm_q is not None and norm_k is not None and rotary_emb is not None, (
            "norm_q, norm_k, and rotary_emb must all be given together"
        )
    fuse_qk = word_ok and want_qk and (N % 3 == 0) and ((N // 3) % HEAD_DIM == 0) and rotary_emb.dtype == torch.float32
    if fuse_qk:
        rot = rotary_emb.reshape(actual_m, HEAD_DIM)
        assert rot.stride(1) == 1
    if word_ok:
        act32 = act.view(torch.int32)
        wgt32 = wgt.view(torch.int32)
        _gemm_w4a4_int4_kernel[grid](
            act32,
            wgt32,
            out2d,
            ascales,
            wscales,
            bias if bias is not None else act,
            lora_act_in if has_lora else act,
            lora_up if has_lora else act,
            wcscales if wcscales is not None else act,
            M,
            N,
            K,
            R,
            actual_m,
            alpha if alpha is not None else 1.0,
            act32.stride(0),
            wgt32.stride(0),
            ascales.stride(0),
            wscales.stride(0),
            out2d.stride(0),
            lora_act_in.stride(0) if has_lora else 0,
            lora_up.stride(0) if has_lora else 0,
            norm_q if fuse_qk else act,
            norm_k if fuse_qk else act,
            rot if fuse_qk else act,
            rot.stride(0) if fuse_qk else 0,
            1e-6,
            ACT_UNSIGNED=act_unsigned,
            HAS_BIAS=bias is not None,
            HAS_LORA=has_lora,
            HAS_WCSCALE=wcscales is not None,
            FUSE_SILU=fuse_silu,
            FUSE_QK_NORM=fuse_qk,
        )
    else:
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

    if want_qk and not fuse_qk:
        _apply_rmsnorm_rope_qk(out2d, norm_q, norm_k, rotary_emb)

    if want_quant:
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
def _rint_small(x):
    """
    Round half to even for ``|x| < 2**22`` via the fp32 magic-number trick:
    adding 1.5 * 2**23 forces rounding at the units place in the current
    (round-to-nearest-even) mode. Two adds instead of ``_rint``'s floor/compare
    chain; exact for the quantizer's ``[-8, 16]`` range.
    """
    return (x + 12582912.0) - 12582912.0


@triton.jit
def _rint(x):
    """Round half to even, matching ``cvt.rni``."""
    f = tl.floor(x)
    d = x - f
    even = (tl.floor(f * 0.5) * 2.0) == f
    up = tl.where(d == 0.5, tl.where(even, 0.0, 1.0), tl.where(d > 0.5, 1.0, 0.0))
    return f + up


@triton.jit
def _quantize_tile_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Quantize a ``[BLOCK_M, K]`` row block, streaming ``BLOCK_K`` columns
    (a whole number of scale groups) at a time.

    Compared to a one-row-per-program kernel:
    every input element is read exactly once (GELU/GLU applied once), the
    LoRA down-projection is a ``tl.dot`` over the tile rather than a per-row
    broadcast-multiply that re-read all of ``lora_down`` for every row, and
    the ``[K // GROUP, M]`` scales are stored coalesced along ``M``.
    """
    pid = tl.program_id(0)
    offs_m = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    m_mask = offs_m < M
    NG: tl.constexpr = BLOCK_K // GROUP

    if USE_FP4:
        qmax = 6.0
    elif UNSIGNED:
        qmax = 15.0
    else:
        qmax = 7.0

    if HAS_LORA:
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < R
        lacc = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask = m_mask[:, None] & (offs_k[None, :] < K)
        x = tl.load(in_ptr + offs_m[:, None] * stride_im + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
        if FUSE_GLU:
            gate = tl.load(in_ptr + offs_m[:, None] * stride_im + K + offs_k[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            x = x * (gate / (1.0 + tl.exp(-gate)))
        if FUSE_GELU:
            x = gelu_tanh(x)

        # LoRA-down consumes the GELU/GLU'd row before shift and smoothing
        # (EpilogueLoraDown runs before EpilogueQuantize in the CUDA kernel).
        if HAS_LORA:
            w = tl.load(
                lora_down_ptr + offs_k[:, None] * stride_ld + offs_r[None, :],
                mask=(offs_k[:, None] < K) & r_mask[None, :],
                other=0.0,
            )
            lacc = tl.dot(x.to(w.dtype), w, lacc)

        x += shift_value
        if HAS_SMOOTH:
            x = x * (1.0 / tl.load(smooth_ptr + offs_k, mask=offs_k < K, other=1.0).to(tl.float32))[None, :]

        xg = tl.reshape(x, (BLOCK_M, NG, GROUP))
        if UNSIGNED:
            gmax = tl.max(xg, axis=2)
        else:
            gmax = tl.max(tl.abs(xg), axis=2)
        gmax = tl.maximum(gmax, 0.0)
        scale = gmax / qmax
        scale_safe = tl.where(scale == 0.0, 1.0, scale)

        offs_g = k0 // GROUP + tl.arange(0, NG)
        tl.store(
            scale_ptr + offs_g[None, :] * stride_sg + offs_m[:, None],
            scale.to(scale_ptr.dtype.element_ty),
            mask=m_mask[:, None] & (offs_g[None, :] < K // GROUP),
        )

        q = tl.reshape(xg * (1.0 / scale_safe)[:, :, None], (BLOCK_M, BLOCK_K))
        if USE_FP4:
            code = _encode_e2m1(q)
        elif UNSIGNED:
            code = tl.minimum(tl.maximum(_rint_small(q), 0.0), 15.0).to(tl.int32) & 0xF
        else:
            code = tl.minimum(tl.maximum(_rint_small(q), -8.0), 7.0).to(tl.int32) & 0xF

        lo, hi = tl.split(tl.reshape(code, (BLOCK_M, BLOCK_K // 2, 2)))
        packed = (lo | (hi << 4)).to(tl.uint8)
        offs_b = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_b[None, :],
            packed,
            mask=m_mask[:, None] & (offs_b[None, :] < K // 2),
        )

    if HAS_LORA:
        tl.store(
            lora_act_ptr + offs_m[:, None] * stride_lam + offs_r[None, :],
            lacc,
            mask=m_mask[:, None] & r_mask[None, :],
        )


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

    # measured on RX 7800 XT: 32x64 is best for large M; small M wants more programs
    BLOCK_M = 32 if M >= 2048 else 16
    BLOCK_K = 64
    _quantize_tile_kernel[(triton.cdiv(M, BLOCK_M),)](
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
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
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
