// SVDQuant W4A4 GEMM for RDNA3 (gfx11), built around v_wmma_i32_16x16x16_iu4.
//
//   out[m, n] = sum_g ascales[g, m] * wscales[g, n] * (A_g[m, :] . W_g[n, :])    int4 x int4 -> int32 per 64-wide
//                                                                               scale group g, exact
//             + lora_act[m, :] . lora_up[n, :] + bias[n]
//
// Same maths and natural layouts as _gemm_w4a4_int4_kernel in nunchaku/ops/triton/gemm_w4a4.py, which Triton
// lowers to int8 WMMA: the int4 WMMA has twice the int8 rate on RDNA3 (~140 vs ~72 TOPS on an RX 7800 XT) and
// Triton cannot emit it.
//
//   act (M, K/2), wgt (N, K/2)   packed nibbles: element k is nibble k % 8 of int32 word k / 8. The WMMA reads the
//                                nibbles of a word in its own order, but A and W go through the same one and the
//                                dot product is invariant to it.
//   ascales (K/64, M), wscales (K/64, N), bias (N,), lora_up (N, R), out (M, N)   T = bf16 or fp16
//   lora_act (M, R)              fp32, rounded to T for the (16-bit) LoRA-up WMMA, as the Triton kernel does
//
// The host (w4a4.py: unsupported_reason) guarantees what this file assumes and falls back to Triton otherwise:
// K % 128 == 0, M % 8 == 0, N % 8 == 0, R % 16 == 0, unit inner strides, and 16-byte aligned rows for every
// operand that is read in 16-byte vectors (act, wgt, ascales, wscales, lora_act, lora_up).
//
// Structure: gemm_w4a4<Tile, A_SIGNED, T, Epi> = LDS-staged main loop -> lora_up_bias -> Epi::apply. The epilogue
// gets the fp32 accumulator tile plus its coordinates (TileCtx) and the LDS buffer, so a fused epilogue (QK
// RMSNorm + RoPE, next-layer quantization, ...) slots in as another Epi without touching the main loop.
#include "hip_prelude.h"

// ---------------------------------------------------------------------------------------------------------------
// 16-bit element types. out / ascales / wscales / bias / lora_up share one dtype T (checked on the host).

struct BF16 {
  typedef v16s frag;  // 16-bit WMMA A/B operand: 16 values of one row
  static __device__ __forceinline__ float to_f32(u16 x) { return bf16_to_f32(x); }
  static __device__ __forceinline__ u16 from_f32(float f) { return f32_to_bf16(f); }
  static __device__ __forceinline__ short frag_elem(float f) { return (short)f32_to_bf16(f); }
  static __device__ __forceinline__ v8f wmma(frag a, frag b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
  }
  // Scale byte-select: the activation scales of a 16-row slice arrive as 8 dwords, dword i holding the scales of
  // rows (2i, 2i+1); accumulator element i of a lane is row 2i + hl (hl = which half-wave). v_perm_b32 with this
  // selector moves the half this lane needs into bits 31:16 and zeroes bits 15:0 -- which *is* the fp32 value of
  // a bf16 -- in one instruction.
  static __device__ __forceinline__ unsigned pair_sel(int hl) { return hl ? 0x03020c0cu : 0x01000c0cu; }
  static __device__ __forceinline__ float pair_to_f32(unsigned pair, unsigned sel) {
    return __builtin_bit_cast(float, __builtin_amdgcn_perm(0u, pair, sel));
  }
};

struct FP16 {
  typedef v16h frag;
  static __device__ __forceinline__ float to_f32(u16 x) { return (float)__builtin_bit_cast(_Float16, x); }
  static __device__ __forceinline__ u16 from_f32(float f) { return __builtin_bit_cast(u16, (_Float16)f); }
  static __device__ __forceinline__ _Float16 frag_elem(float f) { return (_Float16)f; }
  static __device__ __forceinline__ v8f wmma(frag a, frag b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  }
  // as BF16, but the selected half goes to bits 15:0 and is then converted (v_cvt_f32_f16)
  static __device__ __forceinline__ unsigned pair_sel(int hl) { return hl ? 0x0c0c0302u : 0x0c0c0100u; }
  static __device__ __forceinline__ float pair_to_f32(unsigned pair, unsigned sel) {
    return to_f32((u16)__builtin_amdgcn_perm(0u, pair, sel));
  }
};

// ---------------------------------------------------------------------------------------------------------------
// Kernel arguments. Mirrored field for field by _GemmArgs in w4a4.py (ctypes, same natural alignment).

struct GemmArgs {
  const int* act;        // (M, K/2) bytes viewed as int32 words
  const int* wgt;        // (N, K/2)
  u16* out;              // (actual_m, N), T
  const u16* ascales;    // (K/64, M), T
  const u16* wscales;    // (K/64, N), T
  const u16* bias;       // (N,), T; null: no bias
  const float* lora_act; // (M, R) fp32; unused when R == 0
  const u16* lora_up;    // (N, R), T
  int M, N, K, actual_m, R;
  int stride_am, stride_wn;    // in int32 words
  int stride_asg, stride_wsg;  // scale group strides, in elements
  int stride_om, stride_lam, stride_lun;
};

template <int BM_, int BN_, int WM_, int WN_>
struct Tile {
  static constexpr int BM = BM_, BN = BN_;      // workgroup tile
  static constexpr int WM = WM_, WN = WN_;      // wave tile
  static constexpr int TM = WM / 16, TN = WN / 16;  // 16x16 WMMA tiles per wave
  static constexpr int WAVES_N = BN / WN;
  static constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
};

// Where a wave's accumulator lives. acc[t][j][i] (WMMA C/D layout, wave32) holds
//   out[row(t, i), col(j)] = out[m0 + 16 t + 2 i + hl, n0 + 16 j + r16].
template <int TM, int TN>
struct TileCtx {
  int m0, n0;        // wave tile origin in the output
  int r16, hl;       // lane % 16, lane / 16
  int ncol[TN];      // col(j) clamped into [0, N) -- safe to load per-column operands with
  unsigned char* smem;  // the main loop's LDS staging buffer, free for the epilogue after a syncthreads()
  int smem_bytes;
  __device__ __forceinline__ int row(int t, int i) const { return m0 + t * 16 + 2 * i + hl; }
  __device__ __forceinline__ int col(int j) const { return n0 + j * 16 + r16; }
};

// The "magic" accumulator start: the int32 bit pattern of 1.5 * 2^23 (see the main loop).
#define MAGIC_I 0x4B400000
#define MAGIC_F 12582912.0f
#define GROUP_M 8  // grouped tile order (L2 reuse), as the Triton kernel

// ---------------------------------------------------------------------------------------------------------------
// Epilogue stage 1, shared by every epilogue: low-rank (LoRA-up) correction on the 16-bit WMMA, then bias.

template <class T, int TM, int TN>
static __device__ __forceinline__ void lora_up_bias(const GemmArgs& g, v8f (&acc)[TM][TN], const TileCtx<TM, TN>& c) {
  for (int r0 = 0; r0 < g.R; r0 += 16) {  // R is uniform: a scalar loop, skipped entirely without LoRA
    typename T::frag lu[TN];
#pragma unroll
    for (int j = 0; j < TN; j++) {  // B operand: lane r16 holds lora_up[col, r0 : r0 + 16]
      const v4i* p = (const v4i*)(g.lora_up + (long)c.ncol[j] * g.stride_lun + r0);
      v4i x0 = p[0], x1 = p[1];
      lu[j] = __builtin_bit_cast(typename T::frag, (v8i){x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w});
    }
#pragma unroll
    for (int t = 0; t < TM; t++) {  // A operand: lane r16 holds lora_act[row, r0 : r0 + 16], rounded to T
      int m = c.m0 + t * 16 + c.r16;
      m = m < g.M ? m : g.M - 1;
      const v4f* p = (const v4f*)(g.lora_act + (long)m * g.stride_lam + r0);
      typename T::frag la;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        v4f x = p[q];
        la[4 * q] = T::frag_elem(x.x);
        la[4 * q + 1] = T::frag_elem(x.y);
        la[4 * q + 2] = T::frag_elem(x.z);
        la[4 * q + 3] = T::frag_elem(x.w);
      }
#pragma unroll
      for (int j = 0; j < TN; j++) acc[t][j] = T::wmma(la, lu[j], acc[t][j]);
    }
  }
  if (g.bias) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const float bv = T::to_f32(g.bias[c.ncol[j]]);
#pragma unroll
      for (int t = 0; t < TM; t++)
#pragma unroll
        for (int i = 0; i < 8; i++) acc[t][j][i] += bv;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Epilogue stage 2. An epilogue is a type with
//   struct Args { ... };   (its own kernel argument, after GemmArgs; omitted from the signature when empty)
//   template <class T, int TM, int TN>
//   static void apply(const GemmArgs&, const Args&, v8f (&acc)[TM][TN], const TileCtx<TM, TN>&);

// Plain store: round to T, write the rows below actual_m (the activations may be padded past the output's rows).
struct StoreEpilogue {
  struct Args {};
  template <class T, int TM, int TN>
  static __device__ __forceinline__ void apply(const GemmArgs& g, const Args&, v8f (&acc)[TM][TN],
                                               const TileCtx<TM, TN>& c) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const int n = c.col(j);
#pragma unroll
      for (int t = 0; t < TM; t++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int m = c.row(t, i);
          if (m < g.actual_m && n < g.N) g.out[(long)m * g.stride_om + n] = T::from_f32(acc[t][j][i]);
        }
    }
  }
};

// ---------------------------------------------------------------------------------------------------------------
// The GEMM. Each main-loop stage covers K = 128 (two scale groups). The whole workgroup copies the stage's A/B
// tiles and scales global -> registers -> LDS, double-buffered with one barrier per stage, and issues the next
// stage's global loads before the current stage's compute; the waves then read their WMMA operands from LDS. No
// operand byte is fetched from global memory more than once per workgroup.

template <class Tl, bool A_SIGNED, class T, class Epi>
static __device__ __forceinline__ void gemm_w4a4(const GemmArgs& g, const typename Epi::Args& e) {
  constexpr int BM = Tl::BM, BN = Tl::BN, WM = Tl::WM, WN = Tl::WN, TM = Tl::TM, TN = Tl::TN;
  constexpr int NT = Tl::THREADS;
  // LDS stage buffer: A rows | B rows | ascales (2 groups x BM) | wscales (2 groups x BN) | 16-byte dummy slot
  constexpr int PITCH = 80;  // bytes per A/B row: 64 (2 groups of 64 nibbles) + 16 pad against bank conflicts
  constexpr int B_OFF = BM * PITCH;
  constexpr int AS_OFF = B_OFF + BN * PITCH;
  constexpr int WS_OFF = AS_OFF + 2 * BM * 2;
  constexpr int DUMMY_OFF = WS_OFF + 2 * BN * 2;
  constexpr int BUF = DUMMY_OFF + 16;
  constexpr int A_CH = BM * 4, B_CH = BN * 4;     // 16-byte chunks per stage
  constexpr int AS_CH = BM / 4, WS_CH = BN / 4;   // 2 groups x rows x 2 bytes / 16
  constexpr int A_PER = (A_CH + NT - 1) / NT, B_PER = (B_CH + NT - 1) / NT;
  static_assert(AS_CH + WS_CH <= NT, "one scale chunk per thread");
  static_assert(BM % 16 == 0 && BN % 16 == 0 && WM % 16 == 0 && WN % 16 == 0, "tile shape");
  __shared__ __attribute__((aligned(16))) unsigned char smem[2 * BUF];

  // grouped tile order, as _tile_ids in the Triton kernel
  const int M = g.M, N = g.N;
  const int pid = bid_x();
  const int num_pid_m = (M + BM - 1) / BM, num_pid_n = (N + BN - 1) / BN;
  const int num_in_group = GROUP_M * num_pid_n;
  const int first_m = (pid / num_in_group) * GROUP_M;
  const int gsz = (num_pid_m - first_m) < GROUP_M ? (num_pid_m - first_m) : GROUP_M;
  const int bm0 = (first_m + (pid % num_in_group) % gsz) * BM;
  const int bn0 = ((pid % num_in_group) / gsz) * BN;

  const int tid = tid_x();
  const int lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int r16 = lane & 15, hl = lane >> 4;
  const int wm0 = (wave / Tl::WAVES_N) * WM, wn0 = (wave % Tl::WAVES_N) * WN;  // wave tile origin in the block

  // ---- per-thread copy assignments: global source at stage 0 and LDS destination, both in bytes.
  // Rows past M / N are clamped onto the last row: they load valid memory and only feed discarded outputs.
  const unsigned char* a_src[A_PER];
  int a_dst[A_PER];
  const unsigned char* b_src[B_PER];
  int b_dst[B_PER];
#pragma unroll
  for (int q = 0; q < A_PER; q++) {
    int c = tid + q * NT;
    c = c < A_CH ? c : tid;  // spare lanes repeat their first chunk (same data to the same place)
    const int row = c >> 2, part = c & 3;
    const int m = bm0 + row < M ? bm0 + row : M - 1;
    a_src[q] = (const unsigned char*)(g.act + (long)m * g.stride_am) + part * 16;
    a_dst[q] = row * PITCH + part * 16;
  }
#pragma unroll
  for (int q = 0; q < B_PER; q++) {
    int c = tid + q * NT;
    c = c < B_CH ? c : tid;
    const int row = c >> 2, part = c & 3;
    const int n = bn0 + row < N ? bn0 + row : N - 1;
    b_src[q] = (const unsigned char*)(g.wgt + (long)n * g.stride_wn) + part * 16;
    b_dst[q] = B_OFF + row * PITCH + part * 16;
  }
  // scales: thread tid < AS_CH copies 8 ascales (group tid / (BM/8)), the next WS_CH threads 8 wscales, the rest a
  // dummy chunk. Rows past M / N clamp to the last full 8-row chunk (M % 8 == N % 8 == 0; values discarded).
  const u16* s_src = g.ascales + (bm0 < M - 8 ? bm0 : M - 8);
  long s_gstride = 2L * g.stride_asg;  // elements per stage
  int s_dst = DUMMY_OFF;
  if (tid < AS_CH) {
    const int q = tid / (BM / 8), r = (tid % (BM / 8)) * 8;
    const int m = bm0 + r < M ? bm0 + r : M - 8;
    s_src = g.ascales + (long)q * g.stride_asg + m;
    s_dst = AS_OFF + (q * BM + r) * 2;
  } else if (tid < AS_CH + WS_CH) {
    const int u = tid - AS_CH;
    const int q = u / (BN / 8), r = (u % (BN / 8)) * 8;
    const int n = bn0 + r < N ? bn0 + r : N - 8;
    s_src = g.wscales + (long)q * g.stride_wsg + n;
    s_gstride = 2L * g.stride_wsg;
    s_dst = WS_OFF + (q * BN + r) * 2;
  }

  struct Regs {
    v4i a[A_PER];
    v4i b[B_PER];
    v4i s;
  };
  auto gload = [&](int st, Regs& r) {
    const long kb = (long)st * 64;  // byte offset of this stage along a packed row
#pragma unroll
    for (int q = 0; q < A_PER; q++) r.a[q] = *(const v4i*)(a_src[q] + kb);
#pragma unroll
    for (int q = 0; q < B_PER; q++) r.b[q] = *(const v4i*)(b_src[q] + kb);
    r.s = *(const v4i*)(s_src + (long)st * s_gstride);
  };
  auto sstore = [&](int buf, const Regs& r) {
    unsigned char* base = smem + buf * BUF;
#pragma unroll
    for (int q = 0; q < A_PER; q++) *(v4i*)(base + a_dst[q]) = r.a[q];
#pragma unroll
    for (int q = 0; q < B_PER; q++) *(v4i*)(base + b_dst[q]) = r.b[q];
    *(v4i*)(base + s_dst) = r.s;
  };

  v8f acc[TM][TN];
#pragma unroll
  for (int t = 0; t < TM; t++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[t][j] = (v8f){};

  // int -> float without v_cvt_f32_i32: each group's int4 WMMA chain starts from C = the int32 bit pattern of
  // 1.5 * 2^23 instead of 0. Its group sum p satisfies |p| <= 64 * 15 * 8 < 2^22, and in [2^22, 2^24) adding p
  // to that bit pattern adds exactly p to the float it encodes (fp32 has ulp 1 between 2^23 and 2^24), so the
  // result's bits read as the float X = 1.5 * 2^23 + p. Then fma(X, as, -1.5 * 2^23 * as) = round(p * as) with
  // a single rounding (both products are exact), i.e. the same value as float(p) * as -- but in a dual-issuable
  // FMA instead of a conversion. -1.5 * 2^23 * as is exact too: 3 has a 2-bit significand, as at most 11 bits.
  v8i magic = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I};
  // Opaque to the optimizer -- keep this: clang 22 (ROCm 7.2) miscompiles per-element extracts of a WMMA result
  // whose C operand is a constant splat into extracts of element 0.
  __asm__ volatile("" : "+v"(magic));
  const unsigned sel = T::pair_sel(hl);

  auto compute = [&](int buf) {
    const unsigned char* base = smem + buf * BUF;
#pragma unroll
    for (int q = 0; q < 2; q++) {  // the two scale groups of this stage
      v4i bf[TN][2];
      float ws[TN];
#pragma unroll
      for (int j = 0; j < TN; j++) {  // B operand: lane r16 holds 64 nibbles of weight row col(j)
        const unsigned char* p = base + B_OFF + (wn0 + j * 16 + r16) * PITCH + q * 32;
        bf[j][0] = *(const v4i*)p;
        bf[j][1] = *(const v4i*)(p + 16);
        ws[j] = T::to_f32(*(const u16*)(base + WS_OFF + (q * BN + wn0 + j * 16 + r16) * 2));
      }
#pragma unroll
      for (int t = 0; t < TM; t++) {
        const unsigned char* p = base + (wm0 + t * 16 + r16) * PITCH + q * 32;
        const v4i af0 = *(const v4i*)p, af1 = *(const v4i*)(p + 16);
        // the 16 ascales of this slice: a wave-uniform read (LDS broadcast)
        const v4i* sp = (const v4i*)(base + AS_OFF + (q * BM + wm0 + t * 16) * 2);
        const v4i d0 = sp[0], d1 = sp[1];
        const int dd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
        float as[8], nb[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          as[i] = T::pair_to_f32((unsigned)dd[i], sel);
          nb[i] = as[i] * -MAGIC_F;
        }
#pragma unroll
        for (int j = 0; j < TN; j++) {
          // 4 x K16 int4 WMMAs = one 64-wide scale group; each consumes 2 words (16 nibbles) per lane
          v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(A_SIGNED, (v2i){af0.x, af0.y}, true,
                                                             (v2i){bf[j][0].x, bf[j][0].y}, magic, false);
          c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(A_SIGNED, (v2i){af0.z, af0.w}, true,
                                                         (v2i){bf[j][0].z, bf[j][0].w}, c, false);
          c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(A_SIGNED, (v2i){af1.x, af1.y}, true,
                                                         (v2i){bf[j][1].x, bf[j][1].y}, c, false);
          c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(A_SIGNED, (v2i){af1.z, af1.w}, true,
                                                         (v2i){bf[j][1].z, bf[j][1].w}, c, false);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const float y = __builtin_fmaf(i2f(c[i]), as[i], nb[i]);  // = round(p * as), see above
            acc[t][j][i] = __builtin_fmaf(y, ws[j], acc[t][j][i]);
          }
        }
        // keep the scheduler from interleaving sub-tiles (which only raises register pressure)
        __builtin_amdgcn_sched_barrier(0);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  };

  const int S = g.K / 128;
  Regs r0;
  gload(0, r0);
  sstore(0, r0);
  syncthreads();
  for (int st = 0; st < S; st++) {
    // Software pipeline: the next stage's global loads are issued first and only written to LDS after this
    // stage's compute. The scheduler would otherwise sink those loads down to their LDS stores (exposing their
    // whole latency); the sched_barriers pin them. That only works within one basic block, so the body is
    // branch-free: the last iteration re-loads its own stage into the idle buffer instead of skipping the load.
    Regs rn;
    gload(st + 1 < S ? st + 1 : st, rn);
    __builtin_amdgcn_sched_barrier(0);
    compute(st & 1);
    __builtin_amdgcn_sched_barrier(0);
    sstore((st + 1) & 1, rn);
    syncthreads();
  }

  TileCtx<TM, TN> ctx;
  ctx.m0 = bm0 + wm0;
  ctx.n0 = bn0 + wn0;
  ctx.r16 = r16;
  ctx.hl = hl;
#pragma unroll
  for (int j = 0; j < TN; j++) {
    const int n = ctx.col(j);
    ctx.ncol[j] = n < N ? n : N - 1;
  }
  ctx.smem = smem;
  ctx.smem_bytes = (int)sizeof(smem);

  lora_up_bias<T>(g, acc, ctx);
  Epi::template apply<T>(g, e, acc, ctx);
}

// ---------------------------------------------------------------------------------------------------------------
// Instantiations: w4a4_<bf16|fp16>_<s|u>_<BM>x<BN>_<WM>x<WN> (s/u = signed/unsigned activations). The host-side
// tile table (w4a4.py: TILES) must list the same tiles.

#define W4A4_STORE_KERNEL(DT, T, SU, SIGNED, BM, BN, WM, WN)                                                     \
  EXTC __global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)                                             \
      w4a4_##DT##_##SU##_##BM##x##BN##_##WM##x##WN(GemmArgs g) {                                                \
    gemm_w4a4<Tile<BM, BN, WM, WN>, SIGNED, T, StoreEpilogue>(g, StoreEpilogue::Args{});                        \
  }
#define W4A4_TILE(BM, BN, WM, WN)                              \
  W4A4_STORE_KERNEL(bf16, BF16, s, true, BM, BN, WM, WN)       \
  W4A4_STORE_KERNEL(bf16, BF16, u, false, BM, BN, WM, WN)      \
  W4A4_STORE_KERNEL(fp16, FP16, s, true, BM, BN, WM, WN)       \
  W4A4_STORE_KERNEL(fp16, FP16, u, false, BM, BN, WM, WN)

// Measured on an RX 7800 XT at FLUX.1 / FLUX.2 [klein] shapes, these two are within ~4% of the best of
// {64x128, 128x128, 64x256} x {32x32, 32x64, 64x32} everywhere; 64x64 wave tiles spill registers.
W4A4_TILE(64, 128, 32, 32)
W4A4_TILE(128, 128, 32, 64)
