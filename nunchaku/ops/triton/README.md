# Triton kernel backend

A portable reimplementation of Nunchaku's CUDA kernels (`src/kernels/`) in
Triton. It runs on AMD GPUs as well as NVIDIA — the CUDA kernels depend on
NVIDIA-only `mma`/`cp.async` PTX and cannot be built for ROCm at all.

The CUDA backend remains the default wherever it is available.

## Selecting a backend

```bash
NUNCHAKU_BACKEND=triton   # or: cuda (default when nunchaku._C is importable)
```

```python
from nunchaku.ops.backend import set_backend, get_backend
set_backend("triton")
```

When `nunchaku._C` is not importable — the normal case on AMD — the backend
resolves to `triton` automatically.

## What is ported

| CUDA source | Triton module | Notes |
| --- | --- | --- |
| `misc_kernels.cu` | `misc.py` | `topk` defers to `torch.topk` |
| `activation_kernels.cu` | `activation.py` | |
| `layernorm_kernels.cu` | `layernorm.py` | |
| `gemm_w8a8.cu`, `zgemm/gemm_w8a8.cu` | `gemm_w8a8.py` | |
| `gemm_f16.cu`, `gemm_batched.cu` | `gemm_fp16.py` | replaces CUTLASS instantiations |
| `awq/gemv_awq.cu`, `awq/gemm_awq.cu` | `awq.py` | GEMV streams the TinyChat packing (group 64); GEMM decodes, then matmuls |
| `zgemm/gemm_w4a4.cu` | `gemm_w4a4.py` | GEMM, quantizer, RMSNorm+RoPE / next-layer-quant epilogues |
| `zgemm/attention.cu` | `attention.py` | FlashAttention-style |
| checkpoint layout | `layout.py` | converts a loaded checkpoint's mma-fragment tensors to natural layout |

## Measured on MI300X (gfx942, ROCm 7.0, Triton 3.6)

| Kernel | Shape | TFLOP/s |
| --- | --- | --- |
| `gemm_w4a4` (int4) | 4096×3072×3072 | 159 |
| `gemm_w4a4` (int4) | 4096×12288×3072 | 184 |
| `gemm_w8a8` (int8) | 4096×3072×3072 | 535 |
| `attention_fp16` | B1 H24 T4096 D128 | 282 (torch SDPA: 264) |

For reference, dense bf16 `torch.matmul` reaches ~550 TFLOP/s on the same part.

(These kernel numbers predate the int4 GEMM / quantizer rewrite below.)

## End to end, RX 7800 XT (gfx1101, 16 GB, Windows 11, ROCm 7.2.1, triton-windows 3.6)

1024×1024, 4 steps, steady state, int4 r32 checkpoints:

| Model | Setup | Before fusions | After | + native int4 GEMM | + native fused epilogues | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX.1-schnell | model CPU offload | 17.8 s | 12.2 s | 10.8 s | 8.8 s | 9.2 GiB (11.2 before fusions) |
| FLUX.2 [klein] 4B | model CPU offload | 9.6 s | 7.4 s | 6.1 s | 5.8 s | 7.9 GiB |
| FLUX.2 [klein] 9B | prompt encoded on CPU, transformer + VAE resident | 15.3 s | 9.3 s | 6.5 s | 6.9 s * | 11.3 GiB |

\* Same session, Klein 9B: 10.5 s Triton only, 7.7 s native plain GEMMs, 6.9 s with
the fused epilogues too (the machine's non-transformer time was ~1.2 s higher than
in the earlier columns' runs; transformer step 2.10 → 1.39 → 1.17 s).

"Fusions" are the int4 GEMM / quantizer rewrite: a tiled activation
quantizer with matrix-core LoRA-down (30 → ~300 GB/s), an int4 GEMM that
decodes nibbles from 32-bit words (25 → ~37 TFLOP/s), a streaming AWQ GEMV
for the adaLN projections, and the CUDA original's Q/K RMSNorm+RoPE and
GELU + next-layer-quantize epilogues fused back into the GEMM. FLUX.1
transformer step: 3.50 → 2.05 s on the 7800 XT, 501 → 301 ms on MI300X.
Accuracy vs the bf16 model is unchanged (FLUX.1-schnell, 4 prompts: image
PSNR vs bf16 17.73 → 18.12 dB; Klein 4B per-block correlation unchanged).
The native gfx11 kernels keep it that way: on the 7800 XT, scored against
bf16 FLUX.1-schnell outputs recorded on MI300X (same 4 prompts), transformer
rel-L2 is 0.1690 native vs 0.1692 Triton and image PSNR 18.24 vs 18.28 dB.

The Windows driver does not OOM past 16 GB; it silently spills into shared
system memory, which is much slower. Keep `torch.cuda.max_memory_allocated()`
under the card's VRAM (e.g. FLUX.2 [klein] 4B is faster with model offload
than fully resident).

"Native int4 GEMM" is the hand-written RDNA3 kernel described next; "native
fused epilogues" adds its Q/K RMSNorm+RoPE and GELU + next-layer-quantize
epilogues, so QKV projections and FLUX.1's fc1 run natively as well.

## Native int4 GEMM on RDNA3 (gfx11)

RDNA3 has an int4 matrix instruction, `v_wmma_i32_16x16x16_iu4`, with twice
the int8 rate (~140 vs ~72 TOPS on an RX 7800 XT), but Triton cannot emit
it: it decodes the nibbles and runs the int4 GEMM on int8 WMMA. `native/`
holds a hand-written HIP kernel built on the int4 instruction
(`w4a4_gfx11.hip`) that `gemm_w4a4()` uses instead of the Triton kernel
when all of these hold:

- the GPU is gfx11 (RDNA3/3.5; MI300X, RDNA4, NVIDIA etc. stay on Triton);
- int4, not NVFP4 (no `wcscales`, `alpha == 1`);
- no `fuse_silu`; fused epilogues only of the two kinds below (Q/K
  RMSNorm + RoPE, GELU + next-layer quantization);
- the shapes and layouts are supported (below).

Anything else falls back to the Triton kernel. The maths and layouts are the
Triton kernel's (bias, LoRA-up with the fp32 hidden state rounded to 16
bits, signed/unsigned activations, `actual_m` row masking, bf16 or fp16);
the integer part is exact and the result is within the same ~1 output ulp
of an fp64 reference.

**How it is built.** The `.hip` source ships with the package and is
compiled on first use with hipRTC (`hiprtc*.dll` from the ROCm Python
wheels' `_rocm_sdk_core` on Windows, `libhiprtc.so` next to torch's HIP
runtime on Linux) for the exact arch present; no host compiler or MSVC is
needed. The code object is launched through the HIP runtime torch already
loaded, via ctypes, on torch's current stream (~10 µs of CPU per call).
Compiling all variants takes ~1 s (measured, RX 7800 XT); after that the
code object loads from the cache in ~10 ms, plus ~0.5 s for a one-time
self-test of every variant against a torch reference.

**Cache.** `$NUNCHAKU_CACHE_DIR/native/` if set, else
`%LOCALAPPDATA%\nunchaku\cache\native\` on Windows and
`${XDG_CACHE_HOME:-~/.cache}/nunchaku/native/` elsewhere. Entries are keyed
by a hash of the source, the arch, the hipRTC library and version, and the
compile options, and written atomically (concurrent processes are safe).
Deleting the directory only forces a recompile. hipRTC's compiler (comgr)
additionally keeps its own cache (`AMD_COMGR_CACHE_DIR`, by default
`%LOCALAPPDATA%\comgr` / `~/.cache/comgr`).

**Environment.**

```bash
NUNCHAKU_NATIVE_GEMM=0   # never use the native kernel
NUNCHAKU_NATIVE_GEMM=1   # use it on any AMD GPU and raise if it cannot be built,
                         # loaded or launched (tests/debugging); unsupported
                         # calls still go to Triton
# unset / auto: gfx11 only, silently falling back
NUNCHAKU_CACHE_DIR=...   # cache root (code objects go to <dir>/native/)
```

If hipRTC is missing, the compile or the self-test fails, or a launch
fails, the native path is disabled for the rest of the process with one
`RuntimeWarning` and every call runs on Triton.

**Constraints** (checked on the host per call; violations fall back):
`K % 128 == 0`, `M % 8 == 0`, `N % 8 == 0`, LoRA rank `R % 16 == 0` (or no
LoRA); `out`, `ascales`, `wscales`, `bias` and `lora_up` share one dtype
(bf16 or fp16) and `lora_act` is fp32; unit inner strides, and 16-byte
aligned rows and base pointers for `act`, `wgt`, the scales and the LoRA
tensors. Every plain GEMM of FLUX.1-schnell and FLUX.2 [klein] 4B/9B meets
them (the quantizer pads `M` to 256; `K` is 3072, 4096, 9216 or 12288; `R`
is 32), so all of them run natively.

**Kernel speed** (RX 7800 XT, bf16, bias + LoRA r32, best of 3 x 20 runs):

| M x N x K | Where | Triton ms | Native ms | Native TFLOP/s | Speedup |
| --- | --- | --- | --- | --- | --- |
| 4096 x 3072 x 3072 | FLUX.1 attn out | 2.084 | 0.930 | 83 | 2.24x |
| 512 x 3072 x 3072 | FLUX.1 txt attn out | 0.319 | 0.139 | 70 | 2.30x |
| 4608 x 3072 x 12288 | FLUX.1 single fc2 | 8.902 | 3.945 | 88 | 2.26x |
| 512 x 3072 x 12288 | FLUX.1 txt fc2 | 1.069 | 0.490 | 79 | 2.18x |
| 4608 x 18432 x 3072 | Klein 4B single fc1 | 13.937 | 6.254 | 83 | 2.23x |
| 4608 x 3072 x 9216 | Klein 4B single fc2 | 6.736 | 2.996 | 87 | 2.25x |
| 4608 x 24576 x 4096 | Klein 9B single fc1 | 24.442 | 10.936 | 85 | 2.24x |
| 4096 x 4096 x 12288 | Klein 9B fc2 | 10.619 | 4.775 | 86 | 2.22x |
| 512 x 4096 x 4096 | Klein 9B txt attn out | 0.525 | 0.224 | 77 | 2.34x |

End to end the plain GEMMs alone save 1.4 s (FLUX.1-schnell), 1.3 s (Klein
4B) and 2.8 s (Klein 9B) per 1024² 4-step image, and the fused epilogues
another 2.0 s, 0.4 s and 0.8 s (table above; steady state, peak VRAM
unchanged). Transformer step (same session): FLUX.1-schnell 2.06 → 1.70 →
1.19 s, Klein 4B 0.98 → 0.66 → 0.56 s, Klein 9B 2.10 → 1.39 → 1.17 s
(Triton → native plain → native fused).

### Fused epilogues

Two more epilogue types on the same main loop, dispatched from
`gemm_w4a4()` under the same gfx11 / `NUNCHAKU_NATIVE_GEMM` rules (and
self-tested on first use like the plain kernel):

- **Q/K RMSNorm + RoPE** (`QKNormRopeEpilogue`, kernels `w4a4qk_*`; QKV
  projections of FLUX.1 and both Klein models). 128 x 128 workgroup tiles
  hold exactly one 128-wide head of the Q, K or V third. The per-row sum of
  squares is a lane-local sum, a 16-lane DPP butterfly, and a fixed-order
  sum of the two waves' partials through LDS; RoPE gets its pair partner
  from the neighbouring lane (DPP) and loads `(sin, cos)` as one 8-byte
  read. Requires `(N / 3) % 128 == 0`, bf16/fp16 `norm_q`/`norm_k` of the
  output dtype and an fp32 rotary table. Costs ~3% over the plain native GEMM.
- **GELU + next-layer LoRA-down + int4 quantization** (`GeluQuantEpilogue`,
  kernels `w4a4gq_*`; FLUX.1's MLP fc1 via `fused_gelu_mlp`). GELU, the
  next layer's LoRA-down on the 16-bit WMMA (the GELU tile and a transposed
  `lora_down` slab staged in LDS, the slab's loads issued before the GELU
  work), and unsigned int4 quantization per 64-group, which lives inside one
  wave (group max: lane-local + DPP butterfly; 8 lanes OR their nibbles into
  one word, one store per lane). The LoRA-down is reduced like the Triton
  kernel's: one partial per N-tile, summed in a fixed order -- no atomics, so
  results are bit-reproducible. GELU and quantization follow the Triton
  kernel op for op (`exp` as `exp2(x log2 e)`), so the codes match it except
  where an fp32 rounding difference in the GEMM output crosses a rounding
  midpoint: 14 of 56.6 M codes (2.5e-7, never by more than 1) and 4.5e-6 of
  the scales (1 ulp) at 4608 x 12288 x 3072. Requires `M % 16 == 0` (the
  quantizer pads `M` to 256) and `N % 128 == 0`. Costs ~11% over the plain
  native GEMM.

Accuracy: the V third and the optional pre-GELU output are bit-exact on the
integer path; Q/K outputs, `lora_act_out` and the dequantized activations
have the same error against fp64 as the Triton fused kernels
(`tests/triton/test_native_epilogues.py`). Model level, Klein 4B transformer
outputs on real pipeline inputs (6 seeds x 4 steps) are as close to the bf16
model as the Triton kernels' (mean relative L2 0.345 native fused vs 0.338
Triton fused and 0.347 Triton with the unfused epilogue kernels; the 4-bit
model amplifies any rounding difference, so the int4 variants differ from
each other as much as from bf16). Images keep composition and text, except
that small text can change between int4 variants (Klein 4B's sign reads
"hello word" with Triton and "hlo llo word" natively).

| M x N x K | Call | Triton fused ms | Native fused ms | Speedup |
| --- | --- | --- | --- | --- |
| 4608 x 9216 x 3072 | QKV + RMSNorm/RoPE, FLUX.1 / Klein 4B single | 7.079 | 3.235 | 2.19x |
| 4096 x 9216 x 3072 | QKV + RMSNorm/RoPE, img stream | 6.292 | 2.866 | 2.20x |
| 512 x 9216 x 3072 | QKV + RMSNorm/RoPE, txt stream | 0.849 | 0.380 | 2.23x |
| 4608 x 12288 x 4096 | QKV + RMSNorm/RoPE, Klein 9B single | 12.469 | 5.633 | 2.21x |
| 512 x 12288 x 4096 | QKV + RMSNorm/RoPE, Klein 9B txt | 1.467 | 0.639 | 2.29x |
| 4608 x 12288 x 3072 | fc1 + GELU/quantize, FLUX.1 single | 10.032 | 4.687 | 2.14x |
| 4096 x 12288 x 3072 | fc1 + GELU/quantize, img stream | 8.911 | 4.160 | 2.14x |
| 512 x 12288 x 3072 | fc1 + GELU/quantize, txt stream | 1.184 | 0.551 | 2.15x |

(RX 7800 XT, bf16, LoRA r32, median of 7 x 20 runs.) FLUX.2 [klein]'s
SwiGLU MLPs are not fused: `linear_in` runs as a plain native GEMM, then
diffusers' `silu(x[:, :H]) * x[:, H:]` and `linear_out`'s quantizer. Fusing
them would need `linear_in`'s gate/up rows interleaved at load time and is
estimated (from the measured cost of those two steps minus a GELU-like
epilogue's) at ~14 ms per Klein 4B and ~30 ms per 9B transformer step (~2.5%).

**Kernel notes.** Each workgroup stages 128-wide K slices of A, W and both
scale tensors through double-buffered LDS; waves issue four int4 WMMAs per
64-wide scale group and fold the group's scales into an fp32 accumulator.
The int32 group sums are converted to float without `v_cvt_f32_i32` by
starting the WMMA accumulator at the bit pattern of 1.5 x 2^23, which
clang 22 (ROCm 7.2) miscompiles unless the constant is hidden behind an
empty `asm volatile`; `sched_barrier`s and a branch-free loop body keep the
next stage's global loads ahead of the current compute. The epilogue is a
separate stage (`lora_up_bias` then an `Epi::apply` hook with the tile's
coordinates and the LDS buffer), so fused epilogues are added as new
`Epi` types without touching the main loop. See the comments in
`w4a4_gfx11.hip`.

## Status: working end to end

`NunchakuFluxTransformer2DModelV2` (FLUX.1-schnell, real checkpoint) now
generates coherent images on the MI300X via the Triton backend, verified with
`examples/v1/flux.1-schnell-triton.py`. All 145 kernel-level tests in
`tests/triton/` pass.

`NunchakuFlux2Transformer2DModel` (FLUX.2 [klein], `transformer_flux2.py`)
runs the community `tonera/FLUX.2-klein-*-Nunchaku` checkpoints; see
`examples/v1/flux.2-klein-4b.py`. Checked block by block against the bf16
diffusers model on real activations captured from a pipeline run (Klein 4B,
1024², steps 1 and 4): per-block output correlation 0.988–1.000 across all
5 double and 20 single blocks; end-to-end image PSNR vs bf16 15.7 dB with
the same composition and text. The packed-QKV attention fast path of the
fork it was ported from is CUDA-only and not carried over, and the
`Flux2KleinKVPipeline` reference-KV-cache modes are not supported.

### Attention

The model attention processors call `nunchaku.ops.dispatch.scaled_dot_product_attention`,
which runs unmasked fp16/bf16 attention on the Triton flash kernel and falls
back to `torch.nn.functional.scaled_dot_product_attention` otherwise. torch's
SDPA was the bottleneck on RDNA3 under Windows ROCm (~4 TFLOP/s and
run-to-run erratic vs ~34 for the Triton kernel; FLUX.1-schnell 1024² on an
RX 7800 XT went from 36–180 s to a steady ~18.5 s per image), and the Triton
kernel is also slightly ahead on MI300X.

### Bugs found and fixed

- **Real out-of-bounds crash**: `gemm_w4a4()` assumed `act`'s row count
  equals `out`'s row count. `fused_gelu_mlp` pads its intermediate quantized
  tensor to a multiple of 256 rows but allocates `output` at the real batch
  size, so the kernel wrote past the end of `out`. Fixed by threading through
  a separate `actual_m` bound for the store mask (`gemm_w4a4.py`).
- **Checkpoint layout for `qweight`/`wscales`**: confirmed correct via exact
  round-trip against `NunchakuWeightPacker`.
- **Checkpoint layout for `proj_down`/`proj_up`** (the SVD low-rank branch):
  fragment-packed, via `NunchakuWeightPacker.unpack_lowrank_weight`.
  `proj_down` specifically needs an extra transpose after unpacking — the
  on-disk tensor is packed from the *transposed* logical `(rank, in_features)`
  layout, not `(in_features, rank)`.
- **`bias`/`smooth_factor` are NOT fragment-packed** (correcting an earlier,
  incorrect finding in this same investigation): they're stored plain,
  natural-layout. The original "confirmation" that they needed unpacking was
  a false positive — comparing raw checkpoint values against the
  full-precision reference without accounting for the SVDQuant weight
  decomposition (`w_checkpoint = quant(w_ref * smooth - lowrank)`) and the
  unsigned-activation bias-shift correction (`bias' = bias_ref -
  shift * rowsum(w_deq)`) makes correct plain values look permuted. Once the
  comparison correctly reconstructs `main_term + lowrank_term + bias`, plain
  `bias`/`smooth_factor` match the reference almost exactly (corr ≈ 0.998).
- **Missing unsigned (`[0, 15]`) quantization range for the next-layer-quant
  epilogue**: `_quantize_row_kernel` always used the symmetric `[-8, 7]`
  range, even though the CUDA reference (`EpilogueQuantize`'s
  `USE_UNSIGNED = !USE_FP4`) quantizes unsigned whenever the row has already
  been shifted non-negative (i.e. always, for int4, in this epilogue). Fixed
  by adding a `UNSIGNED`/`unsigned` parameter threaded through
  `_quantize_row_kernel`/`_launch_quantize`/`quantize_w4a4_act_fuse_lora`,
  set whenever `gemm_w4a4()`'s own next-layer-quant epilogue is invoked
  (`unsigned=not fp4`).
- **The critical bug**: the LoRA-down projection inside
  `_quantize_row_kernel` incorrectly applied both the unsigned-activation
  shift and the smoothing division before projecting through `lora_down`.
  The CUDA reference (`quantize_w4a4_fuse_lora_kernel`) runs
  `EpilogueLoraDown` directly on the loaded activation fpsum, strictly
  *before* `EpilogueQuantize` applies the shift or smoothing — so the
  LoRA-down projection should consume the raw (GELU'd only) activation. This
  bug scaled the LoRA correction wrong by a layer-dependent factor (as large
  as ~3.4x for `mlp_context_fc1`, where the low-rank branch dominates due to
  outlier channels), while leaving its *direction* correct (corr ~0.996 vs.
  the correctly-scaled target) — the signature of a spurious elementwise
  scale rather than a permutation/orientation bug. Fixing this raised the
  text-stream MLP's correlation from ~0.59 to ~0.99, and per-block
  correlations across all 19 joint + 38 single blocks from ~0.86-0.98 to
  ~0.98-1.00.

### Methodology that found these

Block-by-block diffing against a full-precision reference
(`diffusers.FluxTransformer2DModel`) on identical real activations (captured
via forward hooks on the reference model during an actual pipeline run, not
synthetic random inputs) was essential — synthetic-only tests are
self-consistent by construction and cannot catch bugs where a component
reads its own (consistently wrong) assumption. The specific tells that found
each bug:

- An exact-zero error after applying a permutation fix (confirms
  orientation/permutation bugs).
- A supposedly-corrective term (LoRA) *increasing* error instead of reducing
  it (confirms wrong orientation).
- Correlation staying high (~0.99+) while magnitude is off by a *constant
  scalar factor* across many samples (confirms a missing/extra scale, not a
  permutation) — found by fitting the optimal scalar via least squares and
  checking it explained the residual almost perfectly.
- Progressive per-block hooks (capturing every joint/single block's output
  during one real pipeline run) to localize *where* correlation degrades
  sharply, rather than only inspecting the final image.

## Known gaps

- **Qwen-Image, Z-Image and SDXL loaders don't convert checkpoints to the
  natural layout.** Only the FLUX.1 and FLUX.2 `from_pretrained` call
  `convert_svdq_linear_to_natural_layout` on the Triton backend, so the
  others feed the CUDA mma-fragment layout to kernels that expect row-major
  weights. Z-Image's RoPE hook also emits the CUDA-only `pack_rotemb`
  layout. These models are untested on this backend.
- **Unported `gemm_w4a4` epilogues.** QKV packing (`out_q`/`out_k`/`out_v`,
  only used by the optional `"nunchaku-fp16"` attention processor) and the
  SANA-only linear-attention paths (`poolout`, `out_vk`, `out_linearattn`)
  are not implemented; `nunchaku/ops/gemm.py` raises `NotImplementedError`
  when they are requested.
- **Model-level integration for SANA.** `nunchaku/models/transformers/transformer_sana.py`
  delegates its entire forward pass to the compiled C++ `QuantizedSanaModel`
  and is out of scope for this backend.

## Tests

```bash
pytest tests/triton/            # all pass on MI300X (171, gfx11-only cases skipped) and RX 7800 XT (289)
python tests/triton/bench_kernels.py
```

`test_native_gemm.py` checks the native GEMM against an fp64 reference and
the Triton kernel (gfx11 only), and its dispatch/fallback rules and the
hipRTC compile/cache/launch machinery (any AMD GPU).

Kernel-level tests compare against fp32 torch references computed from the
same packed tensors the kernels read. `test_layout.py` additionally validates
checkpoint-layout conversions against `NunchakuWeightPacker` round-trips.
Neither test suite currently includes a full-model / real-checkpoint
end-to-end check — that comparison was done manually against a
`diffusers`-loaded full-precision reference model (see "Status" above) and
would be a good candidate for a proper regression test.
