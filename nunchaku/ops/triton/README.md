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
| `awq/gemv_awq.cu`, `awq/gemm_awq.cu` | `awq.py` | decodes the TinyChat packing, then matmuls |
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
pytest tests/triton/            # 145 tests, all passing on MI300X
python tests/triton/bench_kernels.py
```

Kernel-level tests compare against fp32 torch references computed from the
same packed tensors the kernels read. `test_layout.py` additionally validates
checkpoint-layout conversions against `NunchakuWeightPacker` round-trips.
Neither test suite currently includes a full-model / real-checkpoint
end-to-end check — that comparison was done manually against a
`diffusers`-loaded full-precision reference model (see "Status" above) and
would be a good candidate for a proper regression test.
