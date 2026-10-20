// Minimal, header-free HIP device prelude for the native kernels.
//
// hipRTC compiles device code only and predefines the HIP attribute macros
// (__global__, __device__, ...), so these kernels need no hip_runtime.h --
// and hence no host C++ toolchain (no MSVC on Windows). The same file also
// works with a plain offline compile for ISA inspection:
//   clang -x hip --offload-arch=gfx1101 --cuda-device-only -nogpuinc -nogpulib -O3 -std=c++17 -S w4a4_gfx11.hip
#pragma once

#ifndef __HIPCC_RTC__  // hipRTC predefines these
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __attribute__((always_inline)) inline
#endif
#undef __launch_bounds__
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))
#define EXTC extern "C"

typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef unsigned short u16;

static __device__ __forceinline__ unsigned tid_x() { return __builtin_amdgcn_workitem_id_x(); }
static __device__ __forceinline__ unsigned bid_x() { return __builtin_amdgcn_workgroup_id_x(); }

// = HIP's __syncthreads(). A bare s_barrier only synchronizes execution; the fences make every LDS write issued
// before the barrier visible to every wave after it (without them the compiler may move LDS accesses across it).
static __device__ __forceinline__ void syncthreads() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// Reinterpret int bits as float. Pass vector elements through this scalar parameter: clang 22 (ROCm 7.2) mis-lowers
// __builtin_bit_cast(float, vec[i]) to a reinterpretation of element 0 regardless of i.
static __device__ __forceinline__ float i2f(int x) { return __builtin_bit_cast(float, x); }

static __device__ __forceinline__ float bf16_to_f32(u16 x) { return __builtin_bit_cast(float, (unsigned)x << 16); }

// fp32 -> bf16, round to nearest even (gfx11 has no conversion instruction). NaN payloads are not preserved.
static __device__ __forceinline__ u16 f32_to_bf16(float f) {
  unsigned u = __builtin_bit_cast(unsigned, f);
  u += 0x7fff + ((u >> 16) & 1);
  return (u16)(u >> 16);
}
