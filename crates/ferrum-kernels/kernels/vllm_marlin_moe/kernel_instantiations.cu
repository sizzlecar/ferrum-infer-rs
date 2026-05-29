// CUDA-13 hidden-default-visibility workaround.
//
// CUDA 13 nvcc generates templated __global__ kernel host stubs with
// hidden ELF visibility by default. The implicit instantiations triggered
// inside the inline dispatcher in ops.cu therefore produce hidden
// symbols, and `ar`-bundled static archives end up rejecting the cross-
// TU lookups rust-lld performs at the final link step (the symbols are
// present but marked hidden, so the linker reports
// `undefined hidden symbol marlin_moe_wna16::Marlin<...>`).
//
// vLLM HEAD works around this with `generate_kernels.py` which emits one
// `.cu` per template configuration; each explicit instantiation at
// namespace scope is required to have external linkage, so the symbol
// is visible to the linker.
//
// We backport just the idiom: a single TU that explicitly instantiates
// every `Marlin<...>` template configuration referenced by the
// dispatcher in ops.cu. This pairs the existing
// `__attribute__((visibility("default")))` on the template definition
// with the explicit-instantiation linkage promotion that CUDA 13
// requires.
//
// The configuration set mirrors `COMMON_GET_IF(kU4B8)` from ops.cu —
// (scalar_t=half, w_type=kU4B8) is M3's GPTQ-INT4 path. If a non-fp16
// MoE model gains traction, add the matching block below.

#include "kernel.h"
#include "marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

// Force explicit instantiation of a single Marlin<...> configuration.
// `pipe_stages` is hard-coded to 4 to match the dispatcher in ops.cu
// (see vLLM `generate_kernels.py` TEMPLATE).
#define _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, THREAD_M_BLOCKS,                \
                     THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8,         \
                     GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT)                   \
  template __global__ void                                                     \
  Marlin<SCALAR_T, vllm::W_TYPE.id(), vllm::S_TYPE.id(), NUM_THREADS,          \
         THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, 4, \
         GROUP_BLOCKS, IS_ZP_FLOAT>(MARLIN_KERNEL_PARAMS);

// Mirror of ops.cu COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS).
#define _COMMON_INSTANTIATE_M1(SCALAR_T, W_TYPE, S_TYPE, N_BLOCKS, K_BLOCKS,   \
                               NUM_THREADS)                                    \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2,       \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4,       \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8,       \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1,     \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8,      \
               NUM_THREADS, false)

// Mirror of ops.cu COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS).
#define _COMMON_INSTANTIATE_M234(SCALAR_T, W_TYPE, S_TYPE, N_BLOCKS, K_BLOCKS, \
                                 NUM_THREADS)                                  \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1,     \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1,     \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1,     \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4,      \
               NUM_THREADS, false)                                             \
  _INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8,      \
               NUM_THREADS, false)

// Mirror of ops.cu COMMON_GET_IF(W_TYPE) — instantiate all tile configs.
#define _COMMON_INSTANTIATE(SCALAR_T, W_TYPE, S_TYPE)                          \
  _COMMON_INSTANTIATE_M1(SCALAR_T, W_TYPE, S_TYPE, 8, 8, 256)                  \
  _COMMON_INSTANTIATE_M1(SCALAR_T, W_TYPE, S_TYPE, 8, 4, 128)                  \
  _COMMON_INSTANTIATE_M1(SCALAR_T, W_TYPE, S_TYPE, 4, 8, 128)                  \
  _COMMON_INSTANTIATE_M234(SCALAR_T, W_TYPE, S_TYPE, 16, 4, 256)               \
  _COMMON_INSTANTIATE_M234(SCALAR_T, W_TYPE, S_TYPE, 8, 4, 128)                \
  _COMMON_INSTANTIATE_M234(SCALAR_T, W_TYPE, S_TYPE, 4, 8, 128)

// === scalar_t=half ===
// M3 primary: Qwen3-30B-A3B-GPTQ-Int4 (fp16 act, INT4 weight).
_COMMON_INSTANTIATE(half, kU4B8, kFloat16)
// Other fp16 quant flavors the dispatcher will accept.
_COMMON_INSTANTIATE(half, kU4, kFloat16)
_COMMON_INSTANTIATE(half, kU8B128, kFloat16)

// === scalar_t=nv_bfloat16 ===
// In case a bf16-activated MoE lands. Skip if it doubles build time
// unacceptably — gate behind FERRUM_VLLM_MOE_BF16 then.
_COMMON_INSTANTIATE(nv_bfloat16, kU4B8, kBFloat16)
_COMMON_INSTANTIATE(nv_bfloat16, kU4, kBFloat16)
_COMMON_INSTANTIATE(nv_bfloat16, kU8B128, kBFloat16)

#undef _COMMON_INSTANTIATE
#undef _COMMON_INSTANTIATE_M1
#undef _COMMON_INSTANTIATE_M234
#undef _INSTANTIATE

}  // namespace MARLIN_NAMESPACE_NAME
