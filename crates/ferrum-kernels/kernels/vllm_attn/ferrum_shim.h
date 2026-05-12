// ferrum_shim.h — minimal stubs for the torch macros vLLM's paged-attention
// kernel headers reach for. Mirrors `vllm_torch_shim.h` from
// vllm_marlin_moe/. Only needed for the few `TORCH_CHECK` sites in
// dtype_fp8.cuh (we never hit the FP8 path; the runtime check there is
// dead code in our kAuto-only instantiation, but it has to compile).
#pragma once

#include <cstdio>
#include <cstdlib>

#define TORCH_CHECK(cond, ...)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr,                                                     \
                   "[vllm_attn] TORCH_CHECK failed at " __FILE__ "\n");        \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
