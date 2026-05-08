// vllm_torch_shim.h — minimal substitutes for vLLM's torch dependencies so
// the marlin_moe_wna16 kernel sources can compile WITHOUT linking against
// libtorch. Only the names actually used by the vendored files are defined.
//
// `TORCH_CHECK(cond, ...)` lives in static + runtime arg validation paths.
// Failures here are programming errors at our integration layer (caller must
// satisfy the contract) — we map them to a stderr printf + abort, matching
// what the original macro does in its eager-mode evaluator.

#pragma once

#include <cstdio>
#include <cstdlib>

// Stub — concatenate all the args after `cond` into a single message; vLLM
// callers pass things like `("prob_n = ", prob_n, " is not divisible by ...")`.
// Variadic streamed printing through cout/cerr is what TORCH_CHECK does on the
// torch side; a simpler approach here: print a fixed prefix + the line, abort.

#define VLLM_SHIM_STRINGIFY_DETAIL(x) #x
#define VLLM_SHIM_STRINGIFY(x) VLLM_SHIM_STRINGIFY_DETAIL(x)

#define TORCH_CHECK(cond, ...)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(                                                            \
          stderr,                                                              \
          "[vllm_marlin_moe] TORCH_CHECK failed at "                           \
          __FILE__ ":" VLLM_SHIM_STRINGIFY(__LINE__) ": "                      \
          #cond "\n");                                                         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define TORCH_CHECK_NOT_IMPLEMENTED(cond, ...)                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(                                                            \
          stderr,                                                              \
          "[vllm_marlin_moe] TORCH_CHECK_NOT_IMPLEMENTED at "                  \
          __FILE__ ":" VLLM_SHIM_STRINGIFY(__LINE__) ": "                      \
          #cond "\n");                                                         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
