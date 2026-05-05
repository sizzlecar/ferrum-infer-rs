# vLLM gptq_marlin port — work in progress

## What's here

Copied from vLLM @ `/Users/chejinxuan/py_ws/vllm/csrc/quantization/marlin/`
plus `csrc/core/scalar_type.hpp`. Goal: replace ferrum's existing
`crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu` (IST-DASLab port,
~800 LOC) with vLLM's more-tuned variant (~5K LOC + 100+ generated
kernel variants). Expected win: ~2× per Marlin GEMM at m=16 (evidence:
op-profile showed ferrum's Marlin call avg 98us vs vLLM's estimated
49us at m=16 on the same RTX 4090 + Llama-3.1-8B-INT4 model).

## What's done

- Copied all .cu/.h/.cuh files (10 files)
- Created `torch_stubs.h` to replace `<torch/library.h>` (TORCH_CHECK,
  TORCH_CHECK_NOT_IMPLEMENTED macros + opaque `torch::Tensor` fwd-decl)
- Patched `scalar_type.hpp` to use `torch_stubs.h` instead of
  `torch/library.h`

## What's blocking compile

1. **`marlin.cu` line 260: `#include "kernel_selector.h"`**. This is a
   GENERATED file — vLLM's build runs `generate_kernels.py` first which
   produces 100+ `kernel_*.cu` instantiations + a `kernel_selector.h`
   dispatcher. We need to either:
   - Run the Python script with the right ARCHS arg, OR
   - Hand-write `kernel_selector.h` for our shape set (Llama-3.x INT4
     desc_act FP16 — covers GPTQ-INT4 b_type with kFloat16 a/c/s)

2. **`#include "core/registration.h"`** in marlin.cu — pulls in PyBind
   registration. Drop or stub out (we don't expose Python bindings).

3. **`marlin_gemm()` torch::Tensor entry point** (line 49). We don't
   need this. Strip and keep only `marlin_mm()` (line 315, raw void*).

4. **`marlin_template.h` is 2081 LOC**. Has #ifdef paths for FP8 (SM89/
   SM120), BF16 (kFloat16+kBFloat16), INT8 act, FP4. Need to verify
   the FP16+INT4 path compiles cleanly when other paths' instantiations
   are absent.

5. **`scalar_type.hpp` uses `constexpr` ScalarType + ScalarTypeId enum**.
   Need to ensure our stubs don't break the `constexpr` evaluation
   that generates the kernel template ids.

## Realistic estimate

Full port: 1-2 weeks of focused CUDA work to get correct + integrated.
Single autonomous-loop iteration is not enough.

## Suggested next steps (multi-session)

Session 1:
- Hand-write a minimal `kernel_selector.h` that supports just our
  shapes (kFloat16 + kU4B8, group_size=128, m_blocks={1,2,3,4})
- Compile a single instantiation standalone (e.g. (1, 8, 8, group=128,
  fp16 a/c)) and see what link errors remain.

Session 2:
- Generate the full per-shape set our model hits (Llama-3.1-8B has
  qkv K=4096 N=6144, gate_up K=4096 N=28672, etc.)
- Wire into bindgen-cuda build path.

Session 3:
- Replace ferrum's `marlin.rs::marlin_gemm` to call vLLM's `marlin_mm`.
- Bench M2 c=4/c=16, verify correctness + speedup.
