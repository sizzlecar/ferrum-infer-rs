# Direct CUDA microbenches

These are standalone .cu programs compiled with `nvcc` directly — they
**bypass the ferrum cargo build** (which can take 30+ min cold). Use
these to verify CUDA-level hypotheses in 5-10 minutes instead of
rebuilding the entire project.

Build helpers that need Marlin or vLLM sources verify and materialize the
versioned source bundle under the external Ferrum cache. Set
`FERRUM_NATIVE_SOURCE_ROOT` only to use an already materialized tree with the
same manifest identity; mismatched or extra files are rejected.

## Building

On any host with `nvcc` + CUDA 12+/13 toolchain:

```bash
# Generic
nvcc -O3 -arch=sm_89 -std=c++17 <source.cu> -o <binary>

```

## Benchmarks

| file | what it verifies |
|---|---|
| `cuda_graph_segment_probe.cu` | Native CUDA probe comparing one monolithic graph with vLLM-style segmented graph replay for a Gemma3-like decode launch count. Use it before changing product graph capture granularity. |
| `layer_split_overlap_probe.cu` | Two-GPU layer-split scheduling probe. Simulates stage0, host bridge, stage1, logits, and microbatch overlap without Rust/Cargo or model loading. Use it before changing the product overlap path. |
| `paged_varlen_window_correctness.cu` | Direct C-ABI correctness probe for Ferrum's paged varlen attention one-pass and split-K kernels. Compares `sliding_window=0` and a local-window case against CPU reference before enabling Gemma3 unified prefill semantics. |
| `paged_varlen_split_qkv_correctness.cu` | Direct correctness probe for split-QKV paged varlen attention. |
| `gemma3_shadow_graph_bench.cu` | Standalone CUDA graph probe for a Gemma3-style 62-layer device F32 residual shadow decode step. Use it to validate graph replay stability and launch-overhead headroom before enabling graph capture on the product Gemma3 shadow path. |
| `dense_marlin_gemma3_perf.cu` | Direct C-ABI benchmark for Ferrum's default dense Marlin GEMM on Gemma3-27B GPTQ qkv/o/gate_up/down shapes. Reports hot event timing, product-profile-style host-sync timing, limited cold-cache timing, multi-weight-cycle timing, and block-policy probes for key auto-tile shapes. Use before changing dense Marlin tile selection or grid policy. |
| `dense_vllm_marlin_gemma3_perf.cu` | Direct C-ABI benchmark for the source-bundled vLLM dense GPTQ-Marlin kernel on Gemma3-27B GPTQ qkv/gate_up/down shapes. The build script uses a temporary minimal selector, so this stays a native diagnostic and does not alter product dense GPTQ routing. |
| `dense_triton_w4a16_gemma3_perf.cu` | Standalone Gemma3 dense W4A16 Triton comparison probe. |
| `gemma3_gate_up_split_perf.cu` | Direct C-ABI benchmark for the Gemma3 GPTQ tail-MLP `gate_up` hotspot. Compares the product fused `n=43008` Marlin projection plus GeGLU against serial and two-stream split `gate`/`up` projections under an 8-layer weight cycle. Use before productizing any split gate/up loader or multi-stream projection path. |
| `build_and_run_gemma3_marlin_cache_policy_perf.sh` | Native A/B for the Gemma3 tail-MLP chain with legacy plain Marlin B-weight `cp.async.cg` (`FERRUM_MARLIN_CP_ASYNC_PLAIN=1`) versus the product-default `L2::evict_first` cache-policy path. Use after Marlin cache-policy changes to confirm the default remains the measured fast path. |
| `fa2_ferrum_shim.cpp` | Out-of-tree C ABI shim for `FERRUM_FA2_DIRECT_FFI=1`; build it with `build_fa2_ferrum_shim.sh` and point `FERRUM_FA2_DIRECT_FFI_SHIM` at the resulting `.so`. |

## Notes

Each microbench prints its findings + a one-line VERDICT. They're
designed to be quick smoke-tests, not full benches. Adding new ones is
encouraged whenever you have a CUDA-level hypothesis to verify
independently from the ferrum runtime.
