# Direct CUDA microbenches

These are standalone .cu programs compiled with `nvcc` directly — they
**bypass the ferrum cargo build** (which can take 30+ min cold). Use
these to verify CUDA-level hypotheses in 5-10 minutes instead of
rebuilding the entire project.

## Building

On any host with `nvcc` + CUDA 12+/13 toolchain:

```bash
# Generic
nvcc -O3 -arch=sm_89 -std=c++17 <source.cu> -o <binary>

# Multi-TU (for cross-TU consistency tests)
nvcc -O3 -arch=sm_89 -std=c++17 -I<include-path> \
    file_a.cu file_b.cu -o binary
```

## Benchmarks

| file | what it verifies |
|---|---|
| `graph_bench.cu` | CUDA Graph capture+replay speedup vs naive serial kernel launches. Measures launch-overhead headroom. |
| `sync_barrier_bench.cu` | Cost of `cuStreamSynchronize` per layer vs async pipeline. Validates the host-route DtoH-barrier hypothesis. |
| `layer_split_overlap_probe.cu` | Two-GPU layer-split scheduling probe. Simulates stage0, host bridge, stage1, logits, and microbatch overlap without Rust/Cargo or model loading. Use it before changing the product overlap path. |
| `scalar_type_id_test.cu` (+ `_other_tu.cu`) | Verifies `vllm::ScalarType::id()` constexpr produces consistent values across translation units, and that template specializations keyed on those IDs dispatch correctly. **Two-TU compile required.** |
| `dense_marlin_gemma3_perf.cu` | Direct C-ABI benchmark for Ferrum's default dense Marlin GEMM on Gemma3-27B GPTQ qkv/o/gate_up/down shapes. Use before changing dense Marlin tile selection or grid policy. |
| `moe_marlin_active65_perf.cu` | Direct C-ABI benchmark for Ferrum's vLLM-Marlin MoE gate/up and down kernels on the real c32 active65 route. Links against existing `ferrum-kernels` build objects, avoiding a full Cargo rebuild. |
| `vllm_flash_attn_varlen_probe.py` | Runs vLLM 0.20.2 `flash_attn_varlen_func` on Qwen3 paged-varlen prefill/mixed shapes. Use it to size the upside before porting a real FA-style varlen kernel into Ferrum. |
| `fa2_direct_ffi_probe.cpp` | Calls the `flash::run_mha_fwd` symbol exported by vLLM's `_vllm_fa2_C.abi3.so` directly, bypassing Python and `torch.ops`, to test whether an opt-in FA2 C-ABI shim is viable. |
| `fa2_ferrum_shim.cpp` | Out-of-tree C ABI shim for `FERRUM_FA2_DIRECT_FFI=1`; build it with `build_fa2_ferrum_shim.sh` and point `FERRUM_FA2_DIRECT_FFI_SHIM` at the resulting `.so`. |
| `fa2_ferrum_source_shim.cu` | Source-built FA2 C ABI shim with no vLLM/Torch runtime link. Build it with `build_fa2_ferrum_source_shim.sh`; this is the bridge toward a Ferrum-owned FA2 integration. |

## Notes

Each microbench prints its findings + a one-line VERDICT. They're
designed to be quick smoke-tests, not full benches. Adding new ones is
encouraged whenever you have a CUDA-level hypothesis to verify
independently from the ferrum runtime.
