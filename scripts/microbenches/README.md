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
| `scalar_type_id_test.cu` (+ `_other_tu.cu`) | Verifies `vllm::ScalarType::id()` constexpr produces consistent values across translation units, and that template specializations keyed on those IDs dispatch correctly. **Two-TU compile required.** |

## Notes

Each microbench prints its findings + a one-line VERDICT. They're
designed to be quick smoke-tests, not full benches. Adding new ones is
encouraged whenever you have a CUDA-level hypothesis to verify
independently from the ferrum runtime.
