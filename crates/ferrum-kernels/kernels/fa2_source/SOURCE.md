# FA2 Source Bridge Inputs

This directory contains the source inputs used by the `fa2-source` Cargo
feature.

- `flash_attn/src/`: FlashAttention source files from
  `vllm-project/flash-attention`, commit
  `f5bc33cfc02c744d24a2e9d50e6db656de40611c`.
- `flash_attn/LICENSE` and `flash_attn/AUTHORS`: upstream FlashAttention
  attribution files.
- `cutlass/include/`: CUTLASS headers from `NVIDIA/cutlass`, commit
  `62750a2b75c802660e4894434dc55e839f322277`.
- `cutlass/LICENSE.txt`: upstream CUTLASS license.
- `stubs/`: Ferrum-owned minimal header stubs used to compile the selected FA2
  templates without Torch/vLLM headers.

Ferrum's own C ABI adapter is
`../../../../scripts/microbenches/fa2_ferrum_source_shim.cu`; `build.rs`
compiles that shim with the selected FA2 template instantiations into the
`libfa2_source.a` static library.
