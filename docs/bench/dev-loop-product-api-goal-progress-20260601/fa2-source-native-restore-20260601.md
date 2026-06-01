# FA2 Source Native Restore - 2026-06-01

Purpose: restore the M3 80% performance path behind `FERRUM_FA2_SOURCE=1` so
it is Ferrum-owned at build/runtime rather than dependent on an external
FlashAttention checkout.

Changed code:

- `crates/ferrum-kernels/kernels/fa2_source/ferrum_fa2_paged_varlen.cu`
  now exports `ferrum_fa2_paged_varlen_fwd` from in-repo CUDA code.
  The native kernel uses warp-partition online softmax partials and block-level
  log-sum-exp merging instead of the earlier full-score shared buffer reader.
- `crates/ferrum-kernels/build.rs` now compiles only that in-repo CUDA file for
  the `fa2-source` static library.
- `scripts/check_fa2_source_native.py` provides a source-boundary guard so the
  product `fa2-source` feature does not silently regress to external
  FlashAttention/CUTLASS source dependencies or runtime source/direct-FFI flag
  conflation. It also rejects the old full-score shared-buffer reader shape for
  the native kernel.
- `Qwen3MoeRuntimeEnv` keeps `fa2_source` distinct from `fa2_direct_ffi`; the
  source path still allocates the FA-compatible K/V pool and calls the same C
  ABI attention entry, but artifacts no longer need to mislabel it as a direct
  runtime shim path.
- The product `fa2-source` build no longer reads `FERRUM_FA2_SRC_DIR`,
  `FERRUM_CUTLASS_INCLUDE_DIR`, `CUTLASS_INCLUDE_DIR`, or a
  `/workspace/vllm-flash-attention-*` checkout.
- Legacy FA2 source-shim microbench scripts are retained only for historical
  replay; they are no longer the product `fa2-source` build path.

Current status:

- Code restoration is present locally.
- Native source-boundary guard passed locally:
  `python3 scripts/check_fa2_source_native.py --self-test` and
  `python3 scripts/check_fa2_source_native.py`.
- Runtime source/direct-FFI flag separation is present locally but still needs
  ordinary Rust test coverage in the next evidence refresh.
- GPU build, Paris/multi-turn smoke, and focused c32 performance validation
  have been exercised during the native FA2 restore loop.
- All-cell N=3 source-FA2 validation now exists at
  `/workspace/m3-fa2-source-current-allcells-n3-20260601`; all c=1/4/16/32
  cells passed Paris, multi-turn, three-turn recall, benchmark completion, and
  artifact validation.
- The all-cell result still does not close the M3 80% target because c32 source
  FA2 measured `1488.08 tok/s`, about `0.754×` of same-pod vLLM `1972.9`.
- Earlier source-linked FA2 smoke artifacts remain performance-direction
  evidence only because they used external FlashAttention source at build time.

Next required GPU evidence:

```bash
cargo build --release -p ferrum-cli \
  --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source

OUT_ROOT=/workspace/m3-fa2-native-restore-c32-smoke-20260601 \
  BUILD=0 FA2_SOURCE=1 FA2_EXTRA_LD_LIBRARY_PATH="" \
  CONCURRENCY=32 NUM_PROMPTS=64 WARMUP_REQUESTS=10 REPEATS=1 \
  bash scripts/m3_fa2_direct_ffi_ab.sh
```

Completion impact:

- This unblocks the intended product shape for the FA2 lever, but it does not
  close the M3 80% target until fresh same-pod correctness and all-cell
  throughput evidence exists for the native in-repo path.
