# W2 Producer-Touch Product CUDA Compile Smoke

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_product_compile_2026-06-16`
- Lane: W2 CUDA compile smoke for producer-touch product source
- Instance: cached Vast 1x RTX 4090 instance `40826362`
- Build command:
  `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Build rc: `0`
- Build result:
  `Finished release profile [optimized] target(s) in 3m 27s`
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 565.77
- Remote base HEAD: `017300426514d62e8e50ac1546ff77d4d54fd6ce`
- Local HEAD at compile time:
  `59a1ab5541a00a4b48c184d5a31663e51128b583`
- Dirty source: product producer-touch source files were synced onto the
  remote working tree for a compile smoke; the local source diff is saved in
  `local/product_touch_source.diff`.

## Interpretation

The CUDA product source compiles with the release CUDA feature set used for
W2 diagnostics. This validates the new CUDA kernel symbol, generated PTX path,
and Rust CUDA feature wiring at compile/link time.

This is not correctness or performance evidence by itself. It is paired with
`w2_producer_touch_product_correctness_2026-06-16` for minimal product
entrypoint correctness.

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
