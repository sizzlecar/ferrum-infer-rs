# W2 Unified Argmax C16 CUDA Diagnostic

- Source SHA: `63b8565eb4f40b3bc94ac48729cd4cd0fd00b2b0`
  (`perf(models): use logits policy in dense unified forward`).
- Binary SHA256:
  `b0676434810f2824094a12a1ccd9bea666a9aaf4e72bfd404c00455888ef407f`.
- GPU: Vast instance `41218739`, 1x RTX 4090, driver `580.95.05`,
  CUDA visible through `nvidia-smi` as 13.0 and `nvcc` as 12.4.
- Cleanup: `cur_state=stopped`, `actual_status=exited` at
  `2026-06-16T18:48:29Z`.

## Correctness

- CUDA release binary build passed; fresh build time was `43m 52s`.
- Product `ferrum run` smoke passed: stdout was `5`, rc `0`.
- Product `ferrum serve` streaming smoke passed in the initial run:
  `SERVE_SMOKE_OK True`, exactly one `[DONE]`, usage present.
- Profile rerun `serve` streaming smoke also passed: `smoke.ok=True`.
- Both c16 `bench-serve --fail-on-error` runs completed `[16]`, errored
  `[0]`, and used `output_token_count_source=usage`.

## Diagnostic Performance

- Initial non-profile product-path diagnostic:
  - output throughput: `273.31374287329345 tok/s`;
  - request throughput: `17.08210892958084 req/s`;
  - TTFT p50/p95: `583.790647 / 584.2009305 ms`;
  - ITL p95: `27.1778395 ms`.
- Profile rerun with typed profile config active:
  - output throughput: `167.05125694272851 tok/s`;
  - request throughput: `10.440703558920532 req/s`;
  - TTFT p50/p95: `777.337679 / 876.2575065 ms`;
  - ITL p95: `85.40438435 ms`.

## Bottleneck Evidence

- Effective profile config included
  `FERRUM_BATCH_DECODE_PROF=1`, `FERRUM_NEXT_BATCH_PROF=1`,
  `FERRUM_UNIFIED_POST_PROF=1`, `FERRUM_DECODE_OP_PROFILE=1`,
  and `FERRUM_MARLIN_PROFILE=1`.
- Target mixed c16 frame:
  `call#21 m_total=897 num_seqs=16 prefill=12 decode=4 total=383684us`.
- Readback dropped from the previous same-shape `22039us` to `516us`
  (`0.0234x` of the previous readback time).
- The remaining largest buckets are dense GPTQ Marlin MLP:
  `gate_up=174891us`, `down=110906us`, `marlin_kernel=312084us`,
  with `lm_head=3167us` and `unwrapped=726us`.

## Status

This is diagnostic evidence, not release evidence. No
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
