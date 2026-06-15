# W2 Marlin Shape Trace Probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_shape_trace_probe_2026-06-15/`.
- Lane: W2 Gemma3 CUDA product-path Marlin shape trace.
- Instance: Vast `40826362`, 1x RTX 4090, cache-retained native CUDA machine.
- Source commit: `b3403dd5394bb044690c918535a71ccc202cd3e7`.
- Binary SHA256: `730df7d84ede559b7ace54abcf1a6c16a3a81e55113789c5e3bc37c9f3844b8f`.
- Cleanup: instance stopped; `vast_shutdown/stopped.json` reached `actual_status=exited`.

## Result

- `cargo_build.rc=0`.
- CUDA/marlin feature compile retry rc `0`.
- `run.status=PASS`.
- Chat smoke content: `5`; usage: `prompt_tokens=23`, `completion_tokens=3`.
- `shape_trace_lines=256`; all 256 lines parsed.

## Shape Trace

The product chat smoke was run with:

```text
FERRUM_MARLIN_TRACE_SHAPES=1
FERRUM_MARLIN_TRACE_SHAPES_MAX=256
FERRUM_DECODE_OP_PROFILE=1
```

Captured Marlin dispatches:

- Calls `0..247`: prefill, `m=23`, 62 layers x 4 projections.
- Calls `248..255`: decode, `m=1`, first 2 layers x 4 projections before the trace cap.

The dispatch labels and shapes were:

- `qkv_proj`: `m={23,1}`, `n=8192`, `k=5376`.
- `o_proj`: `m={23,1}`, `n=5376`, `k=4096`.
- `gate_up_proj`: `m={23,1}`, `n=43008`, `k=5376`.
- `down_proj`: `m={23,1}`, `n=5376`, `k=21504`.

## Interpretation

This probe confirms that the diagnostic trace is wired into the real `ferrum serve`
product path and that a single-request decode feeds Marlin as `m=1`. It does not
yet prove the c16 bottleneck, because the request was intentionally a short
correctness smoke rather than a c16 bench. Combined with the native Ferrum/vLLM
Marlin weight-cycle probe, the next high-value check is a c16 product-path
shape/cadence trace: verify whether concurrent decode consistently reaches
`m ~= 16`, and measure whether time is lost in scheduling gaps, non-Marlin ops,
or per-step synchronization.

## Release-Grade Status

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced. This is diagnostic
evidence only and is not a release-grade performance claim.
