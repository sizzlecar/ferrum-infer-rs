# W2 Gemma3 Producer-Integrated Down Touch Native CUDA Probe

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_native_probe_2026-06-16`
- Lane: W2 native CUDA producer-integrated tail-MLP cache probe
- Instance: cached Vast 1x RTX 4090 instance `40826362`
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 565.77
- Remote base HEAD: `017300426514d62e8e50ac1546ff77d4d54fd6ce`
- Local HEAD at probe time: `f096e96395b11f712a3660999d6b999a0970bc23`
- Dirty source: `scripts/microbenches/gemma3_down_prefetch_overlap_perf.cu`
  was synced to the remote working tree; the local diff is saved in
  `local/probe_source.diff`.
- Binary SHA256:
  `994f828373477f5d9a34f8bd06c42921b1b13cfeb8b28679fd2400fb6f968801`
- Probe retry rc: `0`
- PASS line:
  `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`
- Vast cleanup: `cur_state=stopped actual_status=exited`

The first compile attempt failed because the initial probe used `volatile
uint4` direct copies. The retry uses ordinary `uint4` loads and completed.

## What Changed In The Probe

`scripts/microbenches/gemma3_down_prefetch_overlap_perf.cu` now includes
producer-integrated modes. These modes keep the existing product-shaped
eight-layer rotation and add a GeGLU kernel variant that touches a configurable
slice of the current layer's `down_proj` qweight/scales while computing the
activation. The probe still reports both isolated `down_us` and total
`segment_host_us`.

The important acceptance rule is unchanged: a cache-touch mode is interesting
only if total segment wall time improves. Lowering only `down_us` is not enough.

## Key Rows

m16:

- no prefetch: down `68.773us`, segment `212.295us`
- overlap qweight: down `34.150us`, segment `235.799us` (`+11.07%` segment)
- overlap qweight+scales: down `34.032us`, segment `236.862us` (`+11.57%`)
- producer touch qweight 1x: down `62.787us`, segment `202.341us`
  (`-4.69%`)
- producer touch qweight 4x: down `53.889us`, segment `214.460us` (`+1.02%`)
- producer touch qweight+scales 4x: down `52.139us`, segment `214.662us`
  (`+1.11%`)

m32:

- no prefetch: down `74.286us`, segment `224.566us`
- overlap qweight: down `53.112us`, segment `261.882us` (`+16.62%`)
- overlap qweight+scales: down `52.954us`, segment `263.005us` (`+17.12%`)
- producer touch qweight 1x: down `64.878us`, segment `212.922us`
  (`-5.19%`)
- producer touch qweight 4x: down `53.474us`, segment `240.533us` (`+7.11%`)
- producer touch qweight+scales 4x: down `53.362us`, segment `242.477us`
  (`+7.98%`)

## Interpretation

This is the first native CUDA signal in the cache-residency branch that reduces
total product-shaped tail-MLP segment time instead of merely reducing the
isolated down kernel:

- A small producer-integrated qweight touch (`1x`) improves total segment time
  by about `4.7%` at m16 and `5.2%` at m32.
- More aggressive producer touch (`4x`) improves `down_us` more, but loses on
  total segment wall time.
- External second-stream qweight/scales warm remains rejected: it cuts down
  kernel time but adds more total wall time.

The next implementation decision should therefore not be a full qweight warm or
a simple stream access policy. The viable branch is a small, typed,
producer-adjacent touch/prefetch strategy, or a fused tail-MLP design that can
hide a similarly small amount of down-weight residency work.

## Productization Constraints

This probe does not prove product performance. Before touching product runtime:

- the optimization must be wired through typed projection/layer context, not
  hidden env or diagnostic allocation labels;
- CUDA correctness must pass for both `ferrum run` and `ferrum serve`;
- endpoint performance must be remeasured after correctness, and release-grade
  evidence still requires the W2 final validator PASS line.

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
