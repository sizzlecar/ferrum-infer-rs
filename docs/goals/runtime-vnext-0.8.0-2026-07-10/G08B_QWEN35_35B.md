# G08B: Qwen3.5-35B-A3B Hybrid-MoE 迁移

## 依赖与目标

- 依赖：G08A
- 下游：G08C
- 目标：在 M1 hybrid 基础上加入 256-expert/shared-expert MoE、GPTQ/Marlin 和高压资源路径。

## 必需交付

- CUDA official GPTQ-Int4 full product path。
- Metal Q4_K_S full product path，固定 32GB M1 Max，>=2 GiB 实测 headroom、swap growth 0，无 waiver。
- requested/effective concurrency 分离；CUDA client c32、typed active cap至少 16，并记录
  observed max-active；Metal required client c1/4/16、typed active floor `4`。CUDA/Metal 最高
  cell 的 eligible interval active duty-cycle 均须 `>=0.80`。
- recurrent + KV + scratch 多资源事务 fault grid。
- G00 legacy CUDA binary parity；Metal 使用 HF/CPU + same-GGUF llama.cpp new-lane reference。
- 删除全部 Qwen3.5 family legacy runner/factory/arch-named adapter，包括 G08A test-only adapter。

## 验收

- M2 CUDA/Metal C01-C21 `2/2 PASS`。
- Metal op/layer/full-vocab-logit/token reference 全部满足 MODEL_MATRIX 固定数值门，并绑定 checked-in
  tolerance blob/row；missing/post-hoc-widened tolerance 数量 `0`。
- G02 Qwen3.5 resource/output historical mutations kill `100%`。
- Qwen3.5 family legacy production/test adapter 数量 `0`。
- Qwen3.5 架构专属执行脚手架相对 G00 减少 `>=60%`。
- CUDA client c32/admission-cap 路径资源终态正确，OOM/livelock/leak `0`。
- G08 统一 performance smoke：CUDA `>=0.90x` G00 legacy，Metal `>=0.70x` same-host
  llama.cpp；两者都只作 diagnostic，完整正式门留给 G09。

```text
FERRUM RUNTIME VNEXT G08B QWEN35 35B A3B PASS: <out_dir>
FERRUM GATE vnext-g08b PASS: <out_dir>
```

## CUDA Correctness Checkpoint - 2026-07-23

Clean frozen source `6fa8e21514bcb602e5d21aa2fa367c55159c6d8e` and source tree
`2011052b234ff313fd98eed1c7cf3187172014bb` completed the M2 CUDA C01-C21 matrix on
one RTX 4090. The bound release binary SHA256 was
`4a580c6b3513716c22ae57fc3268728bedfa9d250515b202723260698d17b12b`.

- cases: `703/703 pass`, unexpected/error/known-fail/blocked `0`;
- product command groups: five resident `ferrum run` groups and two isolated
  `ferrum serve` sessions;
- bounded duration: `4817.444734s`;
- bounded peaks: processes `4`, process-group threads `102`, per-process threads `67`;
- driver stderr bytes: `0`.

The canonical runner and unified checkpoint printed:

```text
FERRUM RUNTIME VNEXT G08 MODEL MATRIX SCENARIOS PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z/correctness/m2-qwen35-35b-a3b/cuda/scenario-report.json
FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z/gate-vnext-g08b-cuda
FERRUM GATE vnext-g08b-cuda PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z/gate-vnext-g08b-cuda
```

The complete compressed artifact is stored in the temporary
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z.tar.gz)
with SHA256 `3816f1ea3f696bb3595bd8319cf070d02cabaf7490381a10708211c9df50b2ea`.
The verified local compressed copy is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260723/g08b-cuda-matrix-6fa8e215-github/`;
it was not expanded because local disk space is constrained.

Vast instance `45319871` was stopped only after GitHub upload, local download, and
SHA256 verification. Reconciled state is `cur_state=stopped`,
`actual_status=exited`, and potentially billable sibling count `0`; the stopped
instance retains model/build caches.

This checkpoint is historical/intermediate after later runner and test-policy changes.
It proves M2 CUDA product correctness at `6fa8e215`; it does not prove current-HEAD
freshness or complete G08B. Metal Q4_K_S correctness, CUDA/Metal performance smoke,
legacy/reference parity, historical mutation plus legacy-deletion acceptance, and the
final G08B aggregate remain open.

## CUDA Addressed Paged-Attention Diagnostic - 2026-07-23

Clean source `6a59655836a0a5dc98fc7b361ef85bb374dcfc5e` integrated the existing
vLLM paged-attention v1/v2 implementation with vNext's non-contiguous 64 KiB resource
pages through a typed device-address table. It did not add a second attention
algorithm. The bound CUDA binary SHA256 was
`e8d12cc22b4c1cc88a6473b7cdc17e77ab7a631de6bea0078aec0692f9dd5f72`.

Correctness passed before the performance diagnostic:

- addressed causal-attention provider tests `8/8` and vLLM dispatcher tests `3/3`;
- focused actual-model `c03-001 run`, `c05-001 serve`, and `c06-001 streaming
  serve` all passed;
- the profile-off performance run completed `300/300` measured requests with zero
  request or output-quality errors and usage-derived output-token counts.

The focused runner printed:

```text
FERRUM RUNTIME VNEXT G08B CUDA BUILD READY: /workspace/ferrum-artifacts/runtime-vnext-addressed-pa-6a596558-20260723T152052Z/build/candidate/candidate-build-receipt.json
FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: /workspace/ferrum-artifacts/runtime-vnext-addressed-pa-6a596558-20260723T152052Z/focused-report.json
```

Full operation trace confirmed the exact native path
`vnext.causal_attention.vllm_paged_attention_v1_addressed`. Decode attention fell from
historical `582.551 us/command` and `5.826 ms/wave` to `88.731 us/command` and
`0.887 ms/wave`. Full-profile throughput moved from `6.854` to `7.301 tok/s`, but that
mode remains dominated by about `101 ms/wave` of trace host postprocessing and is not
the product throughput comparison.

The same-hardware profile-off result was `39.5687 +/- 2.0459 tok/s`, `11.16%` above
the historical vNext `35.5956 tok/s`. It missed the predeclared diagnostic KEEP floor
`40.935 tok/s` by `1.366 tok/s` and the `0.90x` historical legacy floor
`76.158 tok/s` by `36.590 tok/s`, so the performance candidate is REJECT:

```text
CUDA ADDRESSED PAGED ATTENTION DIAGNOSTIC REJECT: /workspace/ferrum-artifacts/runtime-vnext-addressed-pa-6a596558-20260723T152052Z/diagnostic-summary.json
```

The next paid run is blocked on a source-level change predicted to reduce profile-off
decode `resource_prepare_attempt` from `4.192 ms` to `<=3.5 ms` and
`submitted_wave_total` from `14.964 ms` to `<=13.5 ms`, while retaining at least
`40.935 tok/s` and all correctness results.

The complete artifact is stored in the temporary
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-addressed-pa-6a596558-20260723T152052Z.tar.zst)
with SHA256 `41543e1e1b4e7acf0decb5fdb31f0bd38b868cdf5d20ebb4a9813384b798769c`.
The GitHub-downloaded local copy was verified at
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260723/github-assets/`.
Vast instance `45319871` is `stopped/exited`, and the reconciled unexpected
billable-or-transitional instance count is `0`.

This diagnostic proves that the typed vNext CUDA path invokes the intended existing
vLLM attention kernel and improves that operation. It does not refresh the current-HEAD
703-case CUDA matrix, satisfy the G08B performance smoke, prove Metal, or complete
G08B/G09.

## CUDA Host-Dispatch Diagnostic - 2026-07-23

Clean source `d0d2e4f5f080b1deb7359569313c27fe15e02454` retained the addressed
vLLM paged-attention path and removed repeated resource-layout compilation, O(N^2)
wave-node lookup, per-node temporary identity vectors, and repeated common-authority
validation. The bound CUDA binary SHA256 was
`3b09cb93f04c2ad9fc255f1f965ee574afc7b1b57d73bce135c16fcd2bf22fc1`.

Correctness passed before performance:

- addressed causal-attention provider tests `8/8` and vLLM dispatcher tests `3/3`;
- actual-model `c03-001 run`, `c05-001 serve`, and `c06-001 streaming serve`
  passed `3/3`;
- the profile-off performance run completed `300/300` measured requests with
  usage-derived output-token counts and zero request, output-quality, malformed-SSE,
  HTTP 500, or panic errors.

The same-hardware profile-off result was `46.0342 +/- 0.7937 tok/s`. This is
`16.34%` above the preceding `39.5687 tok/s` candidate and exceeds the diagnostic
KEEP floor by `5.0993 tok/s`. Decode `submitted_wave_total` fell from `14.964 ms`
to `12.888 ms`, satisfying the `<=13.5 ms` source prediction. Decode
`resource_prepare_attempt` fell from `4.192 ms` to `3.603 ms`, but missed the
predeclared `<=3.5 ms` target by `0.103 ms`. The formal `76.158 tok/s` legacy
90% floor remains open by `30.124 tok/s`, so the scoped diagnostic remains REJECT:

```text
status=REJECT
failure_class=resource-prepare-target-miss-despite-throughput-keep
```

Post-run source audit invalidated the first follow-up hypothesis recorded in the
diagnostic artifact: `bench-serve` c1 constructs `ExecutionBatchParticipants` before
`execute_batch_step` starts `resource_prepare_attempt`, so caching the outer singleton
batch cannot directly explain or close that measured `0.103 ms` gap. The immutable
artifact remains unchanged; this paragraph is the durable correction. A borrowed-span
binding can still remove the inner `spans.to_vec`, and singleton caching can still
reduce unmeasured host work for single-sequence `run`, but neither is accepted as the
next paid-run hypothesis without phase evidence.

Before another paid run, extend the existing typed `wave_timing` metrics to split
`resource_prepare_attempt` into work-shape/request preparation, step admission, and
submission-wave preparation. The next source change must name the dominant measured
phase and predict its metric delta; another GPU sweep must not be used to rediscover
that boundary.

The complete artifact is stored in the temporary
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-host-dispatch-d0d2e4f5-20260723T170239Z.tar.zst)
with SHA256 `72932611450603a67b7ded73be5767079d8f8a616dc690b424e3a6ccba0b724e`.
The GitHub-downloaded local copy was verified at
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260723/github-assets/`.
Vast instance `45319871` is `stopped/exited`, and the reconciled unexpected
billable-or-transitional instance count is `0`.

## Backend-Neutral Workspace Reuse Diagnostic - 2026-07-24

The CUDA correctness checkpoint and host-dispatch diagnostic both ran with
`FERRUM_BATCHED_GRAPH=0`. Their startup preparation reported `enabled=false`, and
the CUDA dynamic pools reported `live_segments=0` after the request. This means the
703-case checkpoint proved the typed vNext product path, but did not prove that
lane-stable reusable workspace backing was active.

Source inspection found that `ReusableExecutionPolicy` creation was incorrectly
conditional on CUDA graph configuration and device executable-capture capability.
The policy owns backend-neutral decode/prefill workspace buckets as well as the
inputs used by device capture. Consequently, Metal and graph-disabled CUDA repeatedly
claimed and released step/submission backing even though executable capture was not
requested.

A dirty local candidate based on `89d3f66d5a709d866ae0accb083f4b528ad62e41`
decouples those responsibilities: the immutable runtime policy always contains
lane-stable workspace buckets, while executable warmup/capture/replay remains gated by
the typed backend capability and `enable_cuda_graphs`. The same real
Qwen3.5-35B-A3B Q4_K_S Metal request produced:

| decode host interval | before | candidate | change |
|---|---:|---:|---:|
| backing claim | 4173.926 us | 261.633 us | -93.73% |
| step admission | 4590.789 us | 692.091 us | -84.92% |
| submission-wave prepare | 2928.569 us | 445.300 us | -84.79% |
| resource-prepare attempt | 7527.991 us | 1170.058 us | -84.46% |

Device capture remained disabled and unsupported with zero captured/replayed
executables. Resource domains 1/4/5 retained `2/2/6` live lane-stable segments,
respectively. The request completed `32/32` waves with zero failed wave and zero
request/sequence/step/wave deferral. Six bounded backing-growth maintenance attempts
were observed during cold allocation, versus four in the baseline; they converged
internally and are not hidden from the artifact.

The diagnostic is deliberately not a correctness or performance PASS: the 32-token
response ended during Qwen reasoning and the source was dirty. The durable artifact is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/workspace-reuse-metal-dirty-89d3f66d/`;
`diagnostic-summary.json` is `KEEP_DIAGNOSTIC`, and the bound Metal binary SHA256 is
`53b5787c97957556c5dcf27e5c600374377a85b1ed1cb898e5c8bd0445be517c`.

Clean source `e66ade7f0c0cd88ffc55c9a3c5a9ac902c68f58d` then completed the
predeclared one-RTX-4090 bounded CUDA diagnostic. The bound release binary SHA256 was
`f35878204dbc0cfbfdf62e8b0ec1304d01b610385244465924c60a31d7ff624f`.
Correctness ran before performance: `c03-001 run`, `c05-001 serve`, and `c06-001
streaming serve` passed `3/3`; the stream emitted one `[DONE]`, one usage event, zero
malformed event, and the exact expected marker.

The profile-off c1 run used the same historical workload (`random 64/32`, `100 x 3`
measured requests, ten warmups per repeat, `--fail-on-error --require-ci`). It
completed `300/300` with zero request or quality error and usage-derived token counts:

- throughput `55.5897 +/- 0.2898 tok/s`, `20.76%` above `d0d2e4f5` at
  `46.0342 tok/s`;
- decode resource preparation `3.6031 ms -> 1.1426 ms` (`-68.29%`), satisfying
  the predeclared `<=3.5 ms` signal;
- decode submitted-wave total `12.8876 ms -> 12.3382 ms` (`-4.26%`);
- graph capture stayed disabled, captured executables stayed `0`, while resource
  domains 3/4/5 retained `2/6/2` lane-stable segments;
- request/sequence/step/wave/extension deferrals and failed waves were all `0`.
  Nine internal cold backing-growth maintenance attempts converged and are preserved
  in the artifact.

The runner printed:

```text
FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: /workspace/ferrum-artifacts/runtime-vnext-workspace-reuse-e66ade7f-20260724/focused-report.json
CUDA WORKSPACE REUSE DIAGNOSTIC KEEP: /workspace/ferrum-artifacts/runtime-vnext-workspace-reuse-e66ade7f-20260724/diagnostic-summary.json
```

This is a scoped `KEEP_DIAGNOSTIC`, not G08B or G09 PASS. It still misses the
`76.1583 tok/s` formal floor by `20.5685 tok/s` (`27.01%`). Client ITL also remains
ineligible for comparison because three of 300 requests had event/usage mismatch,
although request correctness and TPOT evidence were unaffected.

The complete GitHub-transfer artifact is
[runtime-vnext-workspace-reuse-e66ade7f-20260724.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-workspace-reuse-e66ade7f-20260724.tar.zst)
with SHA256 `e25500b9da8ef546f5cb70b0785cd81a15f8b26b0d4a94fa1d2618a439363445`.
The GitHub-downloaded local copy is verified at
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/workspace-reuse-cuda-e66ade7f/`.
Vast instance `45319871` is `stopped/exited`, the cache is retained, and the
reconciled potentially billable/transitional sibling count is `0`.

## CUDA Stable Program Preparation Diagnostic - 2026-07-24

The exact clean `e66ade7f0c0cd88ffc55c9a3c5a9ac902c68f58d` production source and
binary SHA256 `f35878204dbc0cfbfdf62e8b0ec1304d01b610385244465924c60a31d7ff624f`
were reused after the focused correctness and profile-off artifact above. This
bounded full-profile run completed `4/4` requests with zero errors and collected
75 decode waves. Full-profile throughput was `7.4649 tok/s`; it is diagnostic-only
and is not comparable to the profile-off performance result.

The trace localized the remaining host submission cost:

| decode metric | per wave |
|---|---:|
| host encode + submit | `8.5245 ms` |
| provider encode | `1.8789 ms` |
| device runtime submit | `5.6823 ms` |
| enqueue commands | `5.6145 ms` |
| device execution | `9.4945 ms` |
| fence wait | `3.0152 ms` |
| completion-worker queued wait | `0.0458 ms` |
| readback | `0.3570 ms`, `496,640 bytes` |

Every decode wave submitted `174` eager commands: 163 compute commands, ten
dynamic causal-attention bindings, and one token upload. The reusable-execution
metrics reported 40 candidate segments per wave, all 40 outside preparation, and
zero replayed commands. This accepted the predeclared source hypothesis and assigned
failure class `stable-decode-command-program-outside-reusable-preparation`:

```text
CUDA CURRENT HOST DISPATCH PROFILE KEEP: /workspace/ferrum-artifacts/runtime-vnext-current-full-profile-e66ade7f-20260724T0340/diagnostic-summary.json
```

Historical artifacts already prove the existing CUDA reusable-executable path can
replay the stable program: `beb3e63c` reduced enqueue from `6.625 ms` to
`1.819 ms`, and `b38e9645` replayed `2432/2432` candidates with zero request-time
capture. The current miss is therefore a product-policy integration defect, not a
missing paged-attention implementation or a reason to add another command cache.
vNext reusable execution was incorrectly controlled by the legacy, default-off
`FERRUM_BATCHED_GRAPH` policy.

The next candidate must separate legacy whole-model graph policy from typed,
backend-owned reusable program preparation. `ferrum run`, `ferrum serve`, and
configuration must expose the same product-visible policy. Safe capture rejection or
capacity deferral may use visible eager fallback, while indeterminate CUDA, fence,
ownership, or inventory state remains fail-closed.

The next paid CUDA run is bounded by these predeclared checks:

- run `c03-001`, serve `c05-001`, and streaming serve `c06-001` pass before
  performance, with zero panic/OOM/output/stream error;
- startup reports captured, uploaded, and resident executable inventory, with
  `eager_fallback_required=false`;
- request-time capture/upload remains `0`;
- decode replay is `>=150 commands/wave`, eager submission is `<=24 commands/wave`,
  and enqueue is `<=2.0 ms/wave`;
- profile-off throughput exceeds `55.5897 tok/s`; the formal `76.1583 tok/s` floor
  remains unchanged and must still pass before any G08B/G09 performance claim.

The complete GitHub-transfer diagnostic asset is
`runtime-vnext-current-full-profile-e66ade7f-20260724T0340.tar.zst`, asset id
`487573242`, SHA256
`ebb2e401276fc5767ef96bfa66373967f6d242d9bbacd7ccf938dac27fbb59b6`.
The verified local artifact is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/current-sha-full-profile-e66ade7f/`.
The paid window was approximately 12 minutes (`$0.0939`). Vast instance `45319871`
was then polled to `stopped/exited`; no billable or transitional sibling remains.

Clean candidate `a0038a0eb0ec0b83af8f3d34b72d2ec0715f1e55` implemented that
product-policy split without adding a kernel or command cache. Its bound CUDA binary
SHA256 was
`e4a55cddc7883ec65e7fb13dd726ac14692ee3f7b9d0a1f100ff7e42442721eb`.
Local affected-crate gates passed before paid work:

- `ferrum-types` lib `95/95` and config integration `17/17`;
- `ferrum-models` lib `233 pass`, two designed real-model tests ignored, zero fail;
- `ferrum-cli` lib `169/169`;
- formatting and diff checks passed.

On the retained RTX 4090, `c03-001 run`, `c05-001 serve`, and `c06-001 streaming
serve` again passed `3/3`. Default product configuration reported legacy
`FERRUM_BATCHED_GRAPH=0` and typed `FERRUM_REUSABLE_EXECUTION=1`. Startup preparation
reached `ready` in `1.389 s`: all 240 executables were captured, uploaded, and
resident, with rejected/deferred `0` and `eager_fallback_required=false`.

The bounded result nevertheless failed the predeclared performance signal:

| metric | required | observed |
|---|---:|---:|
| request-time capture/upload | `0` | `0/0` |
| candidate/cache-hit/outside-preparation segments per wave | `40/40/0` | `40/40/0` |
| replayed commands per wave | `>=150` | `121` |
| eager commands per wave | `<=24` | `53` |
| enqueue commands | `<=2.0 ms/wave` | `2.509 ms/wave` |
| profile-off throughput | `>55.5897 tok/s` | `55.4898 +/- 2.1109 tok/s` |

Thus typed startup preparation and all 40 segment replays are working, and enqueue
fell from `5.614 ms` to `2.509 ms`, but only `69.54%` of the 174-command wave moved
into replay. The remaining 53 eager commands prevent a performance KEEP and leave
the formal `76.1583 tok/s` floor open by `20.6685 tok/s`. Full-profile diagnostic
throughput was `8.1976 tok/s`; it is not product performance evidence.

The immutable result is:

```text
CUDA REUSABLE PROGRAM INTEGRATION REJECT: /workspace/ferrum-artifacts/runtime-vnext-reusable-program-a0038a0e-20260724T0423/diagnostic-summary.json
```

The first benchmark warmup attempt is preserved as auxiliary failure
`bench-client-served-model-name-mismatch`: the client sent the model directory while
the server exposed `m2-qwen35-35b-a3b`, so it returned HTTP 400 before inference.
The corrected bounded retry reused the same warmed server and completed `300/300`
requests with zero request, quality, panic, HTTP 500, or stream error.

The complete GitHub asset is
[runtime-vnext-reusable-program-a0038a0e-20260724T0423.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-reusable-program-a0038a0e-20260724T0423.tar.zst),
asset id `487626043`, SHA256
`f7ab54160372956f39b868df20cc6ffedca6c06ce953cad43110953d13dbb80d`.
The GitHub-downloaded local copy was verified at
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/reusable-program-cuda-a0038a0e/`.
The paid window was approximately 29 minutes (`$0.2269`). Vast `45319871` is
`stopped/exited`, with no billable/transitional sibling.

No further paid run is allowed until source analysis classifies the 53 eager commands
per decode wave by typed command owner and a source change predicts which owner will
move into prepared replay and reduce enqueue below `2.0 ms`.

The saved full scheduler trace and immutable 163-node Qwen3.5 program now close that
classification. Each decode wave contains one token upload plus 173 provider commands:

| owner | commands/wave | current path |
|---|---:|---|
| RMSNorm (`40` layer + `1` final) | `41` | replayed |
| routed/shared MoE | `40` | replayed |
| residual add | `40` | replayed |
| gated-delta recurrent attention | `30` | eager compute |
| causal paged attention compute | `10` | eager compute |
| causal paged attention address binding | `10` | eager dynamic binding |
| token embedding | `1` | eager compute |
| last-token logits projection | `1` | eager compute |
| token upload | `1` | eager input boundary |

Thus `121 + 53 = 174` is fully owned; there is no unidentified CUDA work. Source
inspection also shows why adding more cache keys would be the wrong fix. The causal
compute command currently carries paged-state regions only to retain their lifetime,
even though the kernel reads their addresses from the typed binding workspace. That
mixes fence retention with captured kernel arguments and strips its reusable-address
contract. Recurrent attention, embedding, and logits still bind request/sequence
addresses directly. They require a plan-owned static command program whose stable
buffers are updated through typed per-wave bindings; dynamic bindings and input
uploads remain explicit ordered boundaries. Capturing those boundaries or marking
request-owned addresses stable is forbidden.

Historical evidence was also re-audited. `beb3e63c` is a real Qwen3.5-4B product
path, but its derived `1.819 ms/wave` mixes prefill and decode and still includes six
request-time captures. `b38e9645` is also a real Qwen3.5-4B serve path, but
`2432/2432` means candidate-segment hits, not command-complete replay: it recorded
`7980` replayed and `2660` eager commands (`75%`) with
`3.27695 ms/wave` enqueue. Neither value is a directly comparable target for the
current Qwen3.5-35B-A3B MoE/GPTQ decode lane. In addition, the current owner trace
used full kernel attribution, which records CUDA events around eager commands; its
command ownership is valid, but its enqueue duration is not directly comparable to
the historical basic-profile runs. The failed `>=150` replay,
`<=24` eager, and `<=2.0 ms` enqueue thresholds remain the immutable predeclared
result for `a0038a0e`; they must not be recycled as evidence for another paid run.

The source work is staged by exact owner movement. First, separate captured launch
regions from fence-only dependencies and restore the ten causal compute commands,
predicting `131 replay / 43 eager`. Second, introduce lane-stable recurrent-state
bindings, predicting `161 / 13`. Third, move embedding/logits through lane-stable
I/O staging, predicting `163 / 11`; the ten causal address bindings and one upload
remain explicit eager boundaries. These are structural predictions, not PASS claims.

The resulting checkpoint is a backend-neutral static-program/binding contract plus
its CUDA implementation, not another replay-key sweep. Before paid
validation it must locally prove that provider node encoding is not rebuilt on a
cache hit, that every dynamic state/IO address is supplied through a typed binding
or explicit eager boundary, that cached programs retain no request-owned resource,
and that fence ownership still covers every binding target until completion. The
next paid prediction must name the exact owner counts and host stages expected to
move; the formal `76.1583 tok/s` floor remains unchanged.

### 2026-07-24 Causal Replay Ownership Result

Clean source `4578e612f5f5c1546be06440a767e8999fbc4cec` first split cached CUDA
executables from fence-only page dependencies. Its CUDA release build and manifest
were READY, but the first real `c03-001 run` stopped the lane before `c05`, `c06`,
or performance: startup replay validation changed the executable inventory from
`80` to `100`. The exact failure class was
`causal-replay-exact-sequence-shape-outside-startup-preparation`; the causal decode
key still contained the current sequence length and block-table length. The
immutable result is:

```text
CUDA CAUSAL REPLAY PREPARATION REJECT: /workspace/ferrum-artifacts/runtime-vnext-causal-replay-4578e612-20260724T0525/diagnostic-summary.json
```

That artifact was transferred through GitHub and verified locally at
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/causal-replay-cuda-4578e612/`.
Its archive is `27,702,492` bytes with SHA256
`995b92995772553eecc8a4b1454778d2973ca3bdf68b61a5e031f6d795d43943`.

Clean follow-up `3ac6b65a9dcbbd9e5751424dfbd65d85583d76d2` introduced a typed causal
decode launch envelope: V1 uses one `<=512` token topology and V2 uses native
512-token partition buckets, while the actual sequence length remains in the
per-wave device binding. The CUDA unit test for those boundaries passed `1/1`;
the candidate release build and model manifest printed their exact READY lines.
The same binary then passed the bounded product correctness gate:

- `c03-001`: resident `ferrum run`, multi-turn memory;
- `c05-001`: OpenAI-compatible non-streaming `ferrum serve`;
- `c06-001`: streaming `ferrum serve`;
- result: `3/3 pass`, request/quality/stream/panic errors `0`.

The predeclared decode owner prediction was exact. Across 75 decode waves, each
wave changed from `121 replay / 53 eager` to `131 replay / 43 eager`: ten causal
compute commands entered replay, ten address-binding commands and one token upload
remained explicit eager boundaries.

The complete performance candidate was nevertheless rejected. The same full-profile
workload exposed cross-topology segment poisoning:

| prefill replay owner | `a0038a0e` | `3ac6b65a` | delta |
|---|---:|---:|---:|
| RMSNorm | `1230` | `955` | `-275` |
| routed/shared MoE | `1200` | `950` | `-250` |
| residual add | `1200` | `950` | `-250` |
| causal attention | `0` | `50` | `+50` |

Making exact-shape varlen prefill commands reusable removed `775` adjacent stable
replays to gain only `50` causal replays, a net prefill loss of `725`. Across the
whole scoped workload, replay therefore increased by only `25`
(`12705 -> 12730`) instead of the decode-only `+750`. This is a static-program
segmentation defect, not a reason to raise cache capacity or add another kernel.

Profile-off `random 64/32`, c1, `100 x 3`, seed `9271` completed `300/300`
requests with usage token counts and zero errors, but throughput fell to
`51.5331 +/- 4.3387 tok/s`. It was `3.9567 tok/s` (`7.13%`) below
`a0038a0e` and missed the unchanged `76.1583 tok/s` floor by
`24.6252 tok/s` (`32.33%`). The immutable result is:

```text
CUDA CAUSAL REPLAY ENVELOPE REJECT: /workspace/ferrum-artifacts/runtime-vnext-causal-envelope-3ac6b65a-20260724T0548/diagnostic-summary.json
```

The complete archive was subsequently transferred through a temporary GitHub
branch and verified locally at
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/causal-envelope-cuda-3ac6b65a/`.
It is `35,154,675` bytes with SHA256
`1e0b9774ff7822ffe3336b39c7afb96b78171a40e5c0a17ba9f4f9863108b8d7`.

The next source hypothesis is topology-typed replay eligibility. Causal decode
V1/V2 remains reusable; exact-shape varlen/fallback prefill remains an explicit
eager boundary until it has a stable prefill envelope. A dynamic-key miss must not
force adjacent stable commands to eager. Before another paid run, local contracts
must prove those boundaries and predict the following scoped full-profile result:
decode `131/43`, prefill stable-owner replay restored to `1230/1200/1200`, total
`13455 replay / 5985 eager`. Model names, GPU names, VRAM sizes, and hidden env
switches are forbidden from that contract.

### 2026-07-24 Topology Barrier Result

Clean source `4df3d63a85bcd69479e827b2c545148ed4c9a91e` made replay eligibility a
typed launch-topology decision. Partition-stable decode V1/V2 commands retain their
bucketed replay key; exact-shape varlen/fallback prefill commands use an explicit
eager command constructor with independent fence dependencies. No model, GPU, VRAM,
or hidden-environment branch selects this behavior.

The candidate release build and execution manifest printed their exact READY lines.
The same binary passed all three bounded product paths:

- `c03-001`: `ferrum run` multi-turn correctness;
- `c05-001`: non-streaming OpenAI-compatible `ferrum serve`;
- `c06-001`: streaming `ferrum serve`;
- result: `3/3 pass`; all execution-envelope, expectation, model-binding, and
  scenario-oracle checks were true.

The full-profile structural prediction was exact:

| signal | predicted | observed |
|---|---:|---:|
| total native replay commands | `13455` | `13455` |
| prefill RMSNorm replay | `1230` | `1230` |
| prefill MoE replay | `1200` | `1200` |
| prefill residual replay | `1200` | `1200` |
| prefill causal replay | `0` | `0` |
| decode owner split | `131/43` | `131/43` across 75 waves |

This accepts the topology isolation source change: exact-shape prefill no longer
poisons adjacent stable replay segments. It does not satisfy G08B or G09. The
profile-off c1 `random 64/32`, `100 x 3`, seed `9271` command completed `300/300`
requests with usage token counts, zero errors, and zero quality issues, but achieved
only `48.3721 +/- 4.9645 tok/s`. It is `12.83%` below the `55.4898 tok/s` stable
Ferrum checkpoint and misses the unchanged `76.1583 tok/s` floor by
`27.7862 tok/s` (`36.48%`). The immutable result is:

```text
CUDA TOPOLOGY REPLAY BARRIER REJECT: /workspace/ferrum-artifacts/runtime-vnext-topology-barrier-4df3d63a-20260723T231634Z/diagnostic-summary.json
```

The archive is `35,174,029` bytes with SHA256
`c433fffb9e77bfea543a66a6e5df5f4420f6cb58d80743e61f4dd2516084657d`.
Its first GitHub upload attempt timed out after packaging, so it remains on retained
instance `45319871`; that instance is confirmed `stopped/exited` and is not billing.
Restoring transfer later does not authorize another benchmark.

The next CUDA source checkpoint moves from command-key work to command-program
ownership: a plan-owned immutable CUDA command program plus typed per-wave binding
patches. Recurrent state must first use lane-stable indirection with RAII/fence
coverage; request-owned state must never be falsely marked stable. Before another
paid run, local contracts must prove program ownership, binding lifetime, state
address scope, and zero model/hardware hard-codes. The next paid artifact must predict
a decode eager count below `43` and recurrent eager count below `30`; otherwise it
cannot justify GPU time.

### 2026-07-24 Recurrent Program-Binding Result

Clean source `393a9a401a17cb261fc8dab159fe412e75437845` contains the recurrent
state-indirection implementation from `c1792e87` plus the standard-operation contract
correction in `393a9a40`. The CUDA release build completed in `310.184429s`; its
binary SHA256 is
`7139a4aca004d1b79008c438cbcbb66c294125e43d1ad94d8117bad5fb00fd3f`.
The same binary then passed the bounded product paths before any performance run:

- `c03-001`: resident `ferrum run` multi-turn correctness;
- `c05-001`: non-streaming OpenAI-compatible `ferrum serve`;
- `c06-001`: streaming `ferrum serve`;
- result: `3/3 pass`, measured request errors `0`, and blocker-scan matches `0`.

The decode structural prediction also matched across all 75 sampled waves. The trace
directly observed `161 replay / 2 node-attributed eager` commands per wave. The
already-tested CUDA coalescer contributes exactly one non-node program-binding
prelude, and the input upload remains one explicit boundary. Therefore the complete
decode split is `161 replay / 3 eager provider / 1 upload`, or `161/4`, versus the
previous `131/42/1` (`131/43`). This is a structural KEEP, not a performance PASS.

The canonical profile-off c1 `random 64/32`, `100 x 3`, ten warmups, seed `9271`
completed `300/300` measured requests with usage token counts, zero errors, and zero
quality issues. Throughput nevertheless regressed to
`44.947749 +/- 3.671515 tok/s`. It is `3.424380 tok/s` (`7.079%`) below the
`4df3d63a` candidate, `10.641951 tok/s` (`19.144%`) below the predeclared
`55.5897 tok/s` KEEP threshold, and `31.210551 tok/s` (`40.981%`) below the
unchanged `76.1583 tok/s` formal floor. The immutable result is:

```text
CUDA PROGRAM BINDING PERFORMANCE REJECT: /workspace/ferrum-artifacts/runtime-vnext-program-binding-393a9a40-20260724T004652Z/diagnostic-summary.json
```

Same-hardware profile-off timing localizes the regression:

| decode host boundary | `4df3d63a` | `393a9a40` | delta |
|---|---:|---:|---:|
| resource prepare | `1.507994ms` | `1.894054ms` | `+25.601%` |
| host encode/submit | `6.122343ms` | `5.066258ms` | `-17.250%` |
| completion round trip | `6.689556ms` | `8.351333ms` | `+24.841%` |
| host postprocess | `0.812439ms` | `1.013550ms` | `+24.754%` |
| submitted wave total | `13.626184ms` | `14.433627ms` | `+5.926%` |

Thus the replay/host-submit direction is validated, but the current binding abstraction
is incomplete. `coalesce_program_bindings` coalesces Rust command ownership while
still executing one native transfer into each provider-owned binding workspace.
Qwen3.5 has 30 recurrent and 10 causal binding patches per wave; the recurrent change
also admits one invocation binding workspace per recurrent node. The next checkpoint
must replace these per-node workspaces with a compiled, typed
`ProgramBindingLayout`, one lane-owned device binding arena, and one contiguous
per-wave patch/upload. Provider compute commands must be compiled once against stable
layout offsets instead of being rebuilt per node per wave.

No repeat of `393a9a40` is authorized. Before another paid run, local evidence must
prove one aggregate binding workspace, at most one native binding upload per wave,
no request-owned address in the cached program, complete RAII/fence retention, and
zero model/GPU/VRAM hard-codes. The next CUDA prediction is to retain `161/4`, reduce
binding native transfers to `<=1`, restore decode resource prepare to
`<=1.507994ms`, completion round trip to `<=6.689556ms`, and total wave time to
`<=13.626184ms` before requiring throughput `>=55.5897 tok/s`.

The complete archive is retained on Vast instance `45319871` as
`/workspace/ferrum-artifacts/runtime-vnext-program-binding-393a9a40-20260724T004652Z.tar.zst`
with SHA256
`38dba973258d622f40d550794b2b2d5b829fe4a85fe0ce075d913382aa2e4146`.
The instance is confirmed `stopped/exited`; no sibling instance is running or
scheduling. GitHub transfer is still pending and does not authorize restarting the
instance solely for another benchmark.

### 2026-07-24 Fused GDN CUDA Checkpoint

Clean source `cefb4de25036a818fd3d0628a63b4fde3b74d81d` and CUDA binary SHA256
`fa76fc9af5cdef07d6a887cc821b4347d139263a6b1618b542af0f29cb947800`
passed the bounded actual-model product paths:

- `c03-001`: resident multi-turn `ferrum run`;
- `c05-001`: non-streaming OpenAI-compatible `ferrum serve`;
- `c06-001`: streaming `ferrum serve`;
- result: `3/3 pass`, usage-derived token counts, and zero blocker-scan match.

The candidate uses the v5 typed GDN contract and packed QKVZ/BA model weights.
CUDA source tests covered packed extents, packed CUDA/CPU numerical parity, and the
production replay kernel symbol. Full-profile evidence observed all 30 recurrent
layers at `11` compute dispatches each, or `330` per correlation, versus
`13`/`390` in the pre-fusion `32c53a6b` artifact. Decode replay elapsed improved
from `7.362019 ms` to `6.888088 ms`; the complete device span improved from
`8.608580 ms` to `8.097945 ms`.

Profile-off c1 random `64/32`, `100 x 3`, ten warmups, and seed `9271` completed
`300/300` with zero request or quality error. Mean throughput was
`50.079974 +/- 11.778071 tok/s`: `9.03%` above the latest available current-path
checkpoint, but still `34.24%` below the unchanged `76.1583 tok/s` formal floor.
This keeps the model/runtime optimization without satisfying G08B or G09:

```text
FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: /workspace/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/focused-c03-c05-c06-report.json
CUDA GDN INPUT FUSION CHECKPOINT KEEP: /workspace/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/diagnostic-summary.json
CUDA GDN INPUT FUSION FORMAL PERFORMANCE REJECT: /workspace/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/diagnostic-summary.json
```

The complete GitHub asset is
[runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z.tar.zst),
asset id `488205719`, size `30,043,998` bytes, SHA256
`0521a5baa3b98398ce2e4683576b3e558500da74f6cf2479b369d53fef41e144`.
The archive and bound binary were revalidated locally under
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/`.
Vast instance `45319871` is confirmed `stopped/exited`; no paid sibling remains.

### 2026-07-24 Single-Token MoE Direct-Alignment Checkpoint

Clean source `992153a4de14bee734c97f54d2b78b754d7737f7` and CUDA binary
SHA256
`393f377659560db9c5df564bf544fbf7d435780059c2fb71fbfbe1e797d1ae1a`
passed the focused actual-model product paths:

- `c03-001`: resident multi-turn `ferrum run`;
- `c05-001`: non-streaming OpenAI-compatible `ferrum serve`;
- `c06-001`: streaming `ferrum serve`;
- result: `3/3 pass`, zero positive blocker-log match, and zero structured
  request-quality issue.

The typed `SingleTokenDirectMarlin` plan fuses route output and Marlin
alignment metadata only for decode. Across 75 decode correlations, all 3,000
single-token MoE node observations moved from `12` to `11` physical compute
dispatches, while prefill remained at `12`. Mean replay time improved from
`6.888088 ms` to `6.233041 ms` (`9.51%`).

The canonical profile-off c1 random `64/32`, `100 x 3`, ten-warmup, seed
`9271` workload completed `300/300` with usage token counts and zero errors.
Its `47.547946 +/- 6.296316 tok/s` mean did not exceed the prior
`50.079974 tok/s` candidate and remains `37.57%` below the formal floor.
Consequently this is a structural optimization checkpoint, not G08B or G09
performance PASS:

```text
FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: /workspace/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/focused-c03-c05-c06-report.json
CUDA MOE DIRECT ALIGN STRUCTURAL KEEP: /workspace/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/diagnostic-summary.json
CUDA MOE DIRECT ALIGN CANONICAL PERFORMANCE REJECT: /workspace/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/diagnostic-summary.json
```

The complete GitHub asset is
[runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z.tar.zst),
asset id `488260567`, size `29,911,858` bytes, SHA256
`0550e682170a20bed55a53627ac879c98cd2a522d1d1f8dc3f115cd02fc51eff`.
The local copy and checksum are under
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/`.
Vast instance `45319871` is confirmed `stopped/exited`; no paid sibling remains.

The local workspace all-target source gate is not claimed: after compiling, two
pre-existing CLI E2E cases were intercepted by the host proxy and received a
Hugging Face `404` instead of their expected local missing-model message.
Targeted kernel contracts and the real CUDA product paths above passed, but
they do not replace the full unit PASS line.

### 2026-07-24 Typed Direct-Program Checkpoint

Clean source `f435bec9498d51e77c0e08b71ea29016f3eb74ed` replaced the cache-hit
path's per-node provider encode with an exact typed CUDA program reference and
per-wave binding patches. The release-feature build printed its required READY
line; the bound binary SHA256 is
`ebc70b8ee0cd55d448db3f3585e45211d350c837476514f77a67b5f813680377`.
The same binary passed `c03-001`, `c05-001`, and `c06-001` (`3/3`) before
performance.

Post-request health evidence proves that this is the production path, not a
source-only abstraction:

- `12,210` direct waves and `32,010` direct segments;
- `1,946,010` logical nodes covered by direct programs;
- `468,600` typed binding-node applications;
- direct fallback and catalog-epoch miss counts both `0`;
- `330` catalog misses remained on non-prepared shapes and used the normal full
  encode path without retrying a possibly-submitted wave.

Canonical profile-off c1 random `64/32`, `100 x 3`, ten warmups, seed `9271`
completed `300/300` measured requests with usage token counts and zero errors.
Throughput was `59.887970 +/- 17.927647 tok/s`, `25.95%` above the
`992153a4` mean, but still `16.270330 tok/s` (`21.36%`) below the unchanged
`76.1583 tok/s` floor. Decode timing localized the remaining wall time to
`2.319187 ms` host encode/submit, `7.482255 ms` completion round trip, and
`10.524822 ms` submitted-wave total. The structural direct-program change is
kept; this artifact is not G08B or G09 performance PASS.

The local evidence root is
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-direct-program-cuda-f435bec9-20260724T115319Z-gated/`.
The large rebuildable binary remains on retained Vast instance `45319871`;
only the build receipt, command lines, correctness cases, health snapshots,
benchmark report, and logs were copied through SSH. The instance was stopped
to `actual_status=exited`, with no paid sibling.

### 2026-07-24 Specialized Binding-Only Result

Clean follow-up `8c58e3ea0017c85865c5b5f56d0b02e94f36063a` makes the CUDA recurrent
and causal attention providers override the binding-only contract. Recurrent
attention now materializes only live convolution/delta-state addresses; causal
attention materializes only the current KV page table and sequence frontier.
Static weights, scratch, launch topology, replay keys, and compute closures are
not rebuilt by those direct helpers. The source contract first failed, then
passed after both overrides were installed; all nine CUDA replay source
contracts passed. The release-feature build printed READY with binary SHA256
`997ec95a9d864e2dbb588f42c5e0993e947d5dfbba3053c8868107128b406b36`.

Actual-model correctness again passed `c03-001`, `c05-001`, and `c06-001`
(`3/3`). Direct coverage was unchanged at `12,210` waves, `32,010` segments,
and `468,600` binding-node applications, with direct fallback and epoch miss
counts both `0`.

The predeclared performance signal did not hold. The same profile-off command
completed `300/300` requests but produced
`50.272873 +/- 3.385823 tok/s`, `9.615097 tok/s` below the preceding mean and
`25.885427 tok/s` (`33.99%`) below the formal floor. The host boundaries moved
together rather than isolating the provider change:

| decode boundary | `f435bec9` | `8c58e3ea` |
|---|---:|---:|
| resource prepare | `1.607868 ms` | `2.435552 ms` |
| host encode/submit | `2.319187 ms` | `3.076369 ms` |
| completion round trip | `7.482255 ms` | `7.628776 ms` |
| host postprocess | `0.721329 ms` | `1.137567 ms` |
| submitted wave total | `10.524822 ms` | `11.845352 ms` |

Because resource preparation and postprocess regressed alongside encode/submit
while completion remained close, this profile-off run cannot attribute the
change to binding-only encoding. Its exact failure class is
`profile-off-host-wide-latency-regression-with-specialized-binding-coverage`.
The source remains a correctness-preserving architectural cleanup, but this
paid result is performance REJECT and authorizes no repeat benchmark.

The diagnostic summary is
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-binding-only-cuda-8c58e3ea-20260724T122544Z/diagnostic-summary.json`.
Vast `45319871` is again `stopped/exited`; active or transitional siblings are
`0`. Before another paid run, existing full-profile stage boundaries or a
bounded same-process attribution must separate provider-node encode cost from
host-wide variance and name the next measurable source signal.

### 2026-07-24 Same-Session Direct-Binding Attribution

The bounded attribution subsequently ran both `f435bec9` and `8c58e3ea`
binaries in `A-B-B-A` order on the same retained RTX 4090 and model cache.
All 100 measured requests passed, all 4,440 waves used the typed direct
program, and direct fallback and catalog-epoch miss remained `0`.

Relative to `f435bec9`, the specialized binding source reduced aggregate
provider-node encode from `1498.316 us` to `860.026 us` (`-42.60%`) and host
encode/submit from `3598.252 us` to `2534.590 us` (`-29.56%`). Completion
round trip was effectively unchanged (`7565.833 us` to `7495.563 us`), while
diagnostic throughput followed the binary in both reversed pairs and moved
from `47.0844` to `54.8719 tok/s`. The source is therefore retained:

```text
CUDA DIRECT BINDING AB ATTRIBUTION KEEP: /Users/chejinxuan/ferrum-artifacts/runtime-vnext-binding-ab-attribution-20260724T130106Z/diagnostic-summary.json
```

This resolves the earlier unpaired host-variance result; it does not satisfy
G08B or the unchanged G09 `76.1583 tok/s` floor.

### 2026-07-24 Single-Token Packed-Decode Rejection

Clean source `67921b1c55a093c43e5e6a4ea5f60c7916a962df` and CUDA binary
SHA256
`1a6c582c6af39bf22bfd9f9d6d9a138334ba1b397a6280d955bb376531b1d8d3`
passed the bounded actual-model product paths:

- `c03-001`: resident multi-turn `ferrum run`;
- `c05-001`: non-streaming OpenAI-compatible `ferrum serve`;
- `c06-001`: streaming `ferrum serve`;
- result: `3/3 pass`, zero request error, zero quality issue, and usage-derived
  token counts.

The typed single-token topology removed two transfers and two compute
dispatches per recurrent layer. Full-profile evidence observed 75 decode
correlations and 2,250 recurrent-layer events, all at
`9 compute / 0 transfer`, versus `11 / 2` before the candidate. The recurrent
decode contribution moved from `330` to `270` compute dispatches per
correlation; physical replay span improved from `6.888088 ms` to
`6.195692 ms` (`10.05%`).

The bounded profile-off random `64/32`, c1, `25 x 3`, five-warmup, seed `9271`
check completed `75/75` without error. Its
`54.612796 +/- 10.026469 tok/s` mean missed the predeclared
`59.887970 tok/s` acceptance line and remained `28.29%` below the formal
floor. The structural result is retained as evidence, but the source candidate
is rejected:

```text
FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: focused-c03-c05-c06-report.json
CUDA PACKED DECODE STRUCTURAL KEEP: dispatch-diagnostic/dispatch-summary.json
CUDA PACKED DECODE CANDIDATE REJECT: diagnostic-summary.json
```

Commit `b0286270` reverted the packed-decode candidate after local replay
contracts passed `9/9`, the ferrum-kernels all-target check passed, and
formatting passed. The active branch therefore does not ship this
performance-rejected path.

The sanitized GitHub asset is
[runtime-vnext-packed-decode-cuda-67921b1c-20260724T140119Z-sanitized.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-packed-decode-cuda-67921b1c-20260724T140119Z-sanitized.tar.zst),
asset id `488525573`, size `28,247,157` bytes, SHA256
`152c6cb4257c7656ab7f3fd3722713da97f0e109b7216b9f2c7970969838cc07`.
The locally verified evidence root is
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-packed-decode-cuda-67921b1c-20260724T140119Z/`.
Vast `45319871` is `stopped/exited`; there is no paid sibling.

No repeat benchmark is authorized. The initial successor hypothesis was to
compute Q/K normalization once per key head in a separate dispatch; the next
section records that this hypothesis has already been tested and rejected.

### 2026-07-24 Separate-Normalization Packed-Decode Rejection

Clean source `a884f5d44e9bb68542e2dfe67d8310fb2071f227`, binary SHA256
`64e137337df0f120d7a4d415198a2d9a7a0921df4c34f3d8b7808fd6992bf357`,
implemented decode conv-pack, a once-per-key-head Q/K normalization dispatch,
and a prenormalized packed-delta kernel. The actual RTX 4090 CUDA parity test
passed `1/1`, and actual-model `c03 run`, `c05 serve`, and `c06 streaming
serve` passed `3/3` with zero request or quality error.

The topology reached `10 compute / 0 transfer`, but mean physical replay over
75 decode correlations was `6.221576 ms`: `0.025885 ms` or `0.4178%` slower
than `67921b1c` at `6.195692 ms`. The predeclared structural stop condition
therefore rejected the candidate before profile-off:

```text
CUDA NORMALIZED PACKED DECODE CANDIDATE REJECT: diagnostic-summary.json
```

Commit `784d5bf2` reverted the source. The verified local artifact is
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-normalized-packed-decode-cuda-a884f5d4-20260724T153606Z/`;
the GitHub archive has asset id `488576890`, size `27,926,534` bytes, and
SHA256
`38e715a12017b0771f8da38b2c80b7b4fbdab358c23a67f61ccba3ad77e67080`.
Vast `45319871` is `stopped/exited` with no paid sibling.

Current vLLM source `426d48bfa149582664d48f89df21ec9beae5c37b`
uses `causal_conv1d_update` followed by packed recurrent delta. Its Triton grid
is `(NV, B * HV)` and normalizes Q/K inside every value tile, deliberately
trading repeated arithmetic for one fewer launch. Thus `67921b1c` was closest
to vLLM; `a884f5d4` was a distinct Ferrum alternative, not a copied design.
A future candidate must avoid both repeated per-value-tile normalization and
the extra normalization dispatch, retain at most `9 compute / 0 transfer`,
and prove direct-path performance with same-session paired evidence. G09 owns
the detailed `full` versus `basic/off` measurement amendment. It permits one
final bounded `A-B-B-A` re-attribution of the already-built baseline and
`67921b1c` binaries because the original rejection compared different
sessions and sample sizes; it does not permit rebuilding or repeating the old
unpaired sweep.

## Metal Matrix Workflow

The Metal lane reuses the same backend-parameterized preparation and checkpoint
contracts as CUDA. The thin Metal entrypoints must not fork the matrix or artifact
schema. The pinned M2 Metal lock is
`scripts/release/configs/runtime_vnext_g08b_m2_metal.models.lock.json`.

Use the staged regression policy from `G02_TEST_EVIDENCE.md`:

1. Build and bind one clean candidate binary.
2. Replay the exact failed case with `--focus-case`; this produces diagnostic
   `KEEP` or `REJECT`, never a formal PASS.
3. Run the affected contract scenario with `--focus-scenario`, plus the
   cross-entrypoint sentinels when shared behavior changed.
4. After source freeze, execute the complete 702-case Metal matrix once.
5. Validate it through `run_gate.py vnext-g08b-metal`.

The formal Metal checkpoint requires ordered C01-C21, `702/702 pass`, both `run`
and `serve`, C18 cells `c1/c4/c16`, typed active floor `4`, active duty-cycle
`>=0.80`, and zero error/OOM/leak counters. Focused artifacts and reports from a
different source SHA, binary SHA256, model lock, model revision, hardware id, or
effective config cannot satisfy this checkpoint.
