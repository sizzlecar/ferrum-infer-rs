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
