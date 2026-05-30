# Structured Profile Schema Status - 2026-05-30

Milestone B is not complete. This checkpoint adds the shared schema,
validators, a native JSONL sink, and the first Rust/C++ profile emitters.

## Added

- `ferrum-bench-core::profile::ProfileEvent` defines the locked JSONL envelope:
  - `event`
  - `commit_sha`
  - `env_hash`
  - `model`
  - `concurrency`
  - `shape`
  - `stage_us`
  - `graph_enabled`
  - `runtime_flags`
  - optional `source`
  - optional `source_line`
- `parse_profile_event_value` checks required-field presence before
  deserialization, so `commit_sha: null` remains distinct from a missing
  `commit_sha` key.
- `profile_event_rejects_every_missing_required_field` and the artifact
  validator self-test now exercise every locked top-level required key rather
  than only a representative missing field.
- `parse_profile_jsonl_str` validates every non-blank JSONL line and preserves
  line numbers in validation errors.
- `require_profile_event_groups` verifies profile artifacts contain expected
  event families such as `unified_prof` and `bucket_prof`.
- `profile_parser_covers_three_fixture_artifact_shapes` covers the Milestone B
  fixture requirement with three representative JSONL artifacts:
  default graph-on (`graph_prof` + `unified_prof`), graph-off route dump
  (`moe_dump` + `vllm_moe_config` + `bucket_prof`), and FA-layout attention
  A/B (`unified_layer_prof` + `unified_prof`).
- `scripts/m3_validate_runner_artifact.py` now mirrors the schema checks for
  runner artifacts: required profile keys, `sha256:` env hash, positive integer
  concurrency, object-shaped `shape` / `stage_us` / `runtime_flags`, and boolean
  `graph_enabled`.
- Artifact validation now re-checks manifest-declared structured profile event
  gates (`required_events` and `required_any_events`) against the actual
  `profile.jsonl` contents, so a missing `moe_dump`, `vllm_moe_config`, or
  equivalent required event group cannot pass by manifest metadata alone.
- `ferrum-bench-core::profile::ProfileJsonlWriter` writes native JSONL events
  from a typed `ProfileSinkConfig`. `ferrum serve --profile-*` initializes the
  Rust writer directly and passes the same typed sink to native kernel bridges.
  The old `FERRUM_PROFILE_*` env bridge remains only as a compatibility
  fallback.
- Native emitters now cover the primary M3 Rust-side profile families:
  `iter_prof`, `unified_prof`, `unified_layer_prof`, `batched_decode_prof`,
  `bucket_prof`, `graph_prof`, and `moe_dump`.
- The vLLM-MoE Marlin C++ config logger now accepts a typed C ABI profile sink
  configured from Rust and appends native `vllm_moe_config` JSONL events without
  requiring `ferrum serve --profile-*` to export `FERRUM_PROFILE_*` env vars.
- `scripts/m3_route_unified_profile.sh` now configures the runner in
  structured profile mode and validates required event groups from
  `profile.jsonl` instead of required grep patterns.
- `scripts/m3_graph_runtime_profile.sh` now configures the runner in
  structured profile mode and validates graph-on runtime profile event groups
  from `profile.jsonl` instead of grepping `server.log`.
- `scripts/m3_ab_runner.py` can now pass structured profile sink metadata to
  `ferrum serve` via typed CLI args for selected cases only via
  `profile.profile_env_cases`, allowing same-binary overhead checks where the
  baseline case is not exposed to the profile sink.
- `scripts/m3_profile_sink_overhead_ab.sh` compares the default runtime
  against a profile-sink-only row without enabling diagnostic timers, dumps, or
  profile event gates.

## Validation

Local:

```bash
cargo test -q -p ferrum-bench-core profile
cargo fmt --all -- --check
cargo check -q -p ferrum-kernels
cargo check -q -p ferrum-cli
python3 -m py_compile scripts/m3_validate_runner_artifact.py scripts/m3_ab_runner.py
python3 scripts/m3_validate_runner_artifact.py --self-test
python3 scripts/m3_ab_runner.py --self-test
VALIDATE_ONLY=1 MODEL_DIR=/tmp BIN=/bin/echo OUT_ROOT=/tmp/m3-route-validate \
  bash scripts/m3_route_unified_profile.sh
VALIDATE_ONLY=1 MODEL_DIR=/tmp BIN=/bin/echo OUT_ROOT=/tmp/m3-graph-runtime-validate \
  bash scripts/m3_graph_runtime_profile.sh
VALIDATE_ONLY=1 MODEL_DIR=/tmp BIN=/bin/echo OUT_ROOT=/tmp/m3-profile-sink-validate \
  bash scripts/m3_profile_sink_overhead_ab.sh
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap
git diff --check
```

Remote GPU pod after rsync to `/workspace/ferrum-codex-clean`:

```bash
cargo test -q -p ferrum-bench-core profile
python3 scripts/m3_ab_runner.py --self-test
python3 scripts/m3_validate_runner_artifact.py --self-test
python3 -m py_compile scripts/m3_ab_runner.py scripts/m3_validate_runner_artifact.py scripts/check_ferrum_env_registry.py
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap
MODEL_DIR=/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441 \
  OUT_ROOT=/tmp/m3-route-validate-structured BIN=/bin/echo VALIDATE_ONLY=1 \
  bash scripts/m3_route_unified_profile.sh
MODEL_DIR=/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441 \
  OUT_ROOT=/tmp/m3-graph-runtime-validate-structured BIN=/bin/echo VALIDATE_ONLY=1 \
  bash scripts/m3_graph_runtime_profile.sh
MODEL_DIR=/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441 \
  OUT_ROOT=/tmp/m3-profile-sink-validate BIN=/bin/echo VALIDATE_ONLY=1 \
  bash scripts/m3_profile_sink_overhead_ab.sh
cargo check -q -p ferrum-kernels --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
cargo check -q -p ferrum-cli
```

Remote `cargo fmt --all -- --check` could not run because the pod's stable
toolchain does not have `rustfmt` installed. The local format check passed.

Local refresh after typed profile sink wiring:

- `cargo test -q -p ferrum-bench-core profile -- --nocapture`: `10 passed`,
  including the three structured profile fixture artifact shapes.
- `cargo check -q -p ferrum-cli`: passed.
- `python3 scripts/m3_ab_runner.py --self-test`: passed.
- `python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap`:
  `147/147` registry coverage, hot direct env reads `4`.
- `VALIDATE_ONLY=1 MODEL_DIR=/tmp BIN=/bin/echo
  OUT_ROOT=/tmp/m3-graph-runtime-validate bash
  scripts/m3_graph_runtime_profile.sh`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

## Real Structured Profile Artifact

Remote c32 route/unified diagnostic profile with explicit artifact verdict:

```bash
cd /workspace/ferrum-codex-clean
OUT_ROOT=/workspace/m3-structured-route-profile-verdict-20260530_001311 \
  BUILD=0 CONCURRENCY=32 NUM_PROMPTS=32 WARMUP_REQUESTS=0 \
  RANDOM_INPUT_LEN=256 RANDOM_OUTPUT_LEN=64 PORT=18731 \
  bash scripts/m3_route_unified_profile.sh
python3 scripts/m3_validate_runner_artifact.py --require-profile-events \
  /workspace/m3-structured-route-profile-verdict-20260530_001311
```

Evidence:

- Runner case `route_profile` passed Paris, bench, structured profile
  validation, and cleanup.
- The root manifest, case manifest, and summary row recorded
  `artifact_verdict=diagnostic-only`, `not_publishable=true`, and
  `not_publishable_reason="graph-off sync-timer route/profile run; throughput is diagnostic-only"`.
- `bench-serve` completed `32/32` requests with `errored=0`.
- Metrics were diagnostic-only because this run set graph-off sync/timing
  profile envs: throughput `691.52 tok/s`, `TTFT p50=867.12 ms`,
  `TPOT p50=31.73 ms`, `ITL p95=144.20 ms`.
- `profile.jsonl` contained `244` native structured events and passed
  `scripts/m3_validate_runner_artifact.py --require-profile-events`.
- Event counts:
  - `batched_decode_prof`: `62`
  - `bucket_prof`: `72`
  - `iter_prof`: `12`
  - `moe_dump`: `1`
  - `unified_prof`: `65`
  - `vllm_moe_config`: `32`
- Required structured profile groups were present:
  `moe_dump`, `vllm_moe_config`, `unified_prof`, `iter_prof`, `bucket_prof`,
  and `batched_decode_prof` satisfying the
  `unified_layer_prof | batched_decode_prof` requirement.
- The route dump captured c32 shape:
  `batch_x_topk=256`, `block_size=16`, `total_post_pad=1008`,
  `active_blocks=63`, `unique_experts=61`.
- The vLLM-MoE config event captured gate/up shape:
  `prob_m=32`, `prob_n=1536`, `prob_k=2048`, `top_k=8`,
  `thread_k=64`, `thread_n=128`, `threads=128`, `blocks_per_sm=3`.
- Cleanup sent `SIGINT`, did not send `SIGKILL`, server return code was `0`,
  and a post-run external process scan found no live `target/release/ferrum`,
  `bench-serve`, `cargo`, `nvcc`, or `vllm` processes.

Earlier full 64-prompt validation at
`/workspace/m3-structured-route-profile-20260529_235552` produced `623`
native events and completed `64/64` requests, but it predates the manifest
verdict fields. The verdict artifact above supersedes it for publishability
metadata evidence.

## Profile Sink Overhead Evidence

Remote c32 same-binary N=3 A/B:

```bash
cd /workspace/ferrum-codex-clean
OUT_ROOT=/workspace/m3-profile-sink-overhead-c32-n3-20260530_002409 \
  BUILD=0 ARTIFACT_VERDICT=pass CONCURRENCY=32 NUM_PROMPTS=128 \
  WARMUP_REQUESTS=10 RANDOM_INPUT_LEN=256 RANDOM_OUTPUT_LEN=128 \
  REPEATS=3 PORT_BASE=18750 \
  bash scripts/m3_profile_sink_overhead_ab.sh
python3 scripts/m3_validate_runner_artifact.py \
  /workspace/m3-profile-sink-overhead-c32-n3-20260530_002409
```

Evidence:

- Both rows passed Paris, completed `384/384` requests, and had `errored=0`.
- The root/case manifests and summary marked the run `pass` and
  `not_publishable=false`.
- `default` recorded `profile_env_enabled=false`; `profile_sink` recorded
  `profile_env_enabled=true`.
- Neither row wrote profile events, confirming no diagnostic profile producers
  were enabled for this overhead check.
- Throughput was `1304.67 ± 14.23 tok/s` for `default` and
  `1296.28 ± 17.69 tok/s` for `profile_sink`, a `-0.64%` N=3 delta versus
  baseline.
- CI95 half-widths were `35.35` and `43.94 tok/s`, so the observed delta is
  well inside run noise for this artifact.
- Cleanup succeeded and the post-run process scan found no live
  `target/release/ferrum`, `bench-serve`, `cargo`, `nvcc`, or `vllm` process.
- Earlier N=1 smoke at
  `/workspace/m3-profile-sink-overhead-smoke-20260530_002108` also passed, with
  a `+1.23%` profile-sink delta, and is diagnostic-only evidence.

## Remaining B Gaps

- Some remaining runtime profile wrappers still consume text logs; the primary
  route/unified and graph-runtime profile wrappers now require native
  structured JSONL events.
- Profile sink overhead now has c32 N=3 same-binary evidence within noise, but
  full diagnostic producer overhead is still intentionally excluded from this
  low-intrusion bound.
- The remaining `FERRUM_PROFILE_*` bridge is now backwards-compatibility only;
  normal `ferrum serve --profile-*` native profile emission uses typed Rust
  configuration plus the vLLM-MoE C ABI sink.
