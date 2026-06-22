# W3 Qwen3.5 Two-Hour Handoff - 2026-06-22

## Current State

- Branch: `goal/w2-w3-release-grade`.
- PR: <https://github.com/sizzlecar/ferrum-infer-rs/pull/237>.
- W3 is not complete. There is no `MODEL_RELEASE_GRADE_W3 PASS`.
- Correctness artifacts for real Qwen3.5 GPTQ product paths exist. Use the
  refreshed L2 artifact with output-hygiene evidence:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_hygiene_from_real_product_20260622_75ec7e6e/w3_l2_quantized.json`.
- The older L2 artifact
  `w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e`
  is no longer sufficient because it lacks `output_hygiene`.
- The current blocking item is still W3 performance versus the vLLM ShareGPT
  baseline, not product-path correctness.

## Effective Progress In This Slice

- Added a checked-in current-evidence config for the W3 final manifest:
  `docs/goals/model-coverage-2026-06-12/w3_qwen35_current_evidence_config.json`.
- `scripts/release/model_release_grade_manifest.py` now accepts `--config` for
  W3/W2 manifest inputs.
- Config paths are resolved relative to the repo root, matching the checked-in
  artifact paths used in this goal.
- Inline `dirty_status` JSON is materialized under
  `<out>/_config_inputs/dirty_status.json`.
- Inline command strings/lists are materialized under
  `<out>/_config_inputs/*.txt`; existing command-file paths are reused.
- The manifest self-test now covers this config path and validates the generated
  synthetic W3 manifest with the final validator.
- Local validation passed:
  `python3 -m py_compile scripts/release/model_release_grade_manifest.py scripts/release/model_release_grade_goal_gate.py`,
  `python3 scripts/release/model_release_grade_manifest.py --self-test`, and
  JSON syntax validation for the checked-in config.
- Running the checked-in current-evidence config produced
  `target/w3_qwen35_current_evidence_probe/model_release_grade_manifest.json`
  and failed the final validator as expected:
  `MODEL_RELEASE_GRADE_W3 FAIL (45 problems)`.
- The failure is now reproducible from one command:

```bash
python3 scripts/release/model_release_grade_manifest.py \
  --config docs/goals/model-coverage-2026-06-12/w3_qwen35_current_evidence_config.json
```

- The current failure buckets are:
  - old L5 artifact lacks `--ignore-eos` command evidence;
  - old L5 artifact lacks `output_tokens_per_request` matrices;
  - old Ferrum and vLLM benchmark commands lack `--ignore-eos`;
  - old Ferrum report does not produce fixed 128 output tokens per request;
  - c1/c4/c16/c32 throughput ratios are still below `0.800`;
  - c1/c4/c16/c32 p95 ITL still exceeds `1.25x` vLLM baseline.
- This is not a W3 completion claim. It is a reproducible final-gate blocker
  snapshot so the next GPU run can replace only the L5/perf artifacts and rerun
  the same config path.

- Added a typed `ferrum bench-serve --ignore-eos` flag in
  `crates/ferrum-cli/src/commands/bench_serve.rs`.
- When the flag is present, the canonical HTTP performance client sends the
  vLLM-compatible request field `"ignore_eos": true` to
  `/v1/chat/completions`.
- Default behavior is unchanged: without `--ignore-eos`, the request body does
  not include `ignore_eos`.
- This uses the existing product path. `crates/ferrum-server` already accepts
  `ignore_eos`, writes `ferrum_ignore_eos` metadata, and
  `crates/ferrum-engine` already uses that metadata when resolving stop
  conditions.
- This is not a hidden env flip and not a model-family default. It is an
  explicit CLI/configured benchmark contract.
- The W3 final validator now enforces this benchmark contract. W3 Ferrum and
  baseline performance commands must include `--ignore-eos`, must record
  `output_tokens_per_request` and `baseline_output_tokens_per_request`, and
  every request in both matrices must equal `--random-output-len`.
- The final validator also reloads the referenced Ferrum and baseline
  `bench-serve` report artifacts and cross-checks their output-token matrices
  against the manifest.
- It also cross-checks original report `n_repeats`, `n_requests_per_run`,
  completed/error counts, and per-run quality counts against the manifest, so
  a hand-written zero-error manifest cannot hide failed measured requests.
- The manifest builder now copies the output-token matrices from
  `bench-serve` reports into the final manifest and validates their shape.
- The W3 L5 concurrency packaging gate now enforces the same fixed-output
  contract before it writes `w3_l5_concurrency.json`: saved L5 commands must
  include `--ignore-eos` and `--random-output-len 128`, and each report cell
  must carry an `output_tokens_per_request` matrix where every request equals
  128.
- The final W3 validator now re-checks those L5 fields, so old L5 artifacts
  without fixed-output evidence no longer pass the W3 L5 correctness link.
- The W3 L2 packaging gate now scans every known-answer output and response
  artifact for release-blocking text such as `<unk>`, `[PAD]`, reserved special
  tokens, mojibake, panic, and KV overflow. The final validator requires this
  `output_hygiene` field.
- A new L2 artifact was generated from the tracked real-product
  `known_answer_report.json`: 11/11 known-answer cases and 11/11 response
  artifacts scanned, with `forbidden_patterns_absent=true`.

## Why This Matters

The existing W3 vLLM and Ferrum performance artifacts are not output-length
equivalent:

| report | c1 mean/min/p50/max output tokens | c32 mean/min/p50/max output tokens |
| --- | ---: | ---: |
| vLLM baseline `w3_vllm_sharegpt_baseline_20260619` | 128 / 128 / 128 / 128 | 128 / 128 / 128 / 128 |
| Ferrum release-shape `w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9` | 47 / 43 / 44 / 71 | 45.013 / 43 / 44 / 71 |

That means the previous Ferrum/vLLM throughput comparison mixes a fixed-output
vLLM run with a natural-EOS-short Ferrum run. The next A/B must make this
explicit by passing `--ignore-eos` through the same `bench-serve` client.

## Local Validation Run

These commands passed on the local Mac workspace:

```bash
cargo fmt --all -- --check
cargo test -p ferrum-cli chat_completion_body -- --nocapture
cargo check -p ferrum-cli
cargo test -p ferrum-cli commands::bench_serve::tests -- --nocapture
cargo run -p ferrum-cli -- bench-serve --help
python3 -m py_compile scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py
python3 scripts/release/model_release_grade_goal_gate.py --self-test
python3 scripts/release/model_release_grade_manifest.py --self-test
git diff --check
```

Important outputs:

- `chat_completion_body_*`: 2 passed.
- `commands::bench_serve::tests`: 18 passed.
- CLI help prints `--ignore-eos`.
- `MODEL RELEASE GRADE GOAL SELFTEST PASS`.
- `MODEL RELEASE GRADE MANIFEST SELFTEST PASS`.
- `W3 L2 QUANTIZED PASS: docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_hygiene_from_real_product_20260622_75ec7e6e`.

## Gate Probe

A local probe using the old Ferrum perf artifact
`w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9` now fails the W3 final
validator for the expected fixed-output reasons:

- `performance.c{1,4,16,32} command missing --ignore-eos`
- `performance.c{1,4,16,32}.baseline command missing --ignore-eos`
- `performance.c{1,4,16,32}.output_tokens_per_request[...] must equal --random-output-len 128`
- `performance.c{1,4,16,32}.artifact.output_tokens_per_request[...] must equal --random-output-len 128`

The old ratio and p95 ITL failures still remain. The important change is that
the old short-output Ferrum artifact can no longer be treated as valid W3 80%
performance evidence.

A direct probe of the old L5 artifact
`w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/l5_concurrency/w3_l5_concurrency.json`
now also fails: it is missing `--ignore-eos` command evidence and its four
concurrency cells do not include `output_tokens_per_request`. Re-run L5 from
the new fixed-output bench report instead of reusing that artifact.

## Next GPU Validation

Use a reachable exact 1x RTX 4090 lane. Do not claim performance from local
CPU/Mac validation.

Correctness first:

- Build CUDA release binary with the W3 CUDA feature set.
- Run product-path smoke for both `ferrum run` and `ferrum serve`.
- Include streaming usage smoke before any performance result is treated as
  evidence.

Then run the same ShareGPT sweep with the new flag:

```bash
/workspace/ferrum-target-w3/release/ferrum bench-serve \
  --base-url http://127.0.0.1:18173 \
  --model 3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b \
  --tokenizer /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b \
  --dataset sharegpt \
  --sharegpt-path /workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl \
  --random-output-len 128 \
  --ignore-eos \
  --concurrency-sweep 1,4,16,32 \
  --num-prompts 100 \
  --warmup-requests 10 \
  --n-repeats 3 \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --timeout 900 \
  --output json \
  --out /workspace/artifacts/<run>/perf/bench_ferrum_sharegpt_sweep_100x3_ignore_eos.json
```

First acceptance check for this diagnostic: every measured request should have
`output_tokens=128` from usage. If not, debug stop-condition propagation before
looking at kernels.

Package L5 from the same fixed-output report, using the exact command string
that produced it:

```bash
python3 scripts/release/w3_l5_concurrency_gate.py \
  --report /workspace/artifacts/<run>/perf/bench_ferrum_sharegpt_sweep_100x3_ignore_eos.json \
  --out /workspace/artifacts/<run>/l5_concurrency \
  --expected-output-len 128 \
  --command "/workspace/ferrum-target-w3/release/ferrum bench-serve --base-url http://127.0.0.1:18173 --model 3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b --tokenizer /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b --dataset sharegpt --sharegpt-path /workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl --random-output-len 128 --ignore-eos --concurrency-sweep 1,4,16,32 --num-prompts 100 --warmup-requests 10 --n-repeats 3 --fail-on-error --require-ci --seed 9271 --timeout 900 --output json --out /workspace/artifacts/<run>/perf/bench_ferrum_sharegpt_sweep_100x3_ignore_eos.json"
```

## Current Performance Gap To Recheck

Historical release-shape numbers before this CLI contract fix:

| concurrency | Ferrum mean tok/s | vLLM mean tok/s | Ferrum p95 ITL ms | vLLM p95 ITL ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 53.806 | 136.143 | 14.730 | 6.792 |
| 4 | 99.130 | 405.420 | 32.615 | 9.120 |
| 16 | 142.177 | 1190.692 | 137.435 | 11.890 |
| 32 | 142.839 | 1708.528 | 198.989 | 15.710 |

Do not use this table as a final performance claim because the output token
distributions were not equivalent.

## GPU Blocker

- Existing Vast instance `41422823` was last verified as stopped/exited.
- SSH to `ssh7.vast.ai:22822` returned `Connection refused`.
- A Vast start attempt returned `resources_unavailable` and remained
  stopped/exited during polling.
- A separate replacement rental attempt was previously blocked by
  `insufficient_credit`.
- Do not keep cycling paid GPU offers until the account/instance state is
  usable.

## If The Two-Hour Cutoff Hits

- Keep this branch clean and pushed to PR #237.
- Do not claim W3 completion.
- State that the pushed progress is:
  - source-level W3 hot-path metadata cleanup,
  - final-validator artifact-path hardening,
  - explicit fixed-output benchmark contract via `bench-serve --ignore-eos`.
  - L5 fixed-output evidence packaging and final-validator cross-checks.
- State that the next required evidence is a 1x RTX 4090 correctness smoke plus
  same-hardware ShareGPT `--ignore-eos` sweep, followed by the W3 final
  validator.

## 2026-06-22 Two-Hour Update

Concrete code progress in this window:

- Added `scripts/release/w3_qwen35_cuda_release_lane.py`, a GPU-first W3 lane
  runner that composes the existing product/L2/L4/L5/final-manifest gates.
- The runner starts HF model snapshot prefetch asynchronously while the CUDA
  release binary builds. It uses `HF_HOME=/workspace/hf-cache` and
  `HF_XET_HIGH_PERFORMANCE=1`; `HF_TOKEN` may be present in the environment but
  is never printed or written to command artifacts.
- The runner records the paid GPU contract, hardware snapshot, git state,
  binary SHA256, sanitized env summary, command JSON/txt files, server logs,
  bench report, L5 artifact, manifest config, and final manifest output under
  one artifact root.
- It enforces the W3 fixed-output perf command shape:
  `bench-serve --dataset sharegpt --random-output-len 128 --ignore-eos
  --concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3
  --fail-on-error --require-ci --seed 9271`.
- `scripts/release/w3_qwen35_real_product_report.py` now accepts typed serve
  CLI knobs for scheduler/prefix-cache validation:
  `--scheduler-prefill-first-until-active`,
  `--scheduler-prefill-step-chunk`,
  `--scheduler-active-decode-prefill-chunk`,
  `--enable-prefix-caching`, and `--disable-prefix-cache`.

Run this when a reachable 1x RTX 4090 host is available:

```bash
python3 scripts/release/w3_qwen35_cuda_release_lane.py \
  --out docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_cuda_release_lane_<YYYYMMDDTHHMMSSZ> \
  --hf-home /workspace/hf-cache \
  --gpu-devices 0
```

Optional typed CLI tuning is passed explicitly, for example:

```bash
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --scheduler-prefill-first-until-active 8
```

Do not use hidden `FERRUM_*` overrides for release evidence; the runner scrubs
them from child processes and records the scrubbed key names.

Validation completed locally:

```bash
python3 -m py_compile \
  scripts/release/w3_qwen35_cuda_release_lane.py \
  scripts/release/w3_qwen35_real_product_report.py \
  scripts/release/w3_l2_quantized_gate.py \
  scripts/release/w3_l4_agent_gate.py \
  scripts/release/w3_l5_concurrency_gate.py \
  scripts/release/model_release_grade_manifest.py
python3 scripts/release/w3_qwen35_cuda_release_lane.py --self-test
python3 scripts/release/w3_qwen35_real_product_report.py --self-test
python3 scripts/release/w3_l2_quantized_gate.py --self-test
python3 scripts/release/w3_l4_agent_gate.py --self-test
python3 scripts/release/w3_l5_concurrency_gate.py --self-test
python3 scripts/release/model_release_grade_manifest.py --self-test
git diff --check
```

Current status after this update:

- Branch has real code progress, but still no new 1x4090 execution artifact.
- No `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` exists.
- The next blocker is external GPU availability, not local command assembly.
- If the lane fails on GPU, stop at the first failing step and inspect that
  step's artifact/log instead of running another full sweep.

## 2026-06-22 Continuation Update

External GPU state:

- Direct SSH to `ssh7.vast.ai:22822` still returned `Connection refused`.
- Vast inventory still lists only instance `41422823`, 1x `RTX 4090`,
  `49140` MiB visible VRAM, `cur_state=stopped`,
  `actual_status=exited`, at about `$0.663/h`.
- One start attempt was made after recording the paid GPU lane/cost/stop/gate
  contract. Vast returned HTTP 200 with
  `success=false`, `error=resources_unavailable`.
- A follow-up query still reported `stopped/exited`, so there was no running
  GPU to use and no idle running instance left behind.
- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_vast_start_41422823_20260622T005359Z/`.

Important runner fix after the GPU blocker:

- The checked-in historical vLLM baseline report is fixed-output, but its saved
  `bench-vllm.command.txt` lacks `--ignore-eos`.
- Current W3 final validation requires `--ignore-eos` on both Ferrum and
  baseline bench commands. Without a fix, the next GPU run could pass Ferrum
  product/L2/L4/L5 and still fail final manifest on baseline command evidence.
- `scripts/release/w3_qwen35_cuda_release_lane.py` now defaults to
  `--baseline-mode auto`:
  - if the historical baseline command/report passes the fixed-output
    contract, it is reused;
  - if it does not, the lane runs a live same-host vLLM OpenAI server and
    re-runs the baseline with the same Ferrum `bench-serve` client and
    `--ignore-eos --random-output-len 128 --concurrency-sweep 1,4,16,32`.
- The script records `baseline_decision.json`, vLLM versions, vLLM server
  command, vLLM bench command, and live baseline report before building the
  final manifest config.
- Use `--baseline-mode historical` only when intentionally requiring a
  pre-existing fixed-output baseline; with the current checked-in baseline
  command, that mode should fail rather than fabricate command evidence.

Validation for this continuation:

```bash
python3 -m py_compile scripts/release/w3_qwen35_cuda_release_lane.py
python3 scripts/release/w3_qwen35_cuda_release_lane.py --self-test
git diff --check
```

All three passed locally.
