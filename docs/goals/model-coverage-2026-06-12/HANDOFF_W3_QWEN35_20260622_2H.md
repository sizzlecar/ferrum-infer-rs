# W3 Qwen3.5 Two-Hour Handoff - 2026-06-22

## Current State

- Branch: `goal/w2-w3-release-grade`.
- PR: <https://github.com/sizzlecar/ferrum-infer-rs/pull/237>.
- W3 is not complete. There is no `MODEL_RELEASE_GRADE_W3 PASS`.
- Correctness artifacts for real Qwen3.5 GPTQ product paths exist and the
  final-validator probe now reaches the performance checks.
- The current blocking item is still W3 performance versus the vLLM ShareGPT
  baseline, not product-path correctness.

## Effective Progress In This Slice

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
- The W3 final validator now enforces this benchmark contract. W3 performance
  cells must include `--ignore-eos`, must record
  `output_tokens_per_request` and `baseline_output_tokens_per_request`, and
  every request in both matrices must equal `--random-output-len`.
- The manifest builder now copies the output-token matrices from
  `bench-serve` reports into the final manifest and validates their shape.

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

## Gate Probe

A local probe using the old Ferrum perf artifact
`w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9` now fails the W3 final
validator for the expected fixed-output reasons:

- `performance.c{1,4,16,32} command missing --ignore-eos`
- `performance.c{1,4,16,32}.output_tokens_per_request[...] must equal --random-output-len 128`

The old ratio and p95 ITL failures still remain. The important change is that
the old short-output Ferrum artifact can no longer be treated as valid W3 80%
performance evidence.

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
- State that the next required evidence is a 1x RTX 4090 correctness smoke plus
  same-hardware ShareGPT `--ignore-eos` sweep, followed by the W3 final
  validator.
