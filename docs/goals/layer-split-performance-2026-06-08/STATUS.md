# Layer Split Performance Status

Last updated: 2026-06-10.

This goal is not complete. Completion still requires the final validator to print:

```text
LAYER_SPLIT_PERF GOAL PASS: <out_dir>
```

## Current State

- Target model: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`.
- Target split: `stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79`.
- Current local base SHA: `7258db1d679dc08f76c010d310a1db248c9d7542`.
- The worktree is intentionally dirty with diagnostic/performance changes; final goal evidence still needs a clean-worktree full run.
- Vast instance `40203673` was used for the latest diagnostics and was stopped after artifacts were copied locally.
- The current candidate config is now `batch-tuned`, not `overlapped`: batch mode with `max_num_seqs=16`, `max_num_batched_tokens=1536`, and `scheduler_prefill_first_until_active=16`.
- The final validator now accepts either `batch` or `overlapped` candidate modes, while preserving the mode-specific pipeline metric checks and the requirement that candidate throughput beat baseline and meet the selected target.

## Latest Product GPU Diagnostic

Artifact copied locally:

```text
/Users/chejinxuan/rust_ws/ferrum-infer-rs-records/layer-split/layer-split-perf-minbatch16-diag-20260609-171056
```

Remote smoke pass line:

```text
LAYER_SPLIT_PERF SMOKE PASS: /workspace/layer-split-perf-minbatch16-diag-20260609-171056
```

This was diagnostic only, not final goal evidence. It ran with dirty-worktree and diagnostic overrides:

```text
max_model_len=4096
kv_capacity=1024
max_num_seqs=16
max_num_batched_tokens=1536
concurrency_cells=4,8,16
num_prompts=48
warmup_requests=4
n_repeats=1
```

Correctness passed for both baseline batch and candidate overlapped:

- `ferrum run` multi-turn
- `ferrum serve` normal answer
- serve multi-turn
- structured output
- streaming usage
- tool-call regression

Throughput:

| Case | c4 output tok/s | c8 output tok/s | c16 output tok/s | c4 TPOT p50 ms | c8 TPOT p50 ms | c16 TPOT p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline batch | 57.581 | 85.816 | 112.394 | 63.626 | 85.049 | 121.818 |
| candidate overlapped | 58.187 | 87.239 | 97.839 | 62.890 | 82.637 | 141.952 |

Relative result:

- c4 candidate: `+1.05%` output tok/s.
- c8 candidate: `+1.66%` output tok/s.
- c16 candidate: `-12.95%` output tok/s and `+16.53%` TPOT p50.

Candidate did exercise overlapped decode:

```text
candidate pipeline_decode.selected_pipeline_mode = overlapped
candidate pipeline_decode.calls = 2545
candidate pipeline_decode.overlapped_calls = 216
candidate pipeline_decode.max_batch = 16
candidate pipeline_decode.microbatch_count_max = 2
candidate pipeline_decode.in_flight_stage_count_max = 2
```

Pipeline timing from the candidate health artifact:

```text
stage_us_avg = [25995, 25978]
logits_us_avg = 4671
bridge_us_avg = 86
host_copy_us_avg = 85
total_us_avg = 54513
```

GPU utilization evidence from bench samples:

| Case | GPU0 util avg/max | GPU1 util avg/max | max memory |
| --- | ---: | ---: | ---: |
| baseline batch | 33.2% / 52% | 58.4% / 100% | ~23.1 GiB/GPU |
| candidate overlapped | 47.8% / 79% | 47.2% / 94% | ~23.1 GiB/GPU |

Key finding: candidate utilization is more balanced, but c16 throughput is worse. Higher GPU utilization is therefore not sufficient evidence; the microbatch split loses more batch efficiency than overlap recovers.

## Native CUDA Overlap Probe

Added direct CUDA microbench:

```text
scripts/microbenches/layer_split_overlap_probe.cu
```

Local artifacts:

```text
/Users/chejinxuan/rust_ws/ferrum-infer-rs-records/layer-split/microbenches/layer_split_overlap_probe_20260609-174309.json
/Users/chejinxuan/rust_ws/ferrum-infer-rs-records/layer-split/microbenches/layer_split_overlap_sweep_20260609-174412/
```

Command shape:

```bash
nvcc -O3 -arch=sm_89 -std=c++17 layer_split_overlap_probe.cu -o layer_split_overlap_probe
./layer_split_overlap_probe \
  --batch-sizes 4,8,16 \
  --microbatch-sizes 4,8 \
  --repeats 30 \
  --warmup 5 \
  --stage0-ms 26 \
  --stage1-ms 26 \
  --logits-ms 4.7 \
  --stage-fixed-frac 0.60
```

Main result:

| Batch | Microbatch | Sequential full-batch ms | Overlapped microbatch ms | Speedup vs full-batch |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 51.431 | 63.396 | 0.811x |
| 16 | 4 | 51.524 | 94.183 | 0.547x |
| 16 | 8 | 51.558 | 63.450 | 0.813x |

The sweep shows the same direction:

- With `stage_fixed_frac=0.00`, overlap can beat full-batch.
- At `stage_fixed_frac=0.20`, only `batch16/microbatch8` still wins slightly.
- At `stage_fixed_frac>=0.40`, `batch8/microbatch4` and `batch16/microbatch8` both lose to full-batch.

Conclusion: the current microbatch overlap route is invalid for the product path. It improves over already-split sequential execution, but it does not beat the existing full-batch path when stage work has meaningful fixed/batch-efficiency cost. This matches the Qwen72B product diagnostic.

## Current Candidate Direction

The release-quality runner no longer defaults to the failed overlapped candidate.

Current configs:

```text
baseline:  scripts/release/configs/layer_split_perf_baseline_batch.json
candidate: scripts/release/configs/layer_split_perf_qwen72b_candidate_batch_tuned.json
smoke:     scripts/release/configs/layer_split_perf_qwen72b_candidate_batch_tuned_smoke.json
```

Candidate intent:

- Keep `selected_pipeline_mode=batch` to preserve full-batch stage efficiency.
- Raise product-visible scheduler/admission settings to allow larger decode cohorts.
- Use the existing `--scheduler-prefill-first-until-active` CLI/config path, not hidden environment variables.
- Treat the previous overlapped artifacts as negative diagnostic evidence, not as the current candidate.

Local validation for this retarget:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release/run_layer_split_perf_goal.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release/layer_split_perf_goal_gate.py --self-test
```

## Previous GPU Diagnostics

Previous diagnostic with actual overlap:

```text
/Users/chejinxuan/rust_ws/ferrum-infer-rs-records/layer-split/layer-split-perf-sched-prefill8-gatefix-diag-20260609-164406
```

Result:

- c4 candidate: about `+1.1%` vs baseline.
- c8 candidate: about `-17.2%` vs baseline.
- Candidate did exercise overlap with `overlapped_calls=563`, `max_batch=8`, and `in_flight_stage_count_max=2`.

Earlier diagnostic:

```text
/Users/chejinxuan/rust_ws/ferrum-infer-rs-records/layer-split/layer-split-perf-maxbatched512-diag-20260609-153106
```

Finding: candidate did not actually exercise overlapped decode:

```text
candidate pipeline_decode.overlapped_calls = 0
candidate pipeline_decode.max_batch = 1
candidate pipeline_decode.in_flight_stage_count_max = 1
```

## Implemented Changes In Current Worktree

- Added `ferrum serve --scheduler-prefill-first-until-active N` so the scheduler cohort policy is visible in product CLI/config artifacts instead of hidden `FERRUM_SCHED_*` state.
- Added CUDA `write_f32_to_activation` so host `f32` hidden activations can be written directly into an existing CUDA activation buffer, avoiding a temporary CUDA allocation plus extra device copy.
- Updated layer-split smoke runner preflights for 2x4090, toolchain, CUDA driver visibility, and dirty diagnostic control.
- Updated layer-split smoke runner to fail the smoke PASS if candidate post-bench health does not prove the expected pipeline mode was actually exercised (`batch` or `overlapped`, with mode-specific metric checks).
- Added native CUDA overlap probe for fast scheduling-shape validation without Rust/Cargo rebuilds.
- Updated Qwen tool-call regression seeding and explicit auto tool-choice prompt so model flake does not mask product-path failures.

## Local Validation

Current checks passed:

```bash
cargo test -p ferrum-models pipeline_
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release/run_layer_split_perf_goal.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release/layer_split_perf_goal_gate.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/release/run_layer_split_perf_goal.py scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py scripts/release/layer_split_perf_goal_gate.py scripts/release/openai_tool_call_regression.py
cargo fmt --all -- --check
git diff --check
```

Earlier checks also passed:

```bash
cargo check -p ferrum-cli --all-targets
cargo test -p ferrum-cli serve_cli_runtime_entries_are_cli_sourced_and_classified
cargo test -p ferrum-cli serve_runtime_snapshot_prefers_cli
cargo test -p ferrum-cli vllm_compat_runtime_flags_follow_existing_precedence
cargo check -p ferrum-kernels -p ferrum-models
```

## Current Direction

Do not continue tuning the current threaded microbatch overlap path as a performance candidate.

Next optimization direction should preserve large-batch stage efficiency:

- Keep decode in full-batch mode for the current Qwen72B product path unless a new candidate demonstrates same-hardware improvement.
- Investigate a final-stage fused path that runs stage1 and logits without materializing hidden back to host and then re-uploading it.
- If pipeline overlap is revisited, use a minimal probe first and preserve large cohort sizes; do not split a single efficient batch into smaller microbatches without evidence that the smaller batch stage kernels remain efficient.
- Only return to a full Rust CUDA build after a native/Python/CUDA probe shows a positive direction.

## 2026-06-09 Batch-Tuned Autopreset Smoke

Artifact copied locally:

```text
/Users/chejinxuan/rust_ws/ferrum-infer-rs-records/layer-split/layer-split-perf-batch-tuned-autopreset-smoke-20260609-184151
```

Required smoke PASS line:

```text
LAYER_SPLIT_PERF SMOKE PASS: /workspace/layer-split-perf-batch-tuned-autopreset-smoke-20260609-184151
```

Candidate source gate PASS line:

```text
G0 SOURCE layer_split_perf_qwen72b_gptq_smoke PASS: /workspace/layer-split-perf-batch-tuned-autopreset-smoke-20260609-184151/candidate-batch-tuned
```

This is dirty diagnostic smoke evidence only. It is not final goal completion evidence and does not replace the full clean-worktree validator.

Same-pod smoke throughput, `n_repeats=1`, `--fail-on-error`, `--seed 9271`, `concurrency-sweep=4,8,16`:

| Concurrency | Baseline output tok/s | Candidate output tok/s | Candidate delta |
| ---: | ---: | ---: | ---: |
| 4 | 57.516 | 54.688 | -4.9% |
| 8 | 80.480 | 79.385 | -1.4% |
| 16 | 79.530 | 103.077 | +29.6% |

Correctness status:

- `ferrum run` multi-turn recall passed.
- `ferrum serve` single-turn, multi-turn, streaming usage, structured output, tool call regression, and concurrency quality checks passed.
- `bench-serve` completed all 24 requests per cell with zero request errors and `output_token_count_source=usage`.

Candidate product-path validation:

- `ferrum serve` command had no tuning flags for layer-split pipeline, max model length, KV capacity, max sequences, max batched tokens, or scheduler admission.
- `serve.effective_config.json` selected preset `qwen25_72b_gptq_int4_2x4090_layer_split`.
- Candidate runtime entries for `FERRUM_LAYER_SPLIT_PIPELINE_MODE`, `FERRUM_MAX_MODEL_LEN`, `FERRUM_KV_MAX_BLOCKS`, `FERRUM_KV_CAPACITY`, `FERRUM_PAGED_MAX_SEQS`, `FERRUM_MAX_BATCHED_TOKENS`, and `FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE` all had `source=default`.
- Effective candidate values were `pipeline=batch`, `max_sequences=16`, `max_batched_tokens=1536`, `kv_block_count=1024`, `kv_capacity=1024`, `scheduler=prefill_first_until_active:16`.

GPU evidence:

- Candidate bench max GPU utilization was `[81%, 53%]`.
- Per-cell max GPU utilization was c4 `[47%, 52%]`, c8 `[35%, 43%]`, c16 `[81%, 53%]`.
- During-bench memory snapshot was about `[23061 MiB, 23003 MiB]`.

Vast cleanup:

- Instance `40203673` was stopped after artifact copyback.
- Vast API status after cleanup: `actual_status=exited`, `cur_state=stopped`, `intended_status=stopped`.
