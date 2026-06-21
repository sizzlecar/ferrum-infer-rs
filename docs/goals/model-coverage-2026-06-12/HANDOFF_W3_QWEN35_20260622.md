# W3 Qwen3.5 Handoff - 2026-06-22

## Status

- Branch: `goal/w2-w3-release-grade`.
- PR: <https://github.com/sizzlecar/ferrum-infer-rs/pull/237>.
- W3 is not complete. There is no `MODEL_RELEASE_GRADE_W3 PASS`.
- Correctness evidence exists for real Qwen3.5 GPTQ product paths, including
  `ferrum run`, `ferrum serve`, streaming usage, tool calls, structured output,
  and L5 zero-error concurrency.
- Performance remains below the W3 80% vLLM target. The accepted vLLM c32 mean
  baseline is `1708.52785 output tok/s`, so the 80% target is
  `1366.82228 output tok/s`. Recent Ferrum c32 diagnostics have been around
  `629-695 output tok/s`.

## Latest Source Progress

This update moves Qwen3.5 GDN prefill toward the vLLM architecture instead of
adding model-name defaults or hidden environment switches.

- Added backend capability:
  `supports_qwen35_packed_gdn_prefill_prepare`.
- Added packed varlen prefill prepare API:
  `linear_attention_prepare_varlen_packed_qkvz_ba_f32`.
- Added CPU reference implementation.
- Added CUDA launcher and `.cu` kernel for packed `[q,k,v,z]` plus `[b,a]`
  varlen GDN prefill prepare.
- Routed Qwen3.5 product prefill through fused `qkvz_proj` and `ba_proj` when
  the backend supports packed prefill prepare.
- Kept separate `qkv/z/b/a` projection fallback for unsupported backends.
- Added prefill profile fields for `qkvz_proj` and `ba_proj`.
- Added CPU packed-vs-separate contract test and a CUDA packed-vs-CPU parity
  test for the next GPU lane.

The key vLLM reference is:

```text
/Users/chejinxuan/py_ws/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
```

The relevant vLLM behavior is that Qwen3.5 prefill uses `in_proj_qkvz` and
`in_proj_ba`, then splits `[q,k,v,z]` and `[b,a]`. Ferrum prefill previously
still launched four projections.

## Local Validation

Passed locally:

```bash
cargo fmt --all
cargo test -p ferrum-kernels --test linear_attention_cpu \
  linear_attention_prepare_varlen_packed_cpu_matches_separate_prepare -- --nocapture
cargo check -p ferrum-kernels -p ferrum-models
cargo test -p ferrum-models \
  linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture
```

Not yet validated locally:

- CUDA feature build, because this Mac has no `nvcc`.
- CUDA packed varlen parity test:

```bash
cargo test -p ferrum-kernels --features cuda --test linear_attention_cuda_eq \
  linear_attention_prepare_varlen_packed_cuda_matches_cpu_reference -- --nocapture
```

## GPU Next Lane

Paid GPU policy for the next run:

- Lane: W3 Qwen3.5 packed-prefill CUDA smoke and c32 diagnostic on exactly
  1x RTX 4090.
- Expected runtime/cost: about 30-60 minutes; on the previous Vast price
  `$0.662962962962963/hour`, expected cost is about `$0.33-$0.66`.
- Stop condition: stop after CUDA build/test failure, product smoke failure,
  one c32 diagnostic artifact, or a clear external blocker.
- Correctness gate:

```bash
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
cargo test -p ferrum-kernels --features cuda --test linear_attention_cuda_eq \
  linear_attention_prepare_varlen_packed_cuda_matches_cpu_reference -- --nocapture
```

Then run real Qwen3.5 product smoke for both `ferrum run` and `ferrum serve`,
including streaming with `stream_options.include_usage=true`.

- Performance command shape:

```bash
ferrum bench-serve \
  --dataset sharegpt \
  --num-prompts 64 \
  --warmup-requests 8 \
  --random-output-len 128 \
  --concurrency 32 \
  --n-repeats 1 \
  --seed 9271 \
  --fail-on-error
```

If c32 materially improves and correctness remains clean, run the release-shape
L5 cells with `c=1/4/16/32`, `--require-ci`, and `--n-repeats 3`.

## Current GPU Blocker

- Existing cached Vast instance `41422823` was last observed stopped/exited.
- A replacement 1x RTX 4090 attempt was externally blocked by Vast
  `insufficient_credit`.
- Do not keep cycling rental attempts until credit is restored.

## What To Watch

- Profile should show prefill projection time moving from separate
  `qkv/z/b/a` entries to `qkvz/ba`.
- CUDA build must confirm the new `.cu` symbols are present.
- If packed prefill improves projection cost but c32 remains far below target,
  continue with profiler-backed bottleneck localization; do not revert blindly
  and do not start env-flag sweeps.
- This update is source progress only. Do not make a performance claim without
  same-hardware artifacts and command logs.
