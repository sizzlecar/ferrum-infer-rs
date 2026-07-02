# W3 Qwen3.5/Qwen3.6 发布级目标

Date: 2026-06-18
Branch: `goal/w2-w3-release-grade`
Current checkpoint: `3b2b55cf fix(w3): parse qwen35 real attention config`

This document is the concrete W3 execution contract for the Qwen3.5/Qwen3.6
Gated DeltaNet family. It refines `W3_CHARTER.md` and inherits the hard release
rules in `RELEASE_GRADE_GOAL.md`. No W3 support claim is valid until the final
validator prints:

```text
MODEL_RELEASE_GRADE_W3 PASS: <out_dir>
```

## Scope

Primary release target:

- Model family: `Qwen/Qwen3.5-35B-A3B`
- Practical 1x4090 validation lane: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- Product entrypoints: both `ferrum run` and `ferrum serve`
- Backend target: CUDA on 1x RTX 4090 first; CPU reference is diagnostic only

Secondary family target:

- `Qwen/Qwen3.6-35B-A3B` must keep L0 template and config compatibility, but it
  does not replace the Qwen3.5 real-model correctness/performance lane.

Explicit non-goals for this W3 lane:

- Qwen3.5 vision tower.
- BF16 full 35B release evidence on 1x4090.
- New live vLLM benchmarking unless the historical baseline is missing or
  invalid for the claimed lane.
- CPU-only product support as a release-grade result.

## Current Context

The previous implementation work had useful building blocks, but it also made
invalid dense/full-attention assumptions. The important correction is that W3
must be driven by official config, vLLM/Transformers behavior, and real weight
metadata before backend code is written.

Official Qwen3.5 facts already verified from Hugging Face/Transformers:

- `architectures = ["Qwen3_5MoeForConditionalGeneration"]`
- `model_type = "qwen3_5_moe"`
- `text_config.model_type = "qwen3_5_moe_text"`
- hidden size: `2048`
- layers: `40`
- layer pattern: `30` linear-attention layers and `10` full-attention layers,
  in a repeated `linear, linear, linear, full` pattern.
- linear attention: `linear_num_key_heads=16`,
  `linear_num_value_heads=32`, `linear_key_head_dim=128`,
  `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`
- full attention: `num_attention_heads=16`, `num_key_value_heads=2`,
  `head_dim=256`
- full attention query width: `num_attention_heads * head_dim = 4096`,
  which is not equal to hidden size `2048`
- full attention `q_proj` width with output gate: `8192`
- `attn_output_gate = true`; attention context must be multiplied by
  `sigmoid(gate)` before `o_proj`
- RoPE: `rope_theta=10000000`, `partial_rotary_factor=0.25`,
  `mrope_interleaved=true`; rotary dim is `64`, not the full `256`
- MoE: `num_experts=256`, `num_experts_per_tok=8`,
  `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`
- GPTQ-Int4 config: `bits=4`, `group_size=128`, `desc_act=false`,
  `sym=true`, `quant_method=gptq`

These facts are now parsed in the typed config path at checkpoint `3b2b55cf`.

## Current Progress

Completed and committed:

- W3 final release-grade validator and manifest scaffolding exist.
- W3 L0 real template/tokenizer gates exist for
  `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B`.
- W3 L1 numeric artifact gate exists and has passed for the CPU reference
  component coverage.
- W3 L2 quantized artifact gate shape exists, but only self-test/synthetic
  validation has passed.
- Qwen35 CPU/FP32 toy reference path can run through explicit
  `--qwen35-reference` product smokes.
- Typed recurrent-state manager plumbing exists for the CPU reference path.
- Native CUDA minimal DeltaNet validation has produced an S0 microbench PASS.
- Latest checkpoint parses real Qwen3.5 attention gate and partial-RoPE config:
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    passed.
  - `cargo test -p ferrum-models qwen35_dense_config_flattens_text_config_without_llama_fallback -- --nocapture`
    passed.
  - `cargo test -p ferrum-models qwen35_moe_config_preserves_shared_expert_shape -- --nocapture`
    passed.
  - `cargo test -p ferrum-models qwen35_model_definition_builds_recurrent_state_spec -- --nocapture`
    passed.

Not completed:

- No real `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` `ferrum run` correctness artifact.
- No real `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` `ferrum serve` correctness artifact.
- No real L2 known-answer report for the full-size quantized model.
- No real L3/L4/L5 product gate for Qwen3.5.
- No W3 CUDA performance artifact and no 80% ratio report.
- No `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`.

## Correctness Acceptance

Correctness is a hard prerequisite for performance. Any failure below stops the
lane; existing performance numbers become diagnostic only.

L0 template/tokenizer:

- Models: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.6-35B-A3B`
- Required PASS line per model:
  `W3 L0 TEMPLATE PASS: <out_dir>`
- Must cover single turn, system prompt, multi-turn, tools, and think-history
  cases.
- EOS/BOS/special token ids must come from model metadata, not family guesses.

L1 numeric/reference:

- Required PASS line:
  `W3 L1 NUMERIC PASS: <out_dir>`
- Must cover linear attention, full attention, DeltaNet, MoE/dense layer path,
  and LM head categories.
- For full attention, tests must include a shape where
  `q_total != hidden_size`, `attn_output_gate=true`, and `rope_dim<head_dim`.

L2 real quantized semantics:

- Model: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- Required PASS line:
  `W3 L2 QUANTIZED PASS: <out_dir>`
- Must run the actual model, not a toy fixture.
- Must include at least `10` known-answer cases.
- Must cover both `ferrum run` and `ferrum serve` command lines.
- `hidden_env` must be empty unless the option is also exposed as typed config,
  CLI, or documented preset.
- No `<unk>`, `[PAD]`, mojibake, malformed UTF-8, missing `[DONE]`, duplicate
  `[DONE]`, panic, or silent fallback is allowed.

L3 behavior:

- Multi-turn KV/recurrent state reuse must produce coherent second-turn output.
- Streaming and non-streaming answers must be semantically consistent.
- Streaming must emit exactly one `data: [DONE]`.
- `stream_options.include_usage=true` must produce usage.
- Stop sequences and natural EOS must work.

L4 agent behavior:

- Required tool-call smoke: `10/10` pass.
- Strict JSON schema smoke: `20/20` pass at `temperature=0`.
- Failures are release blockers, not performance gaps.

L5 concurrency:

- `bench-serve --fail-on-error --seed 9271` is mandatory.
- Release-grade evidence must use `--require-ci --n-repeats 3`.
- Required cells for Qwen3.5 GPTQ: `c=1/4/16/32`.
- Every cell must have full completion and zero request errors.
- If admission or scheduler caps effective concurrency below requested `c`,
  the artifact must record the effective active concurrency and the release
  claim must use the effective value.

## Performance Acceptance

The performance gate starts only after L0-L5 correctness passes.

Baseline:

- Use checked-in historical vLLM data when it matches model family,
  quantization, hardware, prompt shape, output length, and concurrency cell.
- If no valid historical vLLM baseline exists for a cell, that cell has no
  release-grade performance verdict until a valid baseline artifact is added.
- The baseline artifact must record engine version, command, git/build info
  when available, hardware, and runtime config.

Ferrum command:

```bash
ferrum bench-serve ... \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --n-repeats 3
```

Pass criteria:

- Correctness gates must already be PASS.
- Required cells: `c=1/4/16/32`.
- Ferrum lower confidence bound must be at least `80%` of the valid historical
  vLLM baseline for each cell.
- If only means are available, run at least `N=5`; otherwise use `N>=3` with
  saved CI/variance.
- p95 ITL must be no worse than `1.25x` the baseline unless the lane is
  explicitly marked offline-throughput-only.
- No performance claim is allowed without artifact directory, git SHA, dirty
  status, binary SHA256 when available, runtime/config snapshot, hardware
  snapshot, benchmark commands, and final ratio report.

## Implementation Plan

Work one correctness blocker at a time.

1. Real config and weight metadata boundary
   - Keep official `Qwen3.5` config parsing as source of truth.
   - Inspect real GPTQ safetensors index before changing loader assumptions.
   - Reject unsupported or mismatched tensor shapes loudly.

2. Full-attention abstraction repair
   - Remove `q_total == hidden_size` assumptions.
   - Represent `q_proj_total = q_total * 2` when `attn_output_gate=true`.
   - Split `q_proj` into query and attention gate.
   - Apply Q/K RMSNorm over `head_dim`.
   - Apply partial RoPE only over `rope_dim`.
   - Apply `context *= sigmoid(attn_gate)` before `o_proj`.
   - Add CPU reference and backend parity tests for the official-like shape:
     `hidden=2048`, `q_total=4096`, `q_proj_total=8192`,
     `kv_total=512`, `rope_dim=64`.

3. Linear attention and recurrent state
   - Keep native CUDA minimal validation as the first GPU proof point.
   - Reuse typed recurrent-state cache semantics; no hidden CPU fallback.
   - Verify decode uses recurrent state rather than recomputing only for toy
     paths.

4. Sparse MoE backend path
   - Reuse existing Qwen3 MoE/vLLM Marlin assets where shape-compatible.
   - Add shared expert merge semantics explicitly.
   - Validate router top-k, expert layout, and shared expert output before
     product wiring.

5. Product path
   - Replace `--qwen35-reference` toy-only flow with real backend support.
   - `ferrum run` and `ferrum serve` must work with typed user-visible config.
   - The default unsupported error is removed only after real correctness
     gates pass.

6. GPU validation
   - Use one paid GPU lane at a time.
   - Before starting, write lane, expected runtime/cost, stop condition,
     correctness command, and performance command.
   - Correctness first; performance only after L0-L5.

## GPU Lane Contract

First real GPU lane:

- Lane: W3 Qwen3.5 GPTQ-Int4 correctness smoke
- Hardware: 1x RTX 4090, CUDA build with
  `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Expected runtime/cost: state before starting on the selected Vast instance.
- Stop condition: first correctness failure with artifact/log copied back; no
  repeated full sweeps until the failure is understood.
- Correctness command: W3 L2 quantized gate for
  `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- Performance command: not run until L0-L5 correctness passes.

## Open Risks

- Current backend full-attention code still has old dense assumptions from the
  pre-official-config phase. This must be fixed before any real Qwen3.5 run.
- Existing CPU reference/product smokes are toy fixtures; they prove plumbing,
  not release-grade model behavior.
- L2 gate exists but lacks a real full-size quantized report.
- Performance cannot be interpreted until correctness passes.
- Any GPU instance left running without an active lane is a process failure.

## Final Completion Definition

W3 is complete only when all of the following are true:

- `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` is printed by the final validator.
- `<out_dir>` contains L0-L5 correctness manifests for the required models and
  the real Qwen3.5 GPTQ-Int4 product path.
- `<out_dir>` contains Ferrum and baseline performance artifacts for
  `c=1/4/16/32`.
- Every required performance cell is at least `80%` of the accepted baseline.
- Artifacts include commands, git SHA, dirty status, binary SHA256 when
  available, runtime config, hardware snapshot, logs, and ratio report.
- Both `ferrum run` and `ferrum serve` are covered.
