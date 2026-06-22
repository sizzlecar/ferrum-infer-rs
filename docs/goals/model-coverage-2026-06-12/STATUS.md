# STATUS — model-coverage-2026-06-12

进度日志,倒序。

## 2026-06-22 ZZZ73 — `run`/`serve` auto-config now uses real model weight bytes

- Scope:
  - fixed `ferrum run` and `ferrum serve` startup auto-config so
    `model_capabilities.estimated_weight_bytes` is derived from the resolved
    model source when local `.safetensors` / `.bin` shards are present;
  - the file-size path follows Hugging Face cache symlinks via
    `std::fs::metadata`, so cached snapshots account for the real shard bytes;
  - when file sizes are unavailable, the MoE fallback estimate now counts all
    resident experts plus shared expert/router/attention/embedding weights
    instead of reusing dense active-parameter approximation.
- Why:
  - current W3 Qwen3.5 artifacts record
    `model_capabilities.estimated_weight_bytes=907100160`, which is not a
    credible 35B A3B GPTQ footprint;
  - `default_kv_blocks()` consumes that value when deriving KV/admission
    budgets, so a too-low estimate can over-budget KV and hide the real memory
    constraint until runtime pressure or allocation failure;
  - this is a shared typed-runtime fix, not a model-specific default override.
- Validation passed locally:
  - `cargo fmt --all -- --check`;
  - `cargo test -p ferrum-cli model_capabilities -- --nocapture`;
  - `cargo test -p ferrum-cli model_weight_bytes -- --nocapture`;
  - `cargo check -p ferrum-cli`.
- GPU state:
  - SSH to `ssh7.vast.ai:22822` still returned `Connection refused`;
  - sanitized Vast API status for instance `41422823` showed
    `cur_state=stopped`, `actual_status=exited`, `gpu_name=RTX 4090`,
    `num_gpus=1`, `gpu_ram=49140`;
  - no CUDA build, correctness smoke, benchmark, or performance claim ran.
- Status:
  - source/runtime-control progress only;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ72 — 1x4090 W3 diagnostic lane start blocked by Vast resource state

- Scope:
  - attempted to start the existing Vast 1x RTX 4090 instance `41422823`
    after recording the paid-GPU lane contract for W3 Qwen3.5 GPTQ-Int4 CUDA
    build, product correctness smoke, and c32 diagnostic benchmark;
  - saved the non-secret startup artifact under
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_token_rows_cuda_diag_20260621T231558Z_8b33416d/`;
  - removed the raw Vast instance response because it contained a provider
    `jupyter_token`; only sanitized status remains in the committed artifact.
- Result:
  - initial SSH to `ssh7.vast.ai:22822` failed with `Connection refused`;
  - Vast start response was `success=false`, `error=resources_unavailable`,
    `msg="Required resources are currently unavailable, state change queued."`;
  - follow-up instance list still reported `cur_state=stopped` and
    `actual_status=exited` for instance `41422823`.
- Status:
  - no CUDA build, correctness smoke, or c32 benchmark ran;
  - no performance claim;
  - next GPU step still requires a reachable running 1x RTX 4090 instance.

## 2026-06-22 ZZZ71 — W3 S1 archived absolute artifact paths are relocatable

- Scope:
  - fixed `scripts/release/model_release_grade_goal_gate.py` so artifact paths
    that were recorded as absolute `/tmp` or `/workspace` paths can still be
    resolved after the evidence bundle is archived next to its manifest;
  - the resolver now keeps the original absolute candidate but also checks
    artifact-local fallback locations, including `<artifact_dir>/<basename>`,
    `<artifact_dir>/../<basename>`, and the same forms preserving the immediate
    parent directory name;
  - added a W3 final-validator self-test covering an archived S1 artifact whose
    `reference_dump` is recorded as
    `/tmp/original-s1/reference_bundle/reference_dump` while the archived copy
    lives under `reference_bundle/reference_dump`.
- Why:
  - the real archived S1 evidence
    `w3_deltanet_s1_rust_compare_20260617T130232Z_1b480a31/compare/w3_deltanet_s1_layer_compare_manifest.json`
    contains generation-time absolute dump paths under `/private/tmp`;
  - before this change, a clean local final W3 probe could falsely fail S1
    artifact lookup even though the dump directories are present in the checked
    artifact bundle.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - a direct temporary W3 manifest probe using the current real L0-L5,
    S0/S1/S2, historical vLLM baseline, and W3 Ferrum performance artifacts.
- Final-manifest probe result:
  - with the correct S0 design artifact and archived S1 artifact, the final
    validator reached performance evaluation and failed with exactly eight
    performance problems:
    c1/c4/c16/c32 throughput ratio below `0.800`, and c1/c4/c16/c32 p95 ITL
    above the `1.25x` baseline limit;
  - no L0-L5 or S0/S1/S2 correctness artifact problem remained in that probe.
- Status:
  - final-gate reproducibility progress only; no new CUDA build or performance
    claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ70 — Shared MoE expert-count contract fixed for stacked fast paths

- Scope:
  - fixed `crates/ferrum-models/src/moe/dispatch.rs` so
    `ExpertStack::num_experts()` no longer assumes that per-expert
    `gate_up/down` `Vec`s are populated;
  - the method now falls back to the GGUF stacked expert stores and then to
    Marlin stacked expert stores, with debug assertions that paired stacks
    agree on expert count;
  - this closes an abstraction mismatch where the comment described
    stacked-only support but the implementation returned `0` for that shape.
- Why:
  - Qwen3.5 CUDA GPTQ currently builds per-expert Marlin views, so this is not
    the measured W3 CUDA throughput blocker;
  - it is still real shared MoE code debt: GGUF/Metal stacked-only fast paths
    and future Marlin stacked-only paths should not need model-specific
    workarounds just to pass the same MoE dispatch count contract.
- Validation passed locally:
  - `cargo test -p ferrum-models expert_stack_num_experts_uses_stacked_fast_path_count -- --nocapture`;
  - `cargo fmt --all -- --check`.
- Status:
  - source/architecture hygiene progress only; no new GPU correctness or
    performance claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ69 — W3 final manifest probe now reaches the real performance blocker

- Scope:
  - fixed `scripts/release/model_release_grade_goal_gate.py` so nested
    artifacts referenced from a loaded artifact are also resolved relative to
    that artifact's directory;
  - this matters for real W3 S2 product evidence, where
    `w3_s2_whole_model_product_path.json` records `run_stdout.jsonl`,
    `serve.log`, and behavior response files relative to its own artifact
    directory;
  - added a final-validator self-test that moves S2 and its nested evidence
    into a subdirectory, proving the validator accepts artifact-local relative
    paths instead of accidentally depending on files in the final output root.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`;
  - direct probe of the real S2 artifact:
    `W3 S2 REAL ARTIFACT PATH-RESOLUTION PROBE PASS`;
  - direct probe of current real S0/S1/S2 evidence:
    `W3 S0/S1/S2 FINAL-VALIDATOR PROBE PASS`;
  - `git diff --check`.
- Final-manifest probe:
  - built a temporary W3 manifest from the current available evidence:
    current L0-L5 artifacts, S0 design, S0 CUDA microbench, S1 single-layer
    compare, S2 product path, the historical vLLM ShareGPT baseline, and the
    W3 L4/L5 Ferrum performance matrix;
  - the final validator reached performance evaluation and failed with exactly
    eight performance problems:
    c1/c4/c16/c32 throughput ratio below `0.800`, and c1/c4/c16/c32 p95 ITL
    above the `1.25x` baseline limit;
  - diagnostic ratios from that probe were c1 `0.396989`, c4 `0.224899`,
    c16 `0.115835`, and c32 `0.081663`;
  - no L0-L5 or S0/S1/S2 correctness artifact problem remained in that probe.
- Status:
  - this is final-gate/tooling progress and a sharper blocker diagnosis, not a
    performance claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ68 — W3 L1 numeric artifact regenerated with official full-attention shape coverage

- Scope:
  - regenerated the W3 L1 numeric/reference artifact with the current
    `scripts/release/w3_l1_numeric_gate.py`;
  - new artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_l1_numeric_qwen35_family_20260622_da48e058/w3_l1_numeric.json`;
  - official PASS line:
    `W3 L1 NUMERIC PASS: docs/goals/model-coverage-2026-06-12/artifacts/w3_l1_numeric_qwen35_family_20260622_da48e058`;
  - the artifact records `coverage.full_attention_official_shape=true` and
    includes the required official-shape tests:
    `rope_uses_partial_interleaved_rotation`,
    `full_attention_core_applies_qwen35_output_gate`, and
    `dense_full_attention_layer_accepts_qwen35_gate_shape_with_hidden_not_q_total`.
- Why:
  - the older
    `w3_l1_numeric_qwen35_family_20260618/w3_l1_numeric.json` artifact was
    generated before this final-validator requirement and fails current W3 L1
    validation;
  - hand-editing historical artifacts would be invalid, so the gate was rerun
    to produce fresh evidence.
- Validation passed locally:
  - `python3 scripts/release/w3_l1_numeric_gate.py --out docs/goals/model-coverage-2026-06-12/artifacts/w3_l1_numeric_qwen35_family_20260622_da48e058`;
  - direct final-validator probe of the new L1 artifact:
    `W3 L1 FINAL-VALIDATOR PROBE PASS:
    docs/goals/model-coverage-2026-06-12/artifacts/w3_l1_numeric_qwen35_family_20260622_da48e058/w3_l1_numeric.json`;
  - direct W3 L0-L5 final-validator probe using the new L1 artifact plus the
    existing real L0/L2/L3/L4/L5 artifacts:
    `W3 L0-L5 FINAL-VALIDATOR PROBE PASS`.
- Status:
  - source/reference correctness evidence progress only; no new CUDA
    correctness or performance claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ67 — W3 L0-L5 final validator requires official PASS lines

- Scope:
  - hardened `scripts/release/model_release_grade_goal_gate.py` so every W3
    L0-L5 artifact loaded by the final validator must carry the official
    goal-level PASS line prefix:
    `W3 L0 TEMPLATE PASS:`, `W3 L1 NUMERIC PASS:`,
    `W3 L2 QUANTIZED PASS:`, `W3 L3 BEHAVIOR PASS:`,
    `W3 L4 AGENT PASS:`, and `W3 L5 CONCURRENCY PASS:`;
  - updated `scripts/release/model_release_grade_manifest.py --self-test`
    fixtures to emit those pass lines;
  - added a negative final-validator self-test that corrupts the L0 pass line
    and verifies W3 final validation rejects it.
- Why:
  - W3 completion must be based on gate-produced artifacts, not hand-written
    JSON that only happens to match a few summary fields;
  - before this change, L0-L5 artifacts could be accepted by the final
    validator without proving their own gate printed the required PASS line.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`;
  - direct W3 L0-L5 common/pass-line probe across the existing real artifact
    set;
  - `git diff --check`.
- Additional finding:
  - a stricter full-artifact probe of the existing
    `w3_l1_numeric_qwen35_family_20260618` artifact still fails current final
    validation because it lacks `coverage.full_attention_official_shape=true`;
  - that historical artifact was not edited. W3 still needs a real regenerated
    L1 numeric artifact, or a valid newer L1 artifact, before final W3 can pass.
- Status:
  - source/tooling progress only; no new CUDA correctness or performance claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ66 — W3 L3 final validator checks stream/multi-turn case evidence

- Scope:
  - hardened `scripts/release/model_release_grade_goal_gate.py` so W3 L3
    final validation checks the per-case behavior evidence, not only aggregate
    booleans;
  - L3 artifacts must now include behavior cases for multi-turn,
    stream/non-stream matching, natural EOS, custom stop, and reasoning
    extraction;
  - each case must have `passed=true`, a non-empty id, a non-empty artifact
    reference, and a JSON detail object;
  - the stream/non-stream case must explicitly record exactly one
    `stream_done_count` and at least one `stream_usage_chunks`;
  - updated `model_release_grade_manifest.py --self-test` fixtures to include
    L3 case details.
- Why:
  - W3 correctness requires stream behavior, usage, stop/EOS, and multi-turn
    behavior to be proven, not only summarized by booleans;
  - before this change, a hand-written L3 artifact could satisfy the final
    validator with aggregate fields alone.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py scripts/release/w3_qwen35_real_product_report.py`;
  - `python3 scripts/release/w3_qwen35_real_product_report.py --self-test`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`;
  - direct final-validator probe of existing real W3 L3 artifact
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e/real_product_report/w3_l3_behavior.json`;
  - `git diff --check`.
- Status:
  - source/tooling progress only; no new CUDA correctness or performance claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ65 — W3 L4 final validator checks case-level tool/schema evidence

- Scope:
  - hardened `scripts/release/model_release_grade_goal_gate.py` so W3 L4
    final validation no longer trusts only aggregate tool/schema counts;
  - L4 artifacts must now include `negative_contracts.tool_choice_400=true`
    and `negative_contracts.response_format_400=true`;
  - L4 artifacts must include `tool_call_cases` with per-case `passed=true`
    and `finish_reason=tool_calls`;
  - L4 artifacts must include `strict_schema_cases` with per-case
    `passed=true` and non-`length` finish reasons;
  - updated `w3_l4_agent_gate.py --self-test` and
    `model_release_grade_manifest.py --self-test` fixtures to emit those
    details.
- Why:
  - W3 explicitly requires required-tool and strict structured-output behavior;
  - before this change, a hand-written L4 artifact could satisfy the final
    validator with only aggregate counts.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py scripts/release/w3_l4_agent_gate.py`;
  - `python3 scripts/release/w3_l4_agent_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`;
  - direct final-validator probe of existing real W3 L4 artifact
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/l4_agent/w3_l4_agent.json`;
  - `git diff --check`.
- Status:
  - source/tooling progress only; no new CUDA correctness or performance claim;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ64 — W3 manifest self-test matches hardened L2/L5 contracts

- Scope:
  - fixed `scripts/release/model_release_grade_manifest.py --self-test`
    fixtures after the stricter L2/L5 final-validator changes;
  - the W3 manifest self-test L1 artifact now advertises
    `full_attention_official_shape=true`;
  - the W3 manifest self-test L2 artifact now includes real `ferrum run` and
    `ferrum serve` command lines;
  - the W3 manifest self-test L5 artifact now includes a release-shape
    `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
    command covering `c=1/4/16/32`.
- Why:
  - after hardening `model_release_grade_goal_gate.py`, the manifest builder's
    W3 self-test failed because its synthetic fixtures were behind the current
    evidence contract;
  - this keeps the final manifest builder covered by the same checks that will
    reject invalid GPU artifacts.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/model_release_grade_manifest.py scripts/release/model_release_grade_goal_gate.py scripts/release/w3_l2_quantized_gate.py`;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    printed synthetic
    `MODEL_RELEASE_GRADE_W2 PASS: ...`,
    `MODEL_RELEASE_GRADE_W3 PASS: ...`, and
    `MODEL RELEASE GRADE MANIFEST SELFTEST PASS`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - `python3 scripts/release/w3_l2_quantized_gate.py --self-test`;
  - `git diff --check`.
- Status:
  - source/tooling progress only; the self-test PASS lines are temporary
    synthetic artifacts and are not W3 real-model completion evidence;
  - W3 remains incomplete and there is still no real
    `MODEL_RELEASE_GRADE_W3 PASS` artifact directory.

## 2026-06-22 ZZZ63 — W3 L2 now requires real product command evidence

- Scope:
  - hardened `scripts/release/w3_l2_quantized_gate.py` so W3 L2 packaging
    requires real `command_line` evidence for both `ferrum run` and
    `ferrum serve`;
  - declaration-only `product_entrypoints` / `{"entrypoint": ...}` evidence
    no longer counts as product-path proof;
  - the L2 gate now normalizes commands and rejects hidden `FERRUM_*`
    overrides embedded in command lines;
  - hardened `scripts/release/model_release_grade_goal_gate.py` so the final
    W3 validator re-checks L2 command evidence. It accepts older artifacts
    that contain extra declaration-only entries only when real `command_line`
    entries cover both required product commands.
- Why:
  - W3 L2 is the next real Qwen3.5 correctness lane entrypoint, and the goal
    requires both `ferrum run` and `ferrum serve` product evidence;
  - before this change, an L2 report could satisfy entrypoint coverage with
    names only, without proving the actual typed product command.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/w3_l2_quantized_gate.py scripts/release/model_release_grade_goal_gate.py`;
  - `python3 scripts/release/w3_l2_quantized_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - direct final-validator probe of existing real W3 L2 artifact
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e/w3_l2_quantized.json`;
  - re-packaged the historical real
    `w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e/real_product_report/known_answer_report.json`
    into a temporary L2 artifact and got
    `W3 L2 QUANTIZED PASS`;
  - `python3 scripts/release/w3_qwen35_real_product_report.py --self-test`;
  - `git diff --check`.
- GPU status:
  - direct SSH probe to `ssh7.vast.ai:22822` still returned
    `Connection refused`, so no remote CUDA work was started.
- Status:
  - source/tooling progress only; no new CUDA correctness or performance claim;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ62 — W3 L5 now requires release bench command evidence

- Scope:
  - hardened `scripts/release/w3_l5_concurrency_gate.py` so L5 concurrency
    packaging rejects artifacts unless at least one saved `bench-serve`
    command covers `c=1/4/16/32`;
  - the L5 packaging gate now parses command strings with `shlex`, stores
    normalized `command_line` evidence, rejects hidden `FERRUM_*` env
    overrides, rejects `--request-rate`, and requires `--fail-on-error`,
    `--require-ci`, `--seed 9271`, and `--n-repeats 3`;
  - hardened `scripts/release/model_release_grade_goal_gate.py` so the final
    W3 validator re-checks L5 command evidence and refuses hand-built L5
    artifacts that lack compliant release commands.
- Why:
  - W3 goal text requires release-grade L5 evidence to use
    `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`;
  - before this change, L5 validated zero-error report contents but did not
    prove the report came from the required product benchmark command.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/w3_l5_concurrency_gate.py scripts/release/model_release_grade_goal_gate.py`;
  - `python3 scripts/release/w3_l5_concurrency_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - direct validator probe of existing W3 L5 artifact
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/l5_concurrency/w3_l5_concurrency.json`;
  - `git diff --check`.
- Status:
  - source/tooling progress only; no new CUDA correctness or performance claim;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ61 — W3 real-product report records pre-run git evidence

- Scope:
  - fixed `scripts/release/w3_qwen35_real_product_report.py` so release-grade
    evidence captures git status before creating the artifact directory;
  - the S2 whole-model product-path artifact now records the pre-run git
    summary instead of re-reading git after `known_answer_report.json`,
    `w3_l3_behavior.json`, and product logs have been written;
  - the report summary also records that same pre-run git snapshot.
- Why:
  - if a GPU lane writes artifacts inside the repository, reading git after
    artifact generation can make an otherwise clean run appear dirty because of
    its own evidence files;
  - W3 requires current-SHA correctness evidence with explicit dirty status, so
    the runner must preserve the state that existed before inference started.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/w3_qwen35_real_product_report.py`;
  - `python3 scripts/release/w3_qwen35_real_product_report.py --self-test`;
  - `python3 scripts/release/w3_l2_quantized_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`;
  - `git diff --check`.
- Status:
  - local tooling progress only; CUDA correctness/performance still requires a
    runnable 1x4090 and restored Vast credit;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ60 — Current-SHA W3 CUDA lane could not start on cached Vast instance

- Scope:
  - attempted one paid-GPU start of existing cached Vast instance `41422823`
    for the current clean SHA `7ba1f415c54f7eab050563b801a37fb38f0f28af`;
  - lane stated before start: W3 Qwen35 GPTQ-Int4 1x4090 current-SHA CUDA
    correctness smoke;
  - intended correctness path: `w3_qwen35_real_product_report.py` for real
    `ferrum run` + `ferrum serve`, followed by `w3_l2_quantized_gate.py`;
  - performance remained explicitly deferred until correctness; no benchmark or
    performance claim was made.
- Vast/API evidence:
  - sanitized artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_cuda_current_sha_7ba1f415_start_20260621T214634Z/summary.json`;
  - before start, API reported instance `41422823` as `cur_state=stopped`,
    `actual_status=exited`, `intended_status=stopped`, `gpu_name=RTX 4090`,
    `num_gpus=1`, SSH `ssh7.vast.ai:22822`;
  - `PUT state=running` returned an empty response object, and a follow-up API
    query still reported `cur_state=stopped`, `actual_status=exited`,
    `intended_status=stopped`;
  - read-only credit/inventory artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_vast_credit_inventory_20260622T0554CST/summary.json`;
  - that read-only check reported `credit=0`, negative balance state, and only
    the stopped/exited `41422823` instance, so replacement rental attempts must
    stop until credit is restored;
  - no remote SSH, build, correctness gate, or CUDA benchmark command ran.
- Local validator health checks passed while the GPU lane was unavailable:
  - `python3 scripts/release/w3_qwen35_real_product_report.py --self-test`;
  - `python3 scripts/release/w3_l2_quantized_gate.py --self-test`;
  - `python3 scripts/release/w3_l4_agent_gate.py --self-test`;
  - `python3 scripts/release/w3_l5_concurrency_gate.py --self-test`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`.
- Status:
  - W3 current-SHA CUDA evidence is still missing because the cached instance
    did not enter `running`;
  - this is external instance availability, not a Ferrum correctness failure;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ59 — Real Qwen35 GPTQ index matches the loader boundary

- Scope:
  - added `scripts/release/w3_qwen35_weight_index_probe.py`, a dependency-free
    W3 metadata probe that reads only HF `config.json`,
    `model.safetensors.index.json`, and optional `quantize_config.json`;
  - the probe validates the same Qwen3.5 prefix/manifest assumptions as the
    Rust loader boundary, including dense `.weight` tensors, complete GPTQ
    `.qweight/.scales/.qzeros` aliases for linear roles, and sparse MoE expert
    layouts;
  - self-test covers a passing synthetic GPTQ manifest and an incomplete GPTQ
    triplet failure.
- Real metadata artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_weight_index_probe_20260622/w3_qwen35_weight_index_probe.json`;
  - PASS line:
    `W3 QWEN35 WEIGHT INDEX PROBE PASS: docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_weight_index_probe_20260622`.
- Real model facts from the artifact:
  - model/revision:
    `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` at
    `3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`;
  - selected checkpoint prefix: `model.language_model`;
  - index shape: `124611` tensor names across `14` safetensors shards,
    total indexed size `24403162208` bytes;
  - quantization config matches W3 target:
    `quant_method=gptq`, `bits=4`, `group_size=128`, `desc_act=false`,
    `sym=true`;
  - required manifest resolution has zero missing tensors:
    `552` dense `.weight`, `60` non-linear metadata tensors, and one
    top-level `lm_head.weight` alias;
  - sparse MoE per-expert GPTQ coverage is complete:
    `40` layers, `256` experts, `92160` checked
    `.qweight/.scales/.qzeros` tensors, and `g_idx` present for all layers.
- Validation passed locally:
  - `python3 -m py_compile scripts/release/w3_qwen35_weight_index_probe.py`;
  - `python3 scripts/release/w3_qwen35_weight_index_probe.py --self-test`;
  - `python3 scripts/release/w3_qwen35_weight_index_probe.py --out docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_weight_index_probe_20260622 --model-id Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --revision 3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`.
- Status:
  - this closes the W3 implementation-plan item that required inspecting real
    GPTQ safetensors index metadata before changing loader assumptions;
  - source/metadata-boundary progress only; CUDA correctness/performance still
    has not run;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ58 — Qwen35 shared expert gate uses linear loader

- Scope:
  - changed `Qwen35SparseMoeSharedExpertWeights::shared_expert_gate` from a raw
    backend buffer to a `Linear`, matching vLLM's `ReplicatedLinear` modeling
    for the one-output shared expert gate;
  - both Qwen3.5 MoE backend paths now validate `shared_expert_gate` as
    `hidden_size -> 1` and call `Linear::forward` instead of hand-running GEMM
    over a raw tensor;
  - `Qwen35ModelWeights::load` now loads `mlp.shared_expert_gate` through
    `Qwen35WeightPlanLoader::load_layer_linear`, so dense weights and GPTQ
    `.qweight/.scales/.qzeros` aliases use the same loader boundary as router
    and shared expert projections;
  - extended the GPTQ-required-linear manifest regression to include
    `moe_shared_expert_gate`.
- Validation passed locally:
  - `cargo test -p ferrum-models qwen35_weights -- --nocapture`;
  - `cargo test -p ferrum-models sparse_moe_shared_expert_composes_router_fused_experts_and_shared_gate -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - source/loader-boundary progress only; CUDA correctness/performance still
    has not run;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ57 — Qwen35 weight plan accepts GPTQ-only required linears

- Scope:
  - changed Qwen3.5/Qwen3.6 weight inventory resolution so required linear
    roles can be satisfied by either dense `.weight` tensors or a complete
    GPTQ tensor set: `.qweight`, `.scales`, and `.qzeros`;
  - resolved GPTQ linear specs now keep the present `.qweight` name, while
    `Qwen35WeightPlanLoader` still converts it back to the module name before
    calling `WeightLoader::load_linear`, matching the existing
    `NativeSafetensorsLoader` GPTQ path;
  - incomplete GPTQ aliases do not satisfy the manifest, so a lone `.qweight`
    or missing `.qzeros` remains a loud missing-weight failure;
  - this removes a real-model blocker where a Qwen3.5 GPTQ checkpoint with
    quantized required linears could be rejected by the plan layer before the
    GPTQ-capable loader was reached.
- Validation passed locally:
  - `cargo test -p ferrum-models qwen35_weights -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - source/loader-boundary progress only; CUDA correctness/performance still
    has not run;
  - `mlp.shared_expert_gate.weight` was still loaded as a raw tensor at this
    checkpoint; ZZZ58 moved it to the same linear-loader boundary;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ56 — Qwen35 full-attention official-like backend shape is locked

- Scope:
  - added a Qwen3.5 dense full-attention backend/reference parity regression
    for the official scaled shape family: `hidden_size != q_total`,
    `q_proj_total = 2 * q_total`, `attn_output_gate=true`,
    `num_heads > num_kv_heads`, `rope_dim < head_dim`, interleaved partial
    RoPE, and non-zero `position_offset`;
  - the regression runs the full dense full-attention layer backend path, not
    only the CPU reference helper: q projection/gate split, Q/K RMSNorm,
    partial RoPE, head-major attention, attention output gate, `o_proj`, and
    the following dense MLP are all compared against the CPU reference;
  - this closes the previous source-level gap where the official gated
    `hidden != q_total` shape had a CPU acceptance test but the backend parity
    coverage still used the non-gated old shape.
- Validation passed locally:
  - `cargo test -p ferrum-models dense_full_attention_backend_matches_reference_for_qwen35_gated_official_like_shape -- --nocapture`;
  - `cargo test -p ferrum-models dense_full_attention -- --nocapture`;
  - `cargo test -p ferrum-models full_attention_backend_core_matches_reference -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - source/backend-correctness progress only; CUDA correctness/performance still
    has not run;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ55 — LlmExecutor keeps mixed fresh chunk + decode unified

- Scope:
  - added an executor-level regression for `LlmExecutor::unified_decode` with
    one fresh non-final prefill chunk and one decode row in the same
    `UnifiedBatch`;
  - the fake model now records the exact items received by
    `unified_forward_with_logits_policy`, and the regression proves the
    executor forwards both rows in one model call, preserves row order and
    `pos_offset`, returns `None` only for the non-final fresh chunk, and does
    not fall back to split prefill/decode paths;
  - this closes the source-level bridge between the ZZZ54 engine product batch
    and the ZZZ53 Qwen35 mixed paged prefill model path.
- Validation passed locally:
  - `cargo test -p ferrum-models unified_decode_forwards_mixed_fresh_prefill_and_decode_to_unified_model -- --nocapture`;
  - `cargo test -p ferrum-models unified_decode_forwards_logits_policy_to_unified_model -- --nocapture`;
  - `cargo test -p ferrum-models unified_decode_forwards_prefill_logits_policy_to_unified_model -- --nocapture`;
  - `cargo test -p ferrum-models batch_decode_forwards_logits_policy_to_unified_model -- --nocapture`;
  - `cargo check -p ferrum-models`.
- Status:
  - source/product-path bridge progress only; CUDA correctness/performance still
    has not run;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ54 — Engine product path emits active decode + fresh chunk mixed batches

- Scope:
  - added a continuous-engine product-path regression that constructs a real
    `BatchPlan` containing one decode-ready request and one fresh prefill
    request with `chunked_prefill_size=1`;
  - the test calls the real engine `process_batch` path and captures the
    `UnifiedBatch` sent to the model executor, proving the product path emits
    one unified call containing both a fresh non-final first chunk
    (`pos_offset=0`) and an active decode row (`pos_offset>0`);
  - widened `EngineInner::process_batch` visibility only within the
    `continuous_engine` module so this product-path regression can exercise
    the real batch processing path without exposing a public API.
- Validation passed locally:
  - `cargo test -p ferrum-engine process_batch_unified_co_batches_active_decode_with_fresh_prefill_chunk -- --nocapture`;
  - `cargo test -p ferrum-engine process_batch_unified_honors_runtime_chunked_prefill -- --nocapture`;
  - `cargo test -p ferrum-engine process_batch_unified_forwards_prefill_logits_policy -- --nocapture`;
  - `cargo check -p ferrum-engine`.
- GPU status:
  - attempted to reuse existing Vast instance `41422823` (`ssh7.vast.ai:22822`,
    1x RTX 4090) for a W3 Qwen35 mixed-prefill CUDA smoke/c32 diagnostic;
  - SSH returned `connection refused`, and sanitized API evidence shows
    `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`;
  - sanitized local artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_mixed_prefill_cuda_95adb578_20260622/local_vast/status_summary.json`.
- Status:
  - source/product-path progress only; CUDA correctness/performance still has
    not run;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ53 — Qwen35 fresh first chunks can join paged mixed prefill

- Scope:
  - replaced the duplicated Qwen3.5 fresh-only and continuation-only paged
    prefill validation with one typed paged prefill batch entry that supports
    `FreshOnly`, `ContinuationOnly`, and `Mixed` modes;
  - changed Qwen3.5 linear-attention batch prefill initial state from one
    global `fresh_initial_linear_state` bool to a per-row fresh mask, so fresh
    rows keep zero initial conv/GDN state while continuation/decode rows gather
    their existing recurrent state in the same varlen batch;
  - changed `forward_stateful_unified_items` so paged KV frames with any
    prefill row can build one mixed prefill batch containing fresh rows,
    continuation/chunk rows, and eligible decode candidates. The legacy
    no-policy greedy-argmax merge contract remains unchanged;
  - renamed the decode merge helper to `paged_prefill` semantics so the code no
    longer claims this is continuation-only behavior.
- Validation passed locally:
  - `cargo test -p ferrum-models qwen35_unified_forward -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_fresh_prefill_initial_state_slabs_are_zero_not_gathered -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_decode_merge_policy_preserves_legacy_no_policy_contract -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - source/hot-path progress only; this is intended to remove the split fresh
    prefill + decode forward in chunked mixed frames, but CUDA correctness and
    performance still require a reachable 1x4090;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ52 — Runtime chunked prefill stays on unified path

- Scope:
  - removed the stale `FERRUM_CHUNKED_PREFILL`/typed
    `chunked_prefill_size` fallback that forced `process_batch` onto the
    legacy split path;
  - `process_batch_unified` now treats scheduler `tokens_to_process`,
    active-decode prefill chunk, and runtime chunked-prefill size as
    coexisting upper bounds and uses the smallest cap, so one knob cannot
    bypass another;
  - added a product-path regression through `ContinuousBatchEngine::infer`
    proving a 2-token prompt with runtime chunk size `1` emits two unified
    forwards: a non-final prefill chunk at position `0`, then a final prefill
    chunk at position `1`.
- Validation passed locally:
  - `cargo test -p ferrum-engine process_batch_unified_honors_runtime_chunked_prefill -- --nocapture`;
  - `cargo test -p ferrum-engine process_batch_unified_forwards_prefill_logits_policy -- --nocapture`;
  - `cargo check -p ferrum-engine`;
  - `cargo fmt --all`.
- Status:
  - source/product-path progress only; this should let explicit chunked
    prefill use the same unified/paged continuation architecture instead of
    silently reverting to split prefill+decode;
  - CUDA correctness/performance still requires a reachable 1x4090;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ51 — Product unified prefill policy is locked at engine boundary

- Scope:
  - added a `ContinuousBatchEngine` product-path regression that calls the
    public `infer` entrypoint with a native-unified test executor;
  - the test captures the generated `UnifiedBatch` and proves a final prefill
    item carries `LogitsReturnPolicy::GreedyArgmax` with the expected token
    mask, not `FullLogits`;
  - this closes the remaining source-evidence gap between the ZZZ50 model
    support and the actual product batch construction path.
- Validation passed locally:
  - `cargo test -p ferrum-engine process_batch_unified_forwards_prefill_logits_policy -- --nocapture`;
  - `cargo check -p ferrum-engine`;
  - `cargo fmt --all`.
- Status:
  - source/product-path regression only; CUDA correctness/performance still
    requires a reachable 1x4090;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ50 — Qwen35 fresh prefill can honor logits policies

- Scope:
  - added a policy-aware fresh prefill batch entry for Qwen3.5:
    `forward_stateful_prefill_batch_with_logits_return`;
  - `forward_stateful_unified_items` now derives logits return policy for final
    fresh prefill rows from the caller's `LogitsReturnPolicy`, matching the
    policy-aware continuation batch path added in ZZZ48;
  - this lets product unified prefill return model-side greedy argmax
    sentinels for first-token generation when the request policy allows it,
    instead of always forcing full vocab logits readback;
  - added an executor regression proving a final fresh prefill item carries
    `GreedyArgmax` policy into the unified model boundary.
- Validation passed locally:
  - `cargo check -p ferrum-models`;
  - `cargo test -p ferrum-models unified_decode_forwards_prefill_logits_policy_to_unified_model -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_try_argmax_logits_rows_returns_policy_sentinel -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - source progress only; CPU tests prove policy forwarding and the shared
    logits-return helper, but CUDA correctness/performance still needs 1x4090;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ49 — batch_decode unified path forwards logits policy

- Scope:
  - fixed `LlmExecutor::batch_decode` so the fast unified path calls
    `unified_forward_with_logits_policy` instead of the no-policy
    `unified_forward`;
  - fallback still uses `decode_batch_with_logits_policy`, so unsupported
    models keep the previous policy-aware behavior;
  - added an executor regression proving a `GreedyArgmax` decode batch reaches
    the unified model with `requires_full_logits=false` rather than losing the
    policy at the unified boundary.
- Validation passed locally:
  - `cargo check -p ferrum-models`;
  - `cargo test -p ferrum-models batch_decode_forwards_logits_policy_to_unified_model -- --nocapture`;
  - `cargo test -p ferrum-models unified_decode_forwards_logits_policy_to_unified_model -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - this closes a product-path policy hole for decode-only batches and lets
    Qwen3.5's policy-aware continuation/decode logits path from ZZZ48 be used
    from both `batch_decode` and `unified_decode`;
  - source progress only; no CUDA performance claim without 1x4090 evidence;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ48 — Qwen35 paged continuation batch can honor logits policies

- Scope:
  - factored Qwen3.5 decode-batch argmax/readback tail into a shared
    `try_argmax_logits_rows` helper covering raw greedy argmax, token masks,
    and sparse repetition penalties;
  - wired paged continuation varlen prefill batch to accept
    `Qwen35DecodeLogitsReturn`, so merged final rows can return model-side
    greedy argmax sentinels instead of always reading back full vocab logits;
  - changed the mixed decode merge gate so complete explicit
    `LogitsReturnPolicy` batches, including `GreedyArgmax`, can join the
    continuation batch; legacy no-policy `FERRUM_GREEDY_ARGMAX=1` still stays
    on the decode batch path to preserve old no-policy sentinel behavior;
  - added prefill profile `argmax_us`, making continuation-batch argmax vs
    full-logits readback visible in Qwen3.5 profiles.
- Validation passed locally:
  - `cargo check -p ferrum-models`;
  - `cargo test -p ferrum-models qwen35_decode_merge_policy_preserves_legacy_no_policy_contract -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_try_argmax_logits_rows_returns_policy_sentinel -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_multitoken_continuation_matches_stepwise -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - this is source/hot-path progress, not a CUDA performance claim;
  - the intended W3 c32 impact is on policy-driven greedy serving frames where
    continuation/chunked rows and decode rows can now share one paged varlen
    continuation pass while still returning argmax sentinels;
  - CPU tests prove the shared logits-return contract and non-paged fallback;
    CUDA correctness/performance evidence still requires a reachable 1x4090;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ47 — Qwen35 full-logits mixed decode rows can join paged continuation batch

- Scope:
  - changed Qwen3.5 unified item classification to collect decode candidates
    first, then decide whether they can safely join the paged continuation
    varlen prefill batch;
  - when `use_paged_kv` is active, at least one continuation/chunked row is
    present, and all decode rows require full logits, decode rows can now share
    the same `forward_stateful_prefill_continuation_batch` call instead of
    forcing a second decode batch forward;
  - preserved the existing greedy-argmax contract: no-policy decode rows are
    not merged when `FERRUM_GREEDY_ARGMAX=1`, and policy rows are merged only
    when every policy is `FullLogits`;
  - pure decode batches still use the decode batch path, so this targets mixed
    continuation+decode frames rather than replacing the optimized decode path.
- Validation passed locally:
  - `cargo test -p ferrum-models qwen35_decode_merge_policy_preserves_legacy_no_policy_contract -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_multitoken_continuation_matches_stepwise -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_non_final_continuation_skips_logits_tail -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - it should help full-logits mixed frames such as structured/tool paths that
    cannot use model-side greedy argmax, but the primary greedy c32 path still
    needs a policy-aware varlen continuation logits/argmax follow-up;
  - direct SSH to Vast instance `41422823` still returns connection refused,
    and the API reports `cur_state=stopped`, `actual_status=exited`;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ46 — Qwen35 paged continuation rows can use varlen batch prefill

- Scope:
  - added a paged-KV continuation batch path for Qwen3.5 unified prefill rows;
  - when `use_paged_kv` is active, continuation/chunked prefill rows are now
    grouped and sent through the existing varlen batch prefill layer path
    instead of being handled row-by-row;
  - non-paged backends keep the serial stateful fallback, so local CPU tests
    remain valid without pretending CPU proves CUDA paged behavior;
  - generalized `forward_stateful_prefill_batch_taken` with an explicit
    `fresh_initial_linear_state` flag: fresh batch prefill still uses zero
    initial recurrent state, while continuation batch prefill synchronizes
    indexed linear-state slots back to sequence-local state before gathering;
  - fixed batch prefill KV allocation to use
    `state.tokens.len() + q_lens[row]` as the target length, which is required
    for continuation chunks and is unchanged for fresh prefill.
- Validation passed locally:
  - `cargo fmt --all`;
  - `cargo test -p ferrum-models qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_multitoken_continuation_matches_stepwise -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_non_final_continuation_skips_logits_tail -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - local CPU cannot execute the paged varlen continuation batch kernel, so the
    new CUDA path still needs 1x4090 correctness and c32 trace evidence;
  - Vast instance `41422823` still reports `cur_state=stopped`,
    `actual_status=exited`, and SSH still refuses connection;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ45 — Qwen35 continuation chunks advance as one chunk instead of per-token loop

- Scope:
  - removed the model-level Qwen3.5 stateful continuation special case that
    split an already-started multi-token chunk into one `forward_stateful_chunk`
    call per token;
  - final and non-final continuation chunks now use the same multi-token
    stateful layer path, so chunked prefill can amortize layer traversal,
    recurrent/full-attention setup, linear-state slot sync, and final logits
    tail when logits are requested;
  - kept the no-logits path from ZZZ44, so non-final continuation chunks still
    skip final norm/lm_head/readback after advancing the whole chunk;
  - changed the tiny Qwen35 forward test loader from all-zero weights to small
    deterministic non-zero weights, so chunk parity tests exercise real math
    instead of only zero tensors;
  - added a model-level parity regression comparing stepwise continuation
    `[4] + [5]` with a single final continuation chunk `[4, 5]`, including
    both returned logits and the next decode logits.
- Validation passed locally:
  - `cargo fmt --all`;
  - `cargo test -p ferrum-models qwen35_unified_forward_multitoken_continuation_matches_stepwise -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_non_final_continuation_skips_logits_tail -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - CUDA correctness/performance evidence is still blocked on the 1x4090 lane
    becoming reachable;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ44 — Qwen35 non-final continuation chunks skip logits tail

- Scope:
  - split Qwen3.5 stateful chunk execution into logits-returning and
    no-logits paths;
  - non-final continuation/chunked prefill rows in `unified_forward` now
    advance recurrent/full-attention state and sync the linear state slot, then
    return `None` without running final norm, final-token gather, lm_head, or
    logits host readback;
  - kept the existing continuation semantics that split already-started
    multi-token chunks into token-by-token state updates, avoiding an unproven
    behavior change while removing the known logits-tail waste;
  - added a CPU regression with a deliberately broken final norm and panic
    lm_head after the seed prefill, proving a non-final continuation chunk does
    not touch the logits tail while still advancing sequence state.
- Validation passed locally:
  - `cargo fmt --all`;
  - `cargo test -p ferrum-models qwen35_unified_forward_non_final_continuation_skips_logits_tail -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture`;
  - `cargo check -p ferrum-models`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`.
- GPU status:
  - attempted to use existing Vast instance `41422823`
    (`ssh7.vast.ai:22822`, 1x RTX 4090);
  - direct SSH returned `Connection refused`;
  - Vast API showed `cur_state=stopped`, `actual_status=exited`;
  - start request returned
    `Required resources are currently unavailable, state change queued`;
  - a 5-minute poll kept reporting `cur_state=stopped`,
    `actual_status=exited`, so no CUDA build or performance artifact was
    produced this turn.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - same-hardware CUDA evidence is still required for W3;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ43 — Qwen35 unified_forward now handles mixed continuation chunks and decode

- Scope:
  - added a Qwen3.5 model-level `forward_stateful_unified_items` path that
    classifies each unified item as fresh prefill, continuation/chunked
    prefill, or decode instead of only accepting fresh final prefill batches;
  - fresh batch prefill now accepts non-final chunks and returns `None` for
    non-final rows; when a batch has final rows, final norm/lm_head/readback is
    restricted to those final rows instead of all rows;
  - continuation chunks use the existing stateful Qwen35 path and only return
    logits on final chunks;
  - decode rows still use the batched decode path and preserve
    `LogitsReturnPolicy` handling for unified decode;
  - added duplicate `cache_id` rejection and checked `usize -> u32` position
    conversion before decode batching.
- Why:
  - the scheduler can now cap active-decode prefill chunk admission, but the
    model entrypoint also has to accept chunked/mixed work; otherwise Qwen35
    falls back through executor-level split behavior as soon as a mixed frame
    contains continuation prefill;
  - this is an architecture-path cleanup: Qwen35 now has a model-owned unified
    classifier for mixed frames instead of pretending all unified work is fresh
    final prefill.
- Validation passed locally:
  - `cargo fmt --all`;
  - `cargo test -p ferrum-models qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture`;
  - `cargo test -p ferrum-models qwen35_unified_forward_requires_paged_kv_for_fresh_batch_prefill -- --nocapture`;
  - `cargo check -p ferrum-models`.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - same-hardware CUDA evidence is still required to prove scheduler trace and
    throughput impact;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ42 — Scheduler mixed-prefill aggregate cap now honors effective step chunk

- Scope:
  - fixed continuous scheduler active-decode mixed-prefill budgeting so the
    aggregate budget uses the effective chunk
    `min(active_decode_prefill_chunk, prefill_step_chunk)` instead of the raw
    `active_decode_prefill_chunk`;
  - this closes the gap where an explicit large active-decode chunk, for
    example `8192`, could still allow many small prefill chunks into a decode
    iteration even though each individual prefill was capped by
    `prefill_step_chunk`;
  - added a scheduler regression matching the W3 trace shape:
    `max_batch=32`, `max_tokens=8192`, `decode=7`, `waiting_prefill=25`,
    `active_decode_prefill_chunk=8192`, `prefill_step_chunk=64`; the expected
    mixed batch is now `7 decode + 4 prefill chunks` rather than filling all
    25 free batch slots.
- Why:
  - W3 scheduler trace previously showed large mixed frames such as
    `decode=7,prefill=25` and `decode=12,prefill=18` with very high latency;
  - the previous product diagnostic command explicitly used
    `--scheduler-active-decode-prefill-chunk 8192`, while auto-config still
    materialized a smaller prefill-step chunk. Before this fix, the per-request
    cap and aggregate cap disagreed.
- Validation passed locally:
  - `cargo fmt --all -- --check`;
  - `cargo test -p ferrum-scheduler active_decode_prefill -- --nocapture`;
  - `cargo test -p ferrum-scheduler newly_admitted_prefill_uses_remaining_budget_with_decode -- --nocapture`;
  - `cargo test -p ferrum-scheduler -- --nocapture`;
  - `cargo check -p ferrum-types -p ferrum-scheduler -p ferrum-engine -p ferrum-cli`.
- Status:
  - this is source/scheduler progress, not a performance claim;
  - next same-hardware CUDA diagnostic should verify the scheduler trace no
    longer admits large active-decode mixed prefill cohorts under the W3 c32
    command shape;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ41 — Qwen35 fresh GDN prefill skips initial recurrent state gather

- Scope:
  - made Qwen3.5 batch linear-attention prefill carry an explicit
    `fresh_initial_linear_state` semantic from the fresh final prefill entry;
  - fresh batch prefill now passes zero-initialized initial conv/GDN state
    slabs directly into the varlen core instead of gathering per-sequence
    zero state buffers for every linear-attention layer;
  - non-fresh/chunked semantics keep the old gather path so future chunked
    prefill can still feed real recurrent state;
  - added a CPU unit test proving `fresh_initial_linear_state=true` ignores
    dirty sequence-local state and produces zero slabs, while
    `fresh_initial_linear_state=false` still gathers the existing state.
- Why:
  - `forward_stateful_prefill_batch` already rejects non-fresh work:
    `pos_offset != 0`, non-final chunks, non-empty tokens, and non-empty
    full-attention KV all fail before this path;
  - CPU and CUDA `alloc_typed(F32)` are zero-initialized, so this removes
    redundant GPU copies while preserving the current product-path semantics;
  - this complements the previous indexed-pool scatter skip by reducing the
    other side of the per-layer prefill state-copy cost.
- Validation passed locally:
  - `cargo fmt --all -- --check`;
  - `cargo test -p ferrum-models qwen35_fresh_prefill_initial_state_slabs_are_zero_not_gathered -- --nocapture`;
  - `cargo test -p ferrum-models linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture`;
  - `cargo check --workspace --all-targets`.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - local machine still has no `nvcc`, so CUDA feature build, CUDA parity, and
    same-hardware c32 diagnostics are pending on the next available 1x RTX
    4090 lane;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`.

## 2026-06-22 ZZZ40 — Qwen35 GDN prefill now uses vLLM-style packed qkvz/ba source path

- Scope:
  - added backend capability
    `supports_qwen35_packed_gdn_prefill_prepare`;
  - added packed varlen GDN prepare API
    `linear_attention_prepare_varlen_packed_qkvz_ba_f32`;
  - added CPU reference implementation and CUDA launcher/kernel for packed
    `[q,k,v,z]` + `[b,a]` prefill prepare;
  - routed Qwen3.5 product prefill through fused `qkvz_proj` and `ba_proj`
    when the backend advertises the packed prefill capability;
  - routed product varlen prefill through compact core outputs so
    `query/key/value/g/beta/delta_core` debug/reference intermediates are not
    held past the GDN core boundary;
  - when indexed linear state pools are present, product batch prefill now
    writes final recurrent/conv state only to the slot pool and skips the
    duplicate per-sequence state scatter; sequence-local state is still
    synchronized from the slot before any non-indexed use;
  - kept the old separate `qkv/z/b/a` path as the fallback for backends that
    do not support packed prefill prepare.
- vLLM alignment:
  - local vLLM path inspected:
    `/Users/chejinxuan/py_ws/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`;
  - vLLM Qwen3.5 prefill projects with `in_proj_qkvz` and `in_proj_ba`, then
    splits `[q,k,v,z]` and `[b,a]`; Ferrum prefill previously still launched
    four separate projections.
- Validation passed locally:
  - `cargo fmt --all`;
  - `cargo test -p ferrum-kernels --test linear_attention_cpu linear_attention_prepare_varlen_packed_cpu_matches_separate_prepare -- --nocapture`;
  - `cargo check -p ferrum-kernels -p ferrum-models`;
  - `cargo test -p ferrum-models linear_attention_prefill_varlen_compact_core_matches_full_core_outputs -- --nocapture`;
  - `cargo test -p ferrum-models linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture`.
- Added GPU-targeted test:
  - `linear_attention_prepare_varlen_packed_cuda_matches_cpu_reference` in
    `crates/ferrum-kernels/tests/linear_attention_cuda_eq.rs`;
  - local machine has no `nvcc`, so CUDA feature build and this CUDA test are
    pending on the next 1x RTX 4090 lane.
- Status:
  - this is source/hot-path progress, not a performance claim;
  - W3 remains incomplete and there is still no
    `MODEL_RELEASE_GRADE_W3 PASS`;
  - next evidence must be same-hardware CUDA build, CUDA kernel parity test,
    real `ferrum run`/`ferrum serve` smoke, then c32 diagnostic against the
    existing vLLM baseline.

## 2026-06-22 ZZZ39 — W3 scheduler trace localized mixed prefill+decode bottleneck

- Scope:
  - added typed scheduler JSONL tracing for `ferrum serve`;
  - ran real Qwen3.5 GPTQ CUDA product smokes and c32 ShareGPT diagnostics on
    the existing Vast 1x RTX 4090 lane;
  - fixed auto-config so accelerator default `prefill_first_until_active` is
    still materialized when `active_decode_prefill_chunk` is explicitly set.
- Artifact:
  - remote root:
    `/workspace/artifacts/w3_qwen35_sched_trace_20260621T164651Z`;
  - local copyback is pending because Vast instance `41422823` stopped before
    rsync and restart was queued/unavailable.
- Correctness smoke:
  - non-stream chat returned HTTP 200, content `5`, usage present;
  - stream chat with `stream_options.include_usage=true` returned HTTP 200,
    exactly one `[DONE]`, no malformed SSE, content `5`, usage present.
- Diagnostic performance:
  - with cohort prefill policy: `64 completed / 0 errored / 4.5s`,
    `651.4 output tok/s`, TTFT p50/p95 `636.0 / 1121.4 ms`, TPOT p50/p95
    `32.5 / 45.4 ms`;
  - removing `prefill_first_until_active` collapsed the same binary/dataset to
    `22.7 output tok/s`, TTFT p50/p95 `40674.1 / 56641.6 ms`;
  - trace showed mixed prefill+decode outliers such as
    `decode=7,prefill=25` at `56.38s` and `decode=12,prefill=18` at `40.67s`,
    while pure `decode=32` stayed around `19-21 ms`.
- Conclusion:
  - W3 performance gate remains FAIL; no `MODEL_RELEASE_GRADE_W3 PASS`;
  - the next high-return lever is not pure decode scheduling overhead. It is
    either avoiding large mixed prefill+decode until the cohort is formed or
    implementing an efficient vLLM-style mixed/chunked Qwen3.5 GDN path.

## 2026-06-20 ZZZ38 — Qwen35 decode sync fix correctness OK, no material perf gain

- Scope:
  - validated `7852c139 perf(qwen35): avoid decode sync before gpu argmax`
    on the same existing Vast 1x RTX 4090 lane;
  - this is diagnostic evidence only, not release-grade performance evidence:
    `bench-serve` used `n_repeats=1` for c=1/c=32 to avoid wasting paid GPU
    time after the first no-gain signal.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_decode_syncfix_cuda_diag_20260620T040711Z_7852c139`;
  - remote clean checkout `HEAD=7852c13957b9b3085c82aea57e34d4b49fc66947`;
  - CUDA release binary SHA256:
    `154db666e682978ab8f130e0ad4c6771b9a65bf164409279e3b53cbaf7781ebe`;
  - build status `BUILD_PASS`.
- Product smoke:
  - `ferrum run` real CUDA Qwen3.5 GPTQ smoke passed and output `Paris`;
  - `ferrum serve` non-stream and stream smoke passed and output `Paris`;
  - streaming emitted exactly one `[DONE]` and included usage.
- Diagnostic performance:
  - comparison artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_decode_syncfix_cuda_diag_20260620T040711Z_7852c139/perf_ratio_vs_vllm_diag.json`;
  - c=1: Ferrum `53.264` tok/s vs vLLM `136.143` tok/s, mean ratio
    `39.1%`, p95 ITL `2.24x`;
  - c=32: Ferrum `142.443` tok/s vs vLLM `1708.528` tok/s, mean ratio
    `8.3%`, p95 ITL `7.27x`;
  - previous full release-shape sweep was c=1 `53.806` tok/s and c=32
    `142.839` tok/s, so the sync fix does not materially move throughput.
- Conclusion:
  - W3 performance gate remains FAIL; no `MODEL_RELEASE_GRADE_W3 PASS`;
  - the next optimization must be architectural: replace Qwen35 decode's
    per-sequence recurrent state gather/scatter with vLLM-style indexed packed
    conv/GDN state updates.
- GPU lifecycle:
  - Vast instance `41422823` was stopped after copyback;
  - stop check: `cur_state=stopped`, `actual_status=exited`,
    `intended_status=stopped`.

## 2026-06-20 ZZZ37 — W3 Qwen35 L2/L4/L5 correctness PASS, performance ratio FAIL

- Scope:
  - packaged the existing real CUDA product known-answer report into the
    formal W3 L2 quantized artifact;
  - ran real-model W3 L4 agent checks against `ferrum serve`;
  - ran release-shape W3 L5 concurrency with the same ShareGPT/vLLM baseline
    shape: `c=1/4/16/32`, `num_prompts=100`, `warmup=10`,
    `n_repeats=3`, `--fail-on-error`, `--require-ci`, `--seed 9271`;
  - this is correctness evidence, not W3 completion: the vLLM 80% performance
    gate fails in every required cell, and there is still no final
    `MODEL_RELEASE_GRADE_W3 PASS`.
- L2 artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e`;
  - PASS line:
    `W3 L2 QUANTIZED PASS: docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e`;
  - source report was the real Qwen35 CUDA product report from
    `w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e`;
  - report has 11/11 known-answer cases, both `ferrum run` and
    `ferrum serve`, typed CLI product surface, and `hidden_env=[]`.
- L4/L5 CUDA artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9`;
  - remote clean checkout:
    `/workspace/ferrum-w3-unified-75ec7e6e`,
    `HEAD=ba19f2b97457202f9c0dbe108cedf17eca594531`,
    `git status --short` clean before and after;
  - binary SHA256:
    `e32d89a44ac4759cf177ac2d64115389652e27b67c44ceebbbb5ecc3a6eb6c30`;
  - server used typed flags:
    `--backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90
    --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 8192
    --kv-capacity 2048 --scheduler-prefill-first-until-active 32
    --scheduler-active-decode-prefill-chunk 8192 --greedy-argmax`;
  - `HF_HOME=/workspace/hf-cache` was recorded only to select the existing
    Hugging Face cache location; inference behavior remained typed CLI.
- PASS lines:
  - `W3 L4 AGENT PASS: /workspace/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/l4_agent`;
  - `W3 L5 CONCURRENCY PASS: /workspace/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/l5_concurrency`;
  - `W3 QWEN35 L4 L5 CUDA PASS: /workspace/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9`.
- L4 result:
  - required tool-call smoke passed `10/10`;
  - strict JSON schema smoke passed `20/20`;
  - negative contracts returned HTTP 400 for invalid `tool_choice` and invalid
    `response_format`.
- L5 result:
  - c=1 completed `[100,100,100]`, errored `[0,0,0]`;
  - c=4 completed `[100,100,100]`, errored `[0,0,0]`;
  - c=16 completed `[100,100,100]`, errored `[0,0,0]`;
  - c=32 completed `[100,100,100]`, errored `[0,0,0]`;
  - output token count source is `usage`, and the stream/quality zero-error
    fields in the L5 artifact are all zero.
- vLLM 80% ratio status:
  - comparison artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/perf_ratio_vs_vllm.json`;
  - baseline artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_vllm_sharegpt_baseline_20260619/bench_vllm_sharegpt_sweep_100x3.json`;
  - c=1: Ferrum `53.806` tok/s vs vLLM `136.143` tok/s,
    mean ratio `39.5%`, LCB ratio `39.2%`, p95 ITL `2.17x`;
  - c=4: Ferrum `99.130` tok/s vs vLLM `405.420` tok/s,
    mean ratio `24.5%`, LCB ratio `22.5%`, p95 ITL `3.58x`;
  - c=16: Ferrum `142.177` tok/s vs vLLM `1190.692` tok/s,
    mean ratio `11.9%`, LCB ratio `10.9%`, p95 ITL `11.56x`;
  - c=32: Ferrum `142.839` tok/s vs vLLM `1708.528` tok/s,
    mean ratio `8.4%`, LCB ratio `8.1%`, p95 ITL `12.67x`;
  - conclusion: W3 performance gate is currently FAIL, and these numbers are
    diagnostic evidence for the next optimization lane.
- GPU lifecycle:
  - Vast instance `41422823`, 1x `NVIDIA GeForce RTX 4090`, driver
    `580.126.09`, CUDA toolkit `12.4`, dph `$0.662962962962963`;
  - artifacts were copied back locally;
  - stop check after copyback:
    `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`.
- vLLM comparison notes:
  - local vLLM checkout inspected at
    `/Users/chejinxuan/py_ws/vllm`, `HEAD=0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`;
  - relevant files:
    `vllm/model_executor/models/qwen3_5.py`,
    `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
    `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`;
  - vLLM's Qwen3.5 path uses hybrid attention metadata, GDN-specific
    `is_prefilling`, chunked prefill, and packed recurrent decode fast paths;
    Ferrum currently has correctness-clean varlen prefill but does not yet
    match vLLM's packed decode/GDN hot path, which matches the observed
    high-concurrency throughput plateau around `142 tok/s`.
- Next:
  - commit/push this evidence first;
  - then implement the next high-return architecture lever: align Ferrum's
    Qwen35 decode-side GDN path with vLLM's packed recurrent decode structure
    and profile only after the new path has a correctness gate.

## 2026-06-20 ZZZ36 — W3 Qwen35 unified fresh prefill source + CUDA product smoke PASS

- Scope:
  - wired Qwen35 product `unified_forward` to a native fresh final prefill
    batch path for paged KV;
  - the fast path is deliberately narrow and correct: non-empty tokens,
    `pos_offset == 0`, `is_final_chunk == true`, unique cache ids, empty
    sequence state, and empty full-attention KV are required; unsupported
    mixed/decode/chunked shapes fall back through the executor;
  - batch prefill now runs linear-attention layers through the varlen
    prepare/GDN core and full-attention layers through paged varlen
    split-QKV-to-cache plus paged varlen attention, then gathers one final
    hidden row per request for LM head logits;
  - this is still smoke/product correctness evidence only: no c=1/4/16/32
    performance matrix, no same-hardware vLLM ratio, and no final
    `MODEL_RELEASE_GRADE_W3 PASS`.
- Commit:
  - `75ec7e6e perf(qwen35): route fresh prefill through unified forward`;
  - pushed to `origin/goal/w2-w3-release-grade`.
- Local validation:
  - `cargo fmt --all` PASS;
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo check -p ferrum-models --all-targets` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS: 83 matched
    library tests plus `qwen35_config_test` 1 test;
  - added CPU/non-paged product-entry test
    `qwen35_unified_forward_requires_paged_kv_for_fresh_batch_prefill`, so
    unsupported backends keep executor fallback semantics.
- CUDA validation / lifecycle:
  - Vast instance `41422823`, 1x `NVIDIA GeForce RTX 4090`, `49140 MiB`,
    driver `580.126.09`, CUDA toolkit `12.4`, Rust `1.96.0`;
  - clean remote checkout:
    `/workspace/ferrum-w3-unified-75ec7e6e`,
    `HEAD=75ec7e6ebd82e017e74651490ccd1c15f55b1f5a`,
    `git status --short` clean before and after;
  - artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e`;
  - CUDA release build command:
    `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - binary SHA256:
    `e32d89a44ac4759cf177ac2d64115389652e27b67c44ceebbbb5ecc3a6eb6c30`;
  - CUDA unit command:
    `cargo test -p ferrum-models --features cuda linear_attention_prefill_varlen_cuda_backend_matches_per_sequence_stateful_reference -- --nocapture`;
  - real product report PASS line:
    `W3 QWEN35 REAL PRODUCT REPORT PASS: /workspace/artifacts/w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e/real_product_report`;
  - smoke PASS line:
    `W3 QWEN35 UNIFIED PREFILL CUDA SMOKE PASS: /workspace/artifacts/w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e`;
  - Vast stop check after artifact copyback:
    `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`.
- Product result:
  - model: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`;
  - command surface used typed CLI flags only:
    `--backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90
    --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048
    --kv-capacity 2048`;
  - `ferrum run` PASS for known answer `What is 2+3?`, assistant content
    `5`, `finish_reason=stop`;
  - `w3_qwen35_real_product_report.py` PASS with 11 known-answer cases and
    5 behavior cases, covering `ferrum run`, `ferrum serve`, non-stream,
    stream, natural EOS, custom stop, and reasoning extraction;
  - `w3_s2_whole_model_product_path.json` reports:
    `W3 QWEN35 REAL PRODUCT PATH PASS: /workspace/artifacts/w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e/real_product_report`.
- Next:
  - run a targeted concurrent `bench-serve` smoke that forces multiple fresh
    prompts into the new Qwen35 `unified_forward` batch prefill path;
  - then run W3 L5/full c=1/4/16/32 performance and same-hardware vLLM ratio
    only after that correctness smoke remains clean.

## 2026-06-20 ZZZ35 — W3 Qwen35 varlen linear-attention prepare CUDA exec PASS

- Scope:
  - added a backend-native varlen Qwen35 linear-attention prepare primitive:
    depthwise causal conv + Q/K/V split + GDN gate preparation with
    `cu_seqlens` sequence boundaries;
  - the new primitive reads per-sequence initial conv state and writes
    per-sequence final conv state, so batched prefill can avoid cross-request
    conv bleed;
  - added the Qwen35 varlen prefill core that composes varlen prepare,
    existing varlen recurrent GDN, and gated RMSNorm;
  - added CPU reference coverage and a CUDA feature test that executes the new
    CUDA kernel against the same per-sequence stateful reference;
  - this is still prerequisite work only: Qwen35 product `unified_forward` /
    `batch_prefill` has not yet been switched to this path, so no W3 final PASS
    and no performance claim.
- Commit:
  - `a50d42c6 perf(qwen35): add varlen linear attention prepare`.
- Local validation:
  - `cargo fmt --all` PASS;
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo check -p ferrum-kernels --all-targets` PASS;
  - `cargo check -p ferrum-models --all-targets` PASS;
  - `cargo test -p ferrum-models linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS: 82 matched
    library tests plus `qwen35_config_test` 1 test.
- CUDA validation / lifecycle:
  - Vast instance `41422823`, 1x `NVIDIA GeForce RTX 4090`, `49140 MiB`,
    driver `580.126.09`, CUDA toolkit `12.4`, Rust `cargo 1.96.0`;
  - remote clean smoke checkout was `HEAD=d60bb92a` plus exactly the six source
    files from `a50d42c6` before the local commit was created;
  - build artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_varlen_prepare_cuda_smoke_retry_20260620T010212Z`;
  - CUDA build smoke PASS line:
    `W3 QWEN35 VARLEN PREPARE CUDA BUILD SMOKE PASS: /workspace/artifacts/w3_qwen35_varlen_prepare_cuda_smoke_retry_20260620T010212Z`;
  - exec artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_varlen_prepare_cuda_exec_20260620T010925Z`;
  - CUDA execution test command:
    `cargo test -p ferrum-models --features cuda linear_attention_prefill_varlen_cuda_backend_matches_per_sequence_stateful_reference -- --nocapture`;
  - CUDA execution PASS line:
    `W3 QWEN35 VARLEN PREPARE CUDA EXEC SMOKE PASS: /workspace/artifacts/w3_qwen35_varlen_prepare_cuda_exec_20260620T010925Z`;
  - an earlier build-smoke attempt failed with `cargo: command not found`
    because the tmux shell did not load `/root/.cargo/env`; this was an
    environment setup failure, not a source failure;
  - Vast stop check after artifact copyback:
    `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`.
- Next:
  - wire Qwen35 `unified_forward` / product `batch_prefill` to the varlen
    prepare + varlen GDN core with per-request state writeback;
  - then run `ferrum run` and `ferrum serve` correctness before any W3 perf
    comparison.

## 2026-06-20 ZZZ34 — W3 Qwen35 varlen GDN primitive source checkpoint

- Scope:
  - added a native variable-length batched recurrent gated-DeltaNet prefill
    primitive for Qwen35 linear-attention work, matching the `cu_seqlens`
    shape used by vLLM-style chunked GDN prefill;
  - this is an architectural prerequisite only: it adds backend/kernel/API
    surface and CPU reference coverage, but does not yet switch product
    prefill scheduling to the new primitive;
  - no W3 final PASS and no performance claim.
- Commit:
  - `19920b3e perf(qwen35): add varlen gated delta primitive`;
  - pushed to `origin/goal/w2-w3-release-grade`.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo check -p ferrum-kernels --all-targets` PASS;
  - `cargo check -p ferrum-models --all-targets` PASS;
  - `cargo test -p ferrum-models recurrent_delta_rule_varlen_backend_matches_per_sequence_reference -- --nocapture` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS: 81 matched tests
    plus `qwen35_config_test` 1 test.
- CUDA validation / lifecycle:
  - Vast instance `41422823`, 1x `NVIDIA GeForce RTX 4090`, `49140 MiB`,
    driver `580.126.09`, CUDA toolkit `12.4`;
  - artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_varlen_gdn_cuda_build_19920b3e_clean_retry_20260619T235447Z`;
  - clean remote state was `HEAD=19920b3e`, branch ahead of origin only because
    the commit had not been pushed before the remote smoke;
  - `cargo check -p ferrum-kernels --features cuda --all-targets` PASS in
    46.38s;
  - focused remote test PASS:
    `cargo test -p ferrum-models recurrent_delta_rule_varlen_backend_matches_per_sequence_reference -- --nocapture`;
  - smoke PASS line:
    `W3 QWEN35 VARLEN GDN CUDA KERNEL BUILD SMOKE PASS: /workspace/artifacts/w3_qwen35_varlen_gdn_cuda_build_19920b3e_clean_retry_20260619T235447Z`;
  - wider `cargo check -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
    was intentionally stopped at the paid-lane stop condition after roughly
    15 minutes while compiling `vllm_marlin_moe/ops.cu`; it did not fail and
    is not counted as PASS evidence;
  - Vast stop check after artifact copyback:
    `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`.
- Next:
  - route Qwen35 product prefill batches through the varlen primitive and
    write back per-sequence recurrent states;
  - then rerun `ferrum run` / `ferrum serve` correctness and only after that
    same-host diagnostic perf.

## 2026-06-20 ZZZ33 — W3 Qwen35 final-token prefill LM head smoke PASS, perf still prefill-bound

- Scope:
  - targeted 1x Vast CUDA validation for commit
    `adf70f90 perf(qwen35): project only final prefill token`;
  - fixes a real Qwen35 prefill hot-path waste: fresh prefill previously ran
    LM head over every prompt token (`tokens_len * vocab`) and then copied only
    the last logits row; it now projects only the final hidden row because the
    product interface only returns last-token logits for sampling;
  - diagnostic bench ran only after product correctness passed;
  - no W3 final PASS and no release-grade performance claim.
- Local source validation before GPU:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -p ferrum-models --all-targets` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS: 80 matched
    library tests plus `qwen35_config_test` 1 test;
  - `cargo test -p ferrum-models linear_attention_decode_backend_matches_full_reference_last_token -- --nocapture`
    PASS: 1 matched test;
  - `git diff --check` PASS.
- GPU / lifecycle:
  - Vast instance `41422823`, SSH `ssh7.vast.ai:22822`, was started only for
    this targeted lane and stopped after artifact copyback;
  - final stop check: `cur_state=stopped`, `actual_status=exited`;
  - GPU: 1x `NVIDIA GeForce RTX 4090`, `49140 MiB` reported by `nvidia-smi`,
    driver `580.126.09`, compute capability `8.9`;
  - CUDA toolkit `12.4`, Rust `1.96.0`.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_final_prefill_lm_head_adf70f90_20260619T222916Z`;
  - smoke PASS line:
    `W3 QWEN35 VLLM H256 SMOKE PASS: /workspace/artifacts/w3_qwen35_final_prefill_lm_head_adf70f90_20260619T222916Z`;
  - CUDA release build PASS, binary SHA256:
    `2dffff645429f8081edf1d8938137dd1ba148ca5c0b2d63ff8d9498806f9d1f0`.
- Correctness result:
  - `ferrum run` PASS; output:
    `The mysterious ferrum-ok appears only in rare scientific texts.`;
  - `ferrum serve` effective config selected `vllm_paged_attn_v2`, with
    `FERRUM_USE_VLLM_PAGED_ATTN=1` and `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`;
  - non-stream chat HTTP 200, `finish_reason=stop`;
  - stream chat HTTP 200, exactly one `[DONE]`, usage present, final
    `finish_reason=stop`;
  - required tool call HTTP 200, parsed `get_weather({"city":"Paris"})`,
    `finish_reason=tool_calls`;
  - strict structured output HTTP 200, content exactly
    `{"answer":"scenario-ok"}`, `finish_reason=stop`;
  - post-validation rejects `finish_reason=length`, missing tool calls,
    malformed streams, duplicate `[DONE]`, and obvious repetition.
- Diagnostic bench only:
  - command used `bench-serve --dataset sharegpt --num-prompts 8
    --n-repeats 1 --concurrency-sweep 1,32 --fail-on-error --seed 9271`;
  - c=1: 8 completed / 0 errored / 24.2s, output throughput
    `14.60 tok/s`, p50 TTFT `2348.5 ms`,
    `output_token_count_source=usage`;
  - c=32: 8 completed / 0 errored / 20.0s, output throughput
    `19.19 tok/s`, p50 TTFT `18176.8 ms`,
    `output_token_count_source=usage`;
  - this is a bottleneck signal only (`n_repeats=1`, small prompt count), not
    W3 performance evidence.
- Interpretation / next work:
  - the final-token LM head fix is correct and modestly improves the diagnostic
    path versus the previous `13.57`/`16.03 tok/s` artifact;
  - the small delta proves LM head waste was real but not the dominant blocker;
  - c=32 TTFT is still ~18 seconds for 8 prompts, which matches serial
    prefill rather than true multi-request prefill batching;
  - next work should implement or prototype Qwen35 native unified/batch prefill
    instead of further decode-only tuning: full-attention layers can reuse the
    existing paged varlen path, but linear-attention layers need batched
    recurrent/conv prefill state handling and final-row-only LM head.

## 2026-06-20 ZZZ32 — W3 Qwen35 batched linear-attention decode smoke PASS, batch scaling still blocked

- Scope:
  - targeted 1x Vast CUDA validation for commit
    `c61176df perf(qwen35): batch linear attention decode state kernels`;
  - replaces the Qwen35 linear-attention decode per-row compute loop with
    backend batch APIs and CUDA batch kernels for decode preparation and
    recurrent gated delta rule state updates;
  - follows the vLLM direction of batched stateful decode work instead of
    launching one tiny recurrent/prepare kernel sequence per request row;
  - diagnostic bench ran only after CUDA parity tests and product correctness
    passed;
  - no W3 final PASS and no release-grade performance claim.
- Local source validation before GPU:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -p ferrum-kernels -p ferrum-models --all-targets` PASS;
  - `cargo test -p ferrum-kernels --test linear_attention_cpu -- --nocapture`
    PASS: 4 tests;
  - `cargo test -p ferrum-kernels --test gated_delta_rule_cpu -- --nocapture`
    PASS: 3 tests;
  - `cargo test -p ferrum-models linear_attention_decode_backend_matches_full_reference_last_token -- --nocapture`
    PASS: 1 matched test;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS: 81 tests;
  - `git diff --check` PASS.
- GPU / lifecycle:
  - Vast instance `41422823`, SSH `ssh7.vast.ai:22822`, was started only for
    this targeted lane and stopped after artifact copyback;
  - final stop check: `cur_state=stopped`, `actual_status=exited`;
  - GPU: 1x `NVIDIA GeForce RTX 4090`, `49140 MiB` reported by `nvidia-smi`,
    driver `580.126.09`, compute capability `8.9`;
  - CUDA toolkit `12.4`, Rust `1.96.0`.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_batch_linear_decode_c61176df_20260619T215831Z`;
  - smoke PASS line:
    `W3 QWEN35 VLLM H256 SMOKE PASS: /workspace/artifacts/w3_qwen35_batch_linear_decode_c61176df_20260619T215831Z`;
  - CUDA release build PASS, binary SHA256:
    `af47b9fe3573f567a0fe5c63b3300d2931ae05ee1b2e80dfadcc755ef2f8e0c4`.
- CUDA parity result:
  - `cargo test -p ferrum-kernels --features cuda --test linear_attention_cuda_eq -- --nocapture`
    PASS: 6 tests, including
    `linear_attention_decode_prepare_batch_cuda_matches_cpu_reference`;
  - `cargo test -p ferrum-kernels --features cuda --test gated_delta_rule_cuda_eq -- --nocapture`
    PASS: 2 tests, including
    `recurrent_gated_delta_rule_batch_cuda_matches_cpu_reference`.
- Correctness result:
  - `ferrum run` PASS; output:
    `The mysterious ferrum-ok appears only in rare scientific texts.`;
  - `ferrum serve` effective config selected `vllm_paged_attn_v2`, with
    `FERRUM_USE_VLLM_PAGED_ATTN=1` and `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`;
  - non-stream chat HTTP 200, content `The capital of France is Paris.`,
    `finish_reason=stop`;
  - stream chat HTTP 200, exactly one `[DONE]`, usage present, final
    `finish_reason=stop`;
  - required tool call HTTP 200, parsed `get_weather({"city":"Paris"})`,
    `finish_reason=tool_calls`;
  - strict structured output HTTP 200, content exactly
    `{"answer":"scenario-ok"}`, `finish_reason=stop`;
  - post-validation rejects `finish_reason=length`, missing tool calls,
    malformed streams, duplicate `[DONE]`, and obvious repetition.
- Diagnostic bench only:
  - command used `bench-serve --dataset sharegpt --num-prompts 8
    --n-repeats 1 --concurrency-sweep 1,32 --fail-on-error --seed 9271`;
  - c=1: 8 completed / 0 errored / 26.1s, output throughput
    `13.57 tok/s`, `output_token_count_source=usage`;
  - c=32: 8 completed / 0 errored / 22.0s, output throughput
    `16.03 tok/s`, `output_token_count_source=usage`;
  - this is a bottleneck signal only (`n_repeats=1`, small prompt count), not
    W3 performance evidence.
- Interpretation / next work:
  - the batched CUDA linear-attention decode kernels are correct and the product
    smoke remains PASS;
  - performance did not materially improve versus the previous diagnostic
    (`14.5`/`15.3 tok/s`) and remains far below the vLLM 80% targets
    (`107.5`/`1349.9 tok/s`);
  - removing per-row compute kernels was necessary but not sufficient: the next
    investigation should use profiler evidence around the outer decode step to
    separate remaining Qwen35 state pack/copy cost, scheduler admission/step
    behavior, logits/sampling work, and any serialization that prevents true
    c=32 token-step batching.

## 2026-06-20 ZZZ31 — W3 Qwen35 argmax mask model-vocab smoke PASS, performance still blocked

- Scope:
  - targeted 1x Vast CUDA validation for commit
    `e4404604 fix(engine): size argmax masks to model vocab`;
  - fixes the previous model-side argmax correctness regression by building the
    GPU token-validity mask to the model/logits vocab size instead of the
    tokenizer base vocab size;
  - diagnostic bench ran only after correctness passed;
  - no W3 final PASS and no release-grade performance claim.
- Local source validation before GPU:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo test -p ferrum-engine model_decode -- --nocapture` PASS: 5 matched
    tests;
  - `cargo check -p ferrum-engine --all-targets` PASS.
- GPU / lifecycle:
  - Vast instance `41422823`, SSH `ssh7.vast.ai:22822`, was started for this
    targeted lane and stopped after artifact copyback;
  - final stop check: `cur_state=stopped`, `actual_status=exited`;
  - GPU: 1x `NVIDIA GeForce RTX 4090`, `49140 MiB`, driver `580.126.09`,
    compute capability `8.9`;
  - CUDA toolkit `12.4`, Rust `1.96.0`.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_argmax_mask_model_vocab_e4404604_20260619T210406Z`;
  - smoke PASS line:
    `W3 QWEN35 VLLM H256 SMOKE PASS: /workspace/artifacts/w3_qwen35_argmax_mask_model_vocab_e4404604_20260619T210406Z`;
  - CUDA release build PASS, binary SHA256:
    `916e2eb5644a95f1df731aae2bd87fc8fddc6ee81799e269a09265a49dd23f0e`.
- Correctness result:
  - `ferrum run` PASS; output:
    `The mysterious ferrum-ok appears only in rare scientific texts.`;
  - `ferrum serve` effective config selected `vllm_paged_attn_v2`, with
    `FERRUM_USE_VLLM_PAGED_ATTN=1` and `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`;
  - non-stream chat HTTP 200, content `The capital of France is Paris.`,
    `finish_reason=stop`;
  - stream chat HTTP 200, exactly one `[DONE]`, usage present, final
    `finish_reason=stop`;
  - required tool call HTTP 200, parsed `get_weather({"city":"Paris"})`,
    `finish_reason=tool_calls`;
  - strict structured output HTTP 200, content exactly
    `{"answer":"scenario-ok"}`, `finish_reason=stop`.
- Diagnostic bench only:
  - command used `bench-serve --dataset sharegpt --num-prompts 8
    --n-repeats 1 --concurrency-sweep 1,32 --fail-on-error --seed 9271`;
  - c=1: 8 completed / 0 errored / 24.4s, output throughput `14.5 tok/s`;
  - c=32: 8 completed / 0 errored / 23.0s, output throughput `15.3 tok/s`;
  - this is a bottleneck signal only (`n_repeats=1`, small prompt count), not
    W3 performance evidence.
- Interpretation / next work:
  - correctness smoke is back to PASS with model-side argmax enabled;
  - performance regressed from the previous strict-schema diagnostic
    (`26.3`/`35.2` tok/s at c=1/c=32) and remains far below the vLLM 80%
    targets (`107.5`/`1349.9` tok/s);
  - next work should stop optimizing sampling and inspect Qwen35 scheduler/KV
    integration and decode batching, especially why c=32 effective throughput
    is nearly identical to c=1.

## 2026-06-20 ZZZ30 — W3 Qwen35 sparse-repetition CUDA smoke failed at non-stream length

- Scope:
  - targeted 1x Vast CUDA validation for the Qwen35 sparse repetition greedy
    decode work;
  - validates that the CUDA feature build compiles after the GPU-side sparse
    repetition argmax changes;
  - no diagnostic bench was run because correctness failed first;
  - no W3 final PASS and no release-grade performance claim.
- Code:
  - pushed source/perf commit
    `48db0eb5 perf(qwen35): keep repetition greedy decode on gpu`;
  - pushed CUDA build fix commits `31817b49`, `103faeea`, and `6c3aad47`;
  - remote validation head:
    `6c3aad47a63c7e1030a60b00e7d437ec09ac0a79`.
- GPU / lifecycle:
  - Vast instance `41422823`, SSH `ssh7.vast.ai:22822`, was started only for
    this targeted lane and stopped after artifact copyback;
  - final stop check: `cur_state=stopped`, `actual_status=exited`;
  - GPU: 1x `NVIDIA GeForce RTX 4090`, `49140 MiB`, driver `580.126.09`,
    compute capability `8.9`;
  - CUDA toolkit `12.4`, Rust `1.96.0`.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_sparse_repetition_gpu_6c3aad47_20260619T203155Z`;
  - smoke FAIL line:
    `W3 QWEN35 VLLM H256 SMOKE FAIL rc=1: /workspace/artifacts/w3_qwen35_sparse_repetition_gpu_6c3aad47_20260619T203155Z`;
  - CUDA release build PASS, binary SHA256:
    `ee04f09cee1dd91c16b6d6424ebf77a72cdb118b92562fb28d1da79418c2673e`;
  - `ferrum run` validation PASS.
- Correctness failure:
  - `ferrum serve` started and effective config still selected
    `vllm_paged_attn_v2`, with `FERRUM_USE_VLLM_PAGED_ATTN=1` and
    `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`;
  - non-stream chat returned HTTP 200 but failed post-validation because
    `finish_reason=length`;
  - body repeated `Paris` inside `<text>` blocks until the 64-token cap:
    `The capital of France is Paris. <text> Paris </text> ...`;
  - stream/tool/strict-schema request script summary was PASS, but the lane is
    correctly marked FAIL because the non-stream product path is not correct;
  - no `bench-serve` artifact exists for this run.
- Interpretation / next work:
  - the CUDA build part of the sparse repetition argmax fix is now past the
    previous compile blockers;
  - the vLLM-style single-application repetition penalty is not sufficient to
    make the Qwen35 ordinary non-stream greedy chat path stop correctly;
  - next work should inspect whether Qwen35 serve is actually taking the new
    model-side sparse-repetition argmax path and then fix the decode-quality
    architecture before rerunning paid CUDA.

## 2026-06-20 ZZZ29 — W3 Qwen35 greedy repetition stays on GPU source checkpoint

- Scope:
  - Qwen35 decode hot-path architecture fix after the strict-schema smoke showed
    correctness PASS but throughput far below the vLLM 80% targets;
  - source-level validation only at this checkpoint;
  - no CUDA performance artifact, no W3 final PASS, and no release-grade
    performance claim.
- vLLM comparison used:
  - local vLLM source applies repetition penalties on GPU before sampling rather
    than forcing a full `[batch, vocab]` logits download;
  - Ferrum's product default chat path used `repetition_penalty=1.1`, but the
    engine previously treated that as ineligible for model-side greedy argmax.
- Change:
  - `LogitsReturnPolicy::GreedyArgmax` now carries optional sparse
    repetition-penalty metadata;
  - ordinary greedy text decode with token masks and repetition penalty remains
    on the model-side argmax path instead of setting
    `ferrum_require_full_logits`;
  - Qwen35 uploads per-row sparse repeated-token lists and applies the penalty in
    a CUDA logits kernel before the existing masked/unmasked row argmax;
  - the shared CPU/full-logits `RepetitionPenaltyProcessor` now matches the same
    vLLM-style "token appeared" semantics instead of over-penalizing duplicate
    generated tokens by frequency exponent;
  - structured JSON/schema/regex guided requests still request full logits for
    engine-side constrained sampling;
  - non-Qwen35 Llama-family decode conservatively falls back to full logits when
    sparse repetition penalty is requested, until that backend gets the same
    model-side implementation.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo test -p ferrum-interfaces repetition_penalty_applies -- --nocapture`
    PASS: 1 matched test;
  - `cargo check -p ferrum-interfaces -p ferrum-kernels -p ferrum-engine -p ferrum-models --all-targets`
    PASS;
  - `cargo test -p ferrum-engine model_decode -- --nocapture` PASS: 4 matched
    tests;
  - `cargo test -p ferrum-models qwen35_decode_logits_policy_uses_greedy_only_for_consistent_masks -- --nocapture`
    PASS: 1 matched test;
  - `cargo test -p ferrum-models unified_decode_forwards_logits_policy_to_unified_model -- --nocapture`
    PASS: 1 matched test.
- Limitation / next work:
  - CUDA kernel build/runtime behavior still needs same-pod validation on the
    4090 instance;
  - next GPU lane should run the Qwen35 `ferrum run`/`ferrum serve` correctness
    smoke first, then a diagnostic `bench-serve` c=1/32 sweep to measure whether
    this removes the CPU sampling/logits-readback bottleneck.

## 2026-06-20 ZZZ28 — W3 Qwen35 strict-schema product smoke PASS, perf still far below target

- Scope:
  - targeted 1x Vast CUDA product smoke for commit
    `3860d0d3 fix(server): guide strict schema chat sampling`;
  - this validates the previous structured-output/repetition fixes on the real
    Qwen35 GPTQ product path;
  - no W3 final PASS and no release-grade performance claim.
- GPU / lifecycle:
  - Vast instance `41422823`, SSH `ssh7.vast.ai:22822`, stopped after artifact
    copyback;
  - stop check: `cur_state=stopped`, `actual_status=exited`;
  - `nvidia-smi`: 1x `NVIDIA GeForce RTX 4090`, `49140 MiB`, driver
    `580.126.09`, compute capability `8.9`;
  - CUDA toolkit `12.4`, Rust `1.96.0`.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_strict_schema_smoke_3860d0d3_20260619T191231Z`;
  - smoke PASS line:
    `W3 QWEN35 VLLM H256 SMOKE PASS: /workspace/artifacts/w3_qwen35_strict_schema_smoke_3860d0d3_20260619T191231Z`;
  - release CUDA build PASS, binary SHA256:
    `9fb863464d86358dde93674ebf3fdcb02d28f51118a9fa6e96b555b574ea9a55`.
- Correctness result:
  - `ferrum run` PASS; output:
    `The mysterious ferrum-ok appears only in rare scientific texts.`;
  - effective config for both `run` and `serve` selected
    `vllm_paged_attn_v2`, with `FERRUM_USE_VLLM_PAGED_ATTN=1` and
    `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`;
  - non-stream chat HTTP 200, `finish_reason=stop`;
  - stream chat HTTP 200, exactly one `[DONE]`, usage present, final
    `finish_reason=stop`;
  - required tool call HTTP 200, parsed `get_weather({"city":"Paris"})`,
    `finish_reason=tool_calls`;
  - strict structured output HTTP 200, content exactly
    `{"answer":"scenario-ok"}`, `finish_reason=stop`;
  - post-validation additionally rejects `finish_reason=length` and obvious
    repeated token chunks.
- Diagnostic bench only:
  - command used `bench-serve --dataset sharegpt --num-prompts 8
    --n-repeats 1 --concurrency-sweep 1,32 --fail-on-error --seed 9271`;
  - c=1: 8 completed / 0 errored / 36.6s, output throughput `26.3 tok/s`;
  - c=32: 8 completed / 0 errored / 27.3s, output throughput `35.2 tok/s`;
  - this is a bottleneck signal only (`n_repeats=1`, small prompt count), not
    W3 performance evidence.
- Limitation / next work:
  - correctness smoke now passes on the intended vLLM H256/V2 attention route;
  - performance remains orders of magnitude below the recorded vLLM 80%
    targets (`107.5` tok/s at c=1 and `1349.9` tok/s at c=32);
  - next work should focus on the decode throughput architecture: avoid
    CPU-side repetition/sampling bottlenecks and verify Qwen35 uses the shared
    paged KV scheduler/block-table path rather than model-local full-attention
    state.

## 2026-06-20 ZZZ27 — W3 serve strict schema/repetition source fix

- Scope:
  - follow-up source fix for the failed Qwen35 vLLM H256 GPU smoke artifact
    `w3_qwen35_vllm_h256_smoke_178d76fa_20260619T181820Z_hot_target`;
  - no new GPU run yet, no release-grade PASS, no W3 final PASS, and no
    performance claim.
- Root causes addressed:
  - OpenAI chat `response_format.json_schema.strict=true` previously only
    added prompt text and final validation; it did not set engine
    `ResponseFormat::JsonSchema`, so generation was unconstrained and could
    emit trailing text after valid JSON;
  - OpenAI chat serving used `repetition_penalty=1.0` while `ferrum run`
    defaults to a repeat penalty, which matches the GPU smoke's repeated
    `Paris` until `finish_reason=length`.
- Change:
  - strict JSON schema requests now route the schema into guided sampling
    unless a forced tool call already owns the structured response format;
  - non-strict `json_schema` remains prompt/final-validation behavior and is
    not hard-masked;
  - `DEFAULT_CHAT_REPETITION_PENALTY` now lives in `ferrum-types` and is used
    by both `ferrum run` and OpenAI chat serving.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-server strict_json_schema -- --nocapture` PASS:
    11 tests;
  - `cargo test -p ferrum-server chat_accepts_stop_string_and_max_completion_tokens -- --nocapture`
    PASS;
  - `cargo test -p ferrum-cli chat_default_applies_repetition_penalty -- --nocapture`
    PASS;
  - `cargo test -p ferrum-engine --test regex_guided_test -- --nocapture`
    PASS: 3 tests;
  - `cargo test -p ferrum-server --test structured_output_contract -- --nocapture`
    PASS: 3 tests;
  - `git diff --check` PASS.
- Limitation / next work:
  - this is source-level product-path evidence only;
  - restart the stopped Vast GPU lane only after this commit is pushed, then
    rerun the Qwen35 product smoke for `run`, `serve`, streaming, required
    tool, and strict schema before any bench/performance work.

## 2026-06-19 ZZZ26 — W3 Qwen35 vLLM H256 path GPU smoke failed at structured output

- Scope:
  - Qwen35 CUDA product-path smoke for the vLLM-layout paged KV/H256 paged
    attention route;
  - no release-grade PASS, no W3 final PASS, and no performance claim.
- Code:
  - pushed `178d76fa fix(config): enable qwen35 moe vllm paged attention
    defaults` on `goal/w2-w3-release-grade`;
  - this follows `aeb0f33e perf(qwen35): route paged kv through vllm h256
    attention`;
  - root cause fixed: auto-config previously recognized `qwen3_moe` but not
    `qwen3_5_moe`, so Qwen35 product defaults still selected
    `legacy_paged_decode`.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-types auto_config -- --nocapture` PASS: 45 tests;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS: 80 matched
    tests plus Qwen35 config test.
- GPU artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_vllm_h256_smoke_178d76fa_20260619T181820Z_hot_target`;
  - Vast instance `41422823` stopped after triage, stop check:
    `cur_state=stopped`, `actual_status=exited`;
  - CUDA build PASS, release binary SHA256:
    `4052134f6abbc1e9971165d386dac1510172d8a0c259d5a0fecffef1f50ad42f`;
  - `ferrum run` smoke PASS and `run_effective_config.json` confirms
    `selected_attention_impl=vllm_paged_attn_v2`,
    `FERRUM_USE_VLLM_PAGED_ATTN=1`,
    `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`,
    `model_capabilities.architecture=qwen3_5_moe`, `head_dim=256`;
  - `ferrum serve` started and `serve_effective_config.json` confirms the
    same vLLM H256/V2 path.
- Serve request results:
  - non-stream chat: HTTP 200, but answer repeated `Paris` until
    `finish_reason=length`;
  - stream chat: HTTP 200, exactly one `[DONE]`, usage present, but also
    finished by length;
  - required tool call: HTTP 200 and parsed `get_weather({"city":"Paris"})`;
  - strict structured output: HTTP 500,
    `model output did not satisfy response_format.json_schema.strict:
    invalid JSON: trailing characters at line 2 column 1`.
- Limitation / next work:
  - this smoke proves the product defaults now route Qwen35 to the intended
    vLLM H256 paged-attention path;
  - it does not satisfy W3 L2/L3/L4/L5 or the final
    `MODEL_RELEASE_GRADE_W3 PASS`;
  - no diagnostic `bench-serve` was run because correctness failed first;
  - next fix target is structured-output/repetition/length quality on the
    product path, not another VPA routing patch.

## 2026-06-19 ZZZ25 — W3 Qwen35 GPU argmax/readback hot-path checkpoint

- Scope:
  - Qwen35 decode hot-path optimization toward the recorded vLLM 80% targets;
  - local source changes and CPU/Rust validation only at this checkpoint;
  - no new CUDA performance result and no `MODEL_RELEASE_GRADE_W3 PASS:
    <out_dir>` was produced.
- Change:
  - Qwen35 backend model now reads typed runtime snapshot
    `FERRUM_GREEDY_ARGMAX` and can return model-side greedy token sentinels
    instead of downloading `[batch, vocab]` logits on eligible decode rows;
  - Qwen35 `decode_batch_with_logits_policy` now supports consistent
    `GreedyArgmax` token masks via backend masked argmax, and falls back to
    full logits for full-logits requests, mixed masked/unmasked rows, or
    inconsistent masks;
  - `LlmExecutor::decode` now passes `LogitsReturnPolicy` through the single
    decode path and uses the model policy decode path instead of always
    falling back to full-logits `decode`;
  - single-request continuous decode now accepts the same one-element greedy
    sentinel as unified/batch decode and validates it with the existing token
    quality checks before appending the token.
- Validation:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    80 matched tests passed;
  - `cargo test -p ferrum-engine continuous_engine -- --nocapture` PASS:
    27 matched tests passed;
  - `cargo check -p ferrum-interfaces -p ferrum-models -p ferrum-engine
    --all-targets` PASS.
- Limitation:
  - this removes a known logits readback/CPU sampling bottleneck, but it is
    not yet same-hardware performance evidence;
  - the larger structural gap remains Qwen35 full-attention KV layout:
    Qwen35 still uses model-private contiguous full-attention KV state instead
    of the shared vLLM-style paged block table/slot mapping path used by
    existing Llama/Qwen3 MoE model implementations.

## 2026-06-19 ZZZ24 — W3 Qwen35 same-host vLLM ShareGPT baseline recorded

- Scope:
  - W3/Qwen35 same-host vLLM baseline capture for the 80% performance target;
  - 1x Vast CUDA host, vLLM first, ASCII ShareGPT 100 prompts, output length
    128, c=1/4/16/32, `--fail-on-error --require-ci --seed 9271
    --n-repeats 3`;
  - this is optimization input, not a Ferrum release-grade performance PASS.
- Artifact:
  - `docs/goals/model-coverage-2026-06-12/artifacts/w3_vllm_sharegpt_baseline_20260619`;
  - source raw artifact copied from:
    `/tmp/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja`;
  - model snapshot:
    `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4@3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`;
  - dataset SHA256:
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`.
- vLLM baseline and Ferrum targets:
  - c=1: vLLM LCB `134.3690` tok/s, Ferrum 80% target `107.4952`
    tok/s;
  - c=4: vLLM LCB `405.0572` tok/s, Ferrum 80% target `324.0457`
    tok/s;
  - c=16: vLLM LCB `1120.2993` tok/s, Ferrum 80% target `896.2394`
    tok/s;
  - c=32: vLLM LCB `1687.3965` tok/s, Ferrum 80% target `1349.9172`
    tok/s.
- Ferrum diagnostic:
  - the matching Ferrum sweep was stopped early after c=1 repeat 1 because
    `100 completed / 0 errored / 340.3s` implied only about `37.6` output
    tok/s, far below the c=1 80% target;
  - this partial result is a bottleneck signal only and is not used as a
    release-grade performance result.
- Cost/lifecycle:
  - the Vast instance was stopped after artifact copyback and confirmed as
    `cur_state=stopped actual_status=exited`.
- Limitation:
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced;
  - next work is GPU-first optimization toward these 80% targets, followed by
    a fresh full same-host Ferrum/vLLM A/B or a valid final manifest.

## 2026-06-18 ZZZ23 — W3 Qwen35 release-grade goal doc checkpoint

- Scope:
  - W3/Qwen3.5-Qwen3.6 release-grade execution contract and handoff context;
  - documentation-only checkpoint, no code/kernel changes and no GPU execution;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `W3_QWEN35_RELEASE_GRADE_GOAL.md`;
  - records the current official Qwen3.5/Qwen3.6 model facts, current
    checkpoint `3b2b55cf`, completed gates, missing real-model gates, and the
    concrete L0-L5 plus 80% performance acceptance criteria;
  - explicitly names `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` as the practical 1x4090
    real validation lane for the Qwen3.5 family.
- Limitation:
  - this is a planning/handoff artifact only; W3 still needs real backend
    full-attention repair, sparse-MoE backend wiring, real Qwen3.5 GPTQ
    `ferrum run`/`ferrum serve` correctness, L3-L5, and performance evidence.

## 2026-06-18 ZZZ22 — W3 Qwen35 product-path typed recurrent manager checkpoint

- Scope:
  - W3/Qwen35 product-path recurrent-state manager selection;
  - local CPU/Rust tests only, no GPU/CUDA/Metal execution was started;
  - no real Qwen3.5/Qwen3.6 product gate, no performance evidence, and no
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - `EngineBuilder` now installs
    `Qwen35RecurrentStateManager<CpuBackend>` as the default recurrent-state
    manager when the typed product config has `qwen35_reference=true` on CPU;
  - existing CPU non-Qwen35 paths still use `InMemoryRecurrentStateManager`;
  - added a builder test that allocates through the default manager and verifies
    the returned handle is a typed `Qwen35RecurrentStateHandle<CpuBackend>`.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-engine builder -- --nocapture` PASS:
    11 matched tests passed;
  - `cargo test -p ferrum-cli --test qwen35_reference_product -- --nocapture`
    PASS: `ferrum run` and `ferrum serve --qwen35-reference` toy product-path
    tests passed.
- Limitation:
  - this wires the CPU reference product path to typed recurrent state; W3 still
    needs real full-model backend prefill/decode, L2-L5 correctness artifacts,
    concurrency evidence, and 80% performance evidence.

## 2026-06-18 ZZZ21 — W3 Qwen35 reference recurrent-state writeback checkpoint

- Scope:
  - W3/Qwen35 executor-local recurrent-state correctness plumbing;
  - local CPU/Rust tests only, no GPU/CUDA/Metal execution was started;
  - no real Qwen3.5/Qwen3.6 product gate, no performance evidence, and no
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - `Qwen35W3Executor` reference forward now preserves
    `linear_recurrent_states` alongside logits;
  - reference `prefill` and `decode` write DeltaNet final state into
    `Qwen35RecurrentStateHandle<CpuBackend>` when a typed handle is supplied;
  - non-Qwen35 recurrent handles remain pass-through so current product smoke
    paths are not broken before the engine default manager is switched;
  - added tests proving typed recurrent state is populated after prefill and
    updated after decode.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35_w3_reference -- --nocapture` PASS:
    8 matched tests passed;
  - `cargo test -p ferrum-models recurrent_state -- --nocapture` PASS:
    9 matched tests passed.
- Limitation:
  - this removes the executor-level "handle only" gap for the CPU reference
    path; W3 still needs product-path typed manager wiring, real full-model
    backend prefill/decode, L2-L5 correctness artifacts, and 80% performance
    evidence.

## 2026-06-18 ZZZ20 — W3 L2 quantized artifact gate checkpoint

- Scope:
  - W3 L2 real-size quantized semantic correctness artifact packaging;
  - local gate/schema work only, no GPU/CUDA/Metal execution was started;
  - no real Qwen3.5 GPTQ known-answer report was produced, and no
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_l2_quantized_gate.py`;
  - the gate consumes a real known-answer report and writes
    `w3_l2_quantized.json` in the shape accepted by
    `model_release_grade_goal_gate.py`;
  - the gate rejects toy/fixture reports, waived lanes, hidden env, fewer than
    10 known-answer cases, partial semantic pass, and reports missing either
    `ferrum run` or `ferrum serve` product commands;
  - HF/model metadata alone cannot be converted into an L2 PASS artifact.
- Validation:
  - `python3 -m py_compile scripts/release/w3_l2_quantized_gate.py` PASS;
  - `python3 scripts/release/w3_l2_quantized_gate.py --self-test` PASS:
    `W3 L2 QUANTIZED SELFTEST PASS`;
  - synthetic CLI-mode artifact probe PASS:
    `W3 L2 QUANTIZED PASS: /tmp/ferrum-w3-l2-probe-oKme52/out`;
  - final-gate L2 structure probe PASS:
    `W3 L2 FINAL-GATE STRUCTURE PASS: /tmp/ferrum-w3-l2-probe-oKme52/out`.
- Limitation:
  - this closes only the L2 artifact gate gap; W3 still needs a real
    full-size Qwen3.5/Qwen3.6 quantized product run with known-answer
    semantics, plus L3/L4/L5, same-hardware baseline, and 80% performance.

## 2026-06-18 ZZZ19 — W3 L1 numeric artifact gate checkpoint

- Scope:
  - W3 L1 numeric/reference artifact packaging for the Qwen3.5/Qwen3.6 W3
    architecture family;
  - local CPU/Rust test execution only, no GPU/CUDA/Metal execution;
  - no W3 L2-L5 correctness evidence, no real model performance evidence, and
    no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_l1_numeric_gate.py`;
  - the gate runs `cargo test -p ferrum-models qwen35 -- --nocapture`;
  - it verifies that the Rust test log covers the final-gate-required W3 L1
    component categories: linear attention, full attention, DeltaNet, MoE/dense
    path, and LM head;
  - it writes `w3_l1_numeric.json` in the shape accepted by
    `model_release_grade_goal_gate.py`.
- Validation:
  - `python3 -m py_compile scripts/release/w3_l1_numeric_gate.py` PASS;
  - `python3 scripts/release/w3_l1_numeric_gate.py --self-test` PASS:
    `W3 L1 NUMERIC SELFTEST PASS`;
  - `git diff --check -- scripts/release/w3_l1_numeric_gate.py` PASS;
  - real L1 gate PASS:
    `W3 L1 NUMERIC PASS:
    docs/goals/model-coverage-2026-06-12/artifacts/w3_l1_numeric_qwen35_family_20260618`;
  - final-gate L1 structure probe PASS:
    `W3 L1 FINAL-GATE STRUCTURE PASS:
    docs/goals/model-coverage-2026-06-12/artifacts/w3_l1_numeric_qwen35_family_20260618`;
  - artifact records `comparisons_total=14`, `comparisons_passed=14`, and all
    required L1 coverage booleans true.
- Limitation:
  - this is a packaged Rust CPU reference numeric gate; it does not prove W3
    quantized semantics, real-model behavior/tool/schema correctness,
    concurrency, CUDA/Metal execution, or 80% performance.

## 2026-06-18 ZZZ18 — W3 real Qwen3.5/Qwen3.6 L0 template checkpoint

- Scope:
  - real W3 L0 chat-template/tokenizer golden evidence for
    `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B`;
  - local CPU/Rust test execution only, no GPU/CUDA/Metal execution;
  - no W3 L1-L5 correctness evidence, no real model `ferrum run`/`ferrum
    serve` release-grade evidence, and no `MODEL_RELEASE_GRADE_W3 PASS:
    <out_dir>` was produced.
- Change:
  - extended `scripts/gen_chat_template_goldens.py` so W3 target fixtures
    include official HF `generation_config.json`, `tokenizer_config.json`,
    and a compact generated `tokenizer_special_tokens.json` sidecar;
  - avoided checking in full 12 MB `tokenizer.json` files while still recording
    EOS/BOS token-id provenance from `generation_config.json` plus tokenizer
    special-token ids;
  - changed the default generated W3 models to `Qwen/Qwen3.5-35B-A3B` and
    `Qwen/Qwen3.6-35B-A3B`;
  - extended `scripts/release/w3_l0_template_gate.py` to auto-discover fixture
    sidecars and record the token-id source explicitly;
  - added checked-in HF `apply_chat_template` golden fixtures for both W3
    target models, each with `single`, `system`, `multi_turn`, `tools`, and
    `think_history` cases.
- Validation:
  - fixture generation PASS:
    `uv run --with transformers --with jinja2 --with huggingface-hub --with
    socksio python scripts/gen_chat_template_goldens.py
    Qwen/Qwen3.5-35B-A3B Qwen/Qwen3.6-35B-A3B`;
  - `python3 -m py_compile scripts/gen_chat_template_goldens.py
    scripts/release/w3_l0_template_gate.py` PASS;
  - `python3 scripts/release/w3_l0_template_gate.py --self-test` PASS:
    `W3 L0 TEMPLATE SELFTEST PASS`;
  - `git diff --check -- scripts/gen_chat_template_goldens.py
    scripts/release/w3_l0_template_gate.py` PASS;
  - real Qwen3.5 L0 gate PASS:
    `W3 L0 TEMPLATE PASS:
    docs/goals/model-coverage-2026-06-12/artifacts/w3_l0_qwen35_35b_a3b_20260618`;
  - real Qwen3.6 L0 gate PASS:
    `W3 L0 TEMPLATE PASS:
    docs/goals/model-coverage-2026-06-12/artifacts/w3_l0_qwen36_35b_a3b_20260618`;
  - final-gate L0 structure probes PASS for both artifact directories.
- Limitation:
  - this proves only W3 L0 for the two target model families;
  - W3 L1 single-layer/model numeric evidence, L2 quantized semantics, L3/L4
    behavior/tool/schema gates, L5 concurrency, same-hardware baseline, and
    80% performance evidence remain incomplete.

## 2026-06-18 ZZZ17 — W3 L0 template artifact generator checkpoint

- Scope:
  - W3 L0 chat-template/tokenizer golden artifact generation path;
  - local CPU/Rust test execution only, no GPU/CUDA/Metal execution;
  - no real Qwen3.5/Qwen3.6 L0 PASS and no real `MODEL_RELEASE_GRADE_W3
    PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_l0_template_gate.py`;
  - the script validates a checked-in HF `apply_chat_template` golden fixture
    with the existing Rust `ferrum-server` chat-template golden test;
  - it also runs the no-silent-fallback unit test
    `model_template_render_failure_is_an_error_not_a_silent_fallback`;
  - it records a structured `w3_l0_template.json` accepted by the W3 final
    validator, including the five required L0 cases, byte-equality status,
    no hidden env, explicit render-failure behavior, and special-token
    provenance from `generation_config.json`.
- Validation:
  - `python3 -m py_compile scripts/release/w3_l0_template_gate.py` PASS;
  - `python3 scripts/release/w3_l0_template_gate.py --self-test` PASS:
    `W3 L0 TEMPLATE SELFTEST PASS`;
  - `git diff --check -- scripts/release/w3_l0_template_gate.py` PASS;
  - real-mode smoke with existing `Qwen/Qwen3-0.6B` fixture PASS:
    `W3 L0 TEMPLATE PASS: /tmp/ferrum-w3-l0-smoke-fMxTvX/out`;
  - final-gate L0 structure probe on that artifact PASS:
    `W3 L0 FINAL-GATE STRUCTURE PASS:
    /tmp/ferrum-w3-l0-smoke-fMxTvX/out`.
- Limitation:
  - the real-mode smoke used an existing Qwen3 fixture plus temporary
    `generation_config.json`; W3 still needs actual Qwen3.5/Qwen3.6 HF
    golden fixtures and L0 artifact collection before release-grade evidence
    can claim Qwen3.5/Qwen3.6 correctness.

## 2026-06-18 ZZZ16 — W3 L0-L5 correctness artifact gate hardening

- Scope:
  - W3 release-grade correctness validator hardening;
  - no GPU/CUDA/Metal execution was started;
  - no real W3 performance claim and no real `MODEL_RELEASE_GRADE_W3 PASS:
    <out_dir>` was produced.
- Change:
  - `scripts/release/model_release_grade_goal_gate.py` now deep-validates W3
    L0-L5 artifacts instead of accepting shell `status=pass` JSON:
    L0 chat-template golden, L1 numeric/reference coverage, L2 real-size
    quantized semantics, L3 multi-turn/stream/stop behavior, L4 tools plus
    strict JSON schema, and L5 c=1/4/16/32 zero-error concurrency cells;
  - the stricter L0-L5 schema is scoped to W3 so existing W2 artifact formats
    are not broken by this checkpoint;
  - final-gate self-test now includes W3 negative cases for insufficient L4
    strict-schema pass count and nonzero L5 errored requests;
  - `scripts/release/model_release_grade_manifest.py` W3 self-test now emits
    matching structured L0-L5 artifacts before invoking the final validator.
- Validation:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py
    scripts/release/model_release_grade_manifest.py` PASS;
  - `git diff --check -- scripts/release/model_release_grade_goal_gate.py
    scripts/release/model_release_grade_manifest.py` PASS;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS: `MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS, including synthetic `MODEL_RELEASE_GRADE_W2 PASS` and synthetic
    `MODEL_RELEASE_GRADE_W3 PASS`.
- Limitation:
  - synthetic PASS lines remain validator self-tests only; real W3 still needs
    actual model L0-L5 correctness, same-hardware baseline, and c=1/4/16/32
    >=80% performance evidence before release-grade PASS.

## 2026-06-18 ZZZ15 — W3 release-grade manifest builder checkpoint

- Scope:
  - W3 release-grade manifest assembly path;
  - no GPU/CUDA/Metal execution was started;
  - no real W3 performance claim and no real `MODEL_RELEASE_GRADE_W3 PASS:
    <out_dir>` was produced.
- Change:
  - extended `scripts/release/model_release_grade_manifest.py` from W2-only to
    lane-aware `w2`/`w3`;
  - W3 mode requires explicit paths for S0 design, S0 microbench, S1
    single-layer, S2 product path, L0-L5 correctness artifacts, `ferrum run`,
    `ferrum serve`, hardware/runtime/git/binary evidence, Ferrum bench report,
    baseline bench report, and all command-line evidence;
  - W3 perf assembly reuses the release bench schema and requires c=1/4/16/32,
    usage-counted outputs, full completed counts, zero error/quality counts,
    same prompt dataset SHA, same effective concurrency, and bench commands
    suitable for the final validator;
  - builder self-test now creates synthetic W2 and W3 manifests and invokes the
    final validator for both lanes.
- Validation:
  - `python3 -m py_compile scripts/release/model_release_grade_manifest.py`
    PASS;
  - `git diff --check -- scripts/release/model_release_grade_manifest.py`
    PASS;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS, including synthetic `MODEL_RELEASE_GRADE_W2 PASS` and synthetic
    `MODEL_RELEASE_GRADE_W3 PASS`;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS: `MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - W3 missing-args CLI negative test fails as intended and lists all required
    evidence paths.
- Limitation:
  - synthetic manifest PASS lines are validator self-tests only; real W3 still
    needs actual L0-L5 correctness, same-hardware baseline, and c=1/4/16/32
    80% performance evidence before any release-grade claim.

## 2026-06-18 ZZZ14 — W3-S0 design artifact gate checkpoint

- Scope:
  - W3-S0 design evidence generation for the release-grade goal;
  - metadata-only local work, no GPU/CUDA/Metal execution;
  - no performance claim and no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was
    produced.
- Change:
  - added `scripts/release/w3_s0_design_gate.py`;
  - the script validates that `RELEASE_GRADE_GOAL.md` and `W3_CHARTER.md`
    contain the required W3 recurrent-state/paged-KV/ContinuousBatch
    constraints;
  - it writes `w3_s0_design.json` with the recurrent-state manager/spec/handle
    contract, required operations, lifecycle ownership, and coexistence rules
    for paged-KV, ContinuousBatch, preemption, and release;
  - the manifest is shaped for the W3 final validator's
    `w3_s0_design` correctness entry and records `hidden_env=[]`.
- Evidence:
  - artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_s0_design_local_20260617T230640Z/`;
  - script PASS line:
    `W3 S0 DESIGN PASS: docs/goals/model-coverage-2026-06-12/artifacts/w3_s0_design_local_20260617T230640Z`;
  - note: the artifact git summary reports dirty because the local worktree
    already contains many pre-existing untracked historical artifacts, so this
    is S0 design artifact plumbing, not final release-grade evidence.
- Validation:
  - `python3 -m py_compile scripts/release/w3_s0_design_gate.py` PASS;
  - `python3 scripts/release/w3_s0_design_gate.py --self-test` PASS:
    `W3 S0 DESIGN SELFTEST PASS`;
  - `git diff --check -- scripts/release/w3_s0_design_gate.py` PASS;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS: `MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - generated artifact accepted by the final validator's S0 design structure
    probe: `W3 S0 DESIGN ARTIFACT STRUCTURE PASS`.
- Limitation:
  - this closes the S0 design artifact generation gap only; W3 still needs
    real L0-L5 correctness, same-hardware baseline, and c=1/4/16/32 80%
    performance evidence before release-grade PASS.

## 2026-06-18 ZZZ13 — W3 S0/S1/S2 release-grade validator hardening

- Scope:
  - release-grade validator hardening for W3 correctness artifacts;
  - no CUDA/Metal execution was started;
  - no performance claim and no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was
    produced.
- Change:
  - `scripts/release/model_release_grade_goal_gate.py` now deep-validates W3
    S0 design evidence, S0 CUDA delta-rule microbench evidence, S1 single-layer
    compare evidence, and the existing S2 product-path evidence;
  - S0 design evidence must record recurrent-state cache trait/spec semantics
    and coexistence with paged-KV, ContinuousBatch, preemption, and release;
  - S0 microbench evidence must be CUDA mode, include PTX arch, clean git
    summary, deterministic input distribution, reference formula, compile/run
    commands, binary SHA256, and error stats within tolerance;
  - S1 evidence must be real compare mode, not self-test-only evidence, with
    passing delta-rule/layer/expert/router/shared-expert checks and per-tensor
    comparison tolerances;
  - self-test fixtures now include structured W3 S0/S1/S2 artifacts and negative
    cases for S0 tolerance failure, S1 self-test pass-line misuse, and S2 missing
    stream usage.
- Validation:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `git diff --check -- scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS: `MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - current W3 S0 artifact structure probe PASS:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_delta_rule_s0_cuda_20260617T203149Z_c8b8da1f/w3_delta_rule_s0_microbench_manifest.json`;
  - current W3 S1 artifact structure probe PASS:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_deltanet_s1_rust_compare_20260617T130232Z_1b480a31/compare/w3_deltanet_s1_layer_compare_manifest.json`.
- Limitation:
  - this improves the final gate's ability to reject weak W3 evidence; real
    W3 L0-L5 correctness, same-hardware baseline, and c=1/4/16/32 80%
    performance evidence remain incomplete.

## 2026-06-18 ZZZ12 — W3-S2 product smoke artifact script checkpoint

- Scope:
  - W3-S2 artifact-producing product-path smoke for the explicit Qwen3.5 CPU/FP32
    reference executor;
  - no CUDA/Metal execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_qwen35_product_smoke.py`;
  - the script writes a local toy Qwen3.5 safetensors model without requiring
    extra Python packages, runs real `ferrum run`, starts real `ferrum serve`,
    exercises non-streaming and streaming `/v1/chat/completions`, and writes
    `w3_s2_whole_model_product_path.json`;
  - the generated W3-S2 manifest records typed CLI commands, `hidden_env=[]`,
    run JSONL output, serve log, non-stream response, streaming SSE response,
    usage-bearing stream chunk, and exactly one `[DONE]`.
- Evidence:
  - artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_product_smoke_local_20260617T222748Z/`;
  - script PASS line:
    `W3 QWEN35 PRODUCT SMOKE PASS: /Users/chejinxuan/rust_ws/ferrum-infer-rs/docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_product_smoke_local_20260617T222748Z`;
  - note: the artifact git summary reports dirty because the local worktree
    already contains many pre-existing untracked historical artifacts, so this
    remains a toy diagnostic W3-S2 artifact, not release-grade evidence.
- Validation:
  - `python3 -m py_compile scripts/release/w3_qwen35_product_smoke.py` PASS;
  - `python3 scripts/release/w3_qwen35_product_smoke.py --out <out>` PASS;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS: `MODEL RELEASE GRADE GOAL SELFTEST PASS`.
- Limitation:
  - this does not satisfy real Qwen3.5/Qwen3.6 L0-L5 correctness;
  - it does not provide CUDA/Metal execution, same-hardware baseline, or W3
    80% performance evidence.

## 2026-06-18 ZZZ11 — W3 Qwen3.5 reference `ferrum serve` product smoke checkpoint

- Scope:
  - W3-S2 Qwen3.5/Qwen3.6 explicit CPU/FP32 reference execution through both
    real product entrypoints now covered by toy smoke: `ferrum run` and
    `ferrum serve`;
  - no CUDA/Metal execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - extended the Qwen3.5 reference product integration test to spawn a real
    `ferrum serve <model_dir> --host 127.0.0.1 --port <ephemeral>
    --backend cpu --qwen35-reference` subprocess;
  - the test reuses the local toy Qwen3.5 `config.json`, `tokenizer.json`,
    and `model.safetensors` fixture used by the `ferrum run` smoke;
  - non-streaming `/v1/chat/completions` now asserts HTTP success,
    returned model id, `finish_reason=length`, and non-empty content;
  - streaming `/v1/chat/completions` now sends
    `stream_options.include_usage=true` and asserts at least one content delta,
    one usage-bearing chunk, and exactly one `data: [DONE]`.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-cli --test qwen35_reference_product -- --nocapture`
    PASS: `2 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `56 passed`, plus Qwen3.5 config integration coverage `1 passed`;
  - `cargo test -p ferrum-engine qwen35_registry -- --nocapture` PASS:
    `2 passed`;
  - `cargo check -p ferrum-engine -p ferrum-cli` PASS.
- Limitation:
  - this is still a toy CPU/FP32 reference product smoke only;
  - real Qwen3.5/Qwen3.6 model L0-L5 correctness, CUDA/Metal execution,
    release-grade baseline comparison, and W3 80% performance gates remain
    incomplete.

## 2026-06-18 ZZZ10 — W3 Qwen3.5 reference `ferrum run` product smoke checkpoint

- Scope:
  - W3-S2 Qwen3.5/Qwen3.6 explicit CPU/FP32 reference execution through the
    real `ferrum run` product entrypoint;
  - no CUDA/Metal execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added a CLI integration smoke that creates a local toy Qwen3.5 model
    directory with `config.json`, `tokenizer.json`, and `model.safetensors`;
  - the smoke runs the real `ferrum` binary with `run <model_dir> --backend
    cpu --qwen35-reference --output-format jsonl --temperature 0
    --max-tokens 2 --prompt hello`;
  - the assertion checks a successful assistant JSONL event with
    `finish_reason=length`, two generated tokens, and non-empty decoded text;
  - product-path execution exposed a generic engine abstraction gap:
    recurrent-state-capable executors could declare a recurrent-state spec,
    but the default builder did not provide a recurrent-state manager;
  - `EngineBuilder` now installs the default in-memory recurrent-state
    manager for CPU/reference engines when no custom manager is supplied,
    while keeping custom manager overrides intact and leaving GPU backends to
    provide backend-native recurrent-state managers explicitly.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-cli --test qwen35_reference_product -- --nocapture`
    PASS: `1 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `56 passed`, plus Qwen3.5 config integration coverage `1 passed`;
  - `cargo test -p ferrum-engine qwen35_registry -- --nocapture` PASS:
    `2 passed`;
  - `cargo check -p ferrum-engine -p ferrum-cli` PASS.
- Limitation:
  - this is a toy CPU/FP32 reference product smoke only;
  - real Qwen3.5/Qwen3.6 model L0-L5 correctness, `ferrum serve` product
    smoke, CUDA/Metal execution, and W3 80% performance gates remain
    incomplete.

## 2026-06-18 ZZZ9 — W3 Qwen3.5 reference decode replay checkpoint

- Scope:
  - W3-S2 CPU/FP32 reference executor decode semantics after explicit product
    entry wiring;
  - no CUDA/Metal execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - `Qwen35W3Executor` now keeps per-reference-cache token history keyed by
    the returned `GenericKvCacheHandle` cache id;
  - reference `prefill()` records prompt/chunk history and preserves incoming
    recurrent-state handles;
  - reference `decode()` now accepts one token, validates the cache history and
    KV sequence length, replays the dense or sparse-MoE CPU reference model on
    the full sequence, returns `[1, vocab]` logits, and advances KV length;
  - unknown/mismatched reference cache histories are rejected instead of
    silently fabricating state.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `56 passed`, plus Qwen3.5 config integration coverage `1 passed`;
  - `cargo test -p ferrum-engine qwen35_registry -- --nocapture` PASS:
    `2 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine -p ferrum-cli` PASS.
- Limitation:
  - decode is still full-sequence CPU reference replay, not the final
    incremental recurrent-state/KV implementation needed for performance;
  - full `ferrum run` / `ferrum serve` W3 product scenarios, W3 L0-L5
    correctness gates, and W3 80% performance gates remain incomplete.

## 2026-06-18 ZZZ8 — W3 explicit Qwen3.5 reference product-entry checkpoint

- Scope:
  - W3-S2 controlled product-entry bridge for Qwen3.5/Qwen3.6 reference
    execution through the existing `run`/`serve` -> `EngineConfig` ->
    registry -> executor abstraction;
  - no CUDA/Metal product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added user-visible `--qwen35-reference` to `ferrum run` and `ferrum
    serve`;
  - the flag lands as a typed backend option on `EngineConfig`, not as a
    hidden environment-variable combination;
  - `LlmExecutorFactory` now recognizes `Architecture::Qwen35` and
    `Architecture::Qwen35Moe` behind that explicit flag;
  - default Qwen3.5/Qwen3.6 product loading still rejects with a clear
    unsupported message;
  - the explicit path is restricted to CPU/FP32 reference execution and
    materializes dense or sparse-MoE reference runtimes from the existing
    safetensors inventory/weight-plan abstraction.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-cli qwen35_reference -- --nocapture` PASS:
    `2 passed`;
  - `cargo test -p ferrum-engine qwen35_registry -- --nocapture` PASS:
    `2 passed`;
  - `cargo check -p ferrum-engine -p ferrum-cli` PASS.
- Limitation:
  - this proves controlled reference loading and prefill at the registry
    boundary only;
  - decode/recurrent-state incremental semantics, full `ferrum run`/`ferrum
    serve` W3 product scenarios, W3 L0-L5 correctness gates, and W3 80%
    performance gates remain incomplete.

## 2026-06-18 ZZZ7 — W3 sparse-MoE reference runtime/materializer checkpoint

- Scope:
  - W3-S2 Qwen3.5/Qwen3.6 sparse-MoE reference full-model forward and
    safetensors materialization;
  - no product registry wiring was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added typed `norm_topk_prob` parsing to `Qwen35MoeTextConfig`, with
    model config override and Qwen3-MoE compatible default;
  - added `Qwen35SparseMoeReferenceModel` and
    `qwen35_sparse_moe_reference_model_forward_cpu()`;
  - the sparse-MoE reference model composes embeddings, linear/full attention
    layers, router/fused experts/shared expert, final RMSNorm+1, and lm_head;
  - added `Qwen35SparseMoeReferenceRuntime::from_cpu_weight_plan()`;
  - added explicit CPU/FP32
    `Qwen35W3Executor::from_definition_with_sparse_moe_reference_cpu_safetensors()`;
  - `Qwen35W3Executor::prefill()` can now use either dense or sparse-MoE
    reference runtime while default product execution remains disabled.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `54 passed`;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `6 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference materialization only;
  - decode/recurrent-state runtime semantics, product `ferrum run`/`ferrum
    serve`, W3 L0-L5 correctness gates, and W3 80% performance gates remain
    incomplete.

## 2026-06-18 ZZZ6 — W3 S0 native CUDA microbench checkpoint

- Scope:
  - W3-S0 Qwen3.5 delta-rule native CUDA minimal validation on 1x RTX 4090;
  - same SHA clean detached remote worktree:
    `c8b8da1f41ff346809d7bdc88476c755846cdc83`;
  - this does not enable product registry wiring and does not produce
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`.
- Evidence:
  - artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_delta_rule_s0_cuda_20260617T203149Z_c8b8da1f/`;
  - native CUDA/Python reference validator PASS:
    `W3 DELTA RULE S0 MICROBENCH PASS: /workspace/w3_delta_rule_s0_cuda_20260617T203149Z_c8b8da1f`;
  - remote GPU: `NVIDIA GeForce RTX 4090, 24564 MiB, 570.195.03`;
  - remote CUDA compiler:
    `Build cuda_12.4.r12.4/compiler.34097967_0`.
- Validation:
  - `python3 scripts/release/w3_delta_rule_s0_microbench.py --cuda --out <out>`
    PASS;
  - remote minimal Rust install was required because the CUDA devel container
    had no `cargo`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS on the same clean
    worktree:
    `51 passed; 0 failed`, plus `parses_official_qwen35_dense_min_config`
    integration coverage `1 passed`.
- GPU lifecycle:
  - Vast instance `41287720` was stopped after artifact collection;
  - final Vast API cleanup check is saved under
    `local_vast/vast_cleanup_summary.json` and reports
    `cur_state=stopped`, `actual_status=exited`.
- Limitation:
  - this is S0 CUDA kernel-level evidence plus reference executor tests only;
  - Qwen3.5 MoE full-model reference forward, MoE safetensors materialization,
    decode/recurrent-state semantics, product `ferrum run`/`ferrum serve`, W3
    correctness gates, and W3 performance gates remain incomplete.

## 2026-06-18 ZZZ5 — W3 sparse-MoE layer composition checkpoint

- Scope:
  - W3-S1/S2 Qwen3.5 sparse-MoE layer composition before full MoE model
    forward/product wiring;
  - no product registry wiring was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35SparseMoeLinearAttentionLayerShape` and reference output;
  - added `Qwen35SparseMoeFullAttentionLayerShape` and reference output;
  - added `qwen35_sparse_moe_linear_attention_layer_cpu()`;
  - added `qwen35_sparse_moe_full_attention_layer_cpu()`;
  - both paths compose Qwen3.5 RMSNorm+1, linear/full attention, residual,
    post-attention RMSNorm+1, shared-expert sparse MoE, and final residual.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `51 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference layer composition only;
  - MoE full-model reference forward, safetensors materialization into the MoE
    runtime, product `ferrum run`/`ferrum serve`, W3 correctness gates, and W3
    80% performance gates remain incomplete.

## 2026-06-18 ZZZ4 — W3 dense reference safetensors prefill checkpoint

- Scope:
  - W3-S2 dense Qwen3.5 reference executor construction from an actual
    safetensors directory;
  - no product registry wiring was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added
    `Qwen35W3Executor::from_definition_with_dense_reference_cpu_safetensors()`;
  - the constructor keeps the path explicitly FP32 CPU reference-only;
  - it runs safetensors inventory/preflight, resolves the W3 weight plan,
    opens `NativeSafetensorsLoader<CpuBackend>`, materializes the dense
    reference runtime, stores validation/plan evidence on the executor, and
    enables reference `prefill()`;
  - added a temp-safetensors test that writes toy W3 weights, constructs the
    executor from disk, runs `prefill()`, and checks last-token logits plus KV
    sequence length.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `49 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still an explicit CPU-reference constructor, not default product
    support;
  - registry, tokenizer/template product scenarios, `ferrum run`, `ferrum
    serve`, sparse-MoE runtime materialization, decode/recurrent-state
    semantics, and W3 performance gates remain incomplete.

## 2026-06-18 ZZZ3 — W3 dense reference runtime materializer checkpoint

- Scope:
  - W3-S2 dense Qwen3.5 reference runtime construction from the real weight
    planning abstraction;
  - no product registry wiring was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35DenseReferenceRuntime::from_cpu_weight_plan()`;
  - materializes dense W3 reference weights from `Qwen35ResolvedWeightPlan`
    plus `WeightLoader<CpuBackend>`;
  - uses plan roles for embeddings, final norm, optional tied/untied lm head,
    linear-attention projections/state parameters, full-attention projections,
    q/k norms, and dense MLP projections;
  - preserves `norm_eps` and RoPE theta as explicit constructor inputs instead
    of hard-coded product behavior.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `48 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - dense safetensors CPU-reference materialization is now test-covered, but
    product `ferrum run`/`ferrum serve` are still not enabled;
  - sparse-MoE runtime materialization and decode/recurrent-state semantics are
    still incomplete;
  - W3 correctness gates and W3 80% performance gates remain incomplete.

## 2026-06-18 ZZZ2 — W3 dense reference executor prefill checkpoint

- Scope:
  - W3-S2 executor-level dense Qwen3.5 reference prefill boundary;
  - no product registry wiring was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added explicit `Qwen35DenseReferenceRuntime` owned reference weights;
  - added `Qwen35W3Executor::with_dense_reference_runtime()`;
  - default `Qwen35W3Executor::from_definition()` still keeps product
    prefill/decode unsupported;
  - reference-mode `prefill()` now extracts input tokens, runs the dense
    CPU-reference model forward, returns last-token logits as `[1, 1, vocab]`,
    and returns a `GenericKvCacheHandle` with the prompt sequence length;
  - `decode()` remains unsupported until recurrent-state/KV semantics are
    wired instead of faked.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `47 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference prefill only;
  - sparse-MoE model-level executor path is not wired;
  - product `ferrum run`, `ferrum serve`, W3 correctness gates, and W3 80%
    performance gates remain incomplete.

## 2026-06-18 ZZZ1 — W3 dense model CPU-reference forward checkpoint

- Scope:
  - W3-S2 dense Qwen3.5 model-level CPU reference before product executor
    wiring;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35DenseReferenceModel`;
  - added reference layer descriptors for dense linear-attention and
    full-attention layers;
  - added `qwen35_dense_reference_model_forward_cpu()`;
  - the reference forward gathers embeddings, runs linear/full attention
    reference layers in order, applies final RMSNorm+1, and emits lm-head
    logits;
  - captures per-layer hidden states and final recurrent state for each
    linear-attention layer so future executor wiring can compare layer
    boundaries, not only final logits.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `44 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - MoE model-level forward is not implemented in this checkpoint;
  - product `prefill`/`decode`, `ferrum run`, and `ferrum serve` remain
    unwired for W3.

## 2026-06-18 ZZZ — W3 sparse-MoE shared-expert CPU-reference checkpoint

- Scope:
  - W3-S1/S2 Qwen3.5 sparse MoE/shared-expert reference before product
    forward;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35SparseMoeShape`;
  - added `Qwen35SparseMoeReference`;
  - added `qwen35_sparse_moe_shared_expert_cpu()`;
  - the helper uses Ferrum's stable MoE `route()` for top-k ids/weights;
  - fixed CPU reference layout for fused routed experts as
    `[experts, 2 * expert_intermediate, hidden]` for gate/up and
    `[experts, hidden, expert_intermediate]` for down;
  - materializes routed expert output, shared expert gate, shared expert
    output, and final `routed + shared` MoE output.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `42 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - sparse MoE is not yet wired into product `prefill`/`decode`;
  - full W3 model execution, correctness gates, and performance gates remain
    incomplete.

## 2026-06-18 ZZY — W3 dense full-attention layer CPU-reference checkpoint

- Scope:
  - W3-S2 Qwen3.5 dense full-attention decoder layer reference before
    product forward;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35FullAttentionShape`;
  - added `Qwen35FullAttentionReference`;
  - added `Qwen35DenseFullAttentionLayerShape`;
  - added `Qwen35DenseFullAttentionLayerReference`;
  - added `qwen35_full_attention_core_cpu()` with q/k RMSNorm, non-interleaved
    RoPE, GQA head repeat, causal softmax, and token-major context output;
  - added `qwen35_dense_full_attention_layer_cpu()` to compose input
    RMSNorm+1, q/k/v projections, full-attention core, `o_proj`, residual,
    post-attention RMSNorm+1, dense SwiGLU MLP, and final residual;
  - added shared CPU helpers for standard RMSNorm and dense SwiGLU MLP.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `40 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - sparse MoE/shared-expert execution is not implemented here;
  - product `prefill`/`decode` remains unwired;
  - no W3 product correctness or performance gate was run.

## 2026-06-18 ZZX — W3 dense linear-attention layer CPU-reference checkpoint

- Scope:
  - W3-S2 Qwen3.5 dense decoder layer reference before product forward;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35DenseLinearAttentionLayerShape`;
  - added `Qwen35DenseLinearAttentionLayerReference`;
  - added `qwen35_dense_linear_attention_layer_cpu()`;
  - the layer reference composes input RMSNorm+1, qkv/z/a/b projections,
    linear-attention core, attention `out_proj`, residual, post-attention
    RMSNorm+1, dense SwiGLU MLP, and final residual;
  - added CPU helpers for row-major linear projection and Qwen3.5 RMSNorm+1.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `34 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - sparse MoE layer execution and full-attention layer execution are not
    implemented here;
  - product `prefill`/`decode` remains unwired;
  - no W3 product correctness or performance gate was run.

## 2026-06-18 ZZW — W3 linear-attention CPU-reference core checkpoint

- Scope:
  - W3-S2 Qwen3.5 linear-attention core reference before product forward;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35LinearAttentionShape`;
  - added `Qwen35LinearAttentionReference`;
  - added `qwen35_linear_attention_core_cpu()`;
  - the helper composes causal depthwise conv + SiLU, q/k/v split,
    GDN gating, recurrent DeltaNet update, and gated RMSNorm;
  - added public CPU reference helpers for Qwen3.5 depthwise conv, q/k/v
    split, and gated RMSNorm;
  - kept Ferrum recurrent state layout `[value_heads, value_dim, key_dim]`.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `30 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - no `out_proj`, residual, MLP, or product `prefill`/`decode` path has
    been wired here;
  - no W3 product correctness or performance gate was run.

## 2026-06-18 ZZV — W3 GDN attention CPU-reference checkpoint

- Scope:
  - W3-S2 combined Gated DeltaNet attention reference before product forward;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `qwen35_gated_delta_attention_cpu()`;
  - the helper composes `qwen35_gdn_gating_cpu()` with
    `qwen35_recurrent_gated_delta_rule_cpu()`;
  - inputs use projected q/k/v plus a/b/A_log/dt_bias and the Ferrum
    recurrent state layout `[value_heads, value_dim, key_dim]`;
  - returns both attention output and final recurrent state.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `26 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - no depthwise conv/projection wrapper or product `prefill`/`decode` has
    been wired yet;
  - no W3 product correctness or performance gate was run.

## 2026-06-18 ZZU — W3 GDN gating CPU-reference checkpoint

- Scope:
  - W3-S2 Qwen Gated DeltaNet reference math before backend kernel wiring;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Source comparison:
  - checked vLLM CPU/reference GDN path:
    `g = -exp(A_log) * softplus(a + dt_bias)`;
  - checked `beta = sigmoid(b)` from the same path.
- Change:
  - added `qwen35_gdn_gating_cpu()`;
  - validates `A_log`, `dt_bias`, `a`, and `b` lengths;
  - returns `g` and `beta` in `[tokens, value_heads]` layout;
  - shares the reference with the recurrent DeltaNet CPU path added in the
    previous checkpoint.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `25 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still CPU/reference math only;
  - it is not wired into product `prefill`/`decode`;
  - no CUDA/Metal kernel or product correctness gate was run.

## 2026-06-18 ZZT — W3 DeltaNet recurrent CPU-reference checkpoint

- Scope:
  - W3-S2 DeltaNet state-update reference before backend kernel wiring;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Source comparison:
  - checked local vLLM
    `model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py`;
  - matched the recurrent update order:
    decay state, compute `kv_mem`, compute beta-scaled delta, update state
    with `delta outer k`, then produce output from updated state and scaled q;
  - kept q/k head repeat semantics for Qwen3.6 value-head layouts.
- Change:
  - added `Qwen35DeltaRuleShape`;
  - added `qwen35_recurrent_gated_delta_rule_cpu()`;
  - function accepts Ferrum state layout `[value_heads, value_dim, key_dim]`
    and returns output plus final state;
  - supports optional q/k L2 normalization and explicit scale;
  - validates all tensor lengths before compute.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `23 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is a CPU/reference path, not the CUDA/Metal product kernel;
  - it is not wired into `prefill`/`decode` yet;
  - no W3 product entrypoint or performance evidence exists.

## 2026-06-18 ZZS — W3 model-side recurrent-state cache checkpoint

- Scope:
  - W3-S2 recurrent-state runtime storage prerequisite;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35RecurrentStateCache`;
  - added `Qwen35RecurrentStateTensor`;
  - model-side recurrent state can now be allocated from
    `RecurrentStateSpec`;
  - each recurrent tensor records layer index, state name, shape,
    elements-per-slot, and backend buffer;
  - added slot-range calculation so future DeltaNet updates can write only
    the active request slot;
  - cache accounting reports total elements and dtype-based estimated memory.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `20 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not implement DeltaNet state update kernels;
  - this does not implement Qwen3.5/Qwen3.6 prefill/decode;
  - this does not register W3 product execution.

## 2026-06-18 ZZR — W3 runtime-config contract checkpoint

- Scope:
  - W3-S2 runtime contract after typed weight materialization;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - `Qwen35ModelWeights` now carries an explicit `LlmRuntimeConfig`;
  - added `qwen35_runtime_config()` for explicit text-config + vocab/max-seq
    construction;
  - added `qwen35_runtime_config_from_definition()` so product wiring can
    derive scheduler-facing hidden size, layer count, KV heads, head dim,
    vocab size, and max sequence length from `ModelDefinition`;
  - kept `DecoderOnlyLLM` unimplemented because `prefill/decode` cannot
    safely return an unsupported error through that trait today.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `18 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not implement Qwen3.5/Qwen3.6 prefill/decode;
  - this does not register W3 product execution;
  - next W3-S2 work is recurrent-state/runtime storage plus DeltaNet/full
    attention forward wiring.

## 2026-06-18 ZZQ — W3 typed model-weight materialization checkpoint

- Scope:
  - W3-S2 materialization boundary after role-aware weight loading;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `models::qwen35`;
  - added `Qwen35ModelWeights`;
  - added typed layer weights for linear-attention layers, full-attention
    layers, dense MLP layers, and sparse MoE/shared-expert layers;
  - materialization now uses `Qwen35WeightPlanLoader` plus the existing
    backend `WeightLoader<B>` abstraction;
  - dense tied `lm_head` falls back to the embedding linear path, matching the
    existing Qwen3/Qwen3-MoE loader convention;
  - sparse MoE fused expert tensors are loaded as raw backend buffers instead
    of pretending they are rank-2 linears.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `17 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not implement Qwen3.5/Qwen3.6 prefill/decode;
  - this does not register W3 product execution;
  - recurrent-state update and DeltaNet/full-attention forward wiring remain
    the next W3-S2 blockers.

## 2026-06-18 ZZP — W3 role-aware weight-loader adapter checkpoint

- Scope:
  - W3-S2 bridge from resolved weight plan to the existing backend
    `WeightLoader` abstraction;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35WeightPlanLoader`;
  - the adapter loads global and layer tensors by semantic role instead of
    repeating full safetensors names at each call site;
  - the adapter delegates tensor and linear materialization to the existing
    backend `WeightLoader<B>` path;
  - `load_*_linear()` strips the `.weight` suffix before calling
    `WeightLoader::load_linear()`, matching the existing Qwen3/Qwen3-MoE
    loader contract;
  - absent optional tied weights, such as dense `lm_head`, now fail with a
    role-specific error when accidentally loaded directly.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35_weights -- --nocapture` PASS:
    `6 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `15 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not implement Qwen3.5/Qwen3.6 prefill/decode;
  - this does not register W3 product execution;
  - it is the loader abstraction needed before materializing W3 model weights.

## 2026-06-18 ZZO — W3 resolved weight-plan checkpoint

- Scope:
  - W3-S2 loader/executor bridge after safetensors preflight;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35ResolvedWeightPlan`;
  - added resolved global/layer tensor specs with `role`, concrete tensor
    name, required flag, and present flag;
  - wildcard optional expert aliases now resolve to concrete names when a
    safetensors inventory contains them;
  - missing optional tied `lm_head` remains represented as absent instead of
    failing preflight;
  - `Qwen35W3Executor::from_definition_with_weight_preflight()` now stores the
    resolved weight plan alongside the validation summary;
  - added `weight_plan()` and `layer_tensor()` lookup helpers for the next
    tensor materialization step.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35_weights -- --nocapture` PASS:
    `5 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `14 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not materialize tensor data into backend weights;
  - this does not run prefill/decode;
  - registry still rejects Qwen3.5/Qwen3.6 product execution.

## 2026-06-18 ZZN — W3 executor weight-preflight boundary checkpoint

- Scope:
  - W3-S2 executor construction boundary after safetensors inventory support;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35W3Executor::from_definition_with_weight_preflight()`;
  - the constructor reads a model directory via `Qwen35WeightInventory`;
  - it validates the typed `Qwen35WeightManifest` before returning an executor;
  - successful validation is retained on the executor as
    `weight_validation()`;
  - missing required tensors now fail during executor construction with a
    specific missing tensor name;
  - `Qwen35W3Executor::from_definition()` remains available for metadata-only
    tests and performs no filesystem IO.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `12 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - registry still rejects Qwen3.5/Qwen3.6 product execution;
  - this does not materialize tensor data into backend weights;
  - this does not run prefill/decode.

## 2026-06-18 ZZM — W3 Qwen3.5/Qwen3.6 safetensors inventory checkpoint

- Scope:
  - W3-S2 loader preflight after the typed weight manifest;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `qwen35_weights`;
  - added `Qwen35WeightInventory`;
  - added `Qwen35WeightValidation`;
  - inventory reads `model.safetensors` headers via mmap without loading tensor
    data;
  - inventory reads `model.safetensors.index.json` `weight_map` and checks that
    referenced shard files exist;
  - validation compares available tensor names against the typed
    `Qwen35WeightManifest`;
  - prefix detection tries `model.language_model` and `model`, returning the
    first prefix with no missing required tensors;
  - missing required tensors now produce an explicit error listing the missing
    Qwen3.5/Qwen3.6 weight names.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35_weights -- --nocapture` PASS:
    `3 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `10 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not materialize tensors into backend weights;
  - this does not run prefill/decode;
  - it is a fast loader preflight for the real W3 safetensors loader.

## 2026-06-18 ZZL — W3 Qwen3.5/Qwen3.6 weight-manifest checkpoint

- Scope:
  - W3-S2 loader/forward prerequisite after typed layer planning;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Source comparison:
  - reused the existing Ferrum Qwen3.5 S1 replay tensor names for dense
    linear-attention and dense MLP layers;
  - checked local vLLM `qwen3_5.py` / `qwen3_next.py` loader mappings for
    full attention, sparse MoE, fused routed experts, and shared expert names.
- Change:
  - added `tie_word_embeddings` to the typed Qwen3.5 config;
  - added `Qwen35WeightSpec`;
  - added `Qwen35LayerWeightManifest`;
  - added `Qwen35WeightManifest`;
  - added `Qwen35TextConfig::weight_manifest(prefix)`;
  - manifest now emits canonical HF tensor names for:
    - global embedding/final norm/lm head;
    - linear-attention QKV/Z/B/A/conv/A_log/dt_bias/norm/out projection;
    - full-attention q/k/v/o projections and q/k norms;
    - dense MLP gate/up/down;
    - MoE router, shared expert gate, shared expert MLP, fused expert gate-up
      and down weights, plus optional per-expert aliases.
- Evidence:
  - dense Qwen3.5 marks `lm_head` optional because the official artifact ties
    embeddings;
  - Qwen3.6 MoE marks `lm_head` required because the official artifact does
    not tie embeddings;
  - Qwen3.6 layer 0 manifest includes linear-attention and MoE fused expert
    weights;
  - Qwen3.6 layer 3 manifest includes full-attention and shared-expert weights.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `6 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `7 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this still does not load tensors into the W3 executor;
  - this does not run prefill/decode;
  - it fixes the loader contract before the loader implementation.

## 2026-06-18 ZZK — W3 Qwen3.5/Qwen3.6 typed layer-plan checkpoint

- Scope:
  - W3-S2 executor wiring prerequisite after the recurrent-state skeleton;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- vLLM source comparison:
  - checked local `/Users/chejinxuan/py_ws/vllm/vllm/model_executor/models/qwen3_5.py`;
  - `Qwen3_5DecoderLayer` uses `QwenGatedDeltaNetAttention` for
    `linear_attention` layers and `Qwen3NextAttention` for `full_attention`;
  - dense `qwen3_5_text` uses `Qwen3NextMLP` on every layer;
  - `qwen3_5_moe_text` uses `Qwen3NextSparseMoeBlock` on every layer, including
    full-attention layers, with the shared expert block.
- Change:
  - added `Qwen35MlpKind`;
  - added `Qwen35LayerPlan`;
  - added `Qwen35TextConfig::layer_plan()`;
  - added `Qwen35TextConfig::mlp_kind_for_layer()`;
  - added `dense_mlp_layers()` and `sparse_moe_layers()` helpers.
- Evidence:
  - dense `Qwen/Qwen3.5-0.8B` now resolves 24 dense-MLP layers and 0 sparse
    MoE layers;
  - MoE/shared-expert `Qwen/Qwen3.6-35B-A3B` now resolves 40 sparse
    MoE/shared-expert MLP layers and 0 dense-MLP layers;
  - full-attention layers keep `has_recurrent_state=false` but still use the
    MoE MLP in the MoE model, matching vLLM.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `6 passed`;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `7 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this is still config/executor-planning evidence;
  - no weights are loaded and no prefill/decode path is implemented yet.

## 2026-06-18 ZZJ — W3 executor skeleton recurrent-state boundary checkpoint

- Scope:
  - W3-S2 executor boundary work for Qwen3.5/Qwen3.6;
  - no product execution was enabled;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added an unregistered `Qwen35W3Executor` skeleton;
  - the skeleton is constructible from `ModelDefinition`, dtype, and device;
  - it implements `ModelExecutor::recurrent_state_spec()` using the shared
    Qwen3.5/Qwen3.6 recurrent-state spec helper;
  - it keeps `prefill` and `decode` explicitly unsupported;
  - `status()` reports `ExecutorState::Error` and `is_ready=false`, so this
    cannot be mistaken for a runnable product executor;
  - it is exported for the future registry/runtime wiring step, but the
    registry still rejects Qwen3.5/Qwen3.6 product execution.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `7 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this does not yet load Qwen3.5/Qwen3.6 weights;
  - this does not update recurrent state during prefill/decode;
  - `ferrum run` and `ferrum serve` remain intentionally unsupported for W3.

## 2026-06-18 ZZI — W3 recurrent-state spec product-boundary checkpoint

- Scope:
  - W3-S2 bridge work from parsed Qwen3.5/Qwen3.6 product configs to the
    `ModelExecutor::recurrent_state_spec()` allocation contract;
  - no product execution was run;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `Qwen35TextConfig::from_model_definition()` so executor/loader code
    can rebuild the typed W3 config from `ModelDefinition.extra_params`
    without re-reading raw HF JSON or duplicating shape parsing;
  - added `Qwen35TextConfig::to_recurrent_state_spec()` to produce the exact
    `RecurrentStateSpec` needed by ContinuousBatch recurrent-state allocation;
  - validated `max_batch_slots > 0`;
  - kept the registry execution path explicitly unsupported for
    Qwen3.5/Qwen3.6 until the real executor is wired, avoiding a misleading
    `ferrum run` / `ferrum serve` partial path.
- Evidence:
  - dense `Qwen/Qwen3.5-0.8B` produces 18 `delta_state` tensors with shape
    `[16, 128, 128]` and BF16 slot memory `18 * 16 * 128 * 128 * 2`;
  - MoE/shared-expert `Qwen/Qwen3.6-35B-A3B` produces 30 `delta_state` tensors
    with shape `[32, 128, 128]` and FP16 slot memory
    `30 * 32 * 128 * 128 * 2`;
  - the crate-local config chain now covers
    `ConfigManager -> ModelDefinition -> Qwen35TextConfig -> RecurrentStateSpec`.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `6 passed`;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `5 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine` PASS.
- Limitation:
  - this does not allocate/update the recurrent state at runtime yet;
  - Qwen3.5/Qwen3.6 product execution remains intentionally unsupported until
    the W3 executor path calls this spec and implements prefill/decode.

## 2026-06-18 ZZH — W3 Qwen3.6 value-head DeltaNet topology correction checkpoint

- Scope:
  - W3-S1/S2 recurrent-state abstraction correction after checking current
    vLLM Qwen3.5/Qwen3-Next Gated DeltaNet source;
  - no product execution was run;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Correction:
  - vLLM stores Gated DeltaNet temporal state as
    `[value_heads, value_head_dim, key_head_dim]`;
  - Qwen3.6 MoE/shared-expert therefore uses `delta_state` shape
    `[32, 128, 128]`, not the earlier grouped-key-head shape
    `[16, 128, 256]`;
  - the total state elements are unchanged, but the layout semantics are
    different and must be modeled before product prefill/decode wiring.
- Change:
  - updated `Qwen35TextConfig::recurrent_delta_state_shape()` to follow the
    vLLM value-head temporal-state layout;
  - updated the W3 DeltaNet S1 Rust harness and Python comparator with an
    explicit `value_heads` axis;
  - made `delta_beta`, `delta_v`, and `delta_core` value-head-major;
  - added validation that `value_heads` is divisible by q/k `heads`;
  - added a Qwen3.6 MoE topology unit test with `heads=16`,
    `value_heads=32`, `key_dim=128`, and `value_dim=128`;
  - kept expert counts small in the unit test so it stays a fast correctness
    gate rather than a model-performance run.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `4 passed`;
  - `cargo test -p ferrum-models deltanet_s1 -- --nocapture`
    PASS: `5 passed`;
  - `cargo run -p ferrum-models --example w3_deltanet_s1_dump -- --out target/w3_deltanet_s1_rust_qwen36 --tokens 2 --hidden-dim 16 --heads 16 --value-heads 32 --key-dim 128 --value-dim 128 --experts 8 --top-k 2 --expert-hidden-dim 4 --seed 9271`
    PASS line:
    `W3 DELTANET S1 FERRUM DUMP PASS: target/w3_deltanet_s1_rust_qwen36`;
  - `python3 scripts/release/w3_deltanet_s1_layer_compare.py --self-test --out target/w3_deltanet_s1_python_qwen36 --tokens 2 --hidden-dim 16 --heads 16 --value-heads 32 --key-dim 128 --value-dim 128 --experts 8 --top-k 2 --expert-hidden-dim 4 --seed 9271`
    PASS line:
    `W3 DELTANET S1 LAYER COMPARE SELFTEST PASS: /Users/chejinxuan/rust_ws/ferrum-infer-rs/target/w3_deltanet_s1_python_qwen36`;
  - `python3 scripts/release/w3_deltanet_s1_layer_compare.py --compare --reference-dump target/w3_deltanet_s1_python_qwen36/reference_dump --ferrum-dump target/w3_deltanet_s1_rust_qwen36 --out target/w3_deltanet_s1_compare_qwen36 --atol 1e-6`
    PASS line:
    `W3 DELTANET S1 LAYER COMPARE PASS: /Users/chejinxuan/rust_ws/ferrum-infer-rs/target/w3_deltanet_s1_compare_qwen36`;
  - the cross-language compare reported `max_abs = 0.0` for all float tensors
    and `0` mismatches for `router_topk_indices`.
- Limitation:
  - this is still S1 synthetic topology evidence;
  - it does not yet wire the Qwen3.6 product executor, recurrent state update,
    `ferrum run`, or `ferrum serve`.

## 2026-06-17 ZZG — W3 Qwen3.5 recurrent-state shape contract checkpoint

- Scope:
  - W3-S0/S2 bridge work for the Gated-DeltaNet recurrent state contract;
  - derives recurrent-state tensor specs directly from official Qwen3.5 /
    Qwen3.6 HF `text_config`;
  - no product execution was run;
  - no GPU work was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `QWEN35_DELTA_STATE_NAME`;
  - added `Qwen35TextConfig::linear_qk_total_dim()`;
  - added `Qwen35TextConfig::linear_value_total_dim()`;
  - added `Qwen35TextConfig::recurrent_delta_state_shape()`;
  - added `Qwen35TextConfig::recurrent_state_tensor_specs()`;
  - added `Qwen35TextConfig::recurrent_state_elements_per_slot()`.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `4 passed`.
- Evidence from official configs:
  - dense `Qwen/Qwen3.5-0.8B`:
    18 linear-attention layers, `delta_state` shape `[16, 128, 128]`,
    `18 * 16 * 128 * 128` state elements per request slot;
  - MoE/shared-expert `Qwen/Qwen3.6-35B-A3B`:
    30 linear-attention layers, `delta_state` shape `[16, 128, 256]`,
    `30 * 16 * 128 * 256` state elements per request slot;
  - the MoE shape explicitly captures the unequal head topology:
    q/k total dim `2048`, value total dim `4096`, value state grouped across
    16 key heads.
- Limitation:
  - this is a config-derived allocation contract only;
  - it does not update recurrent state during prefill/decode yet;
  - it does not prove Qwen3.6 MoE router/expert/shared-expert numerical
    correctness.
- Next required validation:
  - have the eventual Qwen3.5/Qwen3.6 executor return these specs through
    `ModelExecutor::recurrent_state_spec()`;
  - add S1 tensor evidence for Qwen3.6 MoE router, expert layout, and shared
    expert merge semantics.

## 2026-06-17 ZZF — W3 Qwen3.5/Qwen3.6 product config recognition checkpoint

- Scope:
  - W3-S2 product-path groundwork after the S1 dense first-layer PASS;
  - recognizes official HF `qwen3_5` and `qwen3_5_moe` configs as distinct
    architectures instead of falling through to `Unknown`, `Qwen3`, or
    Llama-family defaults;
  - flattens nested HF `text_config` into `ModelDefinition` for
    Qwen3.5/Qwen3.6, the same way Gemma3 text configs are handled;
  - preserves the typed W3 text shape under
    `extra_params.ferrum_qwen35_text_config`, including layer types,
    linear-attention dimensions, MoE router shape, and shared expert size;
  - updates CLI source-family detection and serve capability snapshots so
    Qwen3.5 dense and Qwen3.5/Qwen3.6 MoE are not silently treated as older
    Qwen3;
  - keeps product execution explicitly unsupported until the real W3 model
    executor is wired, avoiding a false `ferrum run` / `ferrum serve` pass.
- Change:
  - added `Architecture::Qwen35` and `Architecture::Qwen35Moe`;
  - mapped official names:
    `qwen3_5`, `Qwen3_5ForConditionalGeneration`,
    `qwen3_5_moe`, `Qwen3_5MoeForConditionalGeneration`;
  - updated `ConfigManager` to parse `Qwen35TextConfig` for these
    architectures and derive real dimensions from `text_config`;
  - updated `ferrum-cli` source resolver defaults for `qwen3_5` /
    `qwen3_5_moe`;
  - updated `serve` model-capability snapshots to preserve Qwen3.5 MoE expert
    fields.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    `4 passed`;
  - `cargo test -p ferrum-models test_architecture_from_str -- --nocapture`
    PASS: `1 passed`;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS: `4 passed`;
  - `cargo test -p ferrum-cli qwen35 -- --nocapture` PASS: `3 passed`;
  - `cargo check -p ferrum-models -p ferrum-engine -p ferrum-cli` PASS.
- Limitation:
  - this is not an execution-path PASS;
  - `ferrum run` and `ferrum serve` still intentionally reject Qwen3.5/Qwen3.6
    model execution with an explicit unsupported error;
  - no W3 product correctness gate, concurrency gate, or performance gate was
    run;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Next required validation:
  - implement the product model loader/executor path using the preserved W3
    text config rather than ad hoc config parsing;
  - add Qwen3.6 MoE/shared-expert S1 tensor evidence before claiming W3-S1
    complete across dense and MoE variants;
  - only after product execution exists, run `ferrum run` and `ferrum serve`
    correctness before W3-S3 performance.

## 2026-06-17 ZZE — W3 Qwen3.5 S1 Ferrum-vs-HF layer compare PASS

- Scope:
  - W3-S1 correctness evidence for the dense Qwen3.5 first `linear_attention`
    layer;
  - model: `Qwen/Qwen3.5-0.8B`;
  - prompt/layer match the committed HF reference layer dump from checkpoint
    `ZZC`;
  - Ferrum replay reads the real HF safetensors and dumps the same 19 tensor
    checkpoints as the HF reference;
  - no product-path `ferrum run` / `ferrum serve` was run in this checkpoint;
  - no performance benchmark was run;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Evidence:
  - artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_s1_compare_20260617T144200Z_7dc3de10/`;
  - Ferrum dump PASS:
    `W3 QWEN35 FERRUM LAYER DUMP PASS:
    /workspace/w3_qwen35_s1_compare_20260617T144200Z_7dc3de10/ferrum_dump`;
  - comparator PASS:
    `W3 QWEN35 LAYER COMPARE PASS:
    /workspace/w3_qwen35_s1_compare_20260617T144200Z_7dc3de10/compare`;
  - remote git status recorded in both manifests:
    `sha=a98ef736728b1b88637c8751e00f8c25bc5f323d`,
    `is_dirty=false`, empty tracked status, `untracked_count=0`;
  - compared tensors: 19/19 `pass`;
  - largest recorded absolute delta:
    `mixed_qkv_raw max_abs=2.765655517578125e-05,
    mean_abs=9.50081827492492e-07`;
  - final layer output delta:
    `layer_output max_abs=8.642673492431641e-07,
    mean_abs=5.998470555823588e-08`.
- Remote state:
  - retained 1x RTX 4090 host was left usable;
  - no GPU compute process was present after the run;
  - GPU memory check showed 1 MiB used.
- Limitation:
  - this proves the dense Qwen3.5 first layer replay path only;
  - W3 still needs product loader/entrypoint work, `ferrum run` and
    `ferrum serve` correctness, and the W3 performance gate;
  - Qwen3.6 MoE/shared-expert coverage remains open.
- Next required validation:
  - extend this from diagnostic first-layer replay toward the product loader
    path;
  - add Qwen3.6 MoE/shared-expert evidence before any W3 release-grade claim;
  - run W3-S2 product correctness before W3-S3 performance.

## 2026-06-17 ZZD — W3 Qwen3.5 Ferrum S1 replay source checkpoint

- Scope:
  - source checkpoint for Ferrum-owned Qwen3.5 first-layer CPU replay;
  - reads real HF safetensors weights and a matching HF layer dump manifest;
  - targets the same `Qwen/Qwen3.5-0.8B` prompt/layer as the HF reference
    artifact;
  - no remote real-weight Ferrum dump was generated in this checkpoint;
  - no product-path `ferrum run` / `ferrum serve` was run;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `crates/ferrum-models/src/qwen35_s1.rs`;
  - added `crates/ferrum-models/examples/w3_qwen35_s1_dump.rs`;
  - added `scripts/release/w3_qwen35_layer_compare.py`;
  - exported `qwen35_s1` from `crates/ferrum-models/src/lib.rs`;
  - added `memmap2` to `ferrum-models` for diagnostic safetensors mmap.
- Implemented replay pieces:
  - HF safetensors BF16/F16/F32 to f32 materialization;
  - Qwen3.5 `RMSNorm` with `1.0 + weight` semantics;
  - depthwise causal conv + SiLU;
  - first-layer q/k/v/beta/g projection path;
  - single-chunk `torch_chunk_gated_delta_rule` replay for the 5-token
    reference prompt, including q/k L2 normalization and decay mask semantics;
  - gated RMS norm, DeltaNet out projection, post-attention norm, and dense
    gated MLP.
- Validation:
  - `cargo fmt --all` PASS;
  - `python3 -m py_compile scripts/release/w3_qwen35_layer_compare.py` PASS;
  - `cargo check -p ferrum-models --example w3_qwen35_s1_dump` PASS;
  - `cargo test -p ferrum-models qwen35_s1 -- --nocapture` PASS:
    `2 passed`;
  - `python3 scripts/release/w3_qwen35_layer_compare.py --self-test --out
    /tmp/w3_qwen35_layer_compare_selftest` PASS:
    `W3 QWEN35 LAYER COMPARE SELFTEST PASS:
    /private/tmp/w3_qwen35_layer_compare_selftest`.
- Limitation:
  - this checkpoint proves source/schema only; real W3-S1 remains open until
    the example is run against cached `Qwen/Qwen3.5-0.8B` safetensors and the
    comparator prints `W3 QWEN35 LAYER COMPARE PASS`.
- Next required validation:
  - sync this source checkpoint to the retained remote host;
  - run `w3_qwen35_s1_dump` against the cached HF snapshot and compare it with
    the existing HF reference layer dump.

## 2026-06-17 ZZC — W3 Qwen3.5 0.8B HF layer dump PASS

- Scope:
  - paid GPU host was reused, but the HF reference dump ran with CPU torch;
  - real `Qwen/Qwen3.5-0.8B` weights were loaded through HF transformers;
  - selected layer: first `linear_attention` layer, `layer_idx=0`;
  - no Ferrum Qwen3.5 layer dump was generated in this checkpoint;
  - no product-path `ferrum run` / `ferrum serve` was run;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Evidence:
  - artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_08b_hf_layer_dump_20260617T141100Z_2d3092ba/`;
  - required PASS line:
    `W3 QWEN35 HF LAYER DUMP PASS:
    /workspace/w3_hf_dump_artifacts/qwen35_08b_hf_layer_dump_20260617T141100Z_2d3092ba/dump`;
  - manifest git status from remote:
    `sha=2d3092bab3cc4b22c08d97d0e6f0e205b723b9a1`,
    `is_dirty=false`, empty tracked status, `untracked_count=0`;
  - dependencies recorded in manifest:
    `torch=2.12.0+cpu`, `transformers=5.12.1`;
  - captured 19 tensors, including DeltaNet q/k/v/beta/g/core, conv output,
    gated norm output, DeltaNet output, post-attention norm, MLP output, and
    final layer output.
- Remote state:
  - retained 1x RTX 4090 host was left usable;
  - no GPU compute process was present after the dump;
  - GPU memory check showed 1 MiB used.
- Limitation:
  - this is HF reference evidence only; W3-S1 still requires the matching
    Ferrum dump and an explicit Ferrum-vs-HF comparator PASS.
- Next required validation:
  - implement/route Ferrum Qwen3.5 first-layer dump for the same prompt and
    layer;
  - compare against this HF artifact before moving to W3-S2 product paths.

## 2026-06-17 ZZB — W3 Qwen3.5 HF layer dump harness checkpoint

- Scope:
  - source checkpoint for official/HF W3-S1 layer dump extraction;
  - validates the selected Qwen3.5 first `linear_attention` layer contract
    against saved HF config metadata and current Transformers source hooks;
  - no model weights were downloaded;
  - no paid GPU compute was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_qwen35_hf_layer_dump.py`;
  - script modes:
    - `--self-test` validates the dump schema and source-hook contract without
      torch/transformers;
    - `--contract` validates saved HF `config.json` plus Transformers source
      hooks without weights;
    - `--dump` is the real HF/torch path for dumping the selected layer tensors.
  - dump schema captures the first dense Qwen3.5 DeltaNet layer inputs,
    QKV/z/b/a projections, conv output, delta-rule q/k/v/beta/g/core, gated
    norm output, DeltaNet output, residual-after-mixer, post-attention norm,
    MLP output, and layer output.
- Validation:
  - clean-worktree artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_hf_layer_harness_20260617T134658Z_d53683a6/`;
  - artifact manifests were generated from clean detached worktree commit
    `d53683a60029229ff0a60e928df6e88cd3e3e82e`, with
    `is_dirty=false`, empty tracked status, and `untracked_count=0`;
  - `python3 -m py_compile scripts/release/w3_qwen35_hf_layer_dump.py` PASS;
  - `python3 scripts/release/w3_qwen35_hf_layer_dump.py --self-test --out
    /tmp/w3_qwen35_hf_layer_harness_20260617T134658Z_d53683a6/selftest`
    PASS:
    `W3 QWEN35 HF LAYER DUMP SELFTEST PASS:
    /private/tmp/w3_qwen35_hf_layer_harness_20260617T134658Z_d53683a6/selftest`;
  - `python3 scripts/release/w3_qwen35_hf_layer_dump.py --contract --model-id
    Qwen/Qwen3.5-0.8B --config
    docs/goals/model-coverage-2026-06-12/artifacts/w3_hf_config_probe_20260617T131209Z_f97c1d6f/dense_min_reference.config.json
    --out /tmp/w3_qwen35_hf_layer_harness_20260617T134658Z_d53683a6/contract`
    PASS:
    `W3 QWEN35 HF LAYER CONTRACT PASS:
    /private/tmp/w3_qwen35_hf_layer_harness_20260617T134658Z_d53683a6/contract`.
- Limitation:
  - this is still not an official weight-based HF tensor dump and not a
    Ferrum-vs-HF compare artifact;
  - W3-S1 remains open until the real HF dump is compared against a real Ferrum
    dump for the same prompt/layer.
- Next required validation:
  - run `--dump` for `Qwen/Qwen3.5-0.8B` on the retained environment with
    torch/transformers and cached weights;
  - implement/route the matching Ferrum Qwen3.5 layer dump and compare both
    artifacts.

## 2026-06-17 ZZA — W3 Qwen3.5/Qwen3.6 HF config parser checkpoint

- Scope:
  - source checkpoint for W3 loader/config groundwork;
  - parses official/HF nested `text_config` shape into Ferrum-owned typed
    config structures;
  - no model weights were downloaded;
  - no paid GPU compute was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `crates/ferrum-models/src/qwen35_config.rs`;
  - exported `qwen35_config` from `crates/ferrum-models/src/lib.rs`;
  - added `crates/ferrum-models/tests/qwen35_config_test.rs`;
  - parser now preserves linear/full attention layer kinds, linear-attention
    q/k/v head dims, dense intermediate size, MoE expert count/top-k, and
    shared-expert intermediate size.
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models --test qwen35_config_test -- --nocapture`
    PASS (`4 passed`);
  - tests read the committed HF config probe artifact for
    `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.6-35B-A3B`;
  - negative tests reject dense configs with MoE fields and MoE configs missing
    `shared_expert_intermediate_size`.
- Limitation:
  - this is config/loader groundwork only; it does not load weights, emit an
    official/HF hidden-state dump, or run product paths.
- Next required validation:
  - implement official/HF layer-dump extraction for `Qwen/Qwen3.5-0.8B`;
  - map these parsed fields into the real Ferrum W3 model loader/config path.

## 2026-06-17 ZZ — W3 official/HF config probe PASS

- Scope:
  - metadata-only official/HF config probe for W3 reference selection;
  - no model weights were downloaded;
  - generated from clean local worktree at commit
    `f97c1d6f3539ede18621bcb0e10eb7711d3e19bf`;
  - no paid GPU compute was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Evidence:
  - artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_hf_config_probe_20260617T131209Z_f97c1d6f/`;
  - required PASS line:
    `W3 HF CONFIG PROBE PASS:
    /private/tmp/w3_hf_config_probe_20260617T131209Z_f97c1d6f`;
  - clean worktree status in manifest:
    `is_dirty=false`, empty tracked status, `untracked_count=0`;
  - raw `config.json` copies and SHA256s are saved for each model.
- Validated official/HF configs:
  - `Qwen/Qwen3.5-0.8B`: `qwen3_5_text`, 24 layers, 18
    `linear_attention` + 6 `full_attention`, no MoE fields;
  - `Qwen/Qwen3.5-4B`: `qwen3_5_text`, 32 layers, 24
    `linear_attention` + 8 `full_attention`, no MoE fields;
  - `Qwen/Qwen3.6-35B-A3B`: `qwen3_5_moe_text`, 40 layers, 30
    `linear_attention` + 10 `full_attention`, `num_experts=256`,
    `num_experts_per_tok=8`, `moe_intermediate_size=512`,
    `shared_expert_intermediate_size=512`.
- Limitation:
  - this proves the official/HF metadata required to select W3 references; it
    is not a layer dump, product path, or performance artifact.
- Next required validation:
  - use `Qwen/Qwen3.5-0.8B` as the first smallest official dense DeltaNet layer
    reference target;
  - use `Qwen/Qwen3.6-35B-A3B` for the shared-expert / 256-expert MoE variant
    semantic target;
  - implement official/HF layer-dump extraction and compare it against Ferrum
    dumps before W3-S1 can be called real model evidence.

## 2026-06-17 ZY — W3 official/HF config probe source checkpoint

- Scope:
  - source checkpoint for selecting official/HF W3 reference-layer targets;
  - metadata-only, no model weights downloaded;
  - no paid GPU compute was started;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_hf_config_probe.py`;
  - validates nested `text_config` for selected Qwen3.5 dense and Qwen3.6
    MoE references;
  - checks `layer_types` include both `linear_attention` and `full_attention`;
  - checks linear-attention head/key/value/conv fields;
  - checks MoE fields including `num_experts`, `num_experts_per_tok`,
    `moe_intermediate_size`, and `shared_expert_intermediate_size`.
- Validation:
  - `python3 -m py_compile scripts/release/w3_hf_config_probe.py` PASS;
  - `python3 scripts/release/w3_hf_config_probe.py --self-test --out
    target/w3_hf_config_probe_selftest` PASS line:
    `W3 HF CONFIG PROBE SELFTEST PASS:
    /Users/chejinxuan/rust_ws/ferrum-infer-rs/target/w3_hf_config_probe_selftest`;
  - `git diff --check -- scripts/release/w3_hf_config_probe.py` PASS.
- Next required validation:
  - run the probe from a clean worktree against Hugging Face config URLs and
    commit the artifact;
  - use that artifact to choose the first official/HF W3 layer-dump reference.

## 2026-06-17 ZX — W3-S1 Ferrum Rust single-layer compare PASS

- Scope:
  - W3-S1 single-layer checkpoint using a Ferrum-owned Rust dump harness;
  - generated from clean local worktree at commit
    `1b480a31091fb890d753f1e85e008c28db3b1d39`;
  - no paid GPU compute was started;
  - no whole W3 model was loaded;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Evidence:
  - artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_deltanet_s1_rust_compare_20260617T130232Z_1b480a31/`;
  - required comparator PASS line:
    `W3 DELTANET S1 LAYER COMPARE PASS:
    /private/tmp/w3_deltanet_s1_rust_compare_20260617T130232Z_1b480a31/compare`;
  - Ferrum dump PASS line:
    `W3 DELTANET S1 FERRUM DUMP PASS:
    /tmp/w3_deltanet_s1_rust_compare_20260617T130232Z_1b480a31/ferrum_dump`;
  - clean worktree status in comparator manifest:
    `is_dirty=false`, empty tracked status, `untracked_count=0`.
- Validation:
  - reference dump generated by
    `scripts/release/w3_deltanet_s1_layer_compare.py --self-test`;
  - Ferrum dump generated by
    `cargo run -p ferrum-models --example w3_deltanet_s1_dump -- --out ...`;
  - comparator checks all PASS: `delta_rule`, `deltanet_layer`,
    `router_topk`, `expert_layout`, and `shared_expert_merge`;
  - `router_topk_indices` mismatches: `0`;
  - `delta_output`, `routed_expert_output`, `shared_expert_output`,
    `moe_output`, and `layer_output` all report `max_abs=0.0`.
- Limitation:
  - this is real Ferrum Rust single-layer dump evidence against the current
    deterministic S1 reference contract; it is not yet official/HF model-layer
    evidence for Qwen3.5/Qwen3.6, and it does not cover W3-S2 or W3-S3.
- Next required validation:
  - replace the deterministic reference with an official/HF selected W3 model
    layer dump;
  - wire the corresponding real model loader/config path;
  - then run `ferrum run`, `ferrum serve`, L0-L5, and the c=1/4/16/32 80%
    performance gate.

## 2026-06-17 ZW — W3-S1 source checkpoint: Ferrum Rust DeltaNet dump harness

- Scope:
  - source checkpoint for W3-S1 real Ferrum-side single-layer dump generation;
  - no paid GPU compute was started during this checkpoint;
  - no whole W3 model was loaded;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `crates/ferrum-models/src/deltanet_s1.rs`;
  - added `crates/ferrum-models/examples/w3_deltanet_s1_dump.rs`;
  - exported the module from `crates/ferrum-models/src/lib.rs`;
  - implemented deterministic CPU Gated DeltaNet single-layer projection,
    delta-rule core, Ferrum MoE top-k routing, routed expert output, shared
    expert output, MoE merge, and final layer output;
  - the example emits the same dump schema consumed by
    `scripts/release/w3_deltanet_s1_layer_compare.py`.
- Validation:
  - `cargo test -p ferrum-models deltanet_s1 -- --nocapture` PASS
    (`3 passed`);
  - `cargo check -p ferrum-models --example w3_deltanet_s1_dump` PASS;
  - `python3 -m py_compile
    scripts/release/w3_deltanet_s1_layer_compare.py` PASS;
  - local dry-run `W3 DELTANET S1 LAYER COMPARE PASS` was produced under
    `target/w3_deltanet_s1_rust_compare/compare`.
- Limitation:
  - this proves a Ferrum-owned Rust single-layer dump path against the current
    deterministic reference contract; it is not yet official/HF Qwen3.5 or
    Qwen3.6 full model evidence.
- Next required validation:
  - regenerate the Rust-vs-reference compare artifact from a clean worktree and
    commit it;
  - then replace the deterministic reference with the selected official/HF W3
    reference layer dump.

## 2026-06-17 ZV — W3-S1 source checkpoint: DeltaNet layer dump comparator

- Scope:
  - source-only W3-S1 correctness-gate checkpoint;
  - no paid GPU compute was started during this checkpoint;
  - no whole W3 model was loaded;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_deltanet_s1_layer_compare.py`;
  - defined the W3-S1 single-layer dump schema for DeltaNet q/k/v/beta,
    delta-rule core, gated DeltaNet output, router logits/top-k, routed expert
    output, shared expert output, MoE merge, and final layer output;
  - implemented a deterministic CPU reference and comparator for reference dump
    vs Ferrum dump;
  - added self-test mode that writes a reference dump plus synthetic Ferrum dump
    and compares all tensors.
- Validation:
  - `python3 -m py_compile
    scripts/release/w3_deltanet_s1_layer_compare.py` PASS;
  - `git diff --check -- scripts/release/w3_deltanet_s1_layer_compare.py`
    PASS;
  - `python3 scripts/release/w3_deltanet_s1_layer_compare.py --self-test --out
    docs/goals/model-coverage-2026-06-12/artifacts/w3_deltanet_s1_layer_selftest_20260617T124000Z`
    PASS line:
    `W3 DELTANET S1 LAYER COMPARE SELFTEST PASS:
    /Users/chejinxuan/rust_ws/ferrum-infer-rs/docs/goals/model-coverage-2026-06-12/artifacts/w3_deltanet_s1_layer_selftest_20260617T124000Z`;
  - self-test comparisons record zero max_abs for `delta_output`,
    `routed_expert_output`, `shared_expert_output`, `moe_output`, and
    `layer_output`; router top-k indices have zero mismatches.
- Limitation:
  - this is a gate/schema self-test using a synthetic Ferrum dump, not real
    Qwen3.5/Qwen3.6 or HF/reference-vs-Ferrum model evidence.
- Next required validation:
  - implement or expose the real Ferrum DeltaNet single-layer dump using this
    schema;
  - generate the official/HF reference dump for the selected W3 model;
  - rerun this comparator in `--compare` mode and only then count W3-S1 as real
    correctness evidence.

## 2026-06-17 ZU — W3-S0 native CUDA delta-rule microbench PASS

- Scope:
  - paid GPU lane: W3-S0 native CUDA delta-rule microbench on retained 1x RTX
    4090 instance;
  - no whole W3 model was loaded during this checkpoint;
  - no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Evidence:
  - artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_delta_rule_s0_cuda_20260617T123139Z_da2802bd/`;
  - required PASS line:
    `W3 DELTA RULE S0 MICROBENCH PASS:
    /workspace/w3_delta_rule_s0_cuda_20260617T123139Z_da2802bd`;
  - remote git SHA: `da2802bdc67d223c8b94674880a5ca9f03fceb48`;
  - remote git state: clean tracked status, zero untracked files;
  - `ptx_arch=sm_89`;
  - CUDA binary SHA256:
    `6341b5bed1746468b4c15d1fa1acacc3d4700cf8ef325c187560c80dc2367ab1`;
  - input distribution recorded as deterministic centered uniform ranges for
    q/k/v/beta.
- Validation:
  - CUDA source compiled with:
    `nvcc -O2 --generate-line-info -arch=sm_89 ...`;
  - CUDA output compared against the internal Python delta-rule reference;
  - error stats: `max_abs=3.011855029266819e-09`,
    `max_rel=3.429708441964771e-06`,
    `rmse=6.12241793241555e-10`;
  - GPU compute process query after the run returned no active compute apps.
- Next required validation:
  - W3-S1 still needs DeltaNet single-layer CPU/reference vs Ferrum dump and
    MoE variant layout/router/shared-merge coverage;
  - W3-S2 still needs whole-model product-path `ferrum run` and `ferrum serve`;
  - W3-S3 still needs the release-grade 80% performance gate or an explicitly
    documented cap.

## 2026-06-17 ZT — W3-S0 source checkpoint: delta-rule microbench harness

- Scope:
  - source-only W3-S0 microbench harness checkpoint;
  - no paid GPU compute was started during this checkpoint;
  - local self-test is not W3-S0 native CUDA evidence;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `scripts/release/w3_delta_rule_s0_microbench.py`;
  - implemented deterministic Python delta-rule reference and chunked-reference
    comparison;
  - added `--cuda` mode that writes a minimal native CUDA source, builds it
    with `nvcc`, runs it, and compares CUDA output against the same reference;
  - records command line, git state, shapes, seed, tolerance, CUDA metadata,
    build/run commands, process logs, and binary SHA256 when CUDA mode is used.
- Validation:
  - `python3 -m py_compile
    scripts/release/w3_delta_rule_s0_microbench.py` PASS;
  - `python3 scripts/release/w3_delta_rule_s0_microbench.py --self-test --out
    docs/goals/model-coverage-2026-06-12/artifacts/w3_delta_rule_s0_selftest_20260617T060439Z`
    PASS line:
    `W3 DELTA RULE S0 SELFTEST PASS:
    /Users/chejinxuan/rust_ws/ferrum-infer-rs/docs/goals/model-coverage-2026-06-12/artifacts/w3_delta_rule_s0_selftest_20260617T060439Z`;
  - self-test manifest records chunked-reference `max_abs=0.0`,
    `max_rel=0.0`, `rmse=0.0`.
- Next required validation:
  - start retained Vast instance `41241013` only under a stated W3-S0 native
    CUDA lane, run this script with `--cuda`, copy back the artifact, and stop
    the instance immediately after the result.

## 2026-06-17 ZS — Vast cleanup checkpoint: keep one reusable CUDA instance

- Scope:
  - resource-governance checkpoint after user requested: keep one usable Vast
    instance and destroy the rest;
  - no paid GPU compute was started during this checkpoint;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Action:
  - kept `41241013` (`ferrum-w2-vllm-ferrum-c16-ab-20260617`) because it is the
    most useful retained CUDA devel 4090 for W2 same-hardware follow-up;
  - destroyed stopped diagnostic instances `41178475`, `41187356`, `41218739`,
    `41230499`, `41256521`, and `41276321`;
  - did not retain the more expensive `41276321`.
- Evidence:
  - artifact directory:
    `docs/goals/model-coverage-2026-06-12/artifacts/vast_cleanup_keep_one_20260617T055651Z/`;
  - all six destroy summary responses record `success=true`;
  - final API polls 1/2/3 each returned exactly one instance: `41241013`;
  - retained instance final state: `cur_state=stopped`,
    `actual_status=exited`, `gpuCostPerHour=0`, stopped disk `totalHour`
    approximately `$0.111/hr`.
- Next required validation:
  - when using `41241013`, start it only for a stated lane with stop condition
    and command;
  - after each GPU checkpoint, stop it and verify `actual_status=exited`;
  - before final goal completion, destroy it or record explicit approval to keep
    it.

## 2026-06-17 ZR — W3-S0 source checkpoint: model-declared recurrent-state allocation

- Scope:
  - source-only W3-S0 model-declared allocation checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `ModelExecutor::recurrent_state_spec(...)` as the typed hook for
    state-space/hybrid models to declare per-request recurrent-state needs;
  - wired continuous-engine prefill, batched prefill, and unified mixed-batch
    prefill to allocate recurrent state through the configured manager when a
    model returns a spec;
  - disabled current KV-only prefix-cache hits for requests whose model declares
    recurrent state, because no recurrent-state snapshot is stored there yet;
  - preserved in-place recurrent-state handles when model outputs do not return
    a replacement handle;
  - added an engine test proving model-declared recurrent state is allocated and
    deallocated by the in-memory manager across a request lifecycle.
- Validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-engine
    engine_allocates_and_deallocates_model_declared_recurrent_state --
    --nocapture` PASS: `1 passed`;
  - `cargo check -p ferrum-engine --all-targets` PASS;
  - `cargo test -p ferrum-interfaces` PASS: full crate test set `15 passed`;
  - `cargo check -p ferrum-models --all-targets` PASS.
- Next required validation:
  - W3 still needs real DeltaNet model specs and S0 native CUDA/PTX delta-rule
    microbench before product DeltaNet integration;
  - W2 release-grade full matrix still requires restored Vast credit.

## 2026-06-17 ZQ — W3-S0 source checkpoint: in-memory recurrent-state manager

- Scope:
  - source-only W3-S0 concrete manager checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `InMemoryRecurrentStateManager` in `ferrum-engine` for GPU-free
    recurrent-state lifecycle/capacity management;
  - implemented allocation, duplicate rejection, capacity rejection,
    deallocation invalidation, reset invalidation, handle lookup/listing, and
    aggregate stats;
  - re-exported the manager/config/handle from `ferrum-engine` so integration
    wiring can use a concrete manager before backend-specific CUDA/Metal
    managers exist.
- Validation:
  - `cargo test -p ferrum-engine recurrent_state -- --nocapture` PASS:
    `4 passed` under the filter, including 3 in-memory manager lifecycle tests
    plus the builder injection test;
  - `cargo check -p ferrum-engine --all-targets` PASS;
  - `cargo fmt --all` PASS before final checks.
- Next required validation:
  - final local diff/format checks before commit;
  - W3 still needs model-family allocation specs and S0 native CUDA/PTX
    delta-rule microbench before product DeltaNet integration;
  - W2 release-grade full matrix still requires restored Vast credit.

## 2026-06-17 ZP — W3-S0 source checkpoint: recurrent-state manager injection

- Scope:
  - source-only W3-S0 manager-injection checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added optional `RecurrentStateManager` ownership to `ContinuousBatchEngine`
    construction and `EngineInner`;
  - added `EngineBuilder::with_custom_recurrent_state_manager(...)` so tests or
    future model-family wiring can inject a concrete manager without hidden
    environment variables;
  - wired completion and preemption cleanup to call recurrent-state manager
    `deallocate` when a sequence actually owns recurrent state;
  - kept the default product path unchanged: no recurrent manager is installed
    unless typed construction supplies one.
- Validation:
  - `cargo check -p ferrum-engine --all-targets` PASS;
  - `cargo test -p ferrum-engine builder -- --nocapture` PASS:
    builder-filtered test set `10 passed`;
  - `cargo fmt --all` PASS before final checks.
- Next required validation:
  - final local diff/format checks before commit;
  - W3 still needs real recurrent-state manager allocation and S0 native
    CUDA/PTX delta-rule microbench before product DeltaNet integration;
  - W2 release-grade full matrix still requires restored Vast credit.

## 2026-06-17 ZO — W3-S0 source checkpoint: engine recurrent-state lifecycle carrier

- Scope:
  - source-only W3-S0 engine lifecycle checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `SequenceState::recurrent_state` as an optional handle carried next
    to KV state;
  - threaded recurrent-state handles through continuous-engine prefill,
    chunked-prefill, decode, and unified mixed-batch inputs/outputs;
  - cleared recurrent state on preemption reset so resumed requests cannot
    silently reuse stale state;
  - kept current attention-only product paths behaviorally unchanged because no
    recurrent manager allocation path is introduced yet.
- Validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -p ferrum-engine --all-targets` PASS;
  - `cargo test -p ferrum-engine test_sequence_state -- --nocapture` PASS:
    `1 passed`.
- Next required validation:
  - W3 still needs real recurrent-state manager injection/allocation and S0
    native CUDA/PTX delta-rule microbench before product DeltaNet integration;
  - W2 release-grade full matrix still requires restored Vast credit.

## 2026-06-17 ZN — W3-S0 source checkpoint: scheduler recurrent-state resources

- Scope:
  - source-only W3-S0 scheduler/resource-accounting checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added recurrent-state memory/slot fields to scheduler resource constraints,
    allocated resources, batch resource requirements, and resource limits;
  - kept current FIFO/priority/continuous scheduler behavior unchanged by
    setting recurrent-state requirements to zero in existing batch plans;
  - added an interfaces contract test that verifies recurrent-state resource
    fields default to empty/zero.
- Validation:
  - `cargo test -p ferrum-interfaces` PASS:
    full crate test set `15 passed`;
  - `cargo test -p ferrum-scheduler` PASS:
    `53 passed`;
  - `cargo check -p ferrum-engine --all-targets` PASS.
- Next required validation:
  - W3 next local step is engine lifecycle ownership for allocation/deallocation
    of recurrent state handles.

## 2026-06-17 ZM — W3-S0 source checkpoint: model-executor recurrent-state carriers

- Scope:
  - source-only W3-S0 interface integration checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - threaded optional recurrent-state handles through `PrefillInput`,
    `PrefillOutput`, `DecodeInput`, `DecodeOutput`, and `UnifiedBatchItem`;
  - kept existing KV-required decode and unified-batch constructors/fields
    compatible for current KV-only product paths;
  - updated current engine/model unified-batch call sites to set
    `recurrent_state: None`;
  - added a contract test proving model executor inputs and outputs can carry
    recurrent-state handles alongside KV handles.
- Validation:
  - `cargo test -p ferrum-interfaces` PASS:
    recurrent-state tests `5 passed`, full crate test set `14 passed`;
  - `cargo check -p ferrum-engine --all-targets` PASS;
  - `cargo check -p ferrum-models --all-targets` PASS.
- Next required validation:
  - W2 remains blocked on Vast `credit=0` for release-grade full-matrix CUDA;
  - W3 next local step is scheduler/engine lifecycle ownership for recurrent
    state allocation and cleanup, still before product DeltaNet integration.

## 2026-06-17 ZL — W3-S0 source checkpoint: recurrent-state interface contract

- Scope:
  - source-only W3-S0 interface checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `crates/ferrum-interfaces/src/recurrent_state.rs`;
  - exported GPU-free `RecurrentStateSpec`, `RecurrentStateTensorSpec`,
    `RecurrentStateHandle`, `RecurrentStateManager`, stats, and resume policy;
  - added crate-local mock lifecycle tests for allocate/get/list/deallocate,
    reset, capacity rejection, and memory estimation.
- Validation:
  - `cargo test -p ferrum-interfaces` PASS:
    recurrent-state tests `4 passed`, full crate test set `13 passed`;
  - live Vast probe still reports `credit=0`, `running_count=0`, and all known
    instances stopped/exited; saved summary under
    `artifacts/w2_dynamic_kv_full_matrix_samehw_cuda_2026-06-17/local_vast/live_probe_20260617T050350Z/`.
- Next required validation:
  - after credit is restored, resume W2 full-matrix same-hardware CUDA evidence
    first;
  - W3 next local step is to thread optional recurrent-state handles through
    model executor inputs without breaking KV-only models.

## 2026-06-17 ZK — W3-S0 design checkpoint: recurrent-state boundary

- Scope:
  - source/docs-only W3-S0 design checkpoint;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.
- Change:
  - added `W3_S0_RECURRENT_STATE_DESIGN.md`;
  - defined recurrent state as a separate cache abstraction rather than folding
    it into `KvCacheHandle`;
  - captured the interface boundary for paged KV, ContinuousBatch,
    preemption/resume, prefix cache, scheduler resource accounting, and
    model-executor inputs;
  - documented the native CUDA/PTX S0 microbench contract for chunked
    delta-rule before any W3 product-path integration.
- Current external blocker:
  - Vast still reports `credit=0`, no running instances, and only stopped
    49GB RTX 4090 instance `41276321`;
  - higher-priced replacement offers cannot be rented until the external
    account credit state changes.
- Next required validation:
  - after credit is restored, resume W2 full-matrix same-hardware CUDA evidence
    first;
  - for W3, implement the recurrent-state interfaces and S0 microbench before
    touching product `run`/`serve` paths.

## 2026-06-17 ZJ — W2-P0 docs checkpoint: coverage PASS is not release-grade PASS

- Scope:
  - README/support-matrix wording audit for W2-P0 release posture;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Change:
  - `README.md` and `README_zh.md` already described Gemma 3 27B GPTQ as
    functional/known-gap rather than release-grade;
  - tightened the Gemma 3 footnote to say the existing
    `MODEL_COVERAGE_W2 GOAL PASS` is a coverage validator line, not the
    release-grade `MODEL_RELEASE_GRADE_W2 PASS` line.
- Current external blocker:
  - Vast still reports `credit=0`, no running instances, and only stopped
    49GB RTX 4090 instance `41276321`;
  - higher-priced replacement offers cannot be rented until the external
    account credit state changes.

## 2026-06-17 ZI — W2 full-matrix runner checkpoint: auto-generate final manifest

- Scope:
  - source-only paid-GPU workflow hardening for the tracked W2 dynamic-KV full
    matrix runner;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Change:
  - fixed the runner default output directory name from the old
    `w2_dynamic_prefill...` label to
    `w2_dynamic_kv_full_matrix_samehw_cuda_2026-06-17`;
  - after Ferrum and vLLM c=1/4/16/32 sweeps finish, the runner now invokes
    `scripts/release/model_release_grade_manifest.py w2 --source "$OUT"
    --out "$OUT"` so the same artifact directory gets
    `model_release_grade_manifest.json`,
    `model_release_grade_goal_gate.manifest.json`, and the exact final PASS or
    FAIL output;
  - updated the tracked runner SHA256 file after the script change.
- Validation:
  - `bash -n
    docs/goals/model-coverage-2026-06-12/artifacts/w2_dynamic_kv_full_matrix_samehw_cuda_2026-06-17/local_vast/run_remote_full_matrix.sh`
    PASS, with only a local locale warning;
  - `git diff --check --
    docs/goals/model-coverage-2026-06-12/artifacts/w2_dynamic_kv_full_matrix_samehw_cuda_2026-06-17/local_vast/run_remote_full_matrix.sh`
    PASS.
- Next required validation:
  - after Vast credit is available, use the updated runner on a new
    high-availability 49GB RTX 4090 instance; correctness and full matrix must
    still pass before the final W2 release-grade claim is valid.

## 2026-06-17 ZH — W2 release-grade validator checkpoint: bench commands must cover cell concurrency

- Scope:
  - source-only release-grade validator hardening;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Change:
  - `scripts/release/model_release_grade_goal_gate.py` now rejects
    release-grade `bench-serve` evidence when a performance cell's command
    cannot actually produce that closed-loop concurrency;
  - accepted command shapes are either `--concurrency-sweep` containing the
    cell, or a matching single-cell `--concurrency`/`--max-concurrency`;
  - open-loop `--request-rate` is rejected for this release-grade lane because
    it overrides closed-loop concurrency;
  - the W2 manifest generator self-test now records the same
    `--concurrency-sweep 1,4,16,32` command shape as the intended full matrix.
- Validation:
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS;
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS, including the synthetic final validator
    `MODEL_RELEASE_GRADE_W2 PASS: <tmp>/out`;
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py
    scripts/release/model_release_grade_manifest.py
    scripts/release/selftest_g0_validators.py` PASS;
  - `git diff --check -- scripts/release/model_release_grade_goal_gate.py
    scripts/release/model_release_grade_manifest.py` PASS;
  - `python3 scripts/release/selftest_g0_validators.py` PASS.
- Current external blocker:
  - Vast still reports `credit=0`, no running instances, and only stopped
    49GB RTX 4090 instance `41276321`;
  - higher-priced replacement offers cannot be rented until the external
    account credit state changes.
- Next required validation:
  - after Vast credit is available, create a new high-availability 49GB RTX
    4090 instance, run W2 correctness first, then the c=1/4/16/32
    same-hardware Ferrum/vLLM full matrix, then generate the manifest and run
    the final W2 validator.

## 2026-06-17 ZG — W2 CUDA diagnostic: two mixed-prefill chunks pass c16 throughput and p95

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_decode_prefill_budget2_c16_cuda_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ c16 same-pod diagnostic for commit
    `a7444587ebdceb6a62f5ee475e0244c111d340ac`;
  - reused same 1x RTX 4090 Vast instance `41241013`;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- GPU lifecycle:
  - started cached instance `41241013`, 1x RTX 4090, quoted rate
    `0.47111111111111115 USD/h`;
  - verified CUDA with driver `580.95.05`, CUDA toolkit `12.4`, and
    24GB RTX 4090 visibility;
  - synced clean source to commit
    `a7444587ebdceb6a62f5ee475e0244c111d340ac`;
  - copied artifacts back locally, then stopped the instance;
  - final sanitized Vast state recorded `cur_state=stopped`,
    `actual_status=exited`.
- Correctness result:
  - product `ferrum run` smoke passed with stdout content `5`,
    `n_tokens=3`;
  - product `ferrum serve` streaming smoke passed with content `5\n`,
    exactly one `[DONE]`, and usage present
    (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`);
  - c16 `bench-serve` completed `[100,100,100]` requests with
    errors `[0,0,0]`;
  - `output_token_count_source=usage`;
  - no correctness issue was observed in this diagnostic artifact.
- Performance command:
  - `ferrum bench-serve --dataset sharegpt --sharegpt-path
    /workspace/ascii_sharegpt_w2_100.jsonl --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3
    --fail-on-error --require-ci --seed 9271`;
  - same model snapshot and same ASCII ShareGPT 100 dataset as the same-pod
    vLLM reference from `w2_vllm_same_hw_c16_sharegpt_2026-06-17`.
- Result:
  - Ferrum output throughput mean/LCB:
    `463.405 / 460.553 tok/s`;
  - same-pod vLLM reference mean/LCB:
    `500.670 / 478.395 tok/s`;
  - Ferrum LCB / vLLM LCB = `0.9627`, so c16 throughput clears the 80%
    diagnostic line;
  - Ferrum p95 ITL mean `29.247 ms`;
  - same-pod vLLM reference p95 ITL mean `33.070 ms`;
  - Ferrum p95 ITL / vLLM p95 ITL = `0.8844`, so c16 p95 also passes;
  - relative to the one-chunk cap run, c16 LCB improved by
    `127.443 tok/s` while p95 ITL regressed only `2.610 ms`.
- Interpretation:
  - two active-decode mixed-prefill chunks is the first candidate that passes
    both c16 throughput and c16 p95 against the same-pod vLLM reference;
  - this validates the scheduler cadence direction more strongly than the
    one-chunk cap, and avoids returning to the old unbounded mixed-prefill
    behavior;
  - W2 is still not release-grade because this is only c16 diagnostic
    evidence, not the required final W2 gate or full c=1/4/16/32 matrix.
- Next required validation:
  - expand same-hardware validation to c=1/4/32 with the same correctness and
    `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
    standards;
  - only after the expanded matrix passes should the W2 release-grade
    validator be run for `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## 2026-06-17 ZF — W2 source checkpoint: allow two mixed-prefill chunks during active decode

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_scheduler_active_decode_prefill_budget2_source_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ source-only throughput recovery candidate;
  - no paid GPU instance was started for this source checkpoint;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Motivation:
  - the one-chunk aggregate active-decode mixed-prefill cap proved the p95
    root cause by moving c16 p95 ITL from `52.819ms` to `26.637ms`;
  - the same run over-throttled throughput, dropping c16 LCB from
    `414.592 tok/s` to `333.110 tok/s`;
  - therefore the next candidate should recover throughput without going back
    to the old unbounded waiting-request count.
- Source change:
  - introduced `ACTIVE_DECODE_PREFILL_CHUNKS_PER_ITERATION = 2`;
  - when decode requests are scheduled, the aggregate mixed-prefill token
    budget is now `2 * active_decode_prefill_chunk`;
  - each prefill request is still chunked by `active_decode_prefill_chunk`;
  - with the Gemma3 CUDA GPTQ default `active_decode_prefill_chunk=16`, an
    active decode iteration can mix at most two 16-token prefill chunks.
- Local validation:
  - `cargo test -p ferrum-scheduler active_decode_prefill_chunk -- --nocapture`
    PASS: `2 passed`;
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-scheduler` PASS: `52 passed`.
- Required next validation:
  - run Gemma3 product `ferrum run` and `ferrum serve` smoke on native CUDA;
  - then rerun same-pod c16 ShareGPT with
    `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`;
  - diagnostic candidate only passes if throughput LCB clears the vLLM 80%
    line while p95 ITL remains at or below the same-pod vLLM p95.

## 2026-06-17 ZE — W2 CUDA diagnostic: aggregate mixed-prefill cap fixes c16 p95 but over-throttles throughput

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_decode_prefill_budget_c16_cuda_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ post-scheduler-change c16 diagnostic only;
  - reused same 1x RTX 4090 Vast instance `41241013`;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- GPU lifecycle:
  - started cached instance `41241013`, 1x RTX 4090, quoted rate
    `0.47111111111111115 USD/h`;
  - verified CUDA with driver `580.95.05`, CUDA toolkit `12.4`, and
    24GB RTX 4090 visibility;
  - synced clean source to commit
    `699add71ad4a86cfaf6ee6ee00a98d87c27d18d2`;
  - copied artifacts back locally, then stopped the instance;
  - final sanitized Vast state recorded `cur_state=stopped`,
    `actual_status=exited`.
- Source/runtime evidence:
  - remote worktree was clean at
    `699add71ad4a86cfaf6ee6ee00a98d87c27d18d2`;
  - Ferrum binary SHA256
    `49a60008497419336dafd283eb7394c334494ee25ea36fb2f308c87d10c2dee4`;
  - dataset SHA256
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`;
  - model snapshot:
    `/workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2`;
  - effective config check confirmed
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`.
- Correctness result:
  - product `ferrum run` smoke passed with stdout content `5`,
    `n_tokens=3`;
  - product `ferrum serve` streaming smoke passed with content `5\n`,
    exactly one `[DONE]`, and usage present
    (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`);
  - c16 `bench-serve` completed `[100,100,100]` requests with
    errors `[0,0,0]`;
  - `output_token_count_source=usage`;
  - no correctness issue was observed in this diagnostic artifact.
- Performance command:
  - `ferrum bench-serve --dataset sharegpt --sharegpt-path
    /workspace/ascii_sharegpt_w2_100.jsonl --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3
    --fail-on-error --require-ci --seed 9271`;
  - same model snapshot and same ASCII ShareGPT 100 dataset as the same-pod
    vLLM reference from `w2_vllm_same_hw_c16_sharegpt_2026-06-17`.
- Result:
  - Ferrum output throughput mean/LCB:
    `339.927090896005 / 333.10968230699876 tok/s`;
  - same-pod vLLM reference mean/LCB:
    `500.67038762731977 / 478.39462812583776 tok/s`;
  - Ferrum LCB / vLLM LCB = `0.6963073218693773`, so c16 throughput now
    fails the 80% diagnostic line;
  - Ferrum p95 ITL mean `26.63676561666667 ms`;
  - same-pod vLLM reference p95 ITL mean `33.06958213333332 ms`;
  - Ferrum p95 ITL / vLLM p95 ITL = `0.8054763289499648`, so c16 p95
    diagnostic now passes;
  - relative to previous Ferrum c16 same-pod result, LCB changed by
    `-81.4818495619952 tok/s` and p95 ITL changed by `-26.182588216666662 ms`.
- Interpretation:
  - the active-decode mixed-prefill root cause was real: c16 p95 dropped from
    `52.819ms` to `26.637ms`;
  - the aggregate cap at exactly `16` tokens is too strict for throughput:
    it converts a tail-latency win into an overall release-grade failure;
  - this patch is therefore not a final W2 fix and must not be widened to the
    full matrix as-is.
- Next direction:
  - keep the aggregate mixed-prefill concept, but make the active-decode
    budget adaptive or larger under healthy decode cadence so p95 remains
    controlled while c16 LCB returns above the vLLM 80% line;
  - run the next validation as another native CUDA c16 minimum only before
    considering any c=1/4/32 expansion.

## 2026-06-17 ZD — W2 source checkpoint: cap aggregate mixed prefill during active decode

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_scheduler_active_decode_prefill_budget_source_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ source-only p95 latency lever;
  - no paid GPU instance was started;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Bottleneck refinement from existing c16 profile:
  - same-pod c16 throughput already cleared the 80% diagnostic line:
    Ferrum/vLLM LCB ratio `0.8666308262975292`;
  - p95 ITL still failed: Ferrum/vLLM p95 ratio
    `1.597218665188174`;
  - pure decode frames in
    `w2_tail_latency_profile_c16_samepod_2026-06-17` had p95
    `27.978ms`, close to the vLLM c16 p95 ITL `33.0696ms`;
  - mixed prefill+decode frames had p95 `74.050ms`;
  - decode-token-weighted frame p95 was `64.029ms`;
  - `68.9%` of decoded tokens were emitted from frames with at least
    3 active prefill requests mixed into the decode step.
- Source change:
  - `active_decode_prefill_chunk` now acts as an aggregate per-iteration
    mixed-prefill budget when decode requests are scheduled, not only as a
    per-prefill-request chunk cap;
  - with the Gemma3 CUDA GPTQ default `active_decode_prefill_chunk=16`, an
    active decode iteration can admit at most one 16-token prefill chunk
    instead of `N waiting requests * 16 tokens`;
  - added scheduler regression test
    `active_decode_prefill_chunk_caps_aggregate_mixed_prefill_tokens`.
- Local validation:
  - `cargo test -p ferrum-scheduler active_decode_prefill_chunk -- --nocapture`
    PASS;
  - `cargo test -p ferrum-scheduler newly_admitted_prefill_uses_remaining_budget_with_decode -- --nocapture`
    PASS;
  - `cargo test -p ferrum-scheduler` PASS: `52 passed`;
  - `cargo fmt --all -- --check` PASS.
- Next required validation:
  - run Gemma3 product `ferrum run` and `ferrum serve` smoke on native CUDA;
  - then run same-pod c16 ShareGPT A/B with
    `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`;
  - expected profile change: active-decode mixed frames should stop showing
    `prefill=7-11`; if p95 still fails, rerun the focused tail profile with
    both `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1`.

## 2026-06-17 ZC — W2 CUDA diagnostic: c16 tail profile points to Gemma3 GPTQ dense/MLP decode path

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_latency_profile_c16_samepod_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ current-default c16 tail-latency/profile diagnostic
    only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- GPU lifecycle:
  - reused Vast instance `41241013`, 1x RTX 4090, with the retained
    same-pod model/source/build environment from the c16 vLLM/Ferrum A/B;
  - copied artifacts back locally, then stopped the instance;
  - final sanitized Vast state records `cur_state=stopped` and
    `actual_status=exited`.
- Source/runtime evidence:
  - remote worktree was clean at
    `96d2df73e82ab4c0d643ced32d1f424b29dc5353`;
  - Ferrum binary SHA256
    `ca11f78f9e1be27a26bd12f50e377f3def602f14220cb10e1099eadb4f35ca93`;
  - dataset SHA256
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`;
  - model snapshot:
    `/workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2`.
- Correctness result:
  - product `ferrum serve` streaming smoke passed with content `5\n`,
    exactly one `[DONE]`, and usage present
    (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`);
  - diagnostic `bench-serve` completed `[100]` requests with errors `[0]`;
  - `output_token_count_source=usage`;
  - no correctness issue was observed in this diagnostic artifact.
- Diagnostic command:
  - current-default Ferrum product server with
    `FERRUM_DECODE_OP_PROFILE=1`;
  - `ferrum bench-serve --dataset sharegpt --sharegpt-path
    /workspace/ascii_sharegpt_w2_100.jsonl --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 1
    --fail-on-error --seed 9271`;
  - this is not release performance evidence because profile logging changes
    runtime cost and the run intentionally omitted `--require-ci`.
- Profile result:
  - profile parser found `176` `unified-op-profile` rows, including `21`
    decode-only rows;
  - profile-run output throughput mean was `372.00153904953766 tok/s` and
    p95 ITL mean was `57.69527414999998 ms`;
  - decode-only `total_us` mean/p95/max:
    `27052.52380952381 / 27978 / 28153`;
  - decode-only `generic_matmul` share was `0.7742046776728868`;
  - decode-only `gate_up + down` share was `0.5074167888569503`;
  - decode-only attention share was `0.08208370665178674`;
  - decode-only lm_head share was `0.11125447322052515`;
  - `marlin_*` profile buckets were zero in the captured product path.
- Interpretation:
  - the remaining c16 tail issue is not primarily FA2/attention;
  - the current evidence points to Gemma3 GPTQ dense matmul, especially MLP
    `gate_up` and `down`, as the dominant decode cost;
  - the next source step should compare Ferrum's Gemma3 GPTQ dense dispatch
    and packing against vLLM's GPTQ/Marlin path, then use a native CUDA
    microbench for the exact Gemma3 `gate_up/down` shapes before changing
    product defaults.

## 2026-06-17 ZB — W2 CUDA diagnostic: same-pod vLLM/Ferrum c16 throughput passes, p95 remains blocker

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_same_hw_c16_sharegpt_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ same-pod c16 ShareGPT diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- GPU lifecycle:
  - cached instance `41230499` could not be restarted because Vast returned
    `resources_unavailable`, so no more time was spent on that host;
  - created instance `41241013`, 1x RTX 4090, Netherlands, driver
    `580.95.05`, CUDA devel image `nvidia/cuda:12.4.0-devel-ubuntu22.04`,
    quoted total rate `0.47111111111111115 USD/h`;
  - synced clean source by git bundle, prefetched the model into
    `/workspace/hf-cache`, built Ferrum CUDA release with
    `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - copied artifacts back, then stopped the instance; final sanitized Vast
    state recorded `cur_state=stopped`, `actual_status=exited`.
- Source/runtime evidence:
  - remote worktree was clean at
    `96d2df73e82ab4c0d643ced32d1f424b29dc5353`;
  - Ferrum binary SHA256
    `ca11f78f9e1be27a26bd12f50e377f3def602f14220cb10e1099eadb4f35ca93`;
  - dataset SHA256
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`;
  - vLLM baseline used `vllm=0.10.1.1`, `torch=2.7.1+cu126`,
    `transformers=4.55.4`, `fastapi=0.116.1`, `starlette=0.47.2`;
  - Ferrum effective config selected
    `scheduler_admission_policy=active_decode_prefill_chunk:16`;
  - Ferrum still selected `legacy_paged_varlen`, `legacy_paged_decode`,
    `legacy_moe`, and `graph_disabled` for Gemma3 in the product default path.
- Correctness result:
  - vLLM `/v1/models` and streaming smoke passed with content `5\n`, exactly
    one `[DONE]`, and usage present;
  - product `ferrum run` smoke passed with stdout content `5`,
    `n_tokens=3`;
  - product `ferrum serve` streaming smoke passed with content `5\n`, exactly
    one `[DONE]`, and usage present
    (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`);
  - no correctness issue was observed in this diagnostic artifact.
- Performance command:
  - both vLLM and Ferrum used `ferrum bench-serve --dataset sharegpt
    --random-output-len 128 --concurrency-sweep 16 --num-prompts 100
    --n-repeats 3 --fail-on-error --require-ci --seed 9271`;
  - both used the same model snapshot and same ASCII ShareGPT 100 dataset;
  - both had `completed_per_run=[100,100,100]`,
    `errored_per_run=[0,0,0]`, all quality/error counts zero, and
    `output_token_count_source=usage`.
- Same-pod c16 result:
  - vLLM output throughput mean/LCB:
    `500.67038762731977 / 478.39462812583776 tok/s`;
  - Ferrum output throughput mean/LCB:
    `422.34520497237537 / 414.59153186899397 tok/s`;
  - Ferrum LCB / vLLM LCB = `0.8666308262975292`, so the c16 throughput
    diagnostic clears the 80% line on same hardware;
  - vLLM p95 ITL mean `33.06958213333332 ms`;
  - Ferrum p95 ITL mean `52.81935383333333 ms`;
  - Ferrum p95 ITL / vLLM p95 ITL = `1.597218665188174`, so the p95 tail
    diagnostic still fails.
- Interpretation:
  - there is real progress: c16 throughput is no longer below the vLLM 80%
    threshold when measured on the same pod;
  - the remaining c16 blocker is tail latency, not mean/LCB throughput;
  - prior diagnostics already showed typed VPA/FA2 product toggles do not
    materially improve Gemma3 c16 and that decode-step time is dominated by
    Gemma3 tail MLP/GPTQ dense projection, so the next optimization should
    target Gemma3 decode/tail dense path and graph/fast-path integration rather
    than more unscoped env sweeps.
- Required next validation:
  - add a focused tail-latency diagnostic for Gemma3 decode dense buckets
    against this same-pod baseline;
  - once p95 improves, expand the same-hardware matrix to c=1/4/32 before any
    W2 release-grade claim.

## 2026-06-17 ZA — W2 CUDA diagnostic: same-iteration admission clears historical c16 throughput threshold

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_same_iteration_admit_c16_ci_2026-06-17/`.
- Source checkpoint:
  `674c66786f0cf654009c84070b65ec0174a95357`
  (`perf(scheduler): admit prefills with remaining step budget`).
- Scope:
  - W2 Gemma3 CUDA GPTQ c16 ShareGPT CI diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- GPU lifecycle:
  - restarted cached Vast instance `41230499`, 1x RTX 4090, driver
    `590.48.01`, quoted USD `0.5766666666666667/h`;
  - reused existing model/build cache; CUDA release build completed in
    `3m21.770s`;
  - artifacts copied back, then instance stopped; final status
    `cur_state=stopped`, `actual_status=exited`.
- Source/runtime evidence:
  - remote worktree was clean at
    `674c66786f0cf654009c84070b65ec0174a95357`;
  - binary sha256
    `9c7def4de9568657798c3be5dacd3fb6a5b72ced87efa84552966f1bb8320fa6`;
  - dataset sha256
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`;
  - effective config showed
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`.
- Correctness result:
  - product `ferrum run` smoke passed with stdout content `5`,
    `n_tokens=3`;
  - product `ferrum serve` streaming smoke passed with content `5\n`,
    exactly one `[DONE]`, and usage present
    (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`);
  - no correctness issue observed in this diagnostic artifact.
- Performance command:
  - `ferrum bench-serve --dataset sharegpt --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3
    --fail-on-error --require-ci --seed 9271`;
  - completed_per_run `[100, 100, 100]`, errored_per_run `[0, 0, 0]`;
  - all quality/error counts were zero;
  - `output_token_count_source=usage`.
- c16 result:
  - output throughput mean `402.71236961652994 tok/s`;
  - CI half-width `4.727831673618258 tok/s`;
  - LCB `397.9845379429117 tok/s`;
  - previous active-chunk c16 LCB was `386.46217408059744 tok/s`, so this
    checkpoint adds `11.52236386231426 tok/s` LCB;
  - historical same-dataset vLLM c16 LCB from
    `w2_ferrum_natural_c16_same_shape_2026-06-16` was `491.150 tok/s`;
  - diagnostic LCB ratio vs that historical vLLM baseline is `81.03%`;
  - historical 80% threshold is `392.920 tok/s`, so this run is above that
    diagnostic threshold by `5.06453794291167 tok/s`;
  - p95 ITL is `57.782 ms`, only slightly better than previous `58.728 ms`
    and still about `2.05x` the historical vLLM p95 ITL `28.130 ms`.
- Status:
  - c16 throughput now has a credible diagnostic pass against the historical
    same-dataset vLLM LCB threshold;
  - W2 is still not release-grade because it lacks same-hardware vLLM baseline,
    c=1/4/32 cells, full L0-L5 correctness, and final validator PASS;
  - the remaining blocker is tail latency, not c16 throughput mean/LCB.

## 2026-06-17 YZ — W2 source checkpoint: same-iteration admission uses remaining decode-step budget

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_scheduler_same_iteration_admit_source_2026-06-17/`.
- Scope:
  - source-only scheduler checkpoint;
  - no paid GPU instance was started;
  - no release performance claim and no `MODEL_RELEASE_GRADE_W2 PASS` was
    produced.
- Bottleneck interpretation:
  - the latest c16 CI diagnostic moved Ferrum from historical c16 LCB
    `325.184 tok/s` to `386.462 tok/s`, but p95 ITL is still `58.728 ms`;
  - vLLM's scheduler spends remaining per-step token budget on waiting
    requests after running requests in the same scheduler step;
  - Ferrum admitted waiting requests after decode/existing-prefill collection,
    but only scheduled newly admitted prefills in the same iteration when the
    current batch was otherwise empty;
  - this creates a concrete one-iteration delay for closed-loop replacement
    requests and is consistent with the remaining TTFT/ITL tail.
- Source change:
  - factored prefill collection into `add_prefill_requests_to_batch`;
  - added request-ID de-duplication so prefill requests already scheduled
    before admission cannot be scheduled twice;
  - after waiting admission, newly admitted prefills can now use remaining
    batch slot/token budget even when decode work is already present.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-scheduler newly_admitted_prefill_uses_remaining_budget_with_decode -- --nocapture` PASS;
  - `cargo test -p ferrum-scheduler active_decode_prefill_chunk_only_caps_when_decode_is_active -- --nocapture` PASS;
  - `cargo test -p ferrum-scheduler continuous -- --nocapture` PASS;
  - `cargo test -p ferrum-scheduler --lib` PASS;
  - `cargo check -q -p ferrum-scheduler -p ferrum-engine -p ferrum-cli` PASS;
  - `git diff --check` PASS.
- Required next validation:
  - restart the cached 1x4090 instance only for a minimal c16 ShareGPT CI
    diagnostic;
  - run product `ferrum run` and `ferrum serve` correctness first;
  - then rerun the same c16 `bench-serve --fail-on-error --require-ci
    --seed 9271 --n-repeats 3 --num-prompts 100` shape to see whether p95 ITL
    and the remaining ~`6.46 tok/s` historical 80% gap move.

## 2026-06-17 YY — W2 CUDA diagnostic: active chunk reaches 78.69% of historical vLLM c16 LCB

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_chunk_sharegpt_c16_ci_2026-06-17/`.
- Scope:
  - W2 Gemma3 CUDA GPTQ c16 ShareGPT CI diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- GPU lifecycle:
  - restarted cached Vast instance `41230499`, 1x RTX 4090, driver
    `590.48.01`, quoted USD `0.5766666666666667/h`;
  - reused existing model/build cache; CUDA release build completed in
    `0.298s`;
  - artifacts copied back, then instance stopped; final status
    `cur_state=stopped`, `actual_status=exited`.
- Source/runtime evidence:
  - remote worktree was clean at
    `b99afdea19c11cdb4e6244ab2f5bedda20624bdb`;
  - binary sha256
    `426c9b029d08ede6edb986a7dd80e5330e2a9f7489ce7de6224a1b482361d4c7`;
  - dataset sha256
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`;
  - effective config showed
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`.
- Correctness result:
  - product `ferrum run` smoke passed with stdout content `5`,
    `n_tokens=3`;
  - product `ferrum serve` streaming smoke passed with content `5\n`,
    exactly one `[DONE]`, and usage present
    (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`);
  - no correctness issue observed in this diagnostic artifact.
- Performance command:
  - `ferrum bench-serve --dataset sharegpt --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3
    --fail-on-error --require-ci --seed 9271`;
  - completed_per_run `[100, 100, 100]`, errored_per_run `[0, 0, 0]`;
  - all quality/error counts were zero;
  - `output_token_count_source=usage`.
- c16 result:
  - output throughput mean `393.3008267456301 tok/s`;
  - CI half-width `6.8386526650326855 tok/s`;
  - LCB `386.46217408059744 tok/s`;
  - historical same-dataset vLLM c16 LCB from
    `w2_ferrum_natural_c16_same_shape_2026-06-16` was `491.150 tok/s`;
  - diagnostic LCB ratio vs that historical vLLM baseline is `78.69%`;
  - historical 80% threshold is `392.920 tok/s`, so the remaining gap is
    `6.45782591940258 tok/s`;
  - p95 ITL improved from the old Ferrum `83.979 ms` to `58.728 ms`, but this
    still exceeds the historical vLLM p95 ITL `28.130 ms` by about `2.09x`.
- Status:
  - this is a real performance movement from the prior release-shaped Ferrum
    c16 LCB `325.184 tok/s` to `386.462 tok/s`;
  - c16 is now close enough that the next release-grade step should be a
    same-hardware vLLM baseline on the same instance/shape before deciding
    whether to tune the final ~6.5 tok/s gap or proceed to c=1/4/32;
  - W2 remains not release-grade until the final validator prints
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## 2026-06-17 YX — W2 CUDA diagnostic: active-decode prefill chunk removes large mixed prefill/decode frames

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_decode_chunk_c16_cuda_diag_2026-06-17/`.
- Source checkpoint:
  `eed031e334c78cf181a4b1077c1ba2089d0d6d6f`
  (`perf(types): default gemma3 gptq active prefill chunk`).
- GPU lifecycle:
  - reused-start attempts for cached instances were unavailable;
  - run used Vast instance `41230499`, 1x RTX 4090, driver `590.48.01`,
    quoted USD `0.5766666666666667/h`;
  - artifacts copied back, then instance stopped; final status
    `cur_state=stopped`, `actual_status=exited`.
- Build/runtime evidence:
  - CUDA release build passed with
    `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - build time `29m25.806s`;
  - binary sha256
    `426c9b029d08ede6edb986a7dd80e5330e2a9f7489ce7de6224a1b482361d4c7`;
  - remote worktree was clean at
    `eed031e334c78cf181a4b1077c1ba2089d0d6d6f`.
- Correctness result:
  - product `ferrum run` smoke passed with stdout content `5`;
  - product `ferrum serve` streaming smoke passed with content `5\n`,
    exactly one `[DONE]`, and usage present;
  - no correctness issue observed in this diagnostic artifact.
- Runtime config result:
  - effective config materialized
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`;
  - decision trace selected
    `scheduler_admission_policy=active_decode_prefill_chunk:16` from
    `model_metadata`.
- c16 diagnostic performance:
  - command used random 256/128, concurrency `16`, `num-prompts=16`,
    `n-repeats=1`, `--fail-on-error`, seed `9271`;
  - bench completed `[16]`, errored `[0]`,
    `output_token_count_source=usage`;
  - diagnostic output throughput was `294.61885808275144 tok/s`;
  - this is smoke/diagnostic evidence only, not release performance evidence.
- Profile result:
  - summary reported `large_mixed_prefill_decode_lines=[]`;
  - the previous target bad frame was
    `m_total=897 num_seqs=16 prefill=12 decode=4 total=383684us`;
  - this run's largest mixed prefill+decode frame was chunk-shaped
    `m_total=151 num_seqs=16 prefill=9 decode=7`, with sampled examples
    around `79ms` to `107ms`;
  - large full-prompt prefill still appears as pure prefill, e.g.
    `m_total=1866 prefill=7 decode=0`, which is expected.
- Status:
  - active-decode chunking is now the concrete W2 scheduler lever to carry
    forward;
  - still no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - next step is either same-shape default-vs-chunk A/B for a clean delta, or
    the W2 goal gate once the remaining release-grade acceptance matrix is
    ready.

## 2026-06-17 YW — W2 source checkpoint: Gemma3 CUDA GPTQ defaults active-decode prefill chunking

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma3_active_decode_chunk_source_2026-06-17/`.
- Scope:
  - source-only checkpoint; no paid GPU instance was started;
  - no release performance claim and no `MODEL_RELEASE_GRADE_W2 PASS` was
    produced.
- Bottleneck interpretation update:
  - latest c16 profile after the logits-readback fix still names
    MLP/Marlin as the largest bucket, but prior native Ferrum-vs-vLLM dense
    Marlin probes already rejected a direct kernel swap;
  - the actionable new signal is scheduler cadence: sampled frames include
    `prefill=12 decode=4 m_total=897`, while pure decode frames are
    `prefill=0 decode=4` and much smaller;
  - local vLLM source uses running-first scheduling plus token-budgeted
    chunked prefill, while Ferrum had the typed knob but did not select it by
    default for Gemma3 CUDA GPTQ.
- Source change:
  - added a capability-gated Gemma3 CUDA GPTQ/int4 auto-config default:
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`;
  - the default materializes into the runtime snapshot and decision trace as
    `scheduler_admission_policy=active_decode_prefill_chunk:16`;
  - explicit user/config/CLI scheduler choices still override it:
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK`,
    `FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE`, or
    `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE`.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-types auto_config -- --nocapture` PASS;
  - `cargo test -p ferrum-scheduler active_decode_prefill_chunk_only_caps_when_decode_is_active -- --nocapture` PASS;
  - `cargo test -p ferrum-engine continuous_engine_runtime_config_parses_env_snapshot -- --nocapture` PASS;
  - `cargo check -q -p ferrum-types -p ferrum-scheduler -p ferrum-engine -p ferrum-cli` PASS;
  - `git diff --check` PASS.
- Required next validation:
  - one 1x4090 diagnostic with product `ferrum run` and `ferrum serve`
    correctness smoke before performance;
  - c16 `bench-serve --fail-on-error` with the existing profile knobs;
  - accept the lever only if effective config shows the typed chunk default and
    profile frames replace full-prompt mixed batches with small active-decode
    prefill chunks.

## 2026-06-17 YV — W2 CUDA diagnostic: dense unified argmax removes readback bottleneck

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_argmax_c16_cuda_diag_2026-06-17/`.
- Source checkpoint:
  `63b8565eb4f40b3bc94ac48729cd4cd0fd00b2b0`
  (`perf(models): use logits policy in dense unified forward`).
- GPU lifecycle:
  - cached instances `41187356` / `41178475` could not restart
    (`resources unavailable`);
  - created `41218189`, but SSH/proxy never became usable; destroyed it;
  - actual run used Vast `41218739`, 1x RTX 4090, driver `580.95.05`,
    quoted USD `0.4696296296296296/h`;
  - artifacts copied back, then `41218739` stopped; final status
    `cur_state=stopped`, `actual_status=exited`.
- Build/runtime evidence:
  - binary sha256
    `b0676434810f2824094a12a1ccd9bea666a9aaf4e72bfd404c00455888ef407f`;
  - CUDA release build passed; first fresh build took `43m 52s`;
  - product `ferrum run` smoke passed with stdout `5`, rc `0`;
  - product `ferrum serve` streaming smoke passed with exactly one `[DONE]`
    and usage present;
  - c16 `bench-serve --fail-on-error` completed `[16]`, errored `[0]`,
    `output_token_count_source=usage`.
- Product-path diagnostic performance:
  - non-profile c16 random 64/16: `17.0821 req/s`,
    `273.3137 output tok/s`, TTFT p50/p95 `583.8/584.2 ms`,
    ITL p95 `27.18 ms`;
  - this is diagnostic `n=1` evidence only, not release evidence.
- Profile rerun:
  - reran from artifact CWD so `ferrum.toml` profile entries were actually
    loaded;
  - effective config showed `FERRUM_BATCH_DECODE_PROF=1`,
    `FERRUM_NEXT_BATCH_PROF=1`, `FERRUM_UNIFIED_POST_PROF=1`,
    `FERRUM_DECODE_OP_PROFILE=1`, `FERRUM_MARLIN_PROFILE=1`;
  - profile c16 bench also completed `[16]`, errored `[0]`,
    `output_token_count_source=usage`;
  - profile throughput was `167.0513 tok/s` because sync-heavy profiling was
    enabled.
- Bottleneck result:
  - previous same-shape mixed c16 frame had `readback=22039us`;
  - target rerun frame:
    `call#21 m_total=897 num_seqs=16 prefill=12 decode=4 total=383684us`;
  - `readback=516us`, i.e. `0.0234x` of the previous readback time;
  - remaining dominant buckets are MLP/Marlin:
    `gate_up=174891us`, `down=110906us`, `marlin_kernel=312084us`;
  - `lm_head=3167us` and `unwrapped=726us`, so the immediate bottleneck is
    no longer logits readback/lm_head.
- Status:
  - no known correctness issue in this W2 c16 diagnostic;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced;
  - next high-return work is vLLM source comparison plus native CUDA
    minimal validation around Gemma3 GPTQ dense Marlin MLP
    `gate_up/down` projection behavior.

## 2026-06-17 YU — W2 source checkpoint: dense unified logits policy now avoids full readback for greedy rows

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_argmax_source_2026-06-17/`.
- Scope:
  - no GPU instance was started;
  - no performance measurement was taken;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Bottleneck link:
  - `w2_unified_op_profile_c16_rerun_2026-06-16` measured
    `readback=22039us` in a mixed c16 dense unified frame;
  - batched decode already had GPU argmax sentinel support, but dense
    `unified_forward_internal` always downloaded `sampled * vocab` logits.
- Source change:
  - added `DecoderOnlyLLM::unified_forward_with_logits_policy(...)` with a
    backwards-compatible default;
  - `LlmExecutor::unified_decode` now forwards `UnifiedBatchItem.logits_policy`
    to real unified model execution and treats policy-required full logits as a
    full-logits condition;
  - dense `LlamaFamilyModel` unified forward now uses existing GPU
    `argmax_rows_f16` / `argmax_rows_f16_masked` when all sampled rows are
    greedy-compatible and masks are uniform;
  - any full-logits requirement, non-greedy sampling, structured output, or
    incompatible mixed masks still falls back to full logits;
  - default no-prefix-cache final prefill chunks now carry the same greedy
    model-side policy as decode rows, while prefix-cache-enabled runs still
    force full logits for cache storage.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo test -p ferrum-models unified_decode_ -- --nocapture` PASS;
  - `cargo test -p ferrum-engine model_decode_logits_policy -- --nocapture`
    PASS;
  - `cargo check -q -p ferrum-models -p ferrum-engine` PASS.
- Required next validation:
  - one cached 1x4090 diagnostic with product `ferrum run`/`serve` smoke first;
  - then c16 `bench-serve --fail-on-error` with decode op profile enabled;
  - accept this branch only if correctness remains clean and unified
    `readback`/endpoint throughput improve versus the current same-shape
    diagnostic.

## 2026-06-16 YT — W2 c16 token-budget A/B: simple token cap is not the bottleneck fix

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_token_budget_c16_ab_2026-06-16/`.
- Source checkpoint before run:
  `5f01af002d44ec58e2242f63ff085e54ba9a9e8c`
  (`docs(cuda): record w2 unified op profile diagnostic`), clean remote
  worktree.
- GPU lifecycle:
  - old stopped instance `41187356` could not be restarted
    (`resources_unavailable`);
  - new instance `41210668` stayed `actual=loading` with SSH refused and was
    deleted;
  - actual run used Vast `41212840`, 1x RTX 4090, driver `580.119.02`,
    CUDA `12.4`, offer `36846332`, quoted USD `0.4044/h`;
  - artifacts were copied back, then `41212840` was deleted; Vast DELETE
    returned HTTP 200 success.
- Binary evidence:
  `649a73fc2ec46ab4272a14390422a1bfc565243a4b638c6f775e6cc5b15d8962`
  for `/workspace/ferrum-target/release/ferrum`.
- Product path and command shape:
  - `ferrum serve --model gemma3:27b-gptq --backend cuda --kv-capacity 512
    --max-num-seqs 16 --max-num-batched-tokens <1024|512>`;
  - per cell: streaming `2+3` smoke, then `bench-serve --dataset random
    --random-input-len 64 --random-output-len 16 --concurrency 16
    --num-prompts 16 --warmup-requests 4 --n-repeats 1 --fail-on-error
    --seed 9271`.
- Correctness/resource result:
  - `SMOKE_OK 1024 True`;
  - `SMOKE_OK 512 True`;
  - both bench cells completed `[16]`, errored `[0]`;
  - both cells used `output_token_count_source=usage`;
  - log scan found no panic, OOM, illegal address, or CUDA error.
- Diagnostic performance:
  - `max_num_batched_tokens=1024`: `12.722 req/s`,
    `203.552 output tok/s`, TTFT p50/p95 `610.6/662.3 ms`, TPOT p50
    `42.7 ms`, ITL p95 `60.8 ms`;
  - `max_num_batched_tokens=512`: `12.226 req/s`,
    `195.621 output tok/s`, TTFT p50/p95 `536.3/720.2 ms`, TPOT p50
    `49.3 ms`, ITL p95 `159.2 ms`.
- Profile interpretation:
  - `1024` still formed a large mixed-prefill frame:
    `items=16 prefill=11 decode=5 total_q=823`, model batch `334383us`;
  - `512` split the prefill work, e.g.
    `items=16 prefill=3 decode=13 total_q=235`, model batch `118779us`,
    but throughput and tail latency worsened;
  - conclusion: a simple typed token-budget reduction is not the W2
    high-return lever. Next work should focus on Gemma3 GPTQ dense MLP
    Marlin projection behavior, weight residency/permute overhead, or a more
    targeted admission policy than globally reducing
    `max_num_batched_tokens`.
- Gate status:
  - diagnostic only, `n_repeats=1`;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YS — W2 c16 unified op profile: bottleneck is GPTQ Marlin MLP, not attention

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_op_profile_c16_rerun_2026-06-16/`.
- Source checkpoints:
  - `0d52b7b5 test(models): add unified forward op profiling`;
  - `1707b001 test(models): sample large unified op profiles`.
- GPU contract:
  - reused Vast instance `41187356`, 1x RTX 4090, USD `0.38/h`;
  - lane: W2 Gemma3 27B GPTQ c16 minimal diagnostic;
  - correctness gate: product `ferrum serve` streaming smoke, then
    `bench-serve --fail-on-error`;
  - performance command: diagnostic-only c16 random 64/16, n=1, seed `9271`;
  - stop condition: collect target `[unified-op-profile]` frame or failure
    logs, then stop instance.
- Evidence:
  - remote clean worktree:
    `1707b001da835f99484f09dec252a9f3c66823e4`;
  - binary sha256:
    `e2117a1df9613b15a2df470c3f7fa6b50a873b16b6dff61925ce9a9d33d4239f`;
  - GPU: NVIDIA GeForce RTX 4090, driver `580.95.05`;
  - cleanup confirmed `cur_state=stopped`, `actual_status=exited`.
- Correctness/resource result:
  - serve smoke passed: `SMOKE_OK True`;
  - `bench-serve` rc `0`, completed `[16]`, errored `[0]`;
  - output token count source was `usage`;
  - log scan found no panic, OOM, illegal address, or CUDA error.
- Diagnostic performance:
  - request throughput `9.9298 req/s`;
  - output throughput `158.877 tok/s`;
  - TTFT p50/p95 `737.5 ms` / `825.0 ms`;
  - TPOT p50/p95 `57.7 ms` / `86.6 ms`.
- Bottleneck evidence:
  - target frame:
    `call#23 m_total=822 num_seqs=16 prefill=11 decode=5 total=339796us`;
  - major components:
    `gate_up=143991us`, `down=82787us`, `attn=18567us`,
    `readback=22039us`, `qkv=30569us`;
  - Marlin kernels:
    `marlin_gate_up_kernel=141474us`,
    `marlin_down_kernel=71062us`,
    `marlin_qkv_kernel=28120us`,
    `marlin_o_kernel=14974us`.
- Interpretation:
  - the current high-value W2 performance lever is Gemma3 GPTQ dense MLP
    Marlin (`gate_up/down`), not FA2/attention or another broad graph sweep;
  - next work should compare Ferrum and vLLM Marlin projection behavior at the
    same `m_total` shapes (`1`, `4`, `150`, `373`, `822`) using source review
    plus native CUDA/Rust CUDA microbench;
  - if single-op Marlin is already comparable, the lever moves back to
    scheduler/admission token budgeting to avoid TTFT-heavy mixed-prefill
    frames.
- Gate status:
  - diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YR — W2 c16 TTFT split: model prefill batch dominates queue wait

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_ttft_profile_c16_2026-06-16/`.
- Source checkpoint:
  `8eccd1c3 test(engine): add first token ttft profiling`.
- GPU contract:
  - reused Vast instance `41187356`, 1x RTX 4090;
  - expected runtime/cost: 10-20 minutes, hard stop 30 minutes, about
    USD `0.38/h`;
  - correctness gate: `ferrum serve` streaming smoke before profiling;
  - performance command: diagnostic-only c16 `bench-serve --fail-on-error
    --seed 9271 --n-repeats 1`.
- Evidence:
  - remote clean worktree: `8eccd1c33c752937cf903f63638eaa6d51bd643e`;
  - binary sha256:
    `ae817f5b086275a9c8689c8c991d504bb79b73fa39eac8032cbb2368972d5cd1`;
  - diagnostic profile toggles came from saved config-file runtime entries,
    not hidden env-only behavior;
  - cleanup confirmed `cur_state=stopped`, `actual_status=exited`.
- Correctness/resource result:
  - first attempt with `kv-capacity=2048` failed before smoke due CUDA OOM
    on a 128 MiB F16 allocation with about 94 MiB free;
  - retry with `kv-capacity=512`, `max-num-batched-tokens=1024` passed
    streaming smoke and `bench-serve` completed 16/16 requests with 0 errors.
- Diagnostic performance:
  - c16 random 64/16, n=1, throughput `167.9 tok/s`;
  - bench TTFT p50 `674.9 ms`, p95 `781.6 ms`;
  - `first-token-prof`: p50 queue-to-model-start `87.6 ms`, p50 model batch
    `421.6 ms`, p50 queue-to-first-token `559.4 ms`;
  - heaviest observed unified call:
    `items=15 prefill=13 decode=2 total_q=968 elapsed=421564us`.
- Interpretation:
  - this refines the bottleneck away from pure admission queueing: TTFT is
    mainly the large mixed Gemma3 GPTQ prefill/unified model call;
  - next W2 work should compare that same ~1k-token prefill shape against vLLM
    and isolate attention vs dense MLP/Marlin vs packing/logits in the model
    path.
- Gate status:
  - diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YQ — W2 TTFT bottleneck direction: token-budget scheduling, not graph/kernel swapping

- Scope:
  - no GPU instance was started;
  - compared Ferrum continuous-batch scheduling with vLLM v1 scheduler source;
  - added typed-diagnostic profile output on existing `batch_decode_prof` /
    `unified_post_prof` paths:
    - `first-token-prof`: request age at unified-prefill model start, unified
      model batch time, and request age when first token is sampled;
    - `stream-ttft-prof`: request age when the first non-empty SSE chunk is
      emitted.
- Current interpretation:
  - previous evidence showed single prefill around tens of ms while client
    TTFT was hundreds of ms under c=16 ShareGPT;
  - vLLM schedules work through one token-budget model over running/waiting
    requests, while Ferrum still exposes stronger prefill/decode phase queues;
  - the next W2 lever should test scheduler token-budget/admission behavior
    with this instrumentation before touching Marlin or graph code again.
- Local validation:
  - `cargo test -p ferrum-engine --lib continuous_engine` PASS;
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS.
- Gate status:
  - diagnostic instrumentation only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YP — Release-grade validator W3 self-test hardening

- Scope:
  - no GPU instance was started;
  - strengthened `scripts/release/model_release_grade_goal_gate.py --self-test`
    so it now validates a passing W3 manifest as well as the existing W2
    manifest path;
  - added a negative W3 self-test that deletes `w3_s0_microbench` and confirms
    the validator rejects the manifest.
- Why this matters:
  - `RELEASE_GRADE_GOAL.md` defines both W2 and W3 PASS lines through this
    validator;
  - before this checkpoint, the code path for W3 required correctness entries
    existed, but the self-test did not prove the W3-specific S0/S1/S2 fields.
- Local validation:
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS: `MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `python3 scripts/release/selftest_g0_validators.py`
    PASS: `G0 VALIDATOR SELFTEST PASS`.
- Gate status:
  - validator hardening only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` or
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` was produced.

## 2026-06-16 YO — W2 lm-head-eager graph diagnostic: correctness boundary found, not a perf lever

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_lm_head_eager_graph_cuda_smoke_2026-06-16/`.
- Source checkpoint:
  `dded3b7d test(cuda): add lm-head-eager graph scope`.
- GPU contract:
  - lane: `W2 Gemma3 CUDA unified graph lm-head-eager minimal diagnostic`;
  - expected runtime/cost: 15-30 minutes, hard cap 45 minutes, about
    USD `0.38/h` on the selected 1x RTX 4090 instance;
  - stop condition: start/SSH/CUDA/source sync/build failure, product smoke
    failure, graph illegal-address/OOM, or one small c16 diagnostic collected;
  - correctness gate: typed `ferrum run` and `ferrum serve` smoke;
  - performance command: diagnostic-only `bench-serve --fail-on-error --seed
    9271 --n-repeats 1`.
- Evidence:
  - Vast instance `41187356`, 1x RTX 4090, driver `580.95.05`, CUDA `12.4`;
  - model prefetch rc `0`;
  - dense CUDA diagnostic build rc `0`;
  - cleanup confirmed `cur_state=stopped`, `actual_status=exited`.
- Correctness result:
  - `ferrum run` rc `0`, content `5`, `n_tokens=3`;
  - `ferrum serve` chat response content `5`, usage present;
  - repeated same-shape serve requests logged `scope=lm_head_eager` capture and
    replay entries, with no illegal address.
- Diagnostic performance:
  - tiny c16 default: `246.621 tok/s`, completed `[16]`, errored `[0]`;
  - tiny c16 lm-head-eager: `233.070 tok/s`, completed `[16]`, errored `[0]`;
  - this was `n_repeats=1`, random 16/8, and is not release evidence.
- Interpretation:
  - `lm-head-eager` narrows the full unified graph crash suspect to the
    excluded `lm_head` / dense Marlin graph-capture or workspace-aliasing
    region;
  - graph capture is not the current W2 throughput lever, because the clean
    `lm-head-eager` scope did not improve endpoint throughput;
  - next W2 performance work should remain on Gemma3 GPTQ dense tail MLP /
    Marlin path and vLLM comparison, not another broad graph knob sweep.
- Gate status:
  - diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YN — W2 native CUDA graph segment probe: launch count alone does not explain unified-graph OOM

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_cuda_graph_segment_native_probe_2026-06-16/`.
- Source checkpoint:
  `c7250e82 test(cuda): add graph segment probe`.
- Scope:
  - compared vLLM's graph structure against Ferrum's unified graph path;
  - added a native CUDA probe that bypasses Cargo, Torch, vLLM runtime, model
    loading, and Ferrum server startup;
  - ran it once on the cached 1x RTX 4090 Vast instance `40826362`.
- GPU contract:
  - lane: `W2 Gemma3 graph capture granularity native diagnostic`;
  - expected runtime/cost: 5-10 minutes, hard cap 15 minutes, about USD
    `0.04-0.08`;
  - stop condition: compile/run failure or probe `VERDICT`, then collect
    artifacts and stop the instance;
  - correctness gate: process returns 0 and prints
    `VERDICT: CUDA graph segment probe complete`;
  - performance command:
    `./cuda_graph_segment_probe --segment-layers=1 --timed-iters=60 --warmup-iters=6`.
- Evidence:
  - binary SHA256
    `9614573b5df34e77e971e57cc3a43f0b2154368912c4fe9d022eb5fa2cdd2a9b`;
  - GPU `NVIDIA GeForce RTX 4090`, driver `565.77`, `nvidia-smi` CUDA
    `12.7`, `nvcc` `12.4.131`;
  - Vast cleanup confirmed `cur_state=stopped`, `actual_status=exited`,
    `intended_status=stopped`.
- Probe result:
  - eager simplified Gemma3-like launch pattern: `1.108744 ms/step`;
  - one monolithic simple graph: instantiate `1.963643 ms`, replay
    `0.795237 ms/step`;
  - 62 segmented simple graphs: instantiate total `2.662781 ms`, replay
    `0.880079 ms/step`;
  - segmented replay overhead versus monolithic replay: `1.194735x`;
  - verdict printed.
- Interpretation:
  - a Gemma3-like launch count alone does not reproduce Ferrum's prior
    `CUDA_ERROR_OUT_OF_MEMORY` during `--unified-graph` instantiation;
  - the remaining suspect is the real captured content/scope: Marlin and
    attention resource usage, graph memory-pool interaction, and/or final
    norm/lm_head/logit packing being captured into one large graph;
  - next implementation direction is vLLM-style segmented/breakable diagnostic
    graph capture with persistent buffers, not another runtime knob sweep.
- Gate status:
  - diagnostic CUDA evidence only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YM — W2 vLLM source diff checkpoint: dense Marlin ruled out, focus unified graph correctness

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_source_diff_2026-06-16/`.
- Scope:
  - compared local vLLM source at `/Users/chejinxuan/py_ws/vllm`
    (`0b3ba88f1`) with Ferrum Gemma3/GPTQ decode paths;
  - no new paid GPU run was started;
  - local Mac cannot execute vLLM CUDA ops because `torch` is not installed, so
    this checkpoint reuses existing same-4090 artifacts and local source
    inspection.
- Source comparison result:
  - Gemma3 semantics line up: fused QKV, Q/K norm, per-layer sliding window,
    query scale, Gemma sandwich norms, fused `gate_up`, GeGLU tanh, and final
    logits softcap are all represented;
  - dense GPTQ Marlin is not the main remaining gap: existing native
    Ferrum-vs-vLLM Marlin probes show m16 `gate_up` and weight-cycle `down`
    are effectively tied under product-relevant shapes;
  - vLLM's remaining structural advantage is decode integration: persistent
    GPU input buffers plus CUDA graph dispatch/replay for uniform decode;
  - Ferrum's product-clean Gemma3 path is still eager; product
    `--batched-graph` did not help, while `--unified-graph` is the closer match
    to vLLM but is not correctness-clean yet.
- Reused evidence:
  - vLLM ShareGPT c16/c32 baseline: `518.796` / `524.128 tok/s`, zero errors;
  - latest Ferrum default c16 diagnostic: `320.311 tok/s`, correctness clean;
  - typed profile m16 decode: about `30.2 ms` per decode step, with
    `tail_mlp` about `14.8 ms`, `marlin_kernel` about `16.6 ms`, and attention
    about `2.5-2.7 ms`;
  - unified graph c16 diagnostic previously failed under bench with
    `CUDA_ERROR_ILLEGAL_ADDRESS`; a key-fix smoke later avoided that specific
    crash but graph instantiation still hit `CUDA_ERROR_OUT_OF_MEMORY` and
    fell back.
- Correctness status:
  - no new default-path correctness issue found;
  - unified graph remains a correctness blocker and must not be used for
    performance claims.
- Next direction:
  - stop broad runtime knob sweeps;
  - run only a targeted minimal unified-graph correctness/memory validation on
    one Gemma3 decode shape when using the cached 4090 again;
  - collect the exact graph node/failing kernel evidence first, then only
    compare eager vs graph replay after correctness is stable.
- Gate status:
  - diagnostic source-diff checkpoint only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YL — W2 active-decode prefill chunk c16 diagnostic: latency tradeoff, not throughput lever

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_decode_prefill_chunk_c16_diag_2026-06-16/`.
- Paid GPU lane:
  `W2 c16 active-decode prefill chunk diagnostic` on cached Vast instance
  `40826362`, 1x RTX 4090.
- Contract:
  - expected runtime/cost: 25-45 minutes, hard cap 60 minutes, about
    USD 0.18-0.32 at USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/build/`ferrum run`/serve smoke first
    failure, or complete default vs `--scheduler-active-decode-prefill-chunk 32`
    c16 diagnostic, copy artifact, and confirm instance exited;
  - correctness gate: release CUDA build, `ferrum run` 2+3, default and
    chunk32 `ferrum serve` chat smoke;
  - performance command:
    `ferrum bench-serve --dataset sharegpt --random-output-len 128 --concurrency-sweep 16 --num-prompts 64 --n-repeats 1 --fail-on-error --seed 9271`.
- Evidence hygiene:
  - source head `8bc7cf087ae5fe6e7e2e34405ca5781cc8d0acdc`;
  - binary SHA256
    `786bbd8bf2536d46328e1daf4453cc81dbb24213d23914e3f34893582bb32717`;
  - build features `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - GPU `NVIDIA GeForce RTX 4090`, driver `565.77`, `nvidia-smi` CUDA
    `12.7`, `nvcc` `12.4.131`;
  - runtime `2026-06-16T05:43:39Z` to `2026-06-16T06:17:03Z`, estimated
    GPU cost USD `0.237`;
  - Vast cleanup confirmed `actual_status=exited`, `cur_state=stopped`,
    `intended_status=stopped`.
- Correctness:
  - CUDA release build rc `0`;
  - `ferrum run` validation PASS, content `5`;
  - default `ferrum serve` smoke PASS, content `5`;
  - chunk32 `ferrum serve` smoke PASS, content `5`;
  - both bench arms completed `[64]`, errored `[0]`, and used
    `output_token_count_source=usage`.
- Performance result, diagnostic only (`n_repeats=1`, no CI):
  - default c16 throughput `320.311 tok/s`, `65.22%` of vLLM c16 LCB
    `491.150 tok/s`, gap to 80% threshold `72.609 tok/s`;
  - chunk32 c16 throughput `312.911 tok/s`, `63.71%` of vLLM c16 LCB,
    gap to 80% threshold `80.009 tok/s`;
  - chunk32 vs default: throughput `-2.31%`, ITL p95 `-21.26%`,
    TTFT p95 `-18.26%`, TPOT p95 `-5.35%`, E2E p95 `+2.68%`.
- Profile:
  - remote `rg` was unavailable, so profile extracts were regenerated locally
    from `server.log`;
  - default had `73` decode-only rows, p50 `33908 us`, p95 `40314 us`, max
    `42826 us`, and zero mixed prefill+decode rows;
  - chunk32 had `67` decode-only rows, p50 `33090 us`, p95 `40967 us`, max
    `45903 us`, plus `6` bounded mixed rows with p50 `73604 us`, p95
    `84571.5 us`, max `85609 us`.
- Interpretation:
  - chunk32 reduces c16 ITL tail latency but lowers throughput, so it is a
    latency tradeoff rather than the main W2 throughput lever;
  - the default arm did not show mixed prefill+decode rows in this run, so
    continuing to sweep active-decode prefill chunk is not supported by the
    evidence;
  - the remaining gap is now more likely in model-side per-step decode cost,
    batched execution, attention/MLP fusion, or host sync/copy paths.
- Gate status:
  - this checkpoint is diagnostic only;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Next direction:
  - stop sweeping active prefill chunk as the main lever;
  - compare Ferrum against local vLLM source under `/Users/chejinxuan/py_ws/vllm`,
    identify concrete Gemma3/GPTQ decode-path differences, then validate with
    minimal Python or native CUDA probes before any broader GPU sweep.

## 2026-06-16 YK — W2 source checkpoint: expose active decode prefill chunk

- No GPU instance was started in this checkpoint; no performance measurement
  was taken and no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Current bottleneck read from existing evidence:
  - c16 same-shape release-style evidence remains correctness-clean but below
    gate: Ferrum LCB `325.184 tok/s` vs vLLM LCB `491.150 tok/s`, and Ferrum
    p95 ITL `83.979 ms` vs vLLM `28.130 ms`;
  - native tail-MLP probes show the steady decode chain is real model work,
    but the release-style p95 ITL is worse than steady decode alone explains;
  - existing profiler evidence includes long mixed prefill+decode steps such
    as `items=10 prefill=3 decode=7 total=291754us`, which can create the
    observed ITL tail.
- Source change:
  - added `--scheduler-active-decode-prefill-chunk <N>` to `ferrum serve`;
  - added `runtime.scheduler_active_decode_prefill_chunk` to CLI config;
  - both map to the already-supported typed runtime key
    `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK`;
  - defaults are unchanged.
- Why this matters:
  - the next c16 validation can test active-decode prefill chunking through a
    visible product setting rather than a hidden environment variable;
  - this directly targets the suspected mixed prefill+decode ITL tail, not the
    already-rejected `gate_up` split branch.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-cli runtime_cli_config_emits_config_file_source_entries -- --nocapture`
    PASS;
  - `cargo test -p ferrum-cli serve_cli_runtime_entries_are_cli_sourced_and_classified -- --nocapture`
    PASS;
  - `cargo test -p ferrum-cli vllm_compat_runtime_flags_follow_existing_precedence -- --nocapture`
    PASS;
  - `cargo test -p ferrum-cli serve_runtime_snapshot_prefers_cli_over_config_file -- --nocapture`
    PASS.
- Required next validation:
  - one cached 1x4090 c16 diagnostic using the new CLI/config knob, with
    profiler lines enabled, to confirm whether mixed prefill+decode steps and
    p95 ITL fall materially before any default change is considered.

## 2026-06-16 YJ — W2 native gate_up split probe: branch rejected, not a material lever

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gate_up_split_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 gate_up split-vs-fused native probe` on cached Vast instance
  `40826362`, 1x RTX 4090.
- Contract:
  - expected runtime/cost: 10-20 minutes, hard cap 30 minutes, about
    USD 0.07-0.15 at USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, or probe prints
    `VERDICT: gemma3 gate_up split native CUDA probe complete` and the artifact
    is copied back;
  - correctness gate: probe exit `0` plus the native CUDA VERDICT line;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh`.
- Evidence hygiene:
  - source head `50abea26c005c3115a7deb931434f53d0803de51`;
  - source status before remote sync `clean-tracked-before-remote-sync`;
  - binary SHA256
    `f0939e6164e17e6d24b18dc127ff567f5a464913bbcd36b6cfea925caf1140e5`;
  - GPU `NVIDIA GeForce RTX 4090`, driver `565.77`, `nvidia-smi` CUDA
    `12.7`, `nvcc` `12.4.131`;
  - Vast cleanup confirmed
    `poll=00 cur_state=stopped actual_status=exited intended_status=stopped`.
    Final Vast API status is also saved in
    `vast_instance_40826362.final.json`.
- Native CUDA result:
  - probe rc `0`;
  - VERDICT line:
    `VERDICT: gemma3 gate_up split native CUDA probe complete`;
  - serial split `gate`/`up` is slower for every tested `m`;
  - two-stream split is neutral/slightly faster only around `m=10` and `m=16`,
    with maximum isolated segment speedup `1.0136x`;
  - two-stream split regresses at larger local shapes: `0.9828x` at `m=23`
    and `0.9899x` at `m=32`.
- Interpretation:
  - the result rules out split `gate`/`up` productization as the next W2 fix;
  - the branch would add loader, runtime, stream, and correctness risk for at
    most about `1.4%` isolated segment gain near `m=16`, far short of the
    current release-grade gap;
  - next direction should move to another tail-MLP work-reduction/fusion or
    prefill wall-time lever, not more env sweeps around this split branch.
- Gate status:
  - this is diagnostic native CUDA evidence only;
  - no product `ferrum run`/`ferrum serve` release gate was run in this
    checkpoint;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YI — W2 source checkpoint: native gate_up split-vs-fused probe

- No GPU instance was started in this checkpoint; no performance measurement
  was taken and no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Added `scripts/microbenches/gemma3_gate_up_split_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh`.
- Probe scope:
  - directly targets the largest remaining dense tail-MLP hotspot:
    Gemma3 GPTQ `gate_up_proj`;
  - compares the current product-shaped fused Marlin projection
    `hidden -> 2*intermediate` plus GeGLU against split `gate` and `up`
    projections plus a separate GeGLU kernel;
  - includes both serial split and two-stream split variants;
  - cycles eight synthetic layer weight sets so a warm single-layer loop cannot
    be mistaken for product-relevant evidence.
- Why this is the next aligned CUDA check:
  - same-shape c16 evidence now proves W2 is still below the 80% throughput and
    p95 ITL release-grade gates;
  - FA2 source, product batched graph, existing Triton W4A16, direct vLLM dense
    Marlin swap, simple L2 persistence, external prefetch, and the first
    producer-touch product prototype have all failed as safe W2 defaults;
  - the next useful branch is therefore a dense MLP compute/layout question:
    whether the fused `gate_up` Marlin shape itself should be changed before
    touching product loader/runtime code.
- Required next validation:
  - run one cached 1x4090 native CUDA probe:
    `bash scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh`;
  - accept this branch only if total `segment_host_us` improves, not merely if
    one half projection looks faster;
  - if split serial/overlap is not materially faster under eight-layer
    rotation, reject split `gate`/`up` productization and move to another
    tail-MLP work-reduction/fusion lever.
- Local validation:
  - `bash -n scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh`
    PASS;
  - `git diff --check -- scripts/microbenches/gemma3_gate_up_split_perf.cu scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh scripts/microbenches/README.md docs/goals/model-coverage-2026-06-12/STATUS.md`
    PASS;
  - local macOS host has no `nvcc`, so CUDA compile/run validation remains the
    next cached-4090 native probe.

## 2026-06-16 YH — W2 CUDA c16 same-shape validation: correctness clean, performance still below 80%

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_ferrum_natural_c16_same_shape_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 CUDA c16 same-dataset same-shape minimal validation` on cached
  Vast instance `40826362`, 1x RTX 4090.
- Contract:
  - expected runtime/cost: 20-45 minutes, hard cap 60 minutes, about
    USD 0.14-0.32 at USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/build/`ferrum run`/smoke first failure,
    c16 `num_prompts=100,n_repeats=3,--require-ci` artifact copied, or
    60-minute cap;
  - correctness gate: `ferrum run` plus
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 128 --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271`.
- Evidence hygiene:
  - local source head `a45e3caaeb94af5451c64f7542014e580ea613e6`;
  - local tracked dirty count `0`;
  - binary SHA256
    `79379516dc90c958ae03f65aeaa36b706156b5ec1f6e15e14092815f4d62a110`;
  - dataset SHA256
    `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`;
  - Vast cleanup confirmed `actual_status=exited`.
- Correctness:
  - `ferrum run` returned assistant content `5`, finish_reason `stop`,
    `n_tokens=3`;
  - smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`.
- Performance result:
  - c16 completed `[100,100,100]`, errored `[0,0,0]`;
  - all quality/error counts are zero and `output_token_count_source=usage`;
  - Ferrum c16 `332.005 +/- 6.821 tok/s`, LCB `325.184 tok/s`;
  - vLLM c16 same-dataset baseline LCB `491.150 tok/s`;
  - required 80% threshold `392.920 tok/s`;
  - Ferrum LCB / vLLM LCB `66.21%`, gap to 80% threshold
    `67.736 tok/s`;
  - Ferrum p95 ITL `83.979 ms` vs vLLM `28.130 ms`, `2.99x`.
- Interpretation:
  - this closes the previous 32-prompt diagnostic vs 100-prompt baseline
    ambiguity;
  - correctness is clean, but c16 still fails both the throughput and p95 ITL
    release-grade thresholds;
  - W2 remains not release-grade: no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced;
  - the next performance lever should stay on the model-side decode/tail-MLP
    path, not on another full sweep.

## 2026-06-16 YG — W2 release-grade validator checkpoint: prompt dataset evidence must match

- Source checkpoint:
  - `90c48504 test(release): require matching prompt dataset evidence`.
- Scope:
  - no GPU instance was started;
  - no performance measurement was taken;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Validator changes:
  - release-grade `bench-serve` commands must now include `--num-prompts`
    matching the cell `requests_per_run`;
  - each performance cell must include `prompt_dataset_id` and
    `baseline_prompt_dataset_id`, and they must match;
  - each performance cell must include `prompt_dataset_sha256` and
    `baseline_prompt_dataset_sha256`, and they must match;
  - self-tests now reject mismatched `--num-prompts`, prompt dataset id, and
    prompt dataset sha256.
- Local validation:
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS;
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `git diff --check` PASS;
  - `python3 scripts/release/selftest_g0_validators.py` PASS.
- Current W2 state:
  - no new product correctness blocker is known;
  - W2 remains not release-grade because there is no final PASS line and the
    c16/c32 performance gap remains below the 80% mainstream-engine target;
  - the next GPU validation should be a minimal c16 same-dataset,
    same-shape run before any broader release sweep.

## 2026-06-16 YF — W2 release-grade validator checkpoint: baseline cell shape must match Ferrum

- Source checkpoint:
  - `d549c6ed test(release): require matching baseline cell shape`.
- Scope:
  - no GPU instance was started;
  - no performance measurement was taken;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Validator changes:
  - each release-grade performance cell now rejects baseline/Ferrum shape
    mismatches for `n_repeats`;
  - each cell also rejects baseline/Ferrum mismatches for `requests_per_run`;
  - self-tests now cover both mismatches.
- Local validation:
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS;
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `git diff --check` PASS;
  - `python3 scripts/release/selftest_g0_validators.py` PASS.
- Current W2 state:
  - no current evidence shows a new product correctness blocker in
    `ferrum run` or `ferrum serve`;
  - W2 remains not release-grade because there is no final PASS line and
    c16/c32 performance remains below the 80% mainstream-engine line;
  - the latest natural ShareGPT Ferrum/vLLM comparison is still diagnostic:
    Ferrum used `num_prompts=32,n_repeats=1`, while release evidence requires
    `num_prompts=100,n_repeats=3,--require-ci` with matching baseline shape.
- Next direction:
  - before another expensive sweep, run only a minimal same-dataset,
    same-shape c16 validation if GPU is started;
  - continue bottleneck work from the decode tail MLP path, where current
    profiling points to `gate_up -> GeGLU -> down` kernel time rather than
    scheduler/postprocess overhead.

## 2026-06-16 YE — W2 release-grade validator checkpoint: baseline bench cells must be clean

- Source checkpoint:
  - `8cf42094 test(release): require clean baseline bench cells`.
- Scope:
  - no GPU instance was started;
  - no performance measurement was taken;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Validator changes:
  - each performance cell must now include baseline `n_repeats`,
    `requests_per_run`, completed/error counts, quality counts, usage-token
    source, streaming usage flag, and baseline `bench-serve` command line;
  - baseline cells must use `--fail-on-error`, `--require-ci`, `--seed 9271`,
    and matching `--n-repeats`;
  - baseline completed counts must be full, error counts must be zero, and
    bad-output / malformed-stream / missing-DONE / duplicate-DONE /
    zero-output / bulk-flush / HTTP-500 / panic counts must all be zero.
- Local validation:
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS;
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `git diff --check` PASS;
  - `python3 scripts/release/selftest_g0_validators.py` PASS.
- Rationale:
  - the W2 80% denominator must be a correctness-clean same-dataset baseline;
  - the final gate should not accept a bare baseline throughput number without
    the same zero-error/usage-token evidence required from Ferrum.

## 2026-06-16 YD — W2 release-grade validator checkpoint: baseline evidence gate hardened

- Source checkpoints:
  - `d4d73197 test(release): enforce vllm release baselines`;
  - `c881a953 test(release): tighten vllm baseline matching`.
- Scope:
  - no GPU instance was started;
  - no performance measurement was taken;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Validator changes:
  - CUDA HF/safetensors/GPTQ/AWQ release-grade lanes now require a vLLM
    baseline by default;
  - non-vLLM baseline selection is accepted only with explicit
    `selection_exception` evidence proving vLLM is unsupported for that lane;
  - baseline, Ferrum cell artifacts, clean dirty status, and zero bad-output /
    malformed-stream / missing-DONE / duplicate-DONE / zero-output / HTTP-500 /
    panic counts are required before the final gate can pass;
  - misleading engine strings such as `not-vllm` are rejected by self-test.
- Local validation:
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS;
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `git diff --check` PASS;
  - `python3 scripts/release/selftest_g0_validators.py` PASS.
- Current W2 state:
  - latest product-path smokes do not show a known correctness blocker;
  - performance remains below the 80% mainstream baseline line, with current
    ShareGPT diagnostics around 60-65% of vLLM;
  - existing profile evidence still points to Gemma3 GPTQ dense tail MLP,
    especially the `gate_up -> GeGLU -> down` sequence, as the main decode
    bottleneck.
- Next direction:
  - avoid repeated full sweeps until a smaller source/native CUDA lever is
    chosen;
  - continue from the tail-MLP kernel/work-reduction path and keep c32
    effective active concurrency comparable with the vLLM baseline.

## 2026-06-16 YC — W2 CUDA checkpoint: producer-touch product prototype is not safe as a default

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_sharegpt_diag_2026-06-16/`.
- Paid GPU lane:
  `W2 producer-touch ShareGPT endpoint diagnostic` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 15-30 minutes, about USD 0.11-0.22 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/server readiness/chat smoke/bench first
    failure or c16/c32 diagnostic artifact collected, then stop;
  - correctness gate: prior product run/serve smoke plus this run's server
    readiness, chat smoke `5` with usage, bench rc 0, zero request errors, and
    clean server log scan;
  - performance command:
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`.
- Correctness evidence:
  - bench rc `0`;
  - c16/c32 both `16 completed / 0 errored / 0 bad_output / 0 zero_output`;
  - `output_token_count_source=usage`;
  - server error scan has `0` lines;
  - Vast cleanup confirmed `stopped/exited`.
- Results:
  - c16 producer-touch `313.3996 tok/s` vs current default
    `339.9306 tok/s`: `-7.80%`;
  - c32 producer-touch `348.5895 tok/s` vs current default
    `340.5554 tok/s`: `+2.36%`;
  - c16 ratio to vLLM `0.6041`;
  - c32 ratio to vLLM `0.6651`.
- Interpretation:
  - the native producer-touch signal is real, but the product default prototype
    is mixed: it slightly helps c32 and materially hurts c16;
  - this is not a safe default optimization and should not be used for
    release-grade performance work without a narrower variant and fresh product
    c16 evidence;
  - the immediate source follow-up is to return the default product path to the
    previous GeGLU behavior while preserving the diagnostic artifact.
- Scope:
  - this is diagnostic evidence only: `n_repeats=1`, no `--require-ci`;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YB — W2 CUDA checkpoint: producer-touch product prototype compiles and passes run/serve smoke

- Artifacts:
  - compile smoke:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_product_compile_2026-06-16/`;
  - product correctness smoke:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_product_correctness_2026-06-16/`.
- Source change:
  - added typed backend API
    `fused_gelu_tanh_mul_split_with_down_hint(...)`, with default fallback to
    existing GeGLU behavior;
  - CUDA uses the hint only when the downstream projection is a
    `CudaMarlinLinear` backed by Marlin weights;
  - Gemma unified and non-unified paths pass `layer.down_proj` as the hint for
    `Activation::GeluTanh`;
  - added CUDA kernel
    `fused_gelu_tanh_mul_interleaved_f16_touch_down_qweight`, the product
    analogue of the native `producer_touch_qweight_1x` signal.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-kernels -p ferrum-models` PASS.
- Paid GPU compile lane:
  - expected runtime/cost: 10-25 minutes, about USD 0.07-0.18 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/sync/build first failure or build
    artifact collected;
  - correctness gate:
    `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - build rc `0`, release profile finished in `3m 27s`.
- Paid GPU correctness lane:
  - expected runtime/cost: 10-25 minutes, about USD 0.07-0.18;
  - stop condition: `ferrum run`/serve readiness/chat smoke/log scan first
    hard failure or artifact collected, then stop instance;
  - correctness gate: run output `5`, serve chat output `5` with usage, and
    server error scan 0;
  - performance command: none.
- Correctness evidence:
  - binary SHA256
    `5078ea014ee5299a936de62f34475456f9a3c0500d34ab41a96ebcaf9c69fbd8`;
  - `ferrum run` rc `0`, assistant content `5`, finish_reason `stop`,
    `n_tokens=3`;
  - `ferrum serve` readiness passed, chat rc `0`, response content `5`,
    usage `prompt_tokens=23`, `completion_tokens=1`, `total_tokens=24`;
  - `server/error_scan.txt` has `0` lines;
  - `correctness_check.json` reports `ok=true`;
  - Vast cleanup confirmed `stopped/exited`.
- Interpretation:
  - the native producer-touch cache-residency signal has now been converted to
    a typed product prototype and cleared minimal product-entrypoint
    correctness;
  - this is still not performance evidence. The next step is a focused
    same-dataset endpoint diagnostic before deciding whether to keep,
    tune, or revert this product optimization.
- Scope:
  - no release-grade performance matrix was run;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 YA — W2 native CUDA checkpoint: producer-integrated qweight touch has a real segment-time signal

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_native_probe_2026-06-16/`.
- Source change:
  - extended `scripts/microbenches/gemma3_down_prefetch_overlap_perf.cu`
    with producer-integrated GeGLU touch modes;
  - the probe keeps product-shaped eight-layer rotation and reports both
    isolated `down_us` and total `segment_host_us`.
- Paid GPU lane:
  `W2 native CUDA producer-integrated tail-MLP cache probe` on the cached
  1x RTX 4090 Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: SSH/CUDA/sync/compile first failure or probe artifact
    collected, then stop the instance;
  - correctness gate: native probe exit 0 plus
    `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_prefetch_overlap_perf.sh`.
- Evidence:
  - GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 565.77;
  - remote base HEAD `017300426514d62e8e50ac1546ff77d4d54fd6ce`, with the
    local dirty probe source synced over it;
  - local HEAD `f096e96395b11f712a3660999d6b999a0970bc23`;
  - binary SHA256
    `994f828373477f5d9a34f8bd06c42921b1b13cfeb8b28679fd2400fb6f968801`;
  - first compile attempt failed on `volatile uint4` copies and was preserved;
  - retry rc `0`, PASS line present;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 no prefetch: down `68.773us`, segment `212.295us`;
  - m16 external overlap qweight: down `34.150us`, segment `235.799us`
    (`+11.07%` segment, rejected);
  - m16 producer touch qweight 1x: down `62.787us`, segment `202.341us`
    (`-4.69%` segment);
  - m16 producer touch qweight 4x: down `53.889us`, segment `214.460us`
    (`+1.02%` segment, rejected);
  - m32 no prefetch: down `74.286us`, segment `224.566us`;
  - m32 external overlap qweight: down `53.112us`, segment `261.882us`
    (`+16.62%` segment, rejected);
  - m32 producer touch qweight 1x: down `64.878us`, segment `212.922us`
    (`-5.19%` segment);
  - m32 producer touch qweight 4x: down `53.474us`, segment `240.533us`
    (`+7.11%` segment, rejected).
- Interpretation:
  - this is the first cache-residency branch signal that improves total
    product-shaped tail-MLP segment time rather than only improving isolated
    `down_us`;
  - the viable branch is a small, producer-adjacent qweight touch/prefetch, not
    a full external qweight warm and not simple stream access-policy alone;
  - productization still needs typed projection/layer context and full
    `ferrum run`/`ferrum serve` correctness before endpoint performance
    diagnostics.
- Scope:
  - this is native CUDA diagnostic evidence only, not release-grade evidence;
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.

## 2026-06-16 XZ — W2 CUDA checkpoint: next bottleneck lever narrowed to producer-integrated tail-MLP cache test

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_next_bottleneck_lever_2026-06-16/`.
- This checkpoint adds no new GPU run. It consolidates the current bottleneck
  evidence and fixes the next minimal validation target before another paid
  benchmark or product patch.
- Current W2 state:
  - no final `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has been produced;
  - latest product correctness smokes did not expose a new run/serve blocker;
  - performance remains below the 80% same-hardware mainstream baseline line,
    with current ShareGPT diagnostics around 60-65% of vLLM depending on the
    variant.
- Evidence now included in the decision:
  - `tail_mlp` is still the largest profiled decode block;
  - native tail-MLP chain timing matches product timing, so the issue is not
    just endpoint overhead;
  - single-layer down-projection L2 persistence works, but the win disappears
    under product-like eight-layer weight rotation;
  - explicit down-weight warm/prefetch restores the isolated down kernel but
    increases total segment wall time;
  - FA2 source product path is correct but slower on the current Gemma3
    ShareGPT c16 diagnostic;
  - product `--batched-graph` selects the graph path but does not improve
    endpoint throughput.
- Source audit:
  - Gemma3 tail MLP is the direct sequence
    `gate_up_proj.forward -> fused_gelu_tanh_mul_split -> down_proj.forward`
    in both unified and non-unified paths;
  - `CudaMarlinLinear::forward` is currently generic, so a product fix that is
    specific to Gemma3 `down_proj` needs a typed projection/layer context rather
    than relying on diagnostic labels or hidden env.
- Next minimal validation:
  - extend the native CUDA prefetch-overlap probe to test a
    producer-integrated GeGLU variant that touches a configurable slice of
    `down_proj` qweight/scales while preserving eight-layer rotation;
  - accept the branch only if total segment wall time improves, not merely the
    isolated `down_us`;
  - if this fails, abandon the simple cache-warm branch and move to tail-MLP
    work reduction/fusion.
- Scope:
  - this is diagnostic planning/source-audit evidence only, not release-grade
    evidence.

## 2026-06-16 XY — W2 CUDA checkpoint: FA2 source product path is correct but slower on Gemma3 ShareGPT c16

- Artifacts:
  - corrected FA2 product smoke:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_fa2_source_gemma_full_config_smoke_2026-06-16/`;
  - initial minimal-config attempt:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_fa2_source_gemma_smoke_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 typed FA2-source full-config product smoke` on the cached 1x RTX
  4090 Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h for each small smoke attempt;
  - stop condition: startup/SSH/CUDA/config assertion/serve readiness/chat
    smoke/minimal c16 bench first failure, or artifact collected, then stop the
    instance;
  - correctness gate: product `ferrum serve` from artifact-local complete
    `ferrum.toml`, decision trace must select
    `attention_prefill_mixed_backend=fa2_source`, chat smoke must return `5`
    with usage, and bench must return rc 0 with zero request errors;
  - performance command:
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`.
- Config finding:
  - the first `[runtime]`-only config attempt did not inject runtime entries;
    decision trace selected `legacy_paged_varlen`, so no bench was run;
  - the corrected full-config attempt selected `fa2_source` from config-file
    `FERRUM_FA2_SOURCE`, selected decode backend `vllm_paged_attn_v1_short`,
    and autosize logged `KV pool copies=2 (FA-compatible attention path)`.
- Corrected smoke evidence:
  - remote HEAD `017300426514d62e8e50ac1546ff77d4d54fd6ce`;
  - binary SHA256
    `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`;
  - chat smoke response content `"5"`, usage present;
  - bench rc `0`, `output_token_count_source=usage`;
  - server log scan has `0` lines;
  - Vast cleanup confirmed `stopped/exited`.
- Results:
  - c16: `16 completed / 0 errored / 0 bad_output / 0 zero_output`,
    `313.472 tok/s`;
  - compared with the current graph-disabled Ferrum same-dataset c16
    `339.9306 tok/s`, FA2 source is `-26.4586 tok/s`, or `-7.78%`;
  - compared with the clean vLLM ShareGPT c16 baseline `518.796 tok/s`, the
    ratio is `0.6042`.
- Interpretation:
  - FA2's principle is still valid for prefill/mixed attention: fused tiled
    attention reduces HBM traffic and intermediate materialization;
  - however, on current W2 Gemma3 ShareGPT c16, enabling the actual product
    `fa2_source` path is correct but slower than the default path;
  - this rules out "FA2 is missing from the product path" as the current
    14-15 percentage point W2 bottleneck. The next work should return to the
    model-step dominant path, especially Gemma GPTQ dense MLP/tail and decode
    integration, rather than continuing FA2 sweeps.
- Scope:
  - this is diagnostic evidence only: `n_repeats=1`, no `--require-ci`, and no
    final release-grade manifest.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XX — W2 CUDA checkpoint: product batched graph is not the endpoint bottleneck

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_sharegpt_current_diag_2026-06-16/`.
- Paid GPU lane:
  `W2 current HEAD --batched-graph ShareGPT same-dataset diagnostic` on the
  cached 1x RTX 4090 Vast instance.
- Contract:
  - expected runtime/cost: 15-30 minutes, about USD 0.11-0.22 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/sync/serve/bench first failure, or c16/c32
    diagnostic artifact collected, then stop the instance;
  - correctness gate: server readiness, chat smoke response `5` with usage,
    bench rc 0, completed requests, zero request errors, zero bad output, zero
    zero-output responses, zero HTTP 500, and clean server error scan;
  - performance command:
    `ferrum serve --batched-graph` plus
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`.
- Evidence:
  - remote HEAD `017300426514d62e8e50ac1546ff77d4d54fd6ce`;
  - clean remote worktree: `local/remote_clean_worktree.txt` has `0` tracked
    changes;
  - binary SHA256
    `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`;
  - effective graph mode `legacy_batched_decode_graph`;
  - server ready at `ready_at_poll=29`;
  - chat smoke response content `"5"`, usage present;
  - bench rc `0`, `output_token_count_source=usage`;
  - server log scan has `0` lines;
  - Vast cleanup confirmed `stopped/exited` in
    `vast_shutdown/shutdown_complete.txt`.
- Results against the existing clean vLLM ShareGPT baseline:
  - c16: `16 completed / 0 errored / 0 bad_output`,
    `337.6359 tok/s`; ratio `337.6359 / 518.796 = 0.6508`;
  - c32: `16 completed / 0 errored / 0 bad_output`,
    `340.1011 tok/s`; ratio `340.1011 / 524.128 = 0.6489`;
  - compared with the current graph-disabled Ferrum same-dataset diagnostic,
    c16 changed by `-0.675%` and c32 by `-0.133%`.
- Interpretation:
  - no new product `serve` correctness issue was found;
  - product `--batched-graph` is wired through the CLI and selects
    `legacy_batched_decode_graph`, but it does not improve ShareGPT endpoint
    throughput on current HEAD;
  - the W2 performance gap remains about 15 percentage points below the 80%
    same-hardware mainstream baseline target, so the next lever should move
    away from graph-enable diagnostics and toward the model-step dominant path,
    especially dense MLP `gate_up`, work reduction, and launch/graph
    integration backed by profiler evidence.
- Scope:
  - this is diagnostic evidence only: `n_repeats=1`, no `--require-ci`, and no
    final release-grade manifest.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XW — W2 CUDA checkpoint: Marlin evict-first does not move ShareGPT endpoint throughput

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_sharegpt_diag_2026-06-16/`.
- Paid GPU lane:
  `W2 current Ferrum ShareGPT same-dataset diagnostic after Marlin evict-first default`
  on the cached 1x RTX 4090 Vast instance.
- Contract:
  - expected runtime/cost: 20-40 minutes, about USD 0.14-0.28 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/sync/build/serve/bench first failure, or
    diagnostic artifact collected, then stop the instance;
  - correctness gate: prior product `ferrum run`/`ferrum serve` correctness
    artifact plus this run's server readiness, chat smoke, bench rc 0,
    completed requests, zero request errors, and clean server log scan;
  - performance command:
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`.
- Evidence:
  - remote HEAD `7d93c2b481cc3a4d9ae794e2d6a66c3e05a55784`;
  - clean remote worktree: `remote/git_status_short.txt` has `0` lines;
  - binary SHA256
    `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`;
  - server ready at `ready_at_poll=31`;
  - chat smoke response content `"5"`, usage present;
  - bench rc `0`, `output_token_count_source=usage`;
  - server log scan has `0` lines;
  - Vast cleanup confirmed `stopped/exited`.
- Results against the existing clean vLLM ShareGPT baseline
  `w2_vllm_sharegpt_baseline_probe_2026-06-15`:
  - c16: `16 completed / 0 errored / 0 bad_output`,
    `339.9306 tok/s`; ratio `339.9306 / 518.796 = 0.6552`;
  - c32: `16 completed / 0 errored / 0 bad_output`,
    `340.5554 tok/s`; ratio `340.5554 / 524.128 = 0.6498`;
  - compared with the previous Ferrum same-dataset diagnostic, c16 changed by
    `-0.02%` and c32 by `-0.51%`.
- Interpretation:
  - no new product `serve` correctness issue was found;
  - the Marlin B-weight evict-first default is a real native tail-MLP segment
    improvement, but it does not move full ShareGPT endpoint throughput;
  - the W2 performance gap is still about 14-15 percentage points below the
    80% same-hardware mainstream baseline target, so the next lever should move
    to dense MLP `gate_up`, launch count, and batched decode graph/integration
    behavior under product c16/c32.
- Scope:
  - this is diagnostic evidence only: `n_repeats=1`, no `--require-ci`, and no
    final release-grade manifest.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XV — W2 CUDA checkpoint: Marlin evict-first product run/serve correctness passes

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_product_correctness_2026-06-16/`.
- Paid GPU lane:
  `W2 Marlin evict-first product correctness smoke` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 30-60 minutes, about USD 0.21-0.43 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/clean checkout/build/`ferrum run`/
    `ferrum serve` first failure, or run+serve correctness evidence collected;
  - correctness gate: product-default CUDA binary, `ferrum run` and
    `ferrum serve` both return expected `5`, usage present for serve, and log
    scan has no panic/error/NaN/`<unk>`/`[PAD]`/invalid UTF patterns;
  - performance command: none for this lane; correctness-only after the Marlin
    default-path source change.
- Evidence:
  - remote HEAD `212b2bf925c998062ef22767a1da41ba47ed5101`;
  - clean worktree: `remote/git_status_short.txt` has `0` lines;
  - CUDA release build rc `0`;
  - binary SHA256
    `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`;
  - `ferrum run` rc `0`, JSONL assistant content `"5"`,
    `finish_reason=stop`, `n_tokens=3`;
  - `ferrum serve` chat rc `0`, response content `"5"`,
    `finish_reason=length`, usage `prompt_tokens=23`, `completion_tokens=1`,
    `total_tokens=24`;
  - `server/error_scan.txt` has `0` lines;
  - `correctness_check.json` reports `ok=true`;
  - Vast cleanup confirmed `stopped/exited`.
- Note:
  - the first background attempt failed before build because the script did not
    include `/root/.cargo/bin` in `PATH`; preserved under
    `build_initial_env_failure/`;
  - the retry used the same instance and clean worktree and passed.
- Interpretation:
  - the Marlin B-weight `L2::evict_first` default path has now cleared the
    required product-entrypoint correctness smoke for both `ferrum run` and
    `ferrum serve`;
  - this unlocks endpoint performance diagnostics for this source change, but
    it is not itself performance or release-grade evidence.
- Next:
  - run a focused same-dataset Ferrum diagnostic against the existing clean
    vLLM ShareGPT baseline before deciding whether the 1-2% MLP gain moves the
    endpoint ratio materially;
  - continue searching for a higher-return dense MLP `gate_up` /
    work-reduction lever.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XU — W2 native CUDA checkpoint: product-default Marlin evict-first validated

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_default_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Marlin cache-policy product-default native probe` on the cached 1x RTX
  4090 Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`.
- Evidence:
  - remote HEAD `c76bfcfa2b00a73a816e6d44bbd999a621b12a49`;
  - probe rc `0`;
  - legacy plain binary SHA256
    `b0ee9ba92b2a3ab74c382273ea2fc82763277671b436581b5fc47e0d9b896e00`;
  - product default binary SHA256
    `82edfb8e6561f87eef067d3ea7fe5327b54f3cc9450d6c42cf63fe72963aec66`;
  - stdout contains
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 product chain event: legacy plain `215.344us`, product default
    `211.791us` (`-1.6%`);
  - m16 product down: legacy plain `70.496us`, product default `68.852us`;
  - m32 product chain event: legacy plain `227.980us`, product default
    `225.103us` (`-1.3%`);
  - m32 product down: legacy plain `75.653us`, product default `75.414us`.
- Interpretation:
  - after `c76bfcfa`, the product-default Marlin path matches the previously
    validated evict-first variant;
  - this is a real default-path kernel improvement, but only a 1-2% MLP segment
    lever, so it does not close the W2 release-grade performance gap alone.
- Next:
  - validate `ferrum run` and `ferrum serve` correctness on a CUDA product
    binary before any endpoint performance claim;
  - continue searching for a larger dense MLP `gate_up` / work-reduction lever.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `remote/git_status_short.txt` and are
    not used for release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XT — W2 source checkpoint: productize Marlin B-weight evict-first cache policy

- Changed `crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu` so Marlin
  B-weight `cp.async` uses `L2::evict_first` by default for CUDA `sm_80` through
  pre-Blackwell architectures. Blackwell `sm_120` keeps the plain
  `cp.async.cg` fallback because the fractional L2 cache-policy syntax is not
  accepted there.
- Updated `scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`
  to compare legacy plain `cp.async.cg` (`FERRUM_MARLIN_CP_ASYNC_PLAIN=1`)
  against the product-default path.
- Rationale:
  - XS native CUDA evidence showed this cache policy is a small positive
    product-shaped tail-MLP lever rather than a cost-shifting warmup trick;
  - the behavior is now a default CUDA build path, not a hidden user env
    combination.
- Local validation:
  - `bash -n scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`;
  - `git diff --check -- crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh scripts/microbenches/README.md`.
- Required next validation:
  - run the native cache-policy probe on 1x RTX 4090 to confirm the product
    default matches the previously measured evict-first path;
  - before endpoint performance claims, validate `ferrum run` and
    `ferrum serve` correctness on the product binary.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XS — W2 native CUDA checkpoint: Marlin B-weight evict-first is a small positive lever

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 Marlin cache-policy native probe` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh`.
- Evidence:
  - remote HEAD `018ea7bce6494db5539ce32e22f104144fe87eba`;
  - probe rc `0`;
  - baseline binary SHA256
    `50e4ad67f5d79293da1d524eedcae2cde7edb71d7e6d85387e94b5b37cb0ca41`;
  - evict-first binary SHA256
    `69655f683cc80daf98737e290946ca69bbcec87d69c818deff5cf2038e8c8e41`;
  - stdout contains
    `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 product chain event: baseline `215.580us`, evict-first `211.690us`
    (`-1.8%`);
  - m16 product down: baseline `70.549us`, evict-first `68.800us`;
  - m32 product chain event: baseline `227.722us`, evict-first `225.173us`
    (`-1.1%`);
  - m32 product down: baseline `75.659us`, evict-first `75.339us`.
- Interpretation:
  - `FERRUM_MARLIN_CP_ASYNC_EVICT_FIRST=1` compiles on CUDA 12.4 / Ada and
    produces a small positive tail-MLP segment gain;
  - unlike explicit down prefetch, this improves segment wall time rather than
    shifting cost into a warm kernel;
  - the gain is too small by itself to close the W2 gap, so this is a useful
    low-risk kernel lever but not the main missing performance breakthrough.
- Next:
  - either wire this as a typed/default CUDA build policy and validate
    `ferrum run` / `ferrum serve` correctness before endpoint performance, or
    keep searching for a higher-return dense MLP `gate_up` / work-reduction
    lever.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `remote/git_status_short.txt` and are
    not used for release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XR — W2 native CUDA checkpoint: overlap prefetch warms down but worsens wall time

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_prefetch_overlap_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down prefetch-overlap native probe` on the cached 1x RTX 4090
  Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_prefetch_overlap_perf.sh`.
- Evidence:
  - remote HEAD `432e6588bac59902b7488484934494c751534221`;
  - probe rc `0`;
  - binary SHA256
    `58491a34483c8c4ba0ccbd4b1d9c9b127676b1f520a6ba42a0409daae5cc64bc`;
  - stdout contains
    `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 no prefetch: down `69.744us`, segment `216.072us`;
  - m16 overlap qweight: down `34.063us`, segment `239.437us`;
  - m16 overlap qweight+scales: down `32.560us`, segment `238.979us`;
  - m32 no prefetch: down `74.849us`, segment `227.628us`;
  - m32 overlap qweight: down `53.836us`, segment `268.300us`;
  - m32 overlap qweight+scales: down `53.790us`, segment `269.916us`.
- Interpretation:
  - explicit warm/prefetch does make down fast under 8-layer rotation;
  - it increases end-to-end segment wall time, so the current warm kernel shifts
    cost rather than reducing wall time;
  - do not productize this cache-warm branch as W2 performance work unless a
    cheaper producer-integrated prefetch design is found.
- Next:
  - return to dense MLP work-reduction/kernel-design options rather than
    stream-level L2 policy or standalone warm kernels.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XQ — W2 source checkpoint: native down prefetch-overlap probe

- Added `scripts/microbenches/gemma3_down_prefetch_overlap_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_prefetch_overlap_perf.sh`.
- Purpose:
  - XP showed simple L2 access-policy is not enough under 8-layer rotation;
  - explicit down-warm is an upper bound but adds an extra down read;
  - this probe launches a lightweight down qweight/scales read kernel on a
    second CUDA stream while the main stream runs gate_up+GeGLU, then measures
    both down kernel time and host-synchronized segment time.
- Expected GPU use:
  - run one native CUDA validation before touching product code;
  - if overlap prefetch reduces down time but increases segment time by a
    comparable amount, reject it as non-productizable;
  - if it reduces down time and keeps segment time flat or lower, evaluate a
    typed product prefetch policy and then validate `ferrum run`/`serve`
    correctness before endpoint performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XP — W2 native CUDA checkpoint: simple L2 policy fails under multi-layer weight rotation

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_l2_persist_cycle_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down L2 persistence cycle native probe` on the cached 1x RTX
  4090 Vast instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down L2 persistence cycle native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_l2_persist_cycle_perf.sh`.
- Evidence:
  - remote HEAD `357a4b98a2eb80744b8beacf256b91bbff8ae0f2`;
  - probe rc `0`;
  - binary SHA256
    `f9c3e69f4407c4b4bd42b7f28593efcc7eb1c2bc81dff7c10ba98baf10b510f1`;
  - stdout contains
    `VERDICT: gemma3 down L2 persistence cycle native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key rows:
  - m16 single-layer no-policy: `69.832us`;
  - m16 single-layer persist hit60: `34.493us`;
  - m16 8-layer cycle no-policy: `69.736us`;
  - m16 8-layer cycle persist hit60: `69.743us`;
  - m16 8-layer cycle persist plus explicit down-warm: `34.634us`;
  - m32 single-layer no-policy: `75.903us`;
  - m32 single-layer persist hit60: `58.984us`;
  - m32 8-layer cycle no-policy: `75.745us`;
  - m32 8-layer cycle persist hit60: `75.117us`;
  - m32 8-layer cycle persist plus explicit down-warm: `54.067us`.
- Interpretation:
  - XN's single-layer L2 persistence win is real but not sufficient for product
    decode because one layer's next down call is separated by many other layer
    weights;
  - simple per-layer stream access-policy does not improve 8-layer rotation;
  - explicit down-warm remains a useful upper bound but reads down weights an
    extra time, so it is not a free product fix;
  - do not productize simple access-policy alone as the W2 performance lever.
- Next:
  - if staying on this branch, test an overlap/prefetch strategy that can warm
    down qweight concurrently with gate_up work; otherwise return to another
    dense MLP reduction lever.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XO — W2 source checkpoint: native multi-layer L2 persistence cycle probe

- Added `scripts/microbenches/gemma3_down_l2_persist_cycle_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_l2_persist_cycle_perf.sh`.
- Purpose:
  - XN proved stream access-policy can keep a single layer's down qweight hot
    across that same layer's `gate_up -> GeGLU` producer;
  - product decode revisits one layer only after many other layer weights run,
    so the single-layer loop may overstate productizable benefit;
  - this probe allocates 8 synthetic Gemma3 layer weight sets and compares
    single-layer no-policy/persist against 8-layer no-policy/persist and an
    explicit down-warm upper bound.
- Expected GPU use:
  - run one native CUDA validation before productizing L2 policy;
  - if 8-layer persist does not improve over no-policy, reject simple per-layer
    access-policy as insufficient for product performance;
  - if 8-layer persist still helps materially, implement a typed product
    policy and validate `ferrum run` / `ferrum serve` correctness before any
    endpoint performance claim.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XN — W2 native CUDA checkpoint: down qweight L2 persistence restores post-gate_up down speed

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_l2_persist_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down L2 persistence native probe` on the cached 1x RTX 4090 Vast
  instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down L2 persistence native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_l2_persist_perf.sh`.
- Evidence:
  - remote HEAD `6cf26ca99f1958d2e326245bbe55fd8ed22c7e4a`;
  - probe rc `0`;
  - binary SHA256
    `c3fafa5657c5dbc1496f6a9790ffc4440cb4f17ddf01014f55df1212226826f3`;
  - stdout contains
    `VERDICT: gemma3 down L2 persistence native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Device/context:
  - RTX 4090 L2 cache: `75,497,472` bytes;
  - persisting L2 max: `51,904,512` bytes;
  - access window max: `134,213,632` bytes;
  - down qweight policy window: `57,802,752` bytes.
- Key rows:
  - m16 warm repeated baseline: `35.135us`;
  - m16 no-policy after gate_up+GeGLU: `70.342us`;
  - m16 down qweight full-window persist hit100: `35.088us`;
  - m16 down qweight full-window persist hit60: `33.158us`;
  - m32 warm repeated baseline: `55.127us`;
  - m32 no-policy after gate_up+GeGLU: `75.148us`;
  - m32 down qweight full-window persist hit100: `55.545us`;
  - m32 down qweight full-window persist hit60: `54.434us`.
- Interpretation:
  - simple CUDA stream access-policy on down qweight is a real W2 lever;
  - it restores down performance after the product-shaped `gate_up -> GeGLU`
    producer sequence instead of only improving isolated warm microbench rows;
  - expected product upside is bounded to the dense Marlin down component, so it
    will not by itself prove W2 release-grade, but it is the first currently
    measured lever with material tail-MLP savings.
- Next:
  - productize as a typed CUDA runtime/config policy, not a hidden env-only
    requirement;
  - validate `ferrum run` and `ferrum serve` correctness before performance;
  - only after correctness passes, run a focused c16/c32 diagnostic and then
    decide whether to promote to release evidence.
- Scope:
  - this is diagnostic native CUDA evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XM — W2 source checkpoint: native down L2 persistence probe

- Added `scripts/microbenches/gemma3_down_l2_persist_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_l2_persist_perf.sh`.
- Purpose:
  - XL showed down is cold after product-shaped `gate_up -> GeGLU`, even when
    down reads a separate constant input;
  - this probe applies CUDA's stream access-policy window to down `qweight`
    and compares no-policy, full-window, half-window, lower hit-ratio, and
    explicit down-warm cases;
  - it is a native CUDA minimal verification of whether simple persisting L2
    hints are a productizable lever for the `gate_up -> down` sequence.
- Expected GPU use:
  - run one cached 1x4090 native probe, not a release sweep;
  - if no-policy and persisting modes match, reject stream-level L2 persistence
    as a W2 lever;
  - if persisting materially narrows the m16/m32 down gap, inspect whether the
    same policy can be represented as a typed CUDA runtime option without
    hidden env and then validate product correctness before performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XL — W2 native CUDA checkpoint: down slowdown is cache/producer-state, not GeGLU value sensitivity

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_input_source_native_probe_2026-06-16/`.
- Paid GPU lane:
  `W2 Gemma3 down input-source native probe` on the cached 1x RTX 4090 Vast
  instance.
- Contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
    USD 0.42488888888888887/h;
  - stop condition: startup/SSH/CUDA/compile first failure, probe non-zero or
    timeout, or VERDICT plus artifact copyback;
  - correctness gate: native probe exit 0 and
    `VERDICT: gemma3 down input-source native CUDA probe complete`;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_down_input_source_perf.sh`.
- Evidence:
  - remote HEAD `cea63ef0b5c933a2a39802a82010b34eaa1a9d45`;
  - probe rc `0`;
  - binary SHA256
    `dd1a5ba3cd0f244603bc1fbebe8f2a6a224004f98943a5c31a21001e9aa7bfb0`;
  - stdout contains
    `VERDICT: gemma3 down input-source native CUDA probe complete`;
  - Vast cleanup confirmed `stopped/exited`.
- Key m16 rows:
  - constant input baseline: `32.606us`;
  - small constant input baseline: `32.670us`;
  - constant input after gate_up+GeGLU producer: `69.793us`;
  - GeGLU output immediate: `68.356us`;
  - GeGLU output after sync: `70.200us`;
  - copied GeGLU output after sync: `70.098us`;
  - constant input after L2 flush: `90.343us`.
- Interpretation:
  - the isolated Marlin down row is fast only when repeated on warm constant
    input;
  - running the product-shaped gate_up+GeGLU producer immediately before down
    makes down slow even when down reads a separate constant input;
  - small constant input is not slower, so this is not GeGLU numeric magnitude
    or subnormal value sensitivity;
  - the remaining W2 tail-MLP lever is cache/producer-state or weight residency
    around the `gate_up -> down` sequence, not the existing Triton W4A16 path
    or GeGLU data sensitivity.
- Scope:
  - this is diagnostic evidence, not release performance evidence;
  - the remote worktree had old tracked artifact-log modifications after
    syncing `.git`; those are recorded in `git_verify.txt` and are not used for
    release performance claims.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XK — W2 source checkpoint: native down input-source probe

- Added `scripts/microbenches/gemma3_down_input_source_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_down_input_source_perf.sh`.
- Purpose:
  - XG measured the product-shaped tail MLP chain and found m16 `down_proj`
    around `68-71us` when it consumes GeGLU output;
  - XJ measured isolated Marlin down at the same Gemma3 shape around `30-33us`
    at m16 with synthetic constant input;
  - this probe keeps the Marlin down shape fixed and varies only input source /
    producer state: constant input, small constant input, constant after GeGLU,
    constant after L2 flush, immediate GeGLU output, synced GeGLU output, and
    device-copied GeGLU output.
- Expected GPU use:
  - run as a native CUDA minimal verification before any product change;
  - if GeGLU-derived input remains slow after sync/copy, inspect activation
    value range or down-kernel data sensitivity;
  - if constant input slows after preceding GeGLU/flush, inspect cache/producer
    state instead;
  - if the gap disappears, treat the previous difference as measurement setup
    and avoid this branch.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XJ — W2 native CUDA checkpoint: existing Triton W4A16 is slower than Marlin on Gemma3 MLP

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_triton_w4a16_native_probe_2026-06-16/`.
- Lane:
  `W2 Gemma3 dense Triton W4A16 vs Marlin native probe` on cached 1x RTX
  4090 Vast instance `40826362`.
- Paid GPU contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at prior
    USD 0.425/hr;
  - stop condition: startup/SSH/CUDA failure, nvcc compile failure, probe
    nonzero/timeout, or VERDICT line with artifacts copied back;
  - correctness gate: native probe exit 0 and VERDICT line;
  - performance command:
    `bash scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh`.
- Lifecycle:
  - instance started from `stopped/exited`;
  - CUDA verified with 1x RTX 4090, driver 565.77, nvcc 12.4;
  - source synced to remote HEAD
    `2847822395e857cbe23196b9590b88479eadeb60`;
  - remote source status was clean after restoring tracked artifact files
    affected by the source rsync exclude;
  - artifact copied back;
  - instance stopped and final Vast poll confirmed `stopped/exited`.
- Probe result:
  - `probe.rc=0`;
  - stdout contains
    `VERDICT: dense Triton W4A16 Gemma3 native CUDA probe complete`;
  - binary SHA256:
    `83a8112f31951e930b90736fcc7a7a99db69936fdebfa1f92b17449159a6e77c`.
- Key timings:
  - m16 `gate_up`: Marlin product workspace-zero `137.111us`,
    Triton W4A16 `618.924us`, so Triton is `4.51x` slower;
  - m16 `down`: Marlin product workspace-zero `32.527us`, Triton W4A16
    `609.813us`, so Triton is `18.75x` slower;
  - m32 `gate_up`: Marlin `141.253us`, Triton `781.304us`;
  - m32 `down`: Marlin `54.504us`, Triton `749.147us`.
- Interpretation:
  - existing `w4a16_gptq_f16.ptx` is not a W2 dense MLP performance lever;
  - do not productize `FERRUM_TRITON_INT4=1` for W2 release-grade work;
  - any Triton direction would require a new kernel/tile design, not the
    currently committed dense W4A16 PTX.
- Next:
  - continue with levers that can reduce dense GPTQ MLP work or improve the
    Marlin path itself; the direct alternative backend path is now rejected.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XH — W2 source checkpoint: native Triton W4A16 vs Marlin dense MLP probe

- No new GPU run in this checkpoint; this adds the next minimal native CUDA
  probe for the dense MLP compute lever.
- Added `scripts/microbenches/dense_triton_w4a16_gemma3_perf.cu` plus
  `scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh`.
- Probe scope:
  - loads the committed `w4a16_gptq_f16.ptx` through the CUDA Driver API,
    outside Cargo and outside product model loading;
  - compares the existing Marlin path against Triton W4A16 at Gemma3 W2
    `gate_up` (`k=5376,n=43008`) and `down` (`k=21504,n=5376`) shapes;
  - reports `m={1,10,16,23,32}` rows for product Marlin workspace-zero,
    Marlin kernel-only diagnostic, and Triton W4A16.
- Why this is aligned:
  - XG showed the measured product `tail_mlp` cost is explained by dense MLP
    compute across 62 layers, not by a hidden launch-chain overhead;
  - the next useful question is whether an existing alternative dense W4A16
    backend can materially beat Marlin on the exact Gemma3 MLP shapes before
    spending product-code effort on typed runtime wiring.
- Next:
  - run this probe once on the cached 1x4090 lane, capture stdout, binary
    SHA256, and cleanup evidence; if Triton is not faster enough or fails, do
    not wire it into product defaults.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XG — W2 native CUDA checkpoint: tail-MLP chain PASS, bottleneck is dense MLP compute across layers

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_mlp_chain_native_probe_2026-06-16/`.
- Lane:
  `W2 Gemma3 native tail-MLP chain probe` on the cached 1x RTX 4090 Vast
  instance `40826362`.
- Paid GPU contract:
  - expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at prior
    USD 0.425/hr;
  - stop condition: nvcc compile failure, probe nonzero/timeout, or VERDICT
    line with artifacts copied back;
  - correctness gate: native probe exit 0 and VERDICT line;
  - performance command:
    `bash scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh`.
- Lifecycle:
  - instance started from `stopped/exited`;
  - CUDA verified with 1x RTX 4090, driver 565.77, nvcc 12.4;
  - source synced to remote HEAD
    `2c281e56557c11486cbdec5da9dae1234dcae78d`;
  - remote source status was clean after restoring tracked artifact files
    affected by the source rsync exclude;
  - artifact copied back;
  - instance stopped and final Vast poll confirmed `stopped/exited`.
- Probe result:
  - `probe.rc=0`;
  - stdout contains
    `VERDICT: gemma3 tail MLP chain native CUDA probe complete`;
  - binary SHA256:
    `7dd82cd65a02958533c65b45d018e0b49600b1a30d394c7fa567a41f0d4ccca7`.
- Key timings:
  - m16 `product_ws_zero`:
    `chain_event_us=215.750`, `chain_host_sync_us=217.782`;
  - m16 product phase split:
    `pre_norm=5.914us`, `gate_up=139.671us`, `geglu=4.680us`,
    `down=70.903us`, `final_norm=6.352us`;
  - m16 `kernel_only_ws_prezero_diagnostic`:
    `chain_event_us=212.986`.
- Interpretation:
  - the single-layer Gemma3 tail-MLP chain is about `216us`; multiplied by
    62 layers this is about `13.4ms`, matching the earlier product profile
    band where `tail_mlp` was about `13.6-14.9ms` per decode step;
  - this rejects the hypothesis that W2 c16 is blocked by a hidden multi-ms
    launch-chain overhead outside the measured kernels;
  - the remaining W2 performance gap is dense GPTQ MLP compute across layers,
    dominated by `gate_up` and `down`, not HTTP/scheduler/postprocess,
    legacy graph routing, Marlin block-policy, or direct dense Marlin kernel
    swap.
- Next:
  - choose an optimization that changes the dense MLP compute path itself
    rather than doing another scheduling/env sweep: viable candidates are a
    different W4A16 backend for dense Gemma3 MLP, layer/prompt-level MLP
    work reduction if correctness-safe, or a product-profile check that
    compares exact per-layer call counts with the native chain.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XF — W2 source checkpoint: native CUDA Gemma3 tail-MLP chain probe

- No new GPU run in this checkpoint; this adds a minimal native CUDA probe for
  the next paid diagnostic lane.
- Added `scripts/microbenches/gemma3_tail_mlp_chain_perf.cu` plus
  `scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh`.
- Probe scope:
  - models the product Gemma3 tail MLP sequence
    `rms_norm_f32_to_f16 -> Marlin gate_up -> GeGLU -> Marlin down ->
    rms_norm_f16_add_to_f32`;
  - uses product-shaped W2 dimensions (`h=5376`, `intermediate=21504`,
    `gate_up n=43008`, `down n=5376`) and calls the existing product CUDA
    kernels directly;
  - emits phase timing plus full-chain event and host-sync timing for
    `m={1,10,16,23,32}`;
  - keeps `product_ws_zero` as the primary row and labels the workspace-prezero
    kernel-only row as diagnostic only, so it cannot be mistaken for product
    evidence or for an unsafe skip-workspace-zero runtime mode.
- Why this checkpoint matters:
  - earlier evidence already ruled out legacy `--batched-graph`, Marlin block
    policy override, direct vLLM dense Marlin kernel swap, and scheduler/HTTP
    as first-order W2 c16 levers;
  - current c16 profiling points to model-side Gemma3 tail MLP / dense Marlin,
    so this probe is the next smallest native CUDA validation before any full
    product benchmark.
- Next:
  - run this build script once on the cached 1x4090 lane, capture stdout and
    binary SHA256 under a new artifact dir, then choose a concrete optimization
    from the phase split.
- W2 remains blocked on final performance and final validator:
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` has not been produced.

## 2026-06-16 XE — W2 source guard: reject Gemma3 typed unified graph before CUDA runtime

- 本轮无新增 GPU run;这是对 `w2_unified_graph_typed_c16_2026-06-16`
  correctness failure 的 source guard checkpoint。
- Source change:
  - `FerrumConfigBuilder::validate_unified_graph` now rejects
    `FERRUM_UNIFIED_GRAPH=1` when model capabilities report
    `architecture=gemma3`;
  - failure message:`unified decode graph is disabled for Gemma3 sandwich-norm
    models`;
  - `docs/runtime-env-registry.tsv` notes that `FERRUM_UNIFIED_GRAPH` is typed
    but rejected for Gemma3 until graph replay is correctness-safe。
- Rationale:
  - typed unified graph passed one-shot `ferrum run`/`ferrum serve`,but c16
    bench hit `CUDA_ERROR_ILLEGAL_ADDRESS`;
  - since the flag is now product-visible,it must fail early rather than
    allowing users or release scripts to reach a CUDA illegal-address path。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-types unified_graph -- --nocapture` PASS;
  - `cargo test -p ferrum-types batched_graph_override_materializes_decode_graph_policy -- --nocapture` PASS;
  - `cargo test -p ferrum-cli run_effective_runtime_config_records_batched_graph_flag -- --nocapture` PASS。
- Release-grade status:
  - this closes a correctness hazard only;it does not improve W2 performance;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 XD — W2 typed unified graph smoke passes but c16 bench hits CUDA illegal address

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_graph_typed_c16_2026-06-16/`。
- Source/binary:
  - remote source head:`7f15a3ef9a57e2c23d889975ab629d25e8638803`;
  - source status clean for `crates/`,`scripts/`,`Cargo.toml`,`Cargo.lock`,
    and `ferrum.toml`;
  - release binary SHA256:
    `05f18a4cd8d8f34530758584122afad9e12f0bb929b450fc283449bb7d3180bd`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA typed unified graph c16 diagnostic`;
  - reused Vast/cache-retained instance `40826362`,1x RTX 4090,about
    USD `0.425/hr`;
  - stop condition:start/SSH/CUDA/source sync/build,`ferrum run
    --unified-graph`,`ferrum serve --unified-graph`,or c16 bench first
    failure;otherwise copy artifacts and stop the instance;
  - correctness gate:`ferrum run --unified-graph` known-answer smoke plus
    `ferrum serve --unified-graph` chat smoke with usage;
  - performance command:c16-only `bench-serve --fail-on-error --require-ci
    --seed 9271 --n-repeats 3`,diagnostic only。
- Runtime config evidence:
  - `serve_decision_trace.jsonl` selected `decode_graph_policy =
    unified_decode_graph` with `source=cli` and
    `source_key=FERRUM_UNIFIED_GRAPH`;
  - `serve_effective_config.json` selected
    `selected_graph_mode=unified_decode_graph`;
  - this run uses the typed CLI/config path,not a hidden env-only toggle。
- Correctness smoke:
  - `ferrum run --unified-graph` PASS:
    `RUN_SMOKE_PASS content='5' tokens=3`;
  - `ferrum serve --unified-graph` PASS:
    `SERVE_SMOKE_PASS content='5' completion_tokens=3`。
- c16 bench result:
  - repeat 1/3: `16 completed / 0 errored / 3.1s`;
  - repeat 2/3: `16 completed / 0 errored / 3.1s`;
  - repeat 3/3 started,then server hit CUDA illegal address;
  - no throughput result is valid,bench did not complete。
- Failure:
  - server log:
    `[unified-graph] replay err: Unsupported operation: post-launch sync:
    CUDA_ERROR_ILLEGAL_ADDRESS`;
  - follow-on panic:
    `CudaBackend: load_function(rms_norm_f32_to_f16):
    DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was
    encountered")`;
  - run was stopped per first-fail rule;GPU process list was clear afterward。
- Vast cleanup:
  - final poll verified `cur_state=stopped actual_status=exited`;
  - Vast JSON artifacts have `jupyter_token` redacted。
- Interpretation:
  - typed unified graph is now proven product-visible,but not product-safe for
    W2 performance work;
  - it passes one-shot run/serve smoke but fails under c16 bench,so it is a
    correctness blocker and cannot be used as performance evidence;
  - this reinforces the current path:avoid broad graph toggles,keep unified
    graph disabled for release evidence, and pursue the model-side
    Gemma3 MLP/Marlin bottleneck or a native CUDA graph-capture minimal repro
    before re-enabling this path.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 XC — W2 bottleneck narrowed after graph A/B: model-side Gemma3 MLP/Marlin dominates

- 本轮未新增 GPU run,没有新 release-grade artifact;这是基于最新
  `--batched-graph` A/B 与既有 profiler artifact 的 source/evidence
  checkpoint。
- 当前 head:`c5ff183f`。
- Evidence used:
  - `w2_paged_unified_default_path_cuda_smoke_2026-06-16`:
    default paged-unified product path `ferrum run` / `ferrum serve` correctness
    passes, c16 diagnostic `295.806 ± 5.211 tok/s`, health shows
    `decode_batch.calls=0`, `executor_model_lock.samples=4097`,
    `model_execution_time_ms=46.761`;
  - `w2_batched_graph_ab_cuda_diag_2026-06-16`: `--batched-graph`
    correctness passes, c16 diagnostic `287.117 ± 41.633 tok/s`,
    so `--batched-graph/default=0.9706` and graph toggle is not the current
    high-return lever;
  - `w2_typed_decode_profile_2026-06-16`: full `decode=16` iterations had
    mean total `23679.2us`, model time `23311.3us`, decode postprocess
    `347.9us`; model share `98.44%`, postprocess share `1.47%`;
  - `w2_profiler_graph_disabled_retry_2026-06-16` and
    `w2_marlin_typed_profile_2026-06-16`: c16 model-side profile repeatedly
    shows Gemma3 decode dominated by tail MLP / dense Marlin projections:
    `tail_mlp` around `13.6-14.9ms` per step, `matmul` around `7-8ms`,
    attention around `2.7ms`, QKV/RoPE around `0.7ms`; Marlin kernel aggregate
    around `16.5ms`, with `gate_up` around `8.7-9.5ms` and `down` around
    `4.3-5.3ms`.
- Updated bottleneck statement:
  - current c16 requests are reaching the paged unified model path; legacy
    `decode_batch` graph replay metrics stay at zero because that path is not
    serving the steady-state decode;
  - scheduler/HTTP/postprocess is not the primary c16 gap in the profiled
    steady-state path;
  - the highest-confidence bottleneck is Gemma3 model-side decode, especially
    tail MLP dense GPTQ/Marlin `gate_up` and `down` work;
  - dense Marlin grid/block-policy override and legacy batched graph toggle have
    already been falsified as main levers.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - avoid more broad graph/scheduler sweeps;
  - use native CUDA or a very small product profiler cell to test one concrete
    Gemma3 MLP/Marlin lever at a time;
  - after any source change, validate correctness first with product
    `ferrum run` and `ferrum serve`, then a minimal c16 diagnostic before any
    broader performance run。

## 2026-06-16 LXXXIX — W2 CUDA A/B: `--batched-graph` correct but not a c16 performance lever

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_ab_cuda_diag_2026-06-16/`。
- Source/binary:
  - local head at launch:`0adb292a`;
  - remote reused clean source `d6d872c1e12fc364886117b0431aec752b2d78ac`;
  - reused binary SHA256
    `11b26df2b8dccf3138b2fe294e80ef618cc6255e56af626213e6aaabe8b2e48f`;
  - no rebuild/reinstall performed,只复用上一轮环境和模型缓存。
- GPU 执行合同:
  - lane:`W2 batched-graph default-path A/B diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/binary check/run/serve 首败即停,或 run+serve
    correctness + c16 diagnostic 后停止实例;
  - correctness gate:`ferrum run --batched-graph` 与
    `ferrum serve --batched-graph`;
  - performance command:`bench-serve --fail-on-error --require-ci` c16
    diagnostic,非 release evidence。
- Correctness evidence:
  - `ferrum run --batched-graph` rc `0`,output content `"5"`,
    `finish_reason=stop`;
  - `ferrum serve --batched-graph` readiness poll `8`,chat rc `0`,response
    content `"5"`,`finish_reason=length`,`completion_tokens=1`;
  - health after bench:`successful_requests=331`,`failed_requests=0`;
  - server log scan file has `0` lines for panic/error/NaN/`<unk>`/`[PAD]`/
    invalid UTF/fallback/graph-failed/capture-failed patterns;
  - server stopped cleanly,Vast shutdown verified
    `cur_state=stopped actual_status=exited`。
- Effective server config:
  - `selected_graph_mode=legacy_batched_decode_graph`;
  - `selected_kv_layout=paged`;
  - `selected_attention_impl=legacy_paged_decode`;
  - `selected_max_sequences=16`,`selected_kv_capacity=512`,
    `selected_max_batched_tokens=2048`。
- c16 diagnostic performance:
  - command shape:`bench-serve --random-input-len 256 --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error
    --require-ci --seed 9271`;
  - rc `0`,completed per run `[100,100,100]`,errored per run `[0,0,0]`,
    output token count source `usage`;
  - output throughput `287.1167006548677 ± 41.632552793935645 tok/s`;
  - goodput `2.251751298484382 ± 0.3173552733445153 req/s`;
  - TTFT p50 `798.304ms`,TPOT p50 `46.950ms`。
- Interpretation:
  - previous same-binary default-path c16 was
    `295.8064415567493 ± 5.210666937312439 tok/s`;
  - `--batched-graph/default = 0.970624`,so graph replay is not the current
    W2-P2 throughput lever;
  - vs direct random-prompt vLLM diagnostic baseline
    `381.3929242134927 tok/s`,this is `75.2811%`,still below 80%;
  - remaining bottleneck likely sits in decode cadence, scheduler/admission,
    or per-token tail work above/beside graph replay。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - stop spending full sweeps on graph toggle;
  - inspect c16 execution cadence and tail latency with the smallest profiler
    that does not perturb correctness more than necessary;
  - consider comparing default vs graph profile traces only if trace overhead is
    bounded and the hypothesis is specific。

## 2026-06-16 LXXXVIII — W2 CUDA checkpoint: default paged-unified run/serve correctness passes, c16 still below target

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_unified_default_path_cuda_smoke_2026-06-16/`。
- Source checkpoint:
  `d6d872c1e12fc364886117b0431aec752b2d78ac`,远端通过 git bundle clone,
  `git status --short` clean,无 remote diagnostic source patch。
- GPU 执行合同:
  - lane:`W2 default-path paged-unified correctness smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/CUDA/source sync/build/run/serve 任一失败即收集
    日志并停止;run+serve 正确性通过后同次启动内跑一个 c16 diagnostic;
  - correctness gate:默认产品路径 `ferrum run` 和 `ferrum serve`;
  - performance command:correctness clean 后 `bench-serve --fail-on-error
    --require-ci` c16 diagnostic,非 release evidence。
- Build evidence:
  - CUDA release build rc `0`;
  - binary SHA256
    `11b26df2b8dccf3138b2fe294e80ef618cc6255e56af626213e6aaabe8b2e48f`;
  - CUDA environment:driver `565.77`,runtime CUDA `12.7`,`nvcc 12.4.131`。
- Correctness evidence:
  - `ferrum run` rc `0`,one-shot JSONL output content `"5"`,
    `finish_reason=stop`;
  - `ferrum serve` readiness poll `8`,chat rc `0`,response content `"5"`,
    `finish_reason=length`,`usage.prompt_tokens=23`,`completion_tokens=1`;
  - health after bench:`successful_requests=331`,`failed_requests=0`;
  - server log scan file has `0` lines for panic/error/NaN/`<unk>`/`[PAD]`/
    invalid UTF patterns used in this artifact;
  - server stopped cleanly,post-stop `nvidia-smi` shows no running GPU
    processes;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Effective server config:
  - `selected_kv_layout=paged`;
  - `selected_attention_impl=legacy_paged_decode`;
  - `selected_graph_mode=graph_disabled`;
  - `selected_max_sequences=16`,`selected_kv_capacity=512`,
    `selected_max_batched_tokens=2048`。
- c16 diagnostic performance:
  - command shape:`bench-serve --random-input-len 256 --random-output-len 128
    --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error
    --require-ci --seed 9271`;
  - rc `0`,completed per run `[100,100,100]`,errored per run `[0,0,0]`,
    output token count source `usage`;
  - output throughput `295.8064415567493 ± 5.210666937312439 tok/s`;
  - goodput `2.3204614024423846 ± 0.031239672060048216 req/s`;
  - TTFT p50 `798.748ms`,TPOT p50 `45.528ms`。
- Performance interpretation:
  - 直接同形状 random-prompt vLLM artifact 约 `381.5 tok/s`,但该 vLLM
    run 有 `1` 个 bad output/errored request,所以只能做 diagnostic;
    按该不干净 baseline 计算,Ferrum 约 `77.6%`;
  - 更干净的 same-instance vLLM ShareGPT baseline 是 `518.796 tok/s`,但本轮
    Ferrum 没有复跑同一 ShareGPT dataset,不能拿它做严格当前同数据集比例;
  - 因此本轮仍不能证明 `>=80%` mainstream-engine target。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - 先补一个干净 same-dataset current Ferrum/vLLM comparison,再更新性能结论;
  - 当前 correctness blocker 已解除,剩余问题回到性能瓶颈,重点看
    graph-disabled/default runtime policy、decode cadence、scheduler/admission
    与 per-token tail latency。

## 2026-06-16 LXXXVII — W2 source checkpoint: allow paged KV for windowed Gemma3 on varlen backends

- 本轮源码修复:
  - `paged_kv_allowed_for_layer_schedule(...)` 不再对所有
    `sliding_window_pattern != 0` 模型一刀切禁用 paged KV;
  - 新规则: paged enabled 且 (`sliding_window_pattern == 0` 或 backend
    supports varlen QKV);
  - 注释更新为:windowed Gemma3 只有在后端 varlen QKV 路径能接收 per-layer
    sliding-window schedule 时才允许 paged KV;
  - 单测改名并覆盖正反例:
    `paged_kv_layer_schedule_allows_windowed_models_with_varlen_backend`。
- Why:
  - LXXXVI 已用同形状 CUDA product diagnostic 证明,在放开 paged KV guard
    且应用 embed-scale 修复后,`ferrum serve` 最小 chat smoke 从空输出
    恢复为 expected first token `"5"`;
  - LXXXIII 的 native CUDA probe 已排除 split-QKV + paged-varlen attention
    pair 的独立正确性问题;
  - 因此可以把远端 diagnostic guard override 提升为受测试覆盖的源码逻辑,
    但仍需要无 dirty patch 的默认产品路径验证。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models paged_kv_layer_schedule_allows_windowed_models_with_varlen_backend -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests)。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - CUDA default-path correctness smoke without any remote diagnostic source
    patch;
  - because this changes product behavior, validate both `ferrum run` and
    `ferrum serve` before using performance numbers as evidence;
  - only after default-path correctness is clean should c16/c32 same-hardware
    performance comparison resume。

## 2026-06-16 LXXXVI — W2 CUDA diagnostic: embed-scale fix restores first-token correctness

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_unified_embed_scale_fix_cuda_smoke_2026-06-16/`。
- Source checkpoint:
  `fb6789c7f99cc08f05842503846ea42af2be842d` plus a remote-only
  diagnostic patch that temporarily allowed paged KV for windowed Gemma3 when
  CUDA supports varlen QKV。默认 checked-in guard 仍未放开。
- GPU 执行合同:
  - lane:`W2 paged-unified embed-scale fix product smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/source sync/diagnostic guard patch/build/serve/chat
    任一失败,或 fixed-path `[unified-logits]` 与 response evidence collected
    后停机;
  - correctness command:`ferrum serve` + one non-stream chat request with
    `max_tokens=1`;
  - performance command:none。
- Execution evidence:
  - CUDA release build rc `0`;
  - binary SHA256
    `e131ce885efb3f8aeb6049a9181f646638c4c8f81d0c993cfb33da29a4d7bc65`;
  - response content `"5"`,`finish_reason=length`,`completion_tokens=1`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`;
  - `nvidia_smi_after_stop.txt` shows no running GPU processes。
- Key result:
  - `[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23
    attempted_unified=true fallback=false fallback_reason=none elapsed=141708us`;
  - `[unified-logits] call#0 row=0 orig_idx=0 global=22 finite=262208
    nan=0 pos_inf=0 neg_inf=0
    top=[236810:42.031250,239374:20.453125,247918:20.453125,239341:20.187500,242323:20.015625]`。
- Interpretation:
  - LXXXIV 的 pre-fix 同形状 smoke 首 token 直接 EOS/stop;本轮同形状
    smoke 返回 expected first token `"5"`;
  - logits row 仍全 finite,且 EOS token id `106` 不再是 top-1;
  - 这证明 `unified_forward_internal` 漏乘 Gemma3 `embed_scale` 是一个真实
    paged-unified 正确性 bug,不是单纯性能测量噪声。
- Release-grade status:
  - 这是 diagnostic evidence,不是 release evidence,因为远端用了临时
    paged-KV guard override;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。
- Required next:
  - 将 paged-KV guard 放开逻辑以源码形式提交并更新单测;
  - 之后不用远端 dirty patch,按默认产品路径最小验证 `ferrum run` 和
    `ferrum serve`;
  - 默认路径 correctness 过之前不跑 c16/c32 performance。

## 2026-06-16 LXXXV — W2 source checkpoint: apply Gemma embed scale in unified forward

- 本轮源码修复:
  - `unified_forward_internal` 在 `embedding_lookup` 后补上
    `cfg.embed_scale` 的 `B::scale_inplace`;
  - 缩放发生在 `activation_to_f32_shadow` 之前,因此 Gemma3 CUDA 的 F32
    residual shadow 也接收缩放后的 residual;
  - 这与 legacy `decode_batch_internal`,`prefill_internal`,`decode_internal`
    等路径保持一致。
- Why:
  - LXXXIV 的 product logits diagnostic 证明 paged-unified 首步 logits 全
    finite,但 eos/stop token id `106` 排 top;
  - native split-QKV + paged-varlen attention 已在 LXXXIII 通过,所以问题
    更可能在 product unified path 的模型语义差异;
  - 源码对比发现 unified embedding path 漏掉 Gemma3 的
    `embed_scale = sqrt(hidden_size)` 语义,这是与已正确 legacy path 的
    直接差异。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests);
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites -- --nocapture`
    PASS。
- Required next validation:
  - rerun only the LXXXIV minimal CUDA product smoke shape with the same
    diagnostic paged-KV guard override;
  - success criterion for this diagnostic: chat response content should no
    longer be empty/stop-at-first-token, and `[unified-logits]` should no
    longer rank eos token id `106` as top-1;
  - do not run c16/c32 performance until this correctness smoke is clean。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXIV — W2 CUDA product diagnostic: paged-unified logits rank eos top-1

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_unified_logits_product_diag_2026-06-16/`。
- Source checkpoint:
  `b768073a80c8a7519c1107083f5a10b478d0fe1a` plus a remote-only
  diagnostic patch that temporarily allowed paged KV for windowed Gemma3
  when CUDA supports varlen QKV。默认 checked-in path remains protected。
- GPU 执行合同:
  - lane:`W2 paged-unified product logits diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/CUDA/source sync/diagnostic patch/build/serve/chat
    任一失败,或 `[unified-logits]` evidence collected 后停机;
  - correctness command:`ferrum serve` + one non-stream chat smoke with
    `max_tokens=1`;
  - performance command:none。
- Execution evidence:
  - CUDA release build rc `0`;
  - binary SHA256
    `1d046a81f5194f80a946b2c0e2f37f1de97fdde69668ad359135f032e32af5d9`;
  - response empty,`finish_reason=stop`,`completion_tokens=1`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23
    attempted_unified=true fallback=false fallback_reason=none elapsed=136978us`;
  - `[unified-logits] call#0 row=0 orig_idx=0 global=22 finite=262208
    nan=0 pos_inf=0 neg_inf=0
    top=[106:11.039062,108:9.445312,107:8.882812,245526:8.460938,236743:8.304688]`;
  - tokenizer/generation metadata confirms eos token ids `[1,106]`,and token
    id `106` is `<end_of_turn>`。
- Interpretation:
  - paged-unified product path is not producing NaN/Inf or uninitialized
    sampled logits in this repro;
  - the wrong behavior is specifically that the first sampled logits row ranks
    the stop token highest;
  - after LXXXIII ruled out the standalone split-QKV + paged-varlen attention
    chain, the next source diff to fix is product unified model semantics
    above those kernels。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXIII — W2 native checkpoint: split-QKV + paged-varlen combo probe PASS

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_varlen_split_qkv_native_probe_2026-06-16/`。
- Source checkpoint:
  `7dc711ef817af737903098f14c068852c04d7dbf`。
- GPU 执行合同:
  - lane:`W2 paged-varlen split-QKV native correctness probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:启动/SSH/CUDA/source sync/compile/probe 任一失败,
    或 probe PASS 后复制 artifact 并停机;
  - correctness command:
    `bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`;
  - performance command:none。
- Execution evidence:
  - remote/source head `7dc711ef817af737903098f14c068852c04d7dbf`;
  - CUDA environment:driver `565.77`,runtime-reported CUDA `12.7`,
    `nvcc 12.4.131`;
  - `probe/paged_varlen_split_qkv_correctness.rc` = `0`;
  - stdout contains
    `VERDICT: paged varlen split-qkv correctness PASS`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Native CUDA result:
  - `qk_mode=1 sliding_window=0`:q/k/v err all `0`,
    attention err `0.00012147`;
  - `qk_mode=1 sliding_window=3`:q/k/v err all `0`,
    attention err `0.00012141`;
  - `qk_mode=2 sliding_window=3`:q/k/v err all `0`,
    attention err `0.00011945`;
  - `qk_mode=3 sliding_window=3`:q/k/v err all `0`,
    attention err `0.00011978`;
  - `qk_mode=1 semantic_delta_full_vs_window=0.06742159`,证明 full
    causal 和 sliding-window 语义确实不同,不是等价空测。
- Interpretation:
  - standalone native CUDA chain
    `split_qkv_norm_rope_into_paged_cache_varlen_f16` ->
    `paged_varlen_attn_f16` 在合成非零 historical KV、当前 varlen 写入、
    QK-norm/RoPE modes 和 sliding-window attention 上对齐 CPU reference;
  - LXXX 的 Gemma3 paged-unified empty output 更可能在这对 kernels 之上
    或依赖真实 product/model state,例如后续 residual/tail/lm_head、sampled
    logits/stop token,或真实形状/权重数据;
  - 下一步应使用 LXXXI 的 `[unified-logits]` 做一次最小 product smoke,
    不跑 c16/c32,以判断首 token 是否 EOS/stop、logits row 错位或数值异常。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXII — W2 source checkpoint: native split-QKV + paged-varlen combo probe

- 本轮没有启动 GPU,没有新的性能数字,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source/tooling change:
  - 新增
    `scripts/microbenches/paged_varlen_split_qkv_correctness.cu`;
  - 新增
    `scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`;
  - probe 直接用 `nvcc` 链接
    `split_qkv_norm_rope_into_paged_cache.cu` 和
    `paged_varlen_attention.cu`,不走 Cargo,不加载模型;
  - 覆盖 `qk_mode=1`(QK-norm + half-split RoPE, Gemma/Qwen-style),
    `qk_mode=2`,`qk_mode=3`,以及 full causal / sliding-window;
  - cache_k/cache_v 预置非零 historical KV,再由 varlen split 覆盖当前
    q tokens,最后由 paged-varlen attention 消费同一套 Q/K/V buffers。
- Why:
  - XLIII 已证明孤立 `paged_varlen_attention` 的 sliding-window 语义正确;
  - LXXX 失败发生在 product 链路
    `split_qkv_norm_rope_paged` -> `paged_varlen_attention` 之后,所以需要
    native CUDA 组合 probe,而不是继续跑 c16/c32 或只测 attention。
- Local validation:
  - `bash -n scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`
    PASS;
  - `git diff --check -- scripts/microbenches/paged_varlen_split_qkv_correctness.cu scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`
    PASS;
  - local machine has no `nvcc`,so native CUDA compile/run is pending on the
    cached 1x4090 instance。
- Required next validation:
  - run only:
    `bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`;
  - expected PASS line:
    `VERDICT: paged varlen split-qkv correctness PASS`;
  - if it fails, fix the native kernel issue before any product smoke or
    performance sweep;
  - if it passes, use the `[unified-logits]` product smoke from LXXXI to
    classify the remaining empty-output cause。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXXI — W2 source checkpoint: unified logits diagnostic for paged-varlen failure

- 本轮没有启动 GPU,没有新的性能数字,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - 在 `FERRUM_DECODE_OP_PROFILE` 已启用时,`unified_forward_internal`
    读回 sampled logits 后打印 `[unified-logits]`;
  - 每行日志包含前两条 sampled row 的 `orig_idx/global`,
    finite/NaN/+Inf/-Inf 计数,以及 top-5 token id/logit;
  - 诊断采样规则为前 8 次 unified logits readback 必打,之后每 64
    次打一条,避免长 bench 日志爆量;
  - 默认产品路径不启用该日志,不改变 scheduler/KV/sampling 行为。
- Why:
  - LXXX 证明 paged-unified Gemma3 可以去掉
    `fallback_reason=paged_kv_required`,但首个 chat smoke 变成空输出
    且 `finish_reason=stop`;
  - 下一次 CUDA 应该先跑 `max_tokens=1` 的最小 correctness smoke,
    用 `[unified-logits]` 判断是 EOS/stop token 排在 top,logits row
    错位,还是 NaN/Inf/未初始化等数值问题;
  - 在这个结果出来前,不要再跑 c16/c32 性能 sweep。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models logit_row_diagnostics_counts_and_sorts_top_values -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models unified_logits_diag_uses_front_loaded_sampling -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests)。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXX — W2 CUDA checkpoint: paged unified removes fallback but fails chat correctness

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_kv_unified_cuda_smoke_2026-06-16/`。
- Source checkpoint tested:
  `103c7013e849b198cabaa7ad47cd45063bf21e6d`。
- GPU 执行合同:
  - lane:`W2 CUDA paged-KV unified smoke after guard fix`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - stop condition:build + serve/chat smoke + c16 profiler evidence,或首个失败;
  - correctness gate:build rc `0`,serve ready,chat smoke content `5`,
    bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,但本轮因 correctness failure 未进入 bench。
- Execution evidence:
  - remote git head `103c7013e849b198cabaa7ad47cd45063bf21e6d`,
    remote source status clean;
  - binary SHA256
    `0d4595b6dbb6f4920ec5ed4af286ce7fbd89ad936d8dba1c76ad99d15806ac70`;
  - `build/build.rc=0`,serve ready poll `61`;
  - chat smoke failed:content empty,`completion_tokens=1`,
    `finish_reason=stop`;
  - `run_profile.rc=1`;bench did not start;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23
    attempted_unified=true fallback=false fallback_reason=none elapsed=336326us`;
  - `[unified-op]` confirms `split_qkv_norm_rope_paged` and
    `paged_varlen_attention` executed;
  - so `paged_kv_required` was removed, but the paged-unified Gemma3 path
    is not product-correct yet。
- Source follow-up in current HEAD:
  - keep windowed Gemma3 on contiguous KV by default;
  - update the helper/test so `sliding_window_pattern != 0` remains
    non-paged until paged-varlen correctness is fixed;
  - this prevents `103c7013` from leaving a product correctness regression
    active at HEAD。
- Required next work:
  - do not run another c16 perf bench until a smaller correctness repro
    isolates paged-varlen wrong output;
  - preferred next probe: native CUDA/minimal kernel or model-layer smoke for
    `split_qkv_norm_rope_paged` + `paged_varlen_attention` against the
    contiguous Gemma3 path, then re-enable only after chat smoke passes。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXIX — W2 source checkpoint: allow CUDA paged KV for Gemma3 windowed unified path

- 本轮没有再次启动 GPU,不产生新的性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `ensure_kv` 不再用 `sliding_window_pattern == 0` 一刀切禁用 paged KV;
  - 新增 `paged_kv_allowed_for_layer_schedule(...)`,规则是:
    paged enabled 且 (`sliding_window_pattern == 0` 或后端支持
    `varlen_qkv`);
  - 结果:CUDA 这类已有 paged-varlen/sliding-window 参数的后端可以给
    Gemma3 windowed layers 分配 paged pools,从而满足 `unified_forward`
    的 paged KV 前置条件;
  - Metal paged decode 仍未暴露 per-layer window 到 paged dispatch,所以
    Gemma3/windowed family 仍保持 contiguous KV 保护。
- Why:
  - LXXVIII CUDA artifact 显示所有 observed `unified_decode` 都
    `attempted_unified=true`,但全部
    `fallback_reason=paged_kv_required`;
  - 源码确认 `ensure_kv` 里 Gemma3 因 `sliding_window_pattern != 0`
    禁用 paged pools,而 `unified_forward` 又硬要求 `paged_pools`,
    形成真实设计矛盾。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models paged_kv_layer_schedule_allows_windowed_models_only_with_varlen_backend -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests);
  - `git diff --check -- crates/ferrum-models/src/models/llama_family.rs ...`
    PASS。
- Required next validation:
  - one minimal CUDA smoke only: build current checkpoint, serve/chat,
    c16 profiler-on bench;
  - expected first check is that `[unified-decode]` prefill lines no longer
    report `fallback_reason=paged_kv_required`;
  - if correctness fails, stop and inspect token/logit/KV state before any
    performance measurement。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXVIII — W2 CUDA checkpoint: unified_decode fallback is paged_kv_required

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_decode_fallback_reason_cuda_diag_2026-06-16/`。
- Source checkpoint:
  `5e9ae1514247e1ea7c34459dae3e2d1198c5e77d`。
- GPU 执行合同:
  - lane:`W2 unified_decode fallback_reason CUDA diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min;实际包含 release relink;
  - stop condition:build + serve/chat smoke + c16 profiler fallback_reason
    evidence complete,或首个失败;
  - correctness gate:build rc `0`,serve ready,chat smoke pass,bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,diagnostic only。
- Execution evidence:
  - remote git head `5e9ae1514247e1ea7c34459dae3e2d1198c5e77d`,
    remote source status clean;
  - binary SHA256
    `9ef89f43cc5f8675f85aaa32811ba2a3ee9ea704f79f2f87fd250a913363913e`;
  - `build/build.rc=0`,`run_profile.rc=0`,
    `bench/bench_sharegpt_c16.rc=0`;
  - serve ready poll `60`;chat response content `5`,usage present,
    bad_output false;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - c16 diagnostic:16 completed / 0 errored,`output_token_count_source=usage`;
  - throughput `315.39451845233344 tok/s`;
  - orientation-only vLLM baseline `518.7959572662905 tok/s`,Ferrum/vLLM
    ratio `0.6079355747378077`;
  - `[unified-decode]` line count `131`,prefill line count `3`;
  - all observed fallback reasons:
    `{"paged_kv_required": 131}`;
  - c16 prefill cohort:
    `call#3 items=10 prefill=10 decode=0 total_q=1220 attempted_unified=true fallback=true fallback_reason=paged_kv_required elapsed=890406us`;
  - later c16 cohort:
    `call#67 items=16 prefill=16 decode=0 total_q=1952 attempted_unified=true fallback=true fallback_reason=paged_kv_required elapsed=1396576us`。
- Interpretation:
  - LXXV full-logits fix was not the main bottleneck;
  - `LlmExecutor::unified_decode` now proves the model unified path is
    attempted,then falls back because Gemma3 CUDA lacks paged pools;
  - source trace confirms `ensure_kv` disables paged KV for
    `sliding_window_pattern != 0`,while `unified_forward` requires
    `paged_pools`。This is the next real bottleneck。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXVII — W2 source checkpoint: expose unified_decode fallback reason

- 本轮没有启动 GPU,不产生性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `LlmExecutor::unified_decode` 在现有 typed profiler 开关
    `FERRUM_BATCH_PREFILL_PROF` / `FERRUM_BATCH_DECODE_PROF` 下新增
    `[unified-decode]` 结构化日志;
  - 日志记录 `items`、`prefill`、`decode`、`total_q`、
    `attempted_unified`、`fallback`、`fallback_reason` 和 elapsed time;
  - full-logits 不可用时稳定输出
    `fallback_reason=requires_full_logits_unavailable`;
  - `model.unified_forward` 返回 Unsupported 时复用既有短码分类,例如
    `unified_varlen_qkv_disabled`、`sandwich_f32_shadow_required`、
    `paged_kv_required`、`active_lora_adapter`。
- Why:
  - LXXVI 证明 full-logits guard 修复只带来约 `+4.7%`,并且
    `prefill-profile tokens=122` 仍重复出现,说明 Gemma3 c16 prefill cohort
    仍在 `unified_decode` 内部回落到 serial prefill;
  - 继续做 full c16 sweep 前,需要一次最小 CUDA 验证直接读出 fallback reason,
    避免继续猜测瓶颈。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models unified_decode_prof_logs_prefill_fallback_and_sampled_decode -- --nocapture`
    PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (13 tests);
  - `git diff --check -- crates/ferrum-models/src/executor/llm_executor.rs`
    PASS。
- Required next CUDA validation:
  - reuse cached 4090 lane,run minimal serve/chat+c16 diagnostic with profiler on;
  - target evidence is the first `[unified-decode]` prefill line and its
    `fallback_reason`,not a repeated full performance sweep;
  - if it reports a source guard such as `unified_varlen_qkv_disabled` or
    `sandwich_f32_shadow_required`,fix that exact guard before measuring again。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXVI — W2 CUDA checkpoint: full-logits unified prefill fix is insufficient

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_full_logits_unified_prefill_cuda_diag_2026-06-16/`。
- Source checkpoint:
  `40186c75e393ef58e81b9f5acfe529186505a0bc`。
- GPU 执行合同:
  - lane:`W2 full-logits unified-prefill CUDA diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:15-30min,约 USD `0.11-0.22`;
  - stop condition:CUDA build + serve/chat smoke + c16 diagnostic bench +
    log evidence complete,或首个失败;
  - correctness gate:build rc `0`,serve ready,deterministic chat smoke pass,
    bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,diagnostic only。
- Execution evidence:
  - `build/build.rc=0`,`run_profile.rc=0`,
    `bench/bench_sharegpt_c16.rc=0`;
  - smoke response content `5`,usage present,bad_output false;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`;
  - remote source dirty includes the 3 synced source files for
    `40186c75`,so this is not release performance evidence。
- Key result:
  - c16 diagnostic:16 completed / 0 errored,`output_token_count_source=usage`;
  - throughput `298.24957600538823 tok/s` vs previous `284.90049780836483`,
    about `+4.7%`;
  - orientation-only vLLM baseline `518.7959572662905 tok/s`,Ferrum/vLLM
    ratio `0.5748880110341743`;
  - `prefill-profile tokens=122` lines remain `26`;
  - first c16 prefill cohort still shows
    `iter#3 items=10 prefill=10 total=927027us model=924576us` plus repeated
    serial 122-token `prefill-profile` rows。
- Interpretation:
  - LXXV full-logits guard fix was locally correct but insufficient for W2-P2;
  - main TTFT/prefill wall time remains,so do not run another CUDA c16
    diagnostic until `LlmExecutor::unified_decode` records why
    `model.unified_forward` still falls back;
  - next source step: add unified_decode fallback reason observability
    analogous to `batch_prefill`。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXV — W2 source checkpoint: allow full-logits unified prefill

- 本轮没有启动 GPU,不产生性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `DecoderOnlyLLM` 新增 `unified_forward_can_return_full_logits()`,默认
    `true`;
  - `LlmExecutor::unified_decode` 不再因为 batch 中存在
    `ferrum_require_full_logits` 就无条件跳过 `model.unified_forward`;
  - 只有模型声明 unified path 不能返回 full logits 时才保留旧 fallback;
  - Qwen3 MoE 在 `unified_greedy_argmax` sentinel 路径开启时返回 `false`,
    保持 full-logits correctness 保护。
- Why:
  - LXXIV artifact 里的 engine `[unified-prof] items=10 prefill=10` 只能证明
    engine 构造了 unified batch,不能证明 model 层走了 true unified prefill;
  - 同一日志中 `prefill-profile tokens=122` 重复 10 次,与
    `LlmExecutor::unified_decode` 的 full-logits guard 对应:普通请求带
    tokenizer/sampling mask 时会设置 `ferrum_require_full_logits`,从而把
    Gemma3 c16 prefill cohort 退回 serial `model.prefill`;
  - 此改动把产品路径从“full-logits 必定 serial fallback”改为“模型能返回
    full logits 时仍用 unified prefill”,直接对齐 W2-P2 的 batched/unified
    Gemma3 fast path 目标。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llm_executor -- --nocapture` PASS
    (12 tests);
  - targeted tests PASS:
    `unified_decode_uses_unified_forward_when_full_logits_supported`,
    `unified_decode_skips_unified_forward_when_full_logits_unsupported`,
    `unified_decode_full_logits_prefill_uses_unified_forward_and_prepares_kv_capacity_hint`,
    `batch_prefill_falls_back_after_unified_unsupported`。
- Required next CUDA validation:
  - build CUDA release binary on the cached 4090 lane;
  - rerun minimal c16 serve/chat/bench diagnostic;
  - verify `prefill-profile tokens=122` no longer repeats serially before
    the c16 prefill batch, and compare throughput/TTFT against LXXIV。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXIV — W2 CUDA checkpoint: batch-prefill fallback hypothesis rejected

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batch_prefill_fallback_reason_diag_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 batch-prefill fallback-reason diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:15-25min,约 USD `0.11-0.18`;
  - stop condition:CUDA build + serve/chat smoke + c16 diagnostic bench +
    captured prefill/fallback evidence,或首个失败;
  - correctness gate:build rc `0`,serve ready,deterministic chat smoke pass,
    bench rc `0`;
  - performance command:c16/n=16/n_repeats=1 `bench-serve --fail-on-error
    --seed 9271`,diagnostic only。
- Execution evidence:
  - local source checkpoint:`7eb1747703e63ba7ac58ef2133a991f98c21e413`;
  - remote base HEAD:`935777e9feb8c1606631761ec8e0fb6c3f3f0a06`,只同步本轮
    instrumentation diff,因此不作为 release performance evidence;
  - `build/build.rc=0`,`run_profile.rc=0`,
    `bench/bench_sharegpt_c16.rc=0`;
  - smoke response content `5`,usage present,bad_output false;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `FERRUM_BATCH_PREFILL_PROF` 来自 typed config,但完整 server/profile 日志中
    `[batch-prefill]`/`fallback_reason=` 行数为 `0`;
  - continuous unified path 明确已跑 batch prefill:
    `iter#3 items=10 prefill=10 total=946123us model=943620us`;
  - 因此此前“c16 TTFT 主要是 `LlmExecutor::batch_prefill` serial fallback”的
    假设被本轮产品路径证据推翻。
- Diagnostic performance shape:
  - c16 diagnostic:16 completed / 0 errored,`output_token_count_source=usage`;
  - throughput `284.90049780836483 tok/s`;
  - orientation-only vLLM baseline `518.7959572662905 tok/s`,Ferrum/vLLM
    ratio `0.549157127803387`;
  - 该数字为 single-run diagnostic,不是 release 或最终性能声明。
- Interpretation:
  - 下一步不要继续围绕 `LlmExecutor::batch_prefill` fallback 猜测消耗;
  - 真瓶颈转向 unified prefill wall time 本身,并结合已确认的 dense GPTQ
    Marlin MLP kernel 热点和 prefill attention 成本做最小验证;
  - dense Marlin block-policy native probe 已排除 grid override,不能作为
    产品优化杠杆。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXIII — W2 source checkpoint: expose batch-prefill fallback reason

- 本轮没有启动 GPU,不产生性能结论,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `LlmExecutor::batch_prefill` 在 `FERRUM_BATCH_PREFILL_PROF` profiler 行中新增
    `fallback_reason=<code>`;
  - unified prefill `Unsupported` 文本会被归类为稳定短码,包括
    `unified_varlen_qkv_disabled`,`sandwich_f32_shadow_required`,
    `paged_kv_required`,`active_lora_adapter`,`requires_full_logits` 等;
  - 行为不变:unsupported 时仍 fallback 到 serial per-item `model.prefill`。
- Why:
  - W2 c16 TTFT 侧已定位为 Gemma3 serial prefill fallback;
  - 之前 profiler 只能看到 `fallback=true`,不能结构化证明是 varlen/Gemma3
    guard、prefix cache、LoRA、paged KV 还是 full-logits 触发;
  - 下一次 CUDA prefill diagnostic 可以直接验证产品路径的 fallback reason,
    为 narrow Gemma3 cohort-prefill 实现提供可回归证据。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models unified_fallback_reason_code -- --nocapture` PASS;
  - `cargo test -p ferrum-models batch_prefill_falls_back_after_unified_unsupported -- --nocapture`
    PASS;
  - `git diff --check -- crates/ferrum-models/src/executor/llm_executor.rs` PASS。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXII — W2 native checkpoint: dense Marlin block-policy probe rejects grid override

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_block_policy_probe_2026-06-16/`。
- Source checkpoint:
  `da8d8b25 test(cuda): probe dense marlin block policy`。
- GPU 执行合同:
  - lane:`W2 dense Marlin block-policy native probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,约 USD `0.06-0.12`;
  - stop condition:nvcc 编译失败或 native probe 打出 `VERDICT` 后复制
    artifact 并停机;
  - correctness gate:native CUDA compile rc `0`,probe rc `0`;
  - performance command:
    `timeout 1800 bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`。
- Execution evidence:
  - remote base HEAD:`935777e9feb8c1606631761ec8e0fb6c3f3f0a06`;
  - local source checkpoint:`da8d8b25a3f0aa28e826cfd75f3bcfae7b70ea3e`;
  - 本轮为节省时间只同步 native microbench 相关 dirty diff,不作为
    release performance evidence;
  - `probe/dense_marlin_gemma3_perf.rc=0`;
  - stdout 包含 `VERDICT: dense Marlin native CUDA probe complete`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Key result:
  - `gate_up m=16 auto` weight-cycle default `133.956us`,
    `blocks_n_tiles` `134.284us`,`blocks_2sms` `134.647us`;
  - `down m=16 auto` weight-cycle default `68.689us`,
    `blocks_n_tiles` `74.203us`,`blocks_2sms` `74.354us`;
  - `m=23/32` 上 `blocks_n_tiles`/`2sms` 对 `gate_up/down` 更差。
- Interpretation:
  - dense Marlin `gridDim.x`/block policy override 不是当前 W2 产品优化杠杆;
  - 不应把 `blocks=n_tiles` 或 `2sms` 推进到产品内核;
  - 下一步回到 decode integration / non-Marlin scheduling / prefill TTFT,
    或做更窄的 launch-count/overlap native probe。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXXI — W2 CUDA checkpoint: profiler path passes with graph capture disabled

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_profiler_graph_disabled_retry_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 profiler graph-disabled retry`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,hard cap 25min;
  - stop condition:build/server/chat/bench/profile 首败,或最小 c16
    profile 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` c16 single cell;
  - performance command:diagnostic-only c16 ShareGPT single-run profile,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`f7612c3a2a17c7e051f326ed7bac54484b25eb3a`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `2a2ed419f3e80ede06ceaf54ba4495b66265c5ce2ba14b66dc39a35257cb6844`;
  - `ferrum serve` ready after poll `62`,chat smoke passed with content
    `5` and `completion_tokens=3`;
  - `bench_sharegpt_c16.rc=0`,`run_profile.rc=0`,
    `run_remote_profile.outer.rc=0`;
  - c16 diagnostic profile completed `16/16`,errors `0`,
    `output_token_count_source=usage`;
  - `capture_unsupported_panic=false`,`graph_capture_line_count=0`;
  - profile lines: `prefill-profile=297`,`batched-op-profile=128`,
    `unified-prof=67`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Diagnostic performance,not release evidence:
  - profile/eager path output throughput mean:`312.22668693855985 tok/s`;
  - same-artifact orientation vs prior vLLM c16 baseline:
    `60.18294525342617%`;
  - because this is `n_repeats=1`,profiler/eager path,no CI/no
    `--require-ci`,it is diagnostic only。
- Bottleneck signal:
  - c16 decode profile is now stable enough to use;
  - repeated `batched-op-profile m=16` shows `tail_mlp` around
    `13.7ms` of `27.5-28.1ms` total per decode iteration;
  - `tail_gate_up` around `9.0ms`, `tail_down` around `4.7ms`,
    matmul bucket around `7.0ms`, attention around `2.1-2.7ms`;
  - next performance lever should focus on Gemma3 tail MLP / GeGLU
    projection path before broad scheduler or engine work。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXX — W2 source checkpoint: disable graph capture for syncing diagnostics

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `LlamaFamilyRuntimeEnv::graph_capture_allowed()` 新增 single decode
    graph capture 保护;
  - `LlamaBatchedRuntimeConfig::graph_capture_allowed()` 新增 batched/unified
    graph capture 保护;
  - single decode CUDA graph 在 decode/prefill/layer profile、nan trace、
    op dump、layer dump 任一诊断开关启用时不再 capture;
  - batched/unified graph 在 `decode_op_profile`、`unified_profile`、
    `batched_trace` 任一同步型诊断开关启用时不再 capture。
- Why:
  - LXIX 证明 profile+batched graph 组合仍能在 capture window 内走到
    普通 `B::sync`;
  - profiler/trace 本质上依赖同步计时边界,不应和 CUDA graph capture 同时
    生效;
  - 正常产品 graph path 保持可用,diagnostic profile path 改为 eager,
    用于稳定采集热点分解。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llama_batched_runtime_config -- --nocapture`
    PASS,2/2;
  - `cargo test -p ferrum-models llama_family_runtime_env -- --nocapture`
    PASS,2/2;
  - `cargo test -p ferrum-models batched_graph_capture_is_disabled_by_syncing_diagnostics -- --nocapture`
    PASS,1/1;
  - `git diff --check` PASS。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXIX — W2 CUDA checkpoint: capture-lifecycle retry still fails under profile

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_prefill_profile_capture_retry_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 typed prefill profile capture-lifecycle retry`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,hard cap 25min;
  - stop condition:build/server/chat/bench/profile 首败,或最小 c16
    profile 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` c16 single cell;
  - performance command:diagnostic-only c16 ShareGPT single-run profile,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`a9d8b439097f89011fb02dc78e1046ddb07d73e6`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `e111e6ec9653fd141ad5eb8ed504f18997a7d29244dbbe685be9955d2277a350`;
  - `ferrum serve` ready after poll `58`,chat smoke passed with content
    `5` and `completion_tokens=3`;
  - typed profiler config came from config file for
    `FERRUM_BATCH_DECODE_PROF`,`FERRUM_BATCH_PREFILL_PROF`,
    `FERRUM_DECODE_OP_PROFILE`,`FERRUM_PREFILL_OP_PROFILE`,
    `FERRUM_NEXT_BATCH_PROF`,`FERRUM_UNIFIED_POST_PROF`;
  - c16 profile emitted usable partial profile lines before failure:
    `prefill-profile` lines `121`,`batched-op-profile` lines `3`,
    `unified-prof` lines `7`;
  - failure remained:
    `CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")`;
  - run was stopped at first failure;`run_profile.rc=143`,
    `bench_sharegpt_c16.rc=143`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - LXVIII fixed one capture-end condition, but this retry proves the
    profile+batched-graph path can still reach a normal `B::sync` while CUDA
    capture is in flight;
  - this is still a correctness blocker for profiler-backed performance
    diagnosis, even though product chat smoke passed;
  - partial profiles still narrow the hot region: c16 prefill sample shows
    `tail_mlp` about `37-41ms/62 layers`, and batched decode sample shows
    `tail_mlp` about `13.5ms` per m=10 decode iteration。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXVIII — W2 source checkpoint: end capture based on active capture state

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - legacy single decode、unified graph、batched graph 的 capture-end 条件
    从 `should_capture && !*_graph_failed` 改成
    `should_capture && B::graph_capture_in_flight(&ctx)`;
  - 如果 begin capture 成功,即使后续 failure flag 已被置位,也会尝试
    `end_graph_capture` 收口,避免 capture window 泄漏到后续普通
    `B::sync`。
- Why:
  - LXVII retry 说明第一刀 guarded profiler sync 还不够;
  - 失败形态更像 capture window 没有正常结束,导致后续正常同步命中
    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`;
  - 以 backend active-capture state 作为 end 条件比 failure flag 更直接。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-models llama_batched_runtime_config -- --nocapture`
    PASS,2/2。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXVII — W2 CUDA checkpoint: first graph-safe profiler fix still leaves capture open

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_prefill_profile_retry_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 typed prefill profile graph-safe retry`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:8-15min,hard cap 25min;
  - stop condition:启动/SSH/source sync/build/server/smoke/bench 首败,或
    retry artifact 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16 ShareGPT single-run profile,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`f352ff3f6b596418659ff5912995d07f5e9fc1fc`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `f3742953afabfa1ad3ac99d58978d0508825085b4a2b706e4f7e508a1a1944f7`;
  - `ferrum serve` ready after poll `59`,chat smoke passed with content
    `5` and `completion_tokens=3`;
  - retry still hit
    `CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")`;
  - bench was manually stopped after panic;`run_profile.rc=143`,
    `bench_sharegpt_c16.rc=143`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - first source fix guarded profile sync calls, but did not fully fix capture
    lifecycle;
  - evidence points to a capture window remaining open when a later normal
    `B::sync` runs;
  - next source fix should end graph capture whenever
    `B::graph_capture_in_flight(&ctx)` is true, instead of relying on
    `!*_graph_failed` flags。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXVI — W2 source checkpoint: make batched op profiler graph-capture safe

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `Backend` 新增 `graph_capture_in_flight(&Context) -> bool`,默认返回
    `false`;
  - CUDA backend 返回 `CudaState.capture_in_flight`;
  - Llama/Gemma batched/unified decode op profiler 在 graph-capture window
    自动跳过 `B::sync` 型计时边界,避免
    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`。
- Why:
  - LXV 失败证据显示 typed `decode_op_profile/prefill_op_profile` 与
    batched CUDA graph capture 同时开启时,profiler 的同步型计时触发
    CUDA stream-capture 不支持错误;
  - 这不是默认产品 graph 路径的性能问题,而是诊断观测路径的正确性问题;
  - 修复保持 graph 开启,只让 graph capture iteration 不执行 graph-unsafe
    sync profiler。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-kernels cpu_timer -- --nocapture` PASS,2/2;
  - `cargo test -p ferrum-models llama_batched_runtime_config -- --nocapture`
    PASS,2/2。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXV — W2 CUDA checkpoint: typed prefill profile exposes graph-capture profiler correctness bug

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_prefill_profile_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 typed-config ShareGPT prefill profile`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:启动/SSH/source sync/build/server/smoke/bench 首败,或
    profile artifact 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16 ShareGPT
    `bench-serve --fail-on-error --seed 9271 --dataset sharegpt --num-prompts 16 --n-repeats 1`,
    no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`353c1eb2521118c37342def279fe3c22b2715e20`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `138ff2e0000947dafb7299b74d96397fa300b5eb61cc168305fae160d06deeff`;
  - profiler flags came from config-file entries:
    `FERRUM_BATCH_DECODE_PROF`,`FERRUM_BATCH_PREFILL_PROF`,
    `FERRUM_DECODE_OP_PROFILE`,`FERRUM_PREFILL_OP_PROFILE`,
    `FERRUM_NEXT_BATCH_PROF`,`FERRUM_UNIFIED_POST_PROF`;
  - `ferrum serve` ready,chat smoke passed with content `5` and
    `completion_tokens=3`;
  - bench was manually stopped after server panic;`run_profile.rc=143`,
    `bench_sharegpt_c16.rc=143`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Failure:
  - server panic:
    `CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")`;
  - panic occurred after `batched-op-profile` emitted under
    `FERRUM_DECODE_OP_PROFILE` while batched graph was active;
  - this is a correctness issue in diagnostic/profile instrumentation under
    CUDA graph capture,not a release performance data point。
- Interpretation:
  - no W2 performance conclusion from this run;
  - profiler path must become graph-safe before using typed prefill/decode op
    profile to locate the remaining ShareGPT c16 TTFT/decode gap;
  - next source fix should avoid stream synchronization inside CUDA graph
    capture,or automatically disable the graph-unsafe op profiler when graph
    capture is active。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXIV — W2 source checkpoint: expose prefill profiler knobs through runtime config

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - config-file `runtime` 现在可显式开启:
    - `batch_prefill_prof` -> `FERRUM_BATCH_PREFILL_PROF`;
    - `decode_op_profile` -> `FERRUM_DECODE_OP_PROFILE`;
    - `prefill_op_profile` -> `FERRUM_PREFILL_OP_PROFILE`;
  - `LlmExecutorRuntimeEnv` 不再从 `std::env::vars()` 直接冻结 profiler
    flags,改为读取 composition root 安装的 `RuntimeConfigSnapshot`;
  - profiler flags 仍保持 presence-flag 语义:只有 config `true` 才
    materialize runtime entry,`false`/unset 不会生成 `FERRUM_*_PROF=0`。
- Why:
  - 上一轮 typed decode integration profile 已排除 scheduler/postprocess/普通
    host loop gap 为主因;
  - W2 c16 剩余风险集中在 Gemma3 模型侧 decode/prefill,尤其 ShareGPT TTFT
    和 batched prefill 是否仍有 fallback;
  - 下一次 CUDA 只需用 typed config 打开 prefill/decode profiler 做最小
    ShareGPT c16 诊断,不需要隐藏 env 组合或重复 full sweep。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-cli runtime_cli_config -- --nocapture` PASS,3/3;
  - `cargo test -p ferrum-models llm_executor_runtime_env -- --nocapture`
    PASS,3/3;
  - `git diff --check` PASS。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXIII — W2 CUDA checkpoint: typed-config decode integration profile points back to model-side decode

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_decode_profile_2026-06-16/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 typed-config decode integration profile`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:启动/SSH/source sync/build/server/smoke/bench 首败,或 profile
    artifact 完成后复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16
    `bench-serve --fail-on-error --seed 9271 --num-prompts 16 --random-input-len 32 --random-output-len 32`,
    `n_repeats=1`,no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`4fea56ec79d0c8a9edcf99dd90b3889d422869e9`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `23f04a49e361c836ab6a8afb125d68e771df361219013bee0d32ecf630a2559d`;
  - profiler flags came from typed config-file entries:
    `FERRUM_BATCH_DECODE_PROF`,`FERRUM_NEXT_BATCH_PROF`,
    `FERRUM_UNIFIED_POST_PROF`;
  - `ferrum serve` ready after poll `62`,selected graph mode
    `legacy_batched_decode_graph`;
  - chat smoke content `5`,usage completion_tokens `3`;
  - bench rc `0`,`completed_per_run=[16]`,`errored_per_run=[0]`,
    `bad_output_per_run=[0]`,`zero_output_tokens_per_run=[0]`,
    `output_token_count_source=usage`;
  - c16 diagnostic throughput `380.492 tok/s`,TTFT p50 `587.854ms`,
    TPOT p50 `23.822ms`;
  - `output_tokens_per_request`:
    `[[32,32,32,28,32,32,32,31,32,32,32,32,32,30,32,32]]`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Profile result:
  - full `decode=16` iterations:`27`;
  - mean full decode iteration total:`23679.2us`;
  - mean model time:`23311.3us`,mean model share:`98.44%`;
  - mean decode postprocess time:`347.9us`,mean postprocess share:`1.47%`;
  - `bg-loop-gap` mostly single-digit microseconds,scheduler/process loop gap is
    not the main bottleneck。
- Interpretation:
  - 这次最小 profile 没有发现新的正确性问题;
  - engine scheduler、postprocess、streaming、普通 host loop gap 都不足以解释
    W2 c16 与 vLLM 的差距;
  - 继续扫 admission/开关的收益很低,下一步应集中在 Gemma3 模型侧 decode,
    尤其 tail/Marlin/projection behavior 与 weight-residency 类问题。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-16 LXII — W2 source checkpoint: expose decode profiler knobs through runtime config

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - config-file `runtime` 现在可显式开启:
    - `batch_decode_prof` → `FERRUM_BATCH_DECODE_PROF`;
    - `next_batch_prof` → `FERRUM_NEXT_BATCH_PROF`;
    - `rbd_prof` → `FERRUM_RBD_PROF`;
    - `unified_post_prof` → `FERRUM_UNIFIED_POST_PROF`;
  - 这些 profiler 是 presence flags,所以 config `true` 才 materialize runtime
    entry;`false`/unset 都不会生成 `FERRUM_*_PROF=0`,避免误触发。
- Why:
  - 最新 c16 诊断排除了稳定 `m=15` underfill 与输出早停主因;
  - 下一步需要量化 engine/process_batch/model/decode_post/scheduler 开销;
  - 通过 typed config-file 暴露现有 profiler 后,CUDA profile 诊断不需要隐藏
    env 组合,更符合 release-grade 证据路径。
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-cli runtime_cli_config -- --nocapture` PASS,3/3。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 LXI — W2 CUDA checkpoint: c16 output-token/batch-shape diagnostic narrows remaining bottleneck

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_output_token_c16_diag_2026-06-15/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 c16 output-token/batch-shape diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:启动/SSH/build/server/smoke/bench 首败,或 c16 诊断完成后
    复制 artifact 并停机;
  - correctness gate:CUDA release build,`ferrum serve` ready,chat smoke,
    `bench-serve --fail-on-error` 零错误;
  - performance command:diagnostic-only c16
    `bench-serve --fail-on-error --seed 9271 --num-prompts 16 --random-input-len 32 --random-output-len 32`,
    `n_repeats=1`,no CI/no `--require-ci`。
- Execution evidence:
  - remote HEAD:`25c32dac9305eb62acd733bd491b2d1294a3ba64`;
  - non-artifact source status clean;
  - CUDA release build rc `0`,binary SHA256:
    `a7f561a5f49a6858e8a63040a595143395f369ab7da1a5983d94927469b3861a`;
  - chat smoke content `5`,usage completion_tokens `3`;
  - bench rc `0`,`completed=[16]`,`errored=[0]`,
    `output_token_count_source=usage`;
  - new `output_tokens_per_request`:
    `[[32,32,32,28,32,32,32,32,32,30,32,32,32,32,32,32]]`;
  - c16 diagnostic throughput `363.087 tok/s`,TTFT p50 `496.040ms`,
    TPOT p50 `27.226ms`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - short outputs are only `6` tokens below the `16*32=512` cap,so early stop
    cannot explain the remaining W2 performance gap;
  - first batched decode trace saw `m=13`,but the main captured/replayed graph
    was `m=16 m_padded=16`;the previous `m=15` graph capture is not a stable
    sustained-decode bottleneck;
  - drain shape captured at `m=3 m_padded=4`,consistent with the two shorter
    requests finishing before the rest;
  - remaining high-probability bottleneck is still decode integration/host
    scheduling + tail MLP/Marlin projection/weight-residency,not graph replay
    absence,not stable c16 underfill,not output-token early stop。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 LX — W2 source checkpoint: bench-serve records per-request output token counts

- 本轮没有启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - `BenchReport` 增加 `output_tokens_per_request`;
  - `compute_metrics` 从每个 `RunRecord` 写出每次 repeat 的 measured request
    output token 数,失败请求保留记录值(通常为 0);
  - `bench-serve` 失败策略测试现在断言 fail-on-error 路径仍会先写出该字段。
- Why:
  - 最新 batched graph artifact 已证明 graph replay 真实发生,但主 replay shape
    是 `m=15` 而不是满 `m=16`;
  - 旧 report 只有总吞吐,缺少每请求 output token 分布,无法快速区分
    早停/尾部 drain 与持续调度不满批;
  - 下一次 CUDA 最小验证可直接用该字段判断 c16 诊断是否被某些请求短输出
    拉低,避免再做无信息 full sweep。
- Local validation:
  - `cargo test -p ferrum-bench-core` PASS,45/45;
  - `cargo test -p ferrum-cli fail_on_error_still_writes_json_report -- --nocapture`
    PASS;
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS。
- Bottleneck status:
  - 已有证据仍指向 decode integration + tail MLP/Marlin projection + batch
    cadence 的组合问题;
  - workspace zero 和直接 dense vLLM Marlin kernel swap 均不是当前第一杠杆;
  - W2 仍不是 release-grade。

## 2026-06-15 LIX — W2 CUDA checkpoint: batched graph replay confirmed, not sufficient for 80%

- 本轮源码 checkpoint:
  `22f92677 test(cuda): log batched graph replay progress`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_replay_observability_2026-06-15/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 batched graph replay observability smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:10-15min,hard cap 20min;
  - stop condition:收集 capture/replay 日志 + serve correctness smoke,或首败并
    复制 artifact,或 20min hard cap;
  - correctness gate:远端 HEAD `22f92677`,CUDA release build rc `0`,
    `ferrum serve` ready,OpenAI chat smoke 输出 `5`,日志无实际 panic/CUDA
    error/`<unk>`/`[PAD]`;
  - performance command:diagnostic-only c16
    `bench-serve --fail-on-error --seed 9271 --num-prompts 16 --random-output-len 32`,
    `n_repeats=1`,不是 release 性能证据。
- 源码改动:
  - capture 成功时输出 `[batched-graph-capture]`;
  - post-capture replay 与 pure replay 成功时输出低频
    `[batched-graph-replay]`;
  - post-capture replay 现在也计入 `BATCHED_GRAPH_REPLAY_COUNT`;
  - replay 日志按 1/2/4/8/... 次数打印,避免长 bench 刷屏。
- 本地验证:
  - `cargo test -p ferrum-models batched_decode_graph --lib -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`。
- CUDA result:
  - remote HEAD:`22f92677b34bab932407215fcb8c11dd0b372faf`;
  - binary SHA256:
    `f6d6828290c330749f1523c191c3e4034759f97d7c53e0ad4948d7e786995b1b`;
  - `serve_selected_graph_mode=legacy_batched_decode_graph`,
    `serve_selected_max_sequences=16`;
  - chat smoke content `5`;
  - c16 diagnostic bench rc `0`,`completed=16`,`errored=0`,
    output throughput mean `348.0 tok/s`;
  - replay evidence in server log:
    - capture `m=15 m_padded=16 device_shadow=true`;
    - post-capture replay count `1`;
    - pure replay counts `2,4,8,16` on the same `m_padded=16` graph;
    - an additional drain-shape capture `m=7 m_padded=8 device_shadow=true`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - 现在已证明 Gemma3 device-shadow product path 真实进入 legacy batched
    CUDA graph capture/replay;
  - 单次 c16 throughput 没有稳定越过 80%,说明 graph launch overhead 不是
    剩余唯一主瓶颈;
  - 观测到 replay 主 shape 是 `m=15` 而非满 `m=16`,且 drain shape 会额外
    capture;下一步应回到 W2-P2 的剩余两条:batch cadence/TTFT 与
    sustained decode tail MLP/Marlin 投影成本。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LVIII — W2 CUDA checkpoint: batched graph product path enabled, c16 diagnostic improves but remains below 80%

- 本轮源码 checkpoint:
  `2b3b5891 perf(cuda): expose batched decode graph policy`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_batched_graph_product_smoke_2026-06-15/`。
- GPU 执行合同:
  - lane:`W2 Gemma3 batched decode graph product correctness smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:15-25min,hard cap 30min;
  - stop condition:`ferrum run` + `ferrum serve` smoke 通过、首个明确失败并收集
    artifact、或 30min hard cap;
  - correctness gate:CUDA release build,`ferrum run` known-answer,
    `ferrum serve` OpenAI chat smoke,effective config/decision trace 证明
    `decode_graph_policy=legacy_batched_decode_graph`;
  - performance command:diagnostic-only c16 `bench-serve --fail-on-error --seed 9271`,
    `n_repeats=1`,不是 release 性能证据。
- 源码/产品路径:
  - `ferrum run` 与 `ferrum serve` 均新增公开
    `--batched-graph/--disable-batched-graph`;
  - config file 支持 `runtime.batched_graph`;
  - auto-config 新增 `decode_graph_policy` decision,并 materialize
    `FERRUM_BATCHED_GRAPH`;
  - legacy batched decode graph 仍禁止 host residual shadow,但允许 Gemma3
    device residual shadow,并使用独立 graph key namespace。
- 本地验证:
  - `cargo test -p ferrum-types batched_graph -- --nocapture`;
  - `cargo test -p ferrum-cli batched_graph -- --nocapture`;
  - `cargo test -p ferrum-cli runtime_cli_config_emits_config_file_source_entries -- --nocapture`;
  - `cargo test -p ferrum-cli batched_graph_cli_override_records_flag_state -- --nocapture`;
  - `cargo test -p ferrum-models batched_decode_graph --lib -- --nocapture`;
  - `cargo test -p ferrum-types m3_preset_selects_current_safe_fast_path_without_fa2 -- --nocapture`;
  - `cargo fmt --all -- --check`;
  - `git diff --check`。
- CUDA product smoke:
  - remote HEAD:`2b3b5891ff94d6a4d793bb39bd6cab148af49588`;
  - binary SHA256:
    `c31d8b4af03f4669f7fac4fc49035adff97ca4d680d80775703aff99474b3d33`;
  - model path:
    `/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2`;
  - `ferrum run` rc `0`,content `5`,
    `selected_graph_mode=legacy_batched_decode_graph`;
  - `ferrum serve` ready,OpenAI chat content `5`,
    `selected_graph_mode=legacy_batched_decode_graph`,
    `selected_max_sequences=16`;
  - c16 diagnostic bench rc `0`,`16 completed / 0 errored`,
    `372.3 tok/s`;
  - log scan found no actual panic/CUDA error/illegal address/OOM/`<unk>`/`[PAD]`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Interpretation:
  - 产品入口已能用公开 typed path 打开 batched decode graph policy;
  - c16 diagnostic 从 prompt-token admission checkpoint 的 `344.7 tok/s`
    提升到 `372.3 tok/s`,但与同机 vLLM c16 `518.8 tok/s` 相比仍约
    `71.8%`,没有达到 80%;
  - 当前 artifact 证明 policy 生效和产品 correctness smoke 通过,但没有明确
    记录 graph replay counter;下一步应补最小 replay 可观测性,再用同一 c16
    diagnostic 验证 replay 是否确实发生。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LVII — W2 native checkpoint: Gemma3 shadow graph native probe PASS

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma3_shadow_graph_native_probe_2026-06-15/`。
- 源码 checkpoints:
  - `23d8569b test(cuda): add gemma3 shadow graph probe`;
  - `e927e4c1 test(cuda): make shadow graph probe relocatable`;
  - `c46d9540 test(cuda): keep shadow graph probe finite`。
- GPU 执行合同:
  - lane:`W2 Gemma3 shadow graph native CUDA probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD `0.425/hr`;
  - expected runtime/cost:5-10min,hard cap 20min;
  - stop condition:Vast start/SSH/CUDA/nvcc compile/probe 首败,或 probe
    artifact 复制完成后立即停机;
  - correctness gate:`nvcc` 编译 rc `0`,probe rc `0`,checksum finite,
    stdout 含 `VERDICT: Gemma3 shadow graph native CUDA probe complete`;
  - performance command:`bash scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`,
    diagnostic-only,不是 release 性能证据。
- 执行环境:
  - remote HEAD:`c46d95408d12c8c1e177145f7c4c217a34080e62`;
  - GPU:`NVIDIA GeForce RTX 4090`,24564 MiB,driver `565.77`;
  - CUDA compiler:`Build cuda_12.4.r12.4/compiler.34097967_0`;
  - Vast shutdown verified:`cur_state=stopped actual_status=exited`。
- Native probe result:
  - simulated shape:Gemma3-style device F32 residual shadow decode,
    `62` layers,`batch=16`,`hidden=5376`,`498` kernel launches/step;
  - eager ordered state upload:`1.143 ms/step`;
  - eager pre-sync state upload:`1.137 ms/step`;
  - graph ordered state upload:`0.565 ms/step`,`2.02x` vs eager ordered;
  - graph pre-sync state upload:`0.568 ms/step`,`2.00x` vs eager pre-sync;
  - checksum16:`127.94618988`;
  - verdict line present。
- Interpretation:
  - native CUDA 层面已证明 Gemma3 device-shadow-like decode step 可以稳定
    graph capture/replay,并且 launch-overhead headroom 约 `2x`;
  - 这不是产品性能修复;当前 Ferrum product path 仍在 host/device residual
    shadow 路径禁用 legacy batched graph;
  - 下一步应做窄产品改动:shadow-safe graph eligibility/guard,然后先跑
    `ferrum run` + `ferrum serve` correctness smoke,再做 c16 diagnostic。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LVI — W2 source checkpoint: add Gemma3 shadow graph native probe

- 本轮没有启动 GPU,不产生性能结论;这是针对 decode integration/graph
  瓶颈的源码和诊断工具 checkpoint。
- 实质定位:
  - Gemma3 sandwich-norm CUDA 正确性路径依赖 device F32 residual shadow;
  - legacy batched decode graph 入口当前要求
    `FERRUM_BATCHED_GRAPH && !host_residual_shadow && !device_residual_shadow`,
    因此目标 Gemma3 路径即使设置 `FERRUM_BATCHED_GRAPH=1` 也不会进入
    batched graph replay;
  - 这解释了为什么继续扫 graph env 开关不能作为有效性能定位手段。
- 新增最小 native CUDA probe:
  - `scripts/microbenches/gemma3_shadow_graph_bench.cu`;
  - `scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`;
  - probe 模拟 Gemma3-style `62` 层、`batch=16`、device F32 residual
    shadow update 的 decode step,比较 eager launch 与 graph replay;
  - 不加载模型,不跑产品 entrypoint,不改变默认产品路径。
- 本地验证:
  - `git diff --check -- scripts/microbenches/gemma3_shadow_graph_bench.cu scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh scripts/microbenches/README.md` PASS;
  - `bash -n scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`
    PASS;
  - 本地无 `nvcc`,CUDA compile/run 待在已有 1x4090 cache-retained instance 上
    执行单 probe。
- 下一步:
  - 只启动已有 4090 实例跑该 native probe,保存 `VERDICT: Gemma3 shadow graph
    native CUDA probe complete`、stdout、GPU metadata 和 shutdown 记录;
  - 若 native graph replay 稳定且有足够 headroom,再设计产品侧 shadow-safe
    graph eligibility/guard;若不稳定,转向 tail MLP launch/copy fusion,不做
    产品默认 graph 改动。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LV — W2 CUDA checkpoint: prompt-token admission 默认路径正确但不是主瓶颈

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prompt_token_admission_c16_ab_2026-06-15/`。
- 源码 checkpoint:
  `2f732131 perf(scheduler): default to prompt-token admission`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA prompt-token admission c16 A/B diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:start/SSH/CUDA/clean worktree/build/product smoke/c16
    bench 首败,或 artifact 复制完成后停机;
  - correctness gate:`ferrum run` known-answer smoke plus `ferrum serve`
    chat smoke with usage and zero benchmark errors;
  - performance command:diagnostic-only `bench-serve` ShareGPT c16,
    `--num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`。
- 执行环境:
  - remote clean worktree HEAD
    `2f73213181475ba4bdff3e907e45182c24981a0e`;
  - remote `git status --short` 为空;
  - binary SHA256
    `551f83921ea1fb6eb0cfb75170fc2325e31d887530ba084ab72ef77b238ebaf0`;
  - Vast shutdown verified:`cur_state=stopped`,
    `actual_status=exited`。
- Correctness:
  - CUDA release build rc `0`;
  - `ferrum run` rc `0`,answer `5`,n_tokens `3`;
  - `ferrum serve` chat smoke rc `0`,content `5`,usage
    `prompt_tokens=23`,`completion_tokens=3`;
  - `bench-serve --fail-on-error` rc `0`,c16 completed `16/16`,
    `0` errored,`0` bad_output,output token count source `usage`;
  - both run and serve decision traces selected
    `scheduler_admission_policy=prompt_token_estimate` from default。
- Performance diagnostic:
  - prior same-host Ferrum c16 natural ShareGPT baseline:
    `340.003 tok/s`,p50 TTFT `887.683ms`,p50 TPOT `32.817ms`;
  - new default prompt-token admission c16:
    `344.714 tok/s`,p50 TTFT `931.776ms`,p50 TPOT `31.592ms`;
  - delta vs Ferrum baseline:`+1.39%`,single-run diagnostic only;
  - ratio vs same-host vLLM c16 `518.796 tok/s`:`66.4%`,
    still well below W2 80% line。
- Conclusion:
  - 默认 prompt-token admission 是正确的产品默认修复,并由 decision trace
    证明已生效;
  - 但它不是当前 c16 性能主瓶颈;不要继续围绕 admission/env flip
    做 sweep;
  - 下一步回到已经定位的 decode/Marlin tail MLP,尤其是 gate_up/down
    投影与每步 integration 开销。
- Artifact note:
  - remote driver 的最后 summary helper 对 single-c `bench-serve` JSON schema
    假设错误,benchmark 完成后触发 `KeyError: 0`;
  - build/run/smoke/bench rc 均已是 `0`,`summary.json` 由 bench JSON
    重新生成,`run.status` 记录为
    `PASS_CORE_WITH_POSTPROCESS_WARNING`。
- Release status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍不是 release-grade。

## 2026-06-15 LIV — W2 source checkpoint: default scheduler admission uses prompt-token metadata

- 本轮没有启动 GPU,不产生性能结论;这是针对 c16 TTFT/prefill/scheduler
  半边差距的源码 checkpoint。
- 源码改动:
  - `SchedulerConfig::default()` 现在默认
    `prompt_token_estimate=true`;
  - `#[serde(default)]` 改为显式 true default,避免配置文件省略字段时回到
    bool 的 false;
  - auto-config 默认 decision trace 现在选择
    `scheduler_admission_policy=prompt_token_estimate`;
  - 显式 `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE=0` 仍可回到
    `continuous_default`,且 trace 保留该 source key;
  - scheduler 测试保留显式禁用路径,验证旧 admission 行为仍可复现。
- 为什么这个 checkpoint 有意义:
  - 产品 `ferrum run` / `ferrum serve` 已在提交 scheduler 前写入真实
    `ferrum_prompt_tokens`;
  - 旧默认 false 会让初始 prefill admission 用 `prefill_chunk_size=512`
    粗估,在 `max_num_batched_tokens=2048` 下 c16 短 prompt 首批只能进
    约 4 个请求;
  - 默认 true 后,短 prompt admission 会按真实 prompt token 计入预算,目标是
    降低 c16 TTFT/prefill 排队部分,不是直接改 decode kernel。
- 本地验证:
  - `cargo fmt --all -- --check`;
  - `cargo test -p ferrum-types scheduler_config -- --nocapture`;
  - `cargo test -p ferrum-types scheduler_prompt_token_estimate -- --nocapture`;
  - `cargo test -p ferrum-scheduler prompt_token_metadata -- --nocapture`;
  - `cargo test -p ferrum-types engine_config_default_sane -- --nocapture`。
- 下一步:
  - 只做最小 CUDA c16 A/B/product smoke,验证默认 prompt-token admission
    是否实际降低 TTFT/提升 c16 ratio;
  - 若没有明显收益,继续回到已定位的 decode/Marlin tail MLP 半边。
- Correctness/performance status:
  - 本地配置和 scheduler 单元测试通过,没有发现新的 correctness blocker;
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade。

## 2026-06-15 LIII — W2 bottleneck synthesis: c16 gap is split between TTFT/prefill and sustained decode

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_bottleneck_synthesis_2026-06-15/summary.md`。
- 本轮没有新增 GPU run;这是对 2026-06-15 已有 c16 evidence 的综合。
- 使用的主要证据:
  - Ferrum/vLLM ShareGPT c16/c32 baseline:
    `w2_vllm_sharegpt_baseline_probe_2026-06-15`;
  - Ferrum c16 `[batched-op-profile]`:
    `w2_tail_profile_buckets_2026-06-15`;
  - Ferrum Marlin projection split:
    `w2_marlin_projection_profile_2026-06-15`;
  - native Ferrum/vLLM Marlin weight-cycle:
    `w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15`;
  - product Marlin shape trace:
    `w2_marlin_shape_trace_probe_2026-06-15`。
- Same-hardware c16 端到端差距:
  - Ferrum:`340.003 tok/s`,`5.328 req/s`;
  - vLLM:`518.796 tok/s`,`8.106 req/s`;
  - ratio:`65.5%`,仍低于 W2 80% 目标。
- latency 拆分:
  - Ferrum p50 TTFT `887.683ms`,vLLM p50 TTFT `411.903ms`,
    差 `+475.780ms`;
  - Ferrum p50 TPOT `32.817ms`,vLLM p50 TPOT `24.789ms`,
    差 `+8.027ms/token`;
  - 以约 63 个 inter-token gap 估算,TPOT 差距约 `506ms`;
  - c16 batch wall-time 差约 `16/5.328 - 16/8.106 = 1.03s`;
  - TTFT + TPOT 基本解释该差距,说明剩余性能问题不是一个单点内核。
- batch/cadence 结论:
  - 既有 decode batch stats:c16 段 `calls=391`,
    `total_items=5334`,`avg_m=13.642`,`max_m=16`;
  - `w2_tail_profile_buckets` 中 batch `m=16` 有 `118` 行;
  - 因此当前主问题不是“c16 batch 完全没有形成”。
- decode breakdown:
  - `w2_tail_profile_buckets` batch `m=16` mean decode step
    `28.037ms`;
  - `tail_mlp` `13.744ms` (`49.0%`),`matmul` `6.971ms`
    (`24.9%`),`attention` `2.406ms` (`8.6%`),
    `unwrapped` `0.649ms` (`2.3%`);
  - `w2_marlin_projection_profile` batch `m=16` with Marlin profiling:
    Marlin kernels `16.548ms` (`55.0%`),其中 `gate_up`
    `8.728ms`,`down` `4.352ms`,`qkv` `2.132ms`,`o_proj`
    `1.336ms`。
- Current bottleneck statement:
  - c16 batching 大体有效,但平均 batch 仍低于 16,尾段会 drain;
  - 端到端差距约一半来自 TTFT/prefill/scheduling,一半来自持续
    TPOT/decode;
  - decode 内部主要是 Gemma3 tail MLP / Marlin projection 时间;
  - native Ferrum/vLLM Marlin weight-cycle 已经排除“直接换 dense Marlin
    单核即可解决”的主假设。
- 下一步:
  - first-token 侧:查 Ferrum c16 ShareGPT 是否串行/弱批处理 prefill,
    以及 chunked/batched prefill 是否能降低 TTFT;
  - decode 侧:继续从 native CUDA 最小 probe 入手,针对 Gemma3
    tail MLP 的 gate_up/down 调度、activation、residual/norm 边界找可
    落地优化。
- Correctness/performance status:
  - 没有新的 correctness blocker;
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade。

## 2026-06-15 LII — W2 CUDA checkpoint: product Marlin shape trace wired and single-request decode is m=1

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_shape_trace_probe_2026-06-15/`。
- 源码 checkpoint:
  `b3403dd5 test(cuda): add marlin shape trace probe`。
- GPU 执行合同:
  - lane:`W2 Marlin shape-trace compile/product smoke`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:10-20min,hard cap 30min;
  - stop condition:start/SSH/CUDA/source sync/CUDA feature test/build/server/chat
    smoke 首败,或 trace 输出 + artifact 复制后停机;
  - correctness gate:CUDA/marlin feature compile retry rc `0`,release build rc
    `0`,`ferrum serve` ready,non-stream chat smoke 有内容和 usage;
  - performance command:无 release perf;本轮只做 diagnostic product trace。
- 执行结果:
  - remote HEAD:`b3403dd5394bb044690c918535a71ccc202cd3e7`;
  - release CUDA build rc `0`,binary SHA256
    `730df7d84ede559b7ace54abcf1a6c16a3a81e55113789c5e3bc37c9f3844b8f`;
  - `run.status=PASS`;
  - chat smoke 返回 `5`,usage 为 `prompt_tokens=23`,
    `completion_tokens=3`;
  - artifact 已复制回本地;`vast_shutdown/stopped.json` 记录
    `cur_state=stopped actual_status=exited`。
- shape trace 结果:
  - `shape_trace_lines=256`,全部可解析;
  - calls `0..247`:prefill,`m=23`,62 层 × 4 个 Marlin projection;
  - calls `248..255`:decode,`m=1`,trace cap 前 2 层 × 4 个 projection;
  - projection shapes:
    `qkv n=8192 k=5376`,`o n=5376 k=4096`,
    `gate_up n=43008 k=5376`,`down n=5376 k=21504`。
- Interpretation:
  - trace 已接入真实 `ferrum serve` 产品路径,不是 standalone
    microbench;
  - 单请求 decode 明确是 `m=1`;这本身符合预期,但还不能解释 c16
    端到端差距;
  - 结合上一轮 native Ferrum/vLLM Marlin weight-cycle probe,当前主假设应
    继续收敛到 c16 decode integration/cadence:并发 decode 是否稳定形成
    `m≈16`,是否存在 scheduler gap、非 Marlin op 或 per-step sync。
- Correctness/performance status:
  - 该窄 smoke 没有发现新的 correctness blocker;
  - `decode_op_profile.log` 本轮为空,不能据此做非 Marlin 时间结论;
  - 没有生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade;
  - 最新可引用性能仍是 Ferrum c16 `341.24 tok/s` vs vLLM c16
    `518.80 tok/s`,约 `65.8%`。

## 2026-06-15 LI — W2 source checkpoint: add Marlin shape trace for decode integration probe

- 本轮代码改动:
  `crates/ferrum-kernels/src/backend/cuda/marlin.rs`。
- 背景:
  - dense Marlin kernel swap 已被 native hot/weight-cycle A/B 排除;
  - W2 c16 剩余差距应继续追 decode integration/scheduling,而不是继续
    跑 full sweep 或 blind kernel flip;
  - 已有 `FERRUM_MARLIN_PROFILE=1` 能聚合 projection bucket 时间,但缺少
    每次 Marlin dispatch 的 shape/pointer/label trace。
- 改动:
  - 新增默认关闭的 `FERRUM_MARLIN_TRACE_SHAPES=1`;
  - 新增 `FERRUM_MARLIN_TRACE_SHAPES_MAX=<N>`,默认最多 `256` 条;
  - trace 行记录 call id,当前 CUDA alloc label,bucket,m/n/k,group size,
    qweight/scales/workspace len,以及 A/B/C/scales/workspace device pointer;
  - 仅在显式 env 打开时输出,不改变默认产品路径。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `cargo test -p ferrum-kernels cuda_marlin_runtime_config -- --nocapture`
    PASS,但默认 feature 下未命中 CUDA-gated Marlin tests;
  - `cargo test -p ferrum-kernels --features cuda,marlin cuda_marlin_runtime_config -- --nocapture`
    在本机失败,原因是本地无 `nvcc`/`nvidia-smi`,不是代码测试失败。
- 下一步:
  - 在 1x4090 上做最小 CUDA 编译验证,并用
    `FERRUM_MARLIN_TRACE_SHAPES=1 FERRUM_MARLIN_TRACE_SHAPES_MAX=128`
    跑短 product diagnostic,确认 trace 输出可解析且不引入 correctness
    回归。

## 2026-06-15 L — W2 CUDA checkpoint: vLLM dense Marlin weight-cycle 也落在同一瓶颈带

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15/`。
- 源码 checkpoint:
  `2b6e1922 test(cuda): add vllm marlin weight cycle probe`。
- GPU 执行合同:
  - lane:`W2 dense vLLM Marlin weight-cycle native probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:5-12min,hard cap 20min;
  - stop condition:start/SSH/CUDA/source sync/compile/probe 首败,或 verdict
    后复制 artifact 并停机;
  - correctness gate:native compile success + probe rc `0` + verdict line;
  - performance command:diagnostic-only
    `bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`,
    只比较 hot vs weight-cycle,不产生 release 性能声明。
- 执行结果:
  - remote HEAD:`2b6e192205f01ef9106a7c12dbce38198c2584a3`;
  - probe rc `0`,`run.status=PASS`;
  - stdout 含
    `VERDICT: dense vLLM Marlin native CUDA probe complete`;
  - artifact 复制回本地后已停机;`vast_shutdown/cleanup_check.txt`
    记录 `cur_state=stopped actual_status=exited`。
- m=16 A/B/weight-cycle 关键数据:
  - qkv: Ferrum hot `17.005us`,Ferrum weight-cycle `30.278us`,
    vLLM hot `18.315us`,vLLM weight-cycle `30.950us`;
  - gate_up: Ferrum hot `133.715us`,Ferrum weight-cycle `133.985us`,
    vLLM hot `136.988us`,vLLM weight-cycle `137.524us`;
  - down: Ferrum hot `30.356us`,Ferrum weight-cycle `68.651us`,
    vLLM hot `36.027us`,vLLM weight-cycle `69.268us`。
- Interpretation:
  - down projection 的 vLLM dense Marlin hot 看起来快,但一旦模拟 product
    跨层权重轮换,它同样落到 `~69us`,与 Ferrum default weight-cycle
    只差约 `1%`;
  - gate_up 在 hot/weight-cycle 下基本不变,说明它是 compute-bound 大投影;
  - 因此当前剩余 c16 端到端差距不应继续押注 dense Marlin kernel swap;
  - 下一步应转向 decode integration/scheduling:每 token launch 数、非
    Marlin 时间、batch cadence、host/device sync、以及 vLLM 是否在请求调度
    或 decode loop 层面减少了空转/间隙。
- Correctness/performance status:
  - 当前窄产品 correctness 仍无新增 blocker;
  - Ferrum c16 最新 product diagnostic 仍约 `341.24 tok/s`,vLLM same-hardware
    c16 `518.80 tok/s`,约 `65.8%`;
  - W2 仍无 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。

## 2026-06-15 XLIX — W2 source checkpoint: extend vLLM dense Marlin probe with weight-cycle mode

- 本轮代码改动:
  `scripts/microbenches/dense_vllm_marlin_gemma3_perf.cu`。
- 背景:
  - XLVIII 的 native vLLM dense Marlin probe 已经排除 gate_up 上的
    kernel-swap 主假设;
  - 但 down projection 的 product profile 接近 Ferrum default
    weight-cycle,而 XLVIII 只测了 vLLM hot repeated weight;
  - 因此不能只用 vLLM hot down `36.277us` 与 Ferrum weight-cycle
    `68.651us` 做最终判断。
- 改动:
  - 为每个 Gemma3 qkv/gate_up/down shape 和 `m=16/23/32` 同时输出
    `hot` 与 `weight_cycle`;
  - `weight_cycle` 使用 8 组 synthetic qweight/scales/workspace 轮换,
    模拟 product decode 跨层权重切换;
  - 仍是 native CUDA probe,不加载模型,不改变 product dense GPTQ routing。
- 验证状态:
  - 本 checkpoint 未启动 GPU,不产生性能结论;
  - 下一次 paid CUDA 只跑同一 build script,用于确认 vLLM dense Marlin
    down projection 在 weight-cycle 下是否仍明显优于 Ferrum default。

## 2026-06-15 XLVIII — W2 CUDA checkpoint: native vLLM dense Marlin A/B 排除 kernel-swap 主假设

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_native_probe_retry_2026-06-15/`。
- 源码 checkpoint:
  - `09734267 test(cuda): add dense vllm marlin gemma probe`;
  - `5ce9299e fix(cuda): enable dense vllm marlin probe selector`。
- GPU 执行合同:
  - lane:`W2 dense vLLM Marlin native same-shape A/B probe`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - expected runtime/cost:5-12min,hard cap 20min;
  - stop condition:start/SSH/CUDA/source sync/compile/probe 首败,或
    `VERDICT: dense vLLM Marlin native CUDA probe complete` 后复制 artifact
    并停机;
  - correctness gate:native compile success + probe rc `0` + verdict line;
  - performance command:diagnostic-only
    `bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`,
    不产生 release 性能声明。
- 执行结果:
  - first attempt artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_native_probe_2026-06-15/`;
  - first attempt rc `134`,原因是 build script 的 temporary `perl`
    selector include 替换没有命中,临时 `marlin.cu` 仍保留注释 include;
  - 修正脚本后 retry rc `0`,`run.status=PASS`;
  - stdout 含
    `VERDICT: dense vLLM Marlin native CUDA probe complete`;
  - artifact 复制回本地后已停机;`vast_shutdown/cleanup_check.txt`
    记录 `cur_state=stopped actual_status=exited`。
- m=16 native A/B 关键数据:
  - qkv: Ferrum hot `17.005us`,Ferrum weight-cycle `30.278us`,
    vLLM dense Marlin `18.354us`;
  - gate_up: Ferrum hot `133.715us`,Ferrum weight-cycle `133.985us`,
    vLLM dense Marlin `136.581us`;
  - down: Ferrum hot `30.356us`,Ferrum weight-cycle `68.651us`,
    vLLM dense Marlin `36.277us`。
- Product profile 对照:
  - c16 profile batch `16`;
  - product per-layer gate_up kernel 约 `140.77us`,与 native Ferrum/vLLM
    gate_up 基本同量级;
  - product per-layer down kernel 约 `70.20us`,接近 Ferrum weight-cycle
    `68.651us`,而不是 Ferrum hot `30.356us`。
- Interpretation:
  - “直接换 vLLM dense Marlin kernel 能补齐 c16 14 个百分点差距”这个
    主假设已被本轮 native A/B 排除;
  - 当前更可信的瓶颈方向是 product 集成侧的 weight residency/cache-cycle,
    down projection 的权重访问状态,以及 decode launch/host scheduling
    组合开销;
  - 下一步不跑 full sweep,应做最小产品/原生关联 probe:在 decode loop
    里记录 projection 权重地址/调用顺序/stream sync 与 down projection
    cache-cycle 状态,确认为什么 product down 落在 weight-cycle 而不是 hot
    kernel 轨道。
- Release-grade status:
  - 这是诊断证据,不是 release gate;
  - W2 仍无 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。

## 2026-06-15 XLVII — W2 source checkpoint: add dense vLLM Marlin native A/B probe

- 本轮代码改动:
  - `scripts/microbenches/dense_vllm_marlin_gemma3_perf.cu`;
  - `scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`;
  - `scripts/microbenches/README.md`。
- 目的:
  - 当前 c16 产品路径正确性窄门干净,但 unified prefill 与 vLLM paged
    attention 诊断均没有提升吞吐;
  - 已有 op profile 将主要热区压缩到 dense GPTQ decode MLP/Marlin,
    尤其 gate_up/down;
  - 下一步用 native CUDA 同形状 A/B 直接测 vendored vLLM dense GPTQ-Marlin
    在 Gemma3 qkv/gate_up/down 形状上的核耗时,避免继续启动整套
    Cargo/product gate 做盲目验证。
- Probe 设计:
  - 直接调用 vendored `ferrum_marlin_mm_f16_u4b8` C ABI;
  - 覆盖 Gemma3-27B GPTQ 关键 dense 形状:
    `qkv k=5376 n=8192`,`gate_up k=5376 n=43008`,
    `down k=21504 n=5376`;
  - 覆盖 decode 常见 m 值 `16/23/32`;
  - companion build script 只在 `/tmp` 临时副本中打开 minimal
    `kernel_selector.h`,不改变产品 dense GPTQ routing。
- 验证状态:
  - 本 checkpoint 未启动 GPU,不产生性能结论;
  - 下一次 paid CUDA 只需运行
    `bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh`
    并保存 `VERDICT: dense vLLM Marlin native CUDA probe complete`;
  - 如果同形状 vLLM dense Marlin 明显快于当前 `dense_marlin_gemma3_perf`
    的 hot/weight-cycle 数据,再考虑产品侧 selector/核接入;否则继续转向
    host scheduling/weight residency/launch overhead。

## 2026-06-15 XLVI — W2 CUDA checkpoint: Gemma3 unified prefill c16 诊断通过但性能仍约 65.8% vLLM

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_prefill_c16_diag_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090,
  约 USD 0.425/hr。artifact 复制回本地后已停机;`vast_shutdown/cleanup_check.txt`
  记录 `cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA unified-prefill c16 diagnostic`;
  - expected runtime/cost:10-20min,hard cap 30min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/source sync/server readiness 首败,
    chat smoke 首败,`bench-serve` 首败,malformed/missing usage 首败,或
    c16 diagnostic 完成后复制 artifact 并停机;
  - correctness gate:`ferrum serve` readiness + non-stream chat smoke +
    `bench-serve --fail-on-error`;
  - performance command:diagnostic-only
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --concurrency-sweep 16 --num-prompts 16`,
    no `--require-ci`,不产出 release 性能结论。
- Build/runtime evidence:
  - remote HEAD:`d5f82822b56527b47a9d3884639fe737cbb37570`;
  - CUDA release build rc `0`;
  - binary SHA256:
    `4ebf50b5c64a5f72d929e1aeaefde61b0f1bb9ec6fed9dd0d84596f5b803be89`;
  - server log contains
    `Gemma3 family: legacy batched_decode=true varlen_unified=true`;
  - remote git status contains only historical artifact deletion noise from
    rsync artifact exclusion;no non-artifact source dirty rows were present.
- Correctness:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - non-stream chat smoke returned content `"5\n"`,finish_reason `stop`,
    completion_tokens `3`,bad_output `false`;
  - c16 ShareGPT diagnostic reported `completed_per_run=[16]`,
    `errored_per_run=[0]`,`bad_output_per_run=[0]`,
    `zero_output_tokens_per_run=[0]`,`malformed_stream_per_run=[0]`,
    `missing_done_per_run=[0]`,`duplicate_done_per_run=[0]`,
    `panic_per_run=[0]`,`http_500_per_run=[0]`;
  - `output_token_count_source="usage"`;
  - log/bad-marker scan found no product correctness blocker;this checkpoint
    does not replace the full W2 L0-L5 release gate.
- Performance diagnostic:
  - Ferrum unified-prefill c16: `341.2397 tok/s`;
  - same-hardware vLLM c16 baseline:
    `518.7960 tok/s`;
  - Ferrum/vLLM ratio: `65.78%`;
  - previous Ferrum c16 baseline was `340.0029 tok/s`,so enabling
    Gemma3 unified path did not materially close the gap.
- Interpretation:
  - current known correctness state is clean for this narrow product-path
    diagnostic;
  - the main W2 blocker is now performance:still about `14.2` percentage
    points below the 80% vLLM target at c16;
  - next high-value step is not another full sweep;use a minimal targeted
    profile/native CUDA probe to locate the remaining bottleneck, likely in
    decode/projection/attention scheduling rather than the unified-prefill
    enablement guard itself.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XLV — W2 CUDA checkpoint: Gemma3 unified tail 产品 smoke 通过

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma_unified_tail_cuda_smoke_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;`vast_shutdown/cleanup_check.txt` 记录
  `cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA unified-tail product correctness smoke`;
  - expected runtime/cost:10-25min,hard cap 35min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/CUDA/source sync/build 首败、`ferrum run`
    首败、`ferrum serve` smoke 首败、乱码/缺 usage/stream `[DONE]` 等首败,
    或 smoke PASS 后复制 artifact 并停机;
  - correctness gate:CUDA release build + `ferrum run` + `scripts/model_coverage_smoke.sh`;
  - performance command:none;本轮不产出性能结论。
- Build:
  - remote HEAD:`ab0dc99cdc71345e236513dbe0300ce52b162416`;
  - `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
    PASS,`cargo_build.rc=0`;
  - binary SHA256:
    `4ebf50b5c64a5f72d929e1aeaefde61b0f1bb9ec6fed9dd0d84596f5b803be89`;
  - remote git status has only expected artifact noise from rsync artifact exclusion。
- Product correctness:
  - `run.status=PASS`;
  - `ferrum run gemma3:27b-gptq --backend cuda ...` rc `0`;
  - run validation:assistant content `"5"`,finish_reason `stop`,n_tokens `3`,
    bad_output `false`;
  - run stderr contains
    `Gemma3 family: legacy batched_decode=true varlen_unified=true`;
  - `scripts/model_coverage_smoke.sh gemma3:27b-gptq --port 8491 --kv-capacity 2560 --max-seqs 2`
    rc `0`;
  - serve stdout contains `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`;
  - serve log contains
    `Gemma3 family: legacy batched_decode=true varlen_unified=true`;
  - validation scan did not find `panic`,`CUDA_ERROR`,`<unk>`,or `[PAD]` markers.
- Interpretation:
  - Gemma3 unified tail source change is now product-smoke validated on CUDA
    for both `ferrum run` and `ferrum serve`;
  - this is still a smoke checkpoint,not full L0-L5 release evidence and not
    a performance claim;
  - next high-value correctness/perf checkpoint is a tiny c16 ShareGPT
    diagnostic with `--fail-on-error`,checking that fresh prefill no longer
    falls back to serial per-item profiles and measuring the new gap vs vLLM。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XLIV — W2 source checkpoint: Gemma3 unified tail 接入 sandwich/F32 residual 语义

- 本轮没有启动 GPU;在上一个 native CUDA window probe 通过后,继续补
  Gemma3 unified prefill 的源码正确性前置件。
- Source change:
  - unified mixed-batch scratch 增加 `unified_sandwich_tmp`,
    `unified_residual_f32_shadow`, `unified_sandwich_branch_f32`;
  - `ensure_unified_scratch` 会在 sandwich-norm family 且 backend 支持
    device-side F32 residual shadow 时分配 unified F32 residual/branch buffer;
  - `unified_forward_internal` 在 embedding 后把 activation residual 写入
    F32 residual shadow,并在返回前恢复 shadow scratch;
  - `unified_forward_layer` 现在对 sandwich layer:
    - input RMSNorm 从 F32 residual shadow 读;
    - post-attn path 执行 `rms_norm(o_proj_out, post_attn_ln_w)` 后加到
      F32 residual,再对 F32 residual 做 `post_ln_w` pre-MLP norm;
    - gated activation 按 `Activation::GeluTanh` 走 GeGLU,否则走 SwiGLU;
    - post-ffn path 执行 `rms_norm(mlp_out, post_ffn_ln_w)` 后加到
      F32 residual;
    - final norm 从 F32 residual shadow 读;
  - `unified_varlen_qkv_unsupported_reason` 现在把 Gemma3 unified prereq
    明确为:backend varlen QKV + local/global layer pattern +
    device-side F32 residual shadow。
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -p ferrum-models` PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_requires_gemma_sandwich_prerequisites --lib`
    PASS;
  - `cargo test -p ferrum-models llama_attention_semantics_cover_qk_mode_and_layer_windows --lib`
    PASS;
  - `git diff --check` PASS。
- Correctness status:
  - this is not release evidence and not a product correctness PASS;
  - source now has the missing Gemma3 unified tail semantics,so the next
    checkpoint must run CUDA product smoke (`ferrum run` and `ferrum serve`)
    before trusting the newly-enabled unified path;
  - if CUDA smoke fails, stop at the failing artifact and do minimal triage,
    not a full perf sweep。
- Performance status:
  - no performance command in this checkpoint;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains functional / known-gap,not release-grade。
- Next step:
  - reuse the cached 1x4090 only once for a minimal CUDA correctness smoke:
    CUDA build, `ferrum run` Paris/multi-turn smoke, `ferrum serve` non-stream
    and streaming smoke, then a small c16 diagnostic only if correctness is clean。

## 2026-06-15 XLIII — W2 native checkpoint: paged varlen sliding-window CUDA probe 通过

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_varlen_window_native_probe_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090,
  `nvidia/cuda:12.4.0-devel-ubuntu22.04`,driver `565.77`,CUDA compiler
  `12.4`。验证结束后已复制 artifact 并停机;`vast_shutdown/cleanup_check.txt`
  记录 `cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA paged-varlen sliding-window native correctness probe`
    plus `W2 CUDA product build smoke after varlen-window ABI change`;
  - expected runtime/cost:5-15min native probe,额外 3-8min build smoke,
    hard cap 25min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/CUDA/source sync/compile/probe 首败,
    CUDA release build 首败,或 PASS 后复制 artifact 并停机;
  - correctness command:
    `bash scripts/microbenches/build_and_run_paged_varlen_window_correctness.sh`;
  - product build smoke:
    `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`;
  - performance command: none;本轮不产出性能结论。
- Source/tooling change:
  - 新增 `scripts/microbenches/paged_varlen_window_correctness.cu`;
  - 新增 `scripts/microbenches/build_and_run_paged_varlen_window_correctness.sh`;
  - README 记录该 probe 用于 Gemma3 unified prefill 前验证 paged-varlen
    one-pass 和 split-K sliding-window 语义。
- Native CUDA result:
  - `paged_varlen_window_correctness.rc` = `0`;
  - stdout contains `VERDICT: paged varlen window correctness PASS`;
  - `sliding_window=0`: one-pass/split-K max abs err both `0.00003045`;
  - `sliding_window=3`: one-pass/split-K max abs err both `0.00002996`;
  - full causal vs window CPU reference semantic delta `0.02593978`,证明
    probe 实际覆盖了不同语义,不是只测等价路径。
- CUDA product build smoke:
  - `cargo_cuda_build.rc` = `0`;
  - build finished release profile in `3m 46s`;
  - binary SHA256:
    `3d5ce5a0dd931a88f26d2d3e23c27805deeb91c77f804e140dff3304e7afcc4a`;
  - first build attempt failed with rc `127` only because non-login SSH PATH
    lacked `/root/.cargo/bin`;rerun after `source /root/.cargo/env` passed.
- Evidence caveat:
  - remote HEAD was `aa741f90b5a135d974fbb824283252a6b66d5857`;
  - remote git status is dirty because this checkpoint's new microbench files
    were not yet committed at sync time and rsync deliberately excluded local
    historical artifact directories;this is diagnostic correctness evidence,
    not a performance claim.
- Correctness/performance status:
  - the varlen-window CUDA ABI/semantics checkpoint is now validated by native
    CUDA and product CUDA build smoke;
  - Gemma3 unified prefill guard is still not relaxed;W2 still requires
    sandwich norm/GeGLU unified-tail semantics and product `ferrum run`/`serve`
    smoke before performance evidence;
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains functional / known-gap,not release-grade。
- Next step:
  - implement the next Gemma3 unified prefill correctness piece: device-side
    unified tail support for Gemma3 sandwich post-attn/post-ffn norms and GeGLU,
    keeping the guard closed until Paris/chat smoke passes。

## 2026-06-15 XLII — W2 source checkpoint: varlen attention 接入 per-layer sliding-window 语义

- 本轮没有启动 GPU;按最小源码验证推进,避免重复开关机器、重装环境和完整 sweep。
- Source change:
  - `BackendPagedKv::paged_varlen_attention` 增加显式 `sliding_window`
    参数;`0` 表示 full causal,非 0 表示只看最近窗口;
  - CUDA `paged_varlen_attn_f16` 和 split-K phase1 都按
    `attend_start..valid_kv_len` 计算 QK、softmax 和 V 汇聚;
  - Llama/Gemma attention 语义抽成 `llama_qk_mode` 和
    `llama_layer_attention_schedule`,统一 single path、legacy batched
    decode 和 unified varlen path 的 q/k norm、interleaved rope、本地/全局
    layer window 决策;
  - Qwen3 MoE unified callsite 显式传 `0`,保持既有 full causal 行为。
- Why:
  - 上一个 checkpoint 证明 Gemma3 W2 的 c16 fresh prefill cohort 仍走
    serial fallback;直接放开 `supports_varlen_qkv` guard 会绕过 Gemma3
    sandwich norm、GeGLU、local/global window 和 dual RoPE correctness 保护;
  - 本 checkpoint 先补齐 backend varlen attention 的 window 语义,为后续
    Gemma3 unified prefill correctness 做前置切点。
- Validation:
  - `cargo fmt --all` PASS;
  - `cargo test -p ferrum-models llama_attention_semantics_cover_qk_mode_and_layer_windows --lib`
    PASS;
  - `cargo test -p ferrum-models unified_varlen_qkv_rejects_sandwich_configs_even_when_backend_supports_varlen --lib`
    PASS;
  - `cargo check -p ferrum-kernels -p ferrum-models` PASS;
  - `git diff --check` PASS;
  - local Mac 没有 `nvcc`,因此 CUDA kernel 编译/运行验证尚未在本 checkpoint
    本地完成。
- Correctness status:
  - 没有放开 Gemma3 unified varlen guard,所以默认产品路径不应因本轮改变而
    启用未验证 Gemma3 unified prefill;
  - 目前没有新增已知产品 correctness blocker,但 CUDA 编译和产品 smoke 仍是
    下一 checkpoint 必须验证项。
- Performance status:
  - 本轮不是性能证据;Ferrum W2 Gemma3 27B GPTQ 仍约为同机 vLLM c16 baseline
    的 `~65%`,距离 `80%` 目标约 `14-15` percentage points;
  - W2 仍没有 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Next step:
  - 先做 native CUDA 最小编译/attention-window probe 或产品 CUDA smoke,确认
    新 kernel 参数和窗口语义没有破坏现有 path;
  - 再补 Gemma3 unified tail 的 sandwich norm/GeGLU 语义,通过 Paris/chat smoke
    后才考虑放开 Gemma3 unified prefill guard。

## 2026-06-15 XLI — W2 native checkpoint: dense Marlin 多权重轮转 probe 完成

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_weight_cycle_probe_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;`vast_shutdown/cleanup_check.txt` 记录
  `cur_state=stopped actual_status=exited`。
- Source/tooling change:
  - `scripts/microbenches/dense_marlin_gemma3_perf.cu` 增加
    `weight_cycle_kernel` 与 `weight_cycle_ws_plus_kernel` 输出行;
  - 关键 auto-tile `m=16/23/32` 会轮转 `8` 份独立 synthetic
    qweight/scales/workspace,用于判断 Gemma3 gate/up/down Marlin shape 的
    产品侧表现更接近热缓存、冷缓存还是多权重轮转;
  - 更新 `build_and_run_dense_marlin_gemma3_perf.sh` 与 microbench README。
- Why:
  - 既有 native probe 的 repeated-hot timing 对小投影过于乐观,而真实 W2
    Gemma3 27B decode 会在 62 层和多投影权重间切换;
  - 本改动提供比完整 Ferrum release build/bench 更便宜的 native CUDA
    最小验证入口,用于选择下一步是否值得改 Marlin tile/grid/repack path。
- Validation:
  - `git diff --check -- scripts/microbenches/dense_marlin_gemma3_perf.cu scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh scripts/microbenches/README.md`
    PASS;
  - `bash -n scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
    PASS;
  - remote native CUDA command:
    `timeout 1800 bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`;
  - `probe/dense_marlin_gemma3_perf.rc` = `0`;
  - `probe/dense_marlin_gemma3_perf.stdout` contains
    `VERDICT: dense Marlin native CUDA probe complete`;
  - remote HEAD `82fb3272451083bc7f79c7aeca4610793ef579aa`;
  - remote git status is dirty only as diagnostic evidence because rsync excluded
    local artifact directories and the remote checkout reports 812 artifact deletes.
- Key auto-tile results, kernel-only us:
  - `gate_up`: hot/weight-cycle/cold at m16 `133.715/133.985/176.844`,
    m23 `137.396/136.962/181.151`,m32 `138.025/138.386/181.254`;
  - `down`: hot/weight-cycle/cold at m16 `30.356/68.651/93.560`,
    m23 `52.520/72.835/98.045`,m32 `53.017/73.524/99.045`;
  - `qkv` and `o_proj` also show cache sensitivity on small m, but they are not the
    dominant W2 decode bucket.
- Interpretation:
  - `gate_up` does not move under 8-weight cycling,so the large Gemma3 dense GPTQ
    gate/up bucket is compute/path-bound rather than a simple weight-cache artifact;
  - `down` is materially cache sensitive,so product-side timing should be compared
    against weight-cycle/cold-cache brackets rather than repeated-hot microbench rows;
  - this narrows the next useful native lever to shape-specific gate/up Marlin path
    review, while the higher-level W2 gap still also includes Gemma3 serial prefill
    fallback from the previous checkpoint.
- Next step:
  - 不跑新的完整 sweep;先做 `gate_up` Marlin shape-specific source review或更小的
    native CUDA A/B;
  - 真正改产品路径后再按顺序跑 Paris/chat smoke、产品 `ferrum run`/`serve` quick
    regression,最后才进入 W2 release-grade gate。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XL — W2 source trace:Gemma3 prefill 是 serial fallback,不是 unified cohort prefill

- 本轮没有再开 GPU;基于已提交 artifact
  `w2_prefill_bucket_profile_2026-06-15/` 和源码追踪完成定位。
- Evidence:
  - artifact 中 26 个 ShareGPT 测量请求对应 26 条
    `[prefill-profile] tokens=122` row;如果 cohort prefill 真实合批,不应表现为
    每个请求一条 `prefill_internal` profile;
  - `LlamaFamilyModel::load` 对 sandwich-norm families(Gemma 3)设置
    `supports_varlen_qkv = B::supports_varlen_qkv() && !cfg.sandwich_norms`,
    并记录 `Gemma3 family: legacy batched_decode=... varlen_unified=false`;
  - `LlamaFamilyModel::unified_forward` 在 `!supports_varlen_qkv` 时返回
    `Unsupported`,原因是 Gemma3 unified/paged attention 尚未支持 per-layer
    local/global window semantics;
  - `LlmExecutor::batch_prefill` 在 `model.unified_forward(...)` 返回
    `Unsupported` 后,在同一个 model lock 下循环 `model.prefill(cid,toks)`,
    即 serial prefill fallback。
- Interpretation:
  - c16 TTFT p50 约 `0.9s` 与单个 122-token prefill 约 `84ms`
    一致:初始 cohort 大概率被 serial full-prefill 队列放大;
  - 这比单个 Marlin GEMM 微优化更能解释 Ferrum vs vLLM 的 14-15 percentage
    points 缺口;
  - `tail_mlp`/`flash_attn` 仍是单次 prefill 的局部热区,但高杠杆方向是
    Gemma3 unified/batched prefill 或等价 cohort prefill path,而不是继续扫
    `FERRUM_MARLIN_SKIP_WS_ZERO` 这类历史上已显示收益接近 0 的开关。
- Next implementation direction:
  - 不直接打开现有 `supports_varlen_qkv` guard;该 guard 保护 Gemma3 的 sandwich
    norms、dual rope、per-layer local/global attention correctness;
  - 先做一个小型设计/代码切点:复用 Llama unified scaffolding,给 Gemma3 增加
    explicit unsupported reason/profile 或受测的 narrow cohort-prefill path;
  - 任何真正启用 Gemma3 unified prefill 的改动都必须先过 Paris/chat smoke,
    再跑 native CUDA c16 最小验证。
- Release-grade status:
  - 没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXIX — W2 prefill bucket profile: profiler 修复有效,瓶颈落在 tail MLP + prefill attention

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_bucket_profile_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast shutdown poll 3 记录 `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA prefill profile buckets validation`;
  - expected runtime/cost:15-35min,hard cap 45min,约 USD 0.425/hr;
  - stop condition:instance start/SSH/CUDA/source sync/build 首败、serve readiness
    首败、chat smoke 首败、c16 diagnostic 完成,或 45min cap;
  - correctness gate:CUDA release build,serve readiness,non-stream chat smoke,
    then `bench-serve --fail-on-error`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`;
  - profile scope:`FERRUM_PREFILL_OP_PROFILE=1`,只作诊断。
- Build/evidence:
  - remote source commit:`3c407faf25eed833fbb785057c6a7f39d0578e5b`;
  - binary SHA256:
    `5873e674ed0aff9a301af532e0f38c898595d02fd12441125240cf24abea9403`;
  - `cargo_build.rc=0`,release build 用时 `3m27s`;
  - remote full `git status --short` 不干净,原因是本轮为最小源码同步,
    有意没有同步历史 docs artifact;构建相关源码已同步到上述 commit。
- Correctness/perf diagnostic:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c16 with profiler:`321.551 tok/s`,request throughput `5.024 req/s`,
    TTFT p50 `925.570ms`,TTFT p95 `1516.331ms`,TPOT p50 `35.035ms`,
    ITL p50 `26.578ms`,ITL p99 `295.802ms`,
    `output_token_count_source=usage`;
  - 因为本轮开启 profiler,吞吐只用于诊断,不作为正式性能 claim。
- Prefill bucket evidence:
  - captured 27 prefill profiles;其中 ShareGPT `tokens=122` 有 26 个;
  - ShareGPT prefill total mean `83.577ms`,range `83-92ms`;
  - `tail_mlp` mean `37.654ms`,约 `45.1%`;
  - `flash_attn` mean `30.192ms`,约 `36.1%`;
  - ordinary `matmuls` mean `6.000ms`,约 `7.2%`;
  - `qk_norm_rope` mean `1.000ms`,约 `1.2%`;
  - `tail_mlp` 内部:`tail_gate_up` mean `23.115ms`,
    `tail_down` mean `13.115ms`。
- Interpretation:
  - profiler source fix 有效;prefill bucket 不再为空;
  - prefill/TTFT 侧的主热区是 Gemma GPTQ MLP tail 和 prefill attention,
    不是普通 QKV/O matmul;
  - typed vLLM paged attention 已经验证无 end-to-end 收益,下一步应优先看
    Gemma GPTQ MLP tail 的 prefill/decode 共享实现与 kernel launch/Marlin 调度,
    再看 prefill attention 是否仍有可替换路径。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXVIII — W2 prefill/TTFT first profile:正确性干净,发现 prefill profiler bucket 缺口并修复源码

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_ttft_profile_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast shutdown poll 1 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA prefill/TTFT profile diagnostic`;
  - expected runtime/cost:8-20min,hard cap 30min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/server readiness 首败、chat smoke 首败、
    c16 ShareGPT diagnostic 完成并复制 artifact,或 30min cap;
  - correctness gate:`ferrum serve` readiness plus non-stream chat smoke before
    `bench-serve`;`bench-serve` 使用 `--fail-on-error`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`;
  - profile scope:server 使用 `FERRUM_PREFILL_OP_PROFILE=1`,只作诊断。
- Correctness/perf diagnostic:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c16 throughput `340.882 tok/s`,TTFT p50 `889.558ms`,
    TTFT p95 `1452.948ms`,TPOT p50 `32.804ms`,ITL p50 `24.678ms`,
    ITL p99 `281.837ms`;
  - ratio vs clean vLLM c16 baseline:`340.882 / 518.796 = 0.657`,
    距 80% 线约 `14.3` percentage points。
- Prefill observation:
  - captured 27 `[prefill-profile]` total rows;
  - smoke prefill:`tokens=23,total=29ms`;
  - ShareGPT prefills:`tokens=122,total=80-88ms`,median `80ms`;
  - bucket breakdown 为空。
- Source fix in this checkpoint:
  - prefill profile now enables ordinary op timers for `tokens > 1`
    (`decode_op_profile || prefill_op_profile`);
  - prefill start clears stale op/tail counters before timing;
  - prefill summary now drains and prints tail buckets:
    `tail_norm`,`tail_gate_up`,`tail_act`,`tail_down`,`tail_mlp`,
    `tail_resid`;
  - default product path is unchanged; this only affects diagnostic runs with
    `FERRUM_PREFILL_OP_PROFILE=1`。
- Local validation after source fix:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-models/src/models/llama_family.rs
    docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_ttft_profile_2026-06-15`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models
    llama_family_runtime_env_parses_startup_knobs --lib` PASS。
- Next step:
  - rerun the same native CUDA prefill profile after rebuilding on
    `40826362`,then use bucket evidence to choose the next small source lever;
  - do not treat this first profile as release evidence or as proof W2 is
    release-grade。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXVII — W2 typed vLLM paged-attn diagnostic:正确性通过,性能无改善

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_paged_attn_gemma_diag_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast shutdown poll 2 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA typed vLLM paged-attn ShareGPT diagnostic`;
  - expected runtime/cost:10-25min,hard cap 35min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/server readiness 首败、typed attention-selection
    assertion 首败、chat smoke 首败、c16/c32 ShareGPT diagnostic 完成并复制
    artifact,或 35min cap;
  - correctness gate:artifact-local `ferrum.toml` 设置
    `runtime.use_vllm_paged_attn=true` 后 `ferrum serve` readiness、
    decision-trace assertion、non-stream chat smoke,之后才跑 `bench-serve`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`。
- Correctness:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - decision trace 明确 `attention_prefill_mixed_backend=vllm_paged_varlen`,
    source `config_file`,key `FERRUM_USE_VLLM_PAGED_ATTN`;
  - decode selected `vllm_paged_attn_v1_short`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c32 diagnostic cell:`16 completed / 0 errored`,bad_output `[0]`;
  - 本轮没有发现新的 Ferrum product correctness 问题。
- Diagnostic bench(非 release evidence,N=1,无 CI):
  - c16:`340.443 tok/s`,TTFT p50 `890.332ms`,TTFT p95 `1453.858ms`;
  - c32 diagnostic cell:`341.419 tok/s`,TTFT p50 `889.279ms`,
    TTFT p95 `1440.689ms`。
- Diagnostic ratio vs clean vLLM ShareGPT baseline:
  - c16:`340.443 / 518.796 = 0.656`,差距约 `34.4%`,
    距 80% 线约 `14.4` percentage points;
  - c32 diagnostic cell:`341.419 / 524.128 = 0.651`,差距约 `34.9%`,
    距 80% 线约 `14.9` percentage points。
- Interpretation:
  - typed config VPA 路径已经生效,不是 hidden env 组合;
  - 相比 no-VPA Ferrum ShareGPT,c16 只 `+0.13%`,c32 diagnostic cell
    `-0.25%`,没有性能收益;
  - VPA 不是当前缺失的 14-15 percentage points 的主要杠杆,下一步继续回到
    已定位的 Gemma tail/GEMM 热点,尤其 `tail_gate_up` 与 `tail_down`。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXVI — W2 typed prefix-cache ShareGPT diagnostic: 正确性干净,0 hit,性能无改善

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefix_cache_sharegpt_diag_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast poll 1 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA typed prefix-cache ShareGPT diagnostic`;
  - expected runtime/cost:10-25min,hard cap 35min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/server readiness 首败、chat smoke 首败、
    c16/c32 ShareGPT diagnostic 完成并复制 artifact,或 35min cap;
  - correctness gate:`ferrum serve --enable-prefix-cache` readiness plus
    non-stream chat smoke before `bench-serve`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`。
- Correctness:
  - `run.status=PASS`,`bench-serve.rc=0`;
  - chat smoke content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`;
  - c32:`16 completed / 0 errored`,bad_output `[0]`;
  - 本轮没有发现新的 Ferrum product correctness 问题。
- Prefix-cache observation:
  - decision trace 显示 `prefix_cache_policy=prefix_cache_enabled`,
    source `cli`;
  - health after 显示 `enabled=true`,`hits=0`,`misses=53`,
    `saved_prefill_tokens=0`,`entries=0`;
  - 结论:typed product prefix cache 已打开,但没有命中这个 repeated-prompt
    ShareGPT 场景。
- Diagnostic bench(非 release evidence,N=1,无 CI):
  - c16:`340.618 tok/s`,TTFT p50 `889.469ms`,TTFT p95 `1453.788ms`;
  - c32:`342.350 tok/s`,TTFT p50 `887.527ms`,TTFT p95 `1438.820ms`。
- Diagnostic ratio vs clean vLLM ShareGPT baseline:
  - c16:`340.618 / 518.796 = 0.657`,差距约 `34.3%`,
    距 80% 线约 `14.3` percentage points;
  - c32:`342.350 / 524.128 = 0.653`,差距约 `34.7%`,
    距 80% 线约 `14.7` percentage points。
- Interpretation:
  - prefix cache 不是当前差距的现成解;c16 相比 no-prefix 只 `+0.18%`,
    c32 只 `+0.02%`;
  - 下一步若追 prefix-cache,应查 why zero hits/entries,不要重复 full sweep;
  - 否则继续回到已定位的 Gemma tail/GEMM 热点,尤其 `tail_gate_up` 与
    `tail_down`。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXV — W2 vLLM natural ShareGPT baseline clean;Ferrum c16/c32 约 65%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_sharegpt_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast poll 2 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA vLLM ShareGPT baseline-cleanliness probe`;
  - expected runtime/cost:20-45min,hard cap 60min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/vLLM server 首败、baseline smoke 首败、
    c16/c32 ShareGPT diagnostic 完成并复制 artifact,或 60min cap;
  - correctness gate:vLLM `/v1/models` + 非流式 chat smoke;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32,
    `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`。
- vLLM baseline diagnostic:
  - engine:`vllm 0.10.1.1`,GPTQ Marlin,同一 HF/safetensors GPTQ model;
  - `/v1/models` ready,poll 33;
  - chat smoke rc=0,content `5`,usage `completion_tokens=3`;
  - c16:`16 completed / 0 errored`,bad_output `[0]`,
    `518.796 tok/s`;
  - c32:`16 completed / 0 errored`,bad_output `[0]`,
    `524.128 tok/s`。
- Ferrum same-dataset no-profile compare:
  - binary SHA256:
    `3e28a4cf37b2e25b127dbd591e8891b141863d8082d1757486707c785e6869ce`;
  - `ferrum serve --model gemma3:27b-gptq --kv-capacity 512 --max-num-seqs 16`
    ready,poll 29;
  - c16:`16 completed / 0 errored`,bad_output `[0]`,
    `340.003 tok/s`;
  - c32:`16 completed / 0 errored`,bad_output `[0]`,
    `342.284 tok/s`。
- Diagnostic ratio:
  - c16:`340.003 / 518.796 = 0.655`,差距约 `34.5%`;
  - c32:`342.284 / 524.128 = 0.653`,差距约 `34.7%`;
  - 距 80% release-grade 线仍差约 `14.5` percentage points。
- Interpretation:
  - 本轮没有发现新的 Ferrum product correctness 问题;
  - 之前 vLLM random-prompt c16 baseline 自身 invalid-UTF8,不能做 final
    baseline;本轮 natural ShareGPT vLLM c16/c32 是 zero-error,说明 baseline
    路线可以考虑改成自然 prompt 数据集并正式 N>=3 化;
  - 但按这个 clean baseline,Ferrum c16/c32 仍显著低于 80%,W2-P2 仍要继续
    优先优化 Gemma tail/GEMM 路径。
- Release-grade status:
  - 本轮是 N=1 diagnostic,没有 `--require-ci`,没有
    `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXIV — W2 fused sandwich residual-add: native CUDA minimal validation PASS,收益有限

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_fused_sandwich_residual_2026-06-15/`。
- 复用 Vast/cache-retained native CUDA instance `40826362`,1x RTX 4090。验证结束后
  已复制 artifact 并停机;Vast API poll 5 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA fused sandwich residual-add minimal validation`;
  - expected runtime/cost:15-35min,hard cap 45min,约 USD 0.425/hr;
  - stop condition:启动/SSH/CUDA/source sync/build 首败、`ferrum run`
    correctness 首败、serve/bench diagnostic 完成并复制 artifact,或 45min cap;
  - correctness gate:CUDA release build + product `ferrum run`,通过后才进入
    `serve/bench-serve --fail-on-error`;
  - performance command:diagnostic-only natural ASCII ShareGPT c16/c32 小样本,
    `n_repeats=1`,seed 9271,`FERRUM_DECODE_OP_PROFILE=1`。
- 远端源码/构建:
  - git HEAD:`4eeea0ba76a2ac8b0671941bcba0d66020c31ed4`;
  - 本轮 rsync 为减少付费 GPU 空转排除了历史 artifacts,远端 git status
    因旧 artifact 缺失显示 docs 删除;本轮只作为 diagnostic evidence;
  - `CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum
    --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source` PASS;
  - binary SHA256:
    `3e28a4cf37b2e25b127dbd591e8891b141863d8082d1757486707c785e6869ce`。
- Product correctness:
  - `ferrum run gemma3:27b-gptq --backend cuda --max-tokens 64 --temperature 0
    --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl`
    rc=0,assistant content `5`,finish_reason `stop`;
  - `ferrum serve` readiness PASS,poll 29;
  - `bench-serve --fail-on-error` rc=0。
- Diagnostic bench(非 release evidence,N=1,无 CI):
  - c=16:`16 completed / 0 errored`,output token count source `usage`,
    throughput `306.061 tok/s`;
  - c=32:`16 completed / 0 errored`,output token count source `usage`,
    throughput `307.373 tok/s`。
- Profile interpretation:
  - 相比 `w2_tail_gate_down_profile_2026-06-15`,batch=16
    `tail_norm_us_mean` 约 `806.5us -> 685.2us`,
    `tail_resid_us_mean` 约 `567.0us -> 494.8us`;
  - batch=16 total decode step 约 `28.08ms -> 27.82ms`,诊断收益约 `0.9%`;
  - 最大热点仍是 `tail_gate_up` 约 `9.01ms` 与 `tail_down` 约 `4.70ms`,
    因此下一步不应继续围绕 residual add 小项重复验证,应转向 gate/up/down
    或更高收益的 Gemma tail/GEMM 路径。
- Correctness status:
  - 本轮未发现新的 product correctness 问题;
  - 但没有 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未达到 release-grade。

## 2026-06-15 XXXIII — W2 source checkpoint: fuse Gemma sandwich branch norm into F32 residual add

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - added backend trait method `rms_norm_activation_add_to_f32` with a safe
    fallback that materializes the F32 branch then calls `add_inplace`;
  - added CUDA kernel `rms_norm_f16_add_to_f32` in `sandwich_norm.cu`;
  - CUDA backend now launches the fused kernel for F16 activation +
    F16 norm weight + F32 residual shadow;
  - Gemma sandwich device-shadow path now uses the fused helper for
    post-attention and post-FFN residual updates;
  - `nan_trace` keeps the old two-step path so `post_attn_norm` /
    `post_ffn_norm` intermediate dumps remain available for diagnostics.
- Expected effect:
  - removes one F32 residual-add kernel launch and one F32 branch scratch
    write/read at each Gemma sandwich branch residual update;
  - affects default CUDA Gemma3 device-shadow path only,not CPU/Metal
    fallback semantics.
- Local validation:
  - `cargo fmt --all` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/traits.rs
    crates/ferrum-kernels/kernels/sandwich_norm.cu
    crates/ferrum-kernels/src/backend/cuda/mod.rs
    crates/ferrum-models/src/models/llama_family.rs` PASS;
  - `cargo check -q -p ferrum-kernels -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-kernels --lib` PASS,8/8 tests;
  - `cargo test -q -p ferrum-models --lib` PASS,124/124 tests.
- Validation still required:
  - CUDA build must compile the new `sandwich_norm.cu` symbol;
  - run a minimal product correctness check on the same cache-retained 4090
    before any performance measurement;
  - if correctness passes, run a small same-dataset diagnostic to see whether
    the fused branch update moves W2 throughput or profile buckets.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXXII — W2 native CUDA dense Marlin probe: gate/up still top target, no tile-default change

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_native_probe_2026-06-15/`。
- Scope:
  - 这是 native CUDA kernel-ceiling diagnostic,不是 release-grade gate;
  - 没有运行 `ferrum run` 或 `ferrum serve`;
  - 没有生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不是
    release-grade。
- GPU execution:
  - reused Vast instance `40826362`,1x RTX 4090,约 USD 0.425/hr;
  - source sync 后远端 HEAD:`951348b23956caab8c459823708ddc4b63b90a8e`;
  - first native probe rc=`0`,printed
    `VERDICT: dense Marlin native CUDA probe complete`;
  - host-sync/cold-cache probe rc=`0`,printed the same `VERDICT`;
  - artifact copied back locally;
  - Vast stop poll verified `cur_state=stopped`,`actual_status=exited` at
    `2026-06-15T06:52:02Z`。
- Source/tooling change in this checkpoint:
  - extended `scripts/microbenches/dense_marlin_gemma3_perf.cu` with
    product-profile-style `host_sync_kernel` / `host_sync_ws_plus_kernel`
    modes;
  - added limited `cold_cache_kernel` rows for auto-tile `m=16/23/32`
    by flushing a 256MiB scratch buffer before timing;
  - updated `scripts/microbenches/README.md`.
- Key m=16 auto-tile native timings:
  - `qkv`: hot event `17.207 us`,host-sync `18.887 us`,cold-cache
    `39.929 us`;
  - `o_proj`: hot event `12.058 us`,host-sync `13.695 us`,cold-cache
    `24.447 us`;
  - `gate_up`: hot event `133.650 us`,host-sync `135.924 us`,cold-cache
    `177.144 us`;
  - `down`: hot event `30.395 us`,host-sync `32.049 us`,cold-cache
    `93.558 us`.
- Interpretation:
  - host-sync overhead alone does not explain product-profile `qkv/o/down`
    time; repeated-hot native timing was too optimistic for smaller
    projections because the same synthetic weight buffer is reused;
  - forced cold-cache timing is too pessimistic but brackets product behavior,
    confirming cache residency is a major measurement variable;
  - tile override evidence is weak: `64x256` only marginally improves hot
    `gate_up` and regresses `down`,so this checkpoint does not justify a
    default tile change;
  - no new product correctness issue was found in this diagnostic.
- Local validation:
  - `git diff --check -- scripts/microbenches/dense_marlin_gemma3_perf.cu`
    PASS;
  - `bash -n scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
    PASS.
- Next step:
  - continue from the correct default path, but choose the next lever based on
    product-representative weight/cache behavior rather than repeated-hot
    synthetic kernel timings alone;
  - do not claim performance or release readiness from this native diagnostic.

## 2026-06-15 XXXI — W2 source checkpoint: add native CUDA dense Marlin Gemma3 probe

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source/tooling change:
  - added `scripts/microbenches/dense_marlin_gemma3_perf.cu`;
  - added `scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`;
  - updated `scripts/microbenches/README.md`.
- Probe purpose:
  - bypass Cargo, model loading, tokenizer, server, and bench client;
  - compile only the probe plus `crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu`
    with `nvcc`;
  - call `marlin_cuda` directly on synthetic buffers for Gemma3-27B GPTQ
    `qkv`, `o_proj`, `gate_up`, and `down` shapes;
  - report `kernel_only` and `ws_plus_kernel` µs/call plus useful and padded
    TFLOPS for `m={1,3,6,9,12,16,23,32}` and tile choices
    `auto`, `128x128`, `64x256`.
- Why this checkpoint matters:
  - before changing dense Marlin tile selection or grid policy, we can now get
    a native CUDA kernel-ceiling result in minutes on the cached 4090 instead
    of paying a full Ferrum release build/product run for each hypothesis.
- Local validation:
  - `git diff --check -- scripts/microbenches/dense_marlin_gemma3_perf.cu
    scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh
    scripts/microbenches/README.md` PASS;
  - `bash -n scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
    PASS;
  - local machine has no `nvcc`,so native CUDA compile/run is pending on the
    CUDA host.
- Next step:
  - on the same cache-retained 4090, run only this native CUDA probe first;
  - keep the machine running during tight source/probe iterations; stop only
    after artifacts are copied or the iteration is no longer active.

## 2026-06-15 XXX — W2 source checkpoint: restore dense vLLM Marlin guard after first-fail evidence

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- Source change:
  - restored `reject_dense_vllm_marlin_if_requested`;
  - dense GPTQ load now rejects `FERRUM_VLLM_MARLIN=1` before building a
    dense vLLM Marlin store;
  - removed the diagnostic dense vLLM load path added in `ce960292`,because
    first-fail evidence showed it reaches vendored vLLM Marlin `abort()`
    before generation;
  - updated the unsupported message with the real blocker: vendored dense
    vLLM Marlin currently compiles with `kernel_selector.h` disabled for the
    CUDA hidden-symbol workaround,so it cannot select a real GEMM kernel
    safely;
  - default dense Marlin remains unchanged; vLLM Marlin MoE remains behind
    `FERRUM_VLLM_MOE` for stacked MoE weights.
- Local validation:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-kernels --tests` PASS;
  - `cargo test -q -p ferrum-kernels --lib` PASS,8/8 tests;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/quant.rs`
    PASS.
- Next step:
  - do not spend more time on dense vLLM Marlin unless explicitly taking on
    the broader `kernel_selector.h` / CUDA hidden-symbol linker problem;
  - continue W2 performance work on the existing correct default path with
    minimal same-pod validation.

## 2026-06-15 XXIX — W2 dense vLLM Marlin first-fail: prefill launch config aborts before generation

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_diag_2026-06-15/`。
- Scope:
  - 这不是 release-grade gate,也没有生成 `MODEL_RELEASE_GRADE_W2 PASS:
    <out_dir>`;
  - 失败只覆盖新接入的诊断路径 `FERRUM_VLLM_MARLIN=1`,默认 dense GPTQ
    Marlin 路径未因本轮诊断改判为失败;
  - remote git HEAD:`ce960292cf3132b982770a4cc727a9a6b19d2f4e`;
  - remote git status 因 artifacts 目录 rsync 排除显示旧 artifact 删除,所以本轮
    只能作为 first-fail/debug evidence,不是 clean performance evidence。
- GPU execution:
  - lane:`W2 Gemma3 CUDA dense vLLM Marlin first-fail diagnostic`;
  - Vast instance:`40826362`,1x RTX 4090,约 USD 0.425/hr;
  - release CUDA build PASS in `3m 28s`;
  - binary SHA256:
    `abd576f024776ed6df39c9e4c939b28344d93e6e69429cb663a749de28a1f3c8`;
  - sensitive scan of copied artifact: no `VAST_API_KEY`,`HF_TOKEN`,
    private-key,`jupyter_token`,or startup-script hits.
- Product-path result:
  - command: `FERRUM_VLLM_MARLIN=1 target/release/ferrum run
    gemma3:27b-gptq --backend cuda --prompt "What is 2+3? Answer with just
    the number." --max-tokens 64 --temperature 0 --kv-capacity 2560
    --max-num-seqs 2 --output-format jsonl`;
  - `run.status=FAIL`;
  - `correctness/run.rc=134`,`nohup.rc=134`;
  - model load completed,then the first dense vLLM Marlin launch aborted before
    token generation:
    `m=23 n=8192 k=5376 group_size=128`;
  - vLLM Marlin error:
    `Invalid thread config: thread_m_blocks = 1, thread_k = -1,
    thread_n = -1, num_threads = -1 for MKN = [23, 5376, 8192] and
    num_bits = 4, prob_m_split = 16, group_size = 128`;
  - server readiness and `bench-serve` did not run because the correctness
    first-fail stop condition triggered.
- Interpretation:
  - dense vLLM Marlin load/repack path is wired far enough to reach the kernel
    launch;
  - the current launch path is not safe for the skinny prefill shape seen by
    `ferrum run`,so it is a blocker for using `FERRUM_VLLM_MARLIN=1` as a
    product path;
  - this does not invalidate the previously collected default-path Ferrum
    zero-error diagnostics, but W2 still remains not release-grade until the
    final validator prints the required PASS line.
- GPU cleanup:
  - artifact copied locally before shutdown;
  - stop poll reached `cur_state=stopped actual_status=exited`;
  - stopped timestamp:`2026-06-15T06:23:32Z`.
- Next step:
  - inspect vLLM Marlin shape/config constraints and make the smallest safe
    source change: either route unsupported skinny prefill shapes to the
    existing IST-DASLab Marlin path,or correct the vLLM launch config;
  - validate on the same cache-retained 4090 with a minimal `ferrum run`
    first,then only run c16/c32 diagnostic if correctness passes.

## 2026-06-15 XXVIII — W2 source checkpoint: wire diagnostic dense vLLM Marlin GPTQ load path

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_VLLM_MARLIN=1` 不再在 dense GPTQ dispatch 入口被提前拒绝;
  - 当二进制带 `vllm-marlin` feature 且设置 `FERRUM_VLLM_MARLIN=1` 时,
    dense `load_gptq` 会:
    - 上传 GPTQ qweight;
    - 通过已有 `ferrum_vllm_gptq_marlin_repack` FFI 生成
      vLLM Marlin tile qweight;
    - 使用与 vLLM stacked path 一致的 Marlin scale permutation;
    - 构造 `MarlinWeight` 后复用现有 `launch_vllm_marlin` dispatch;
  - 如果设置 `FERRUM_VLLM_MARLIN=1` 但未编译 `vllm-marlin`,现在会在
    load 阶段明确报错;
  - 默认路径不变:未设置 `FERRUM_VLLM_MARLIN=1` 时仍走
    IST-DASLab Marlin repack/dispatch。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/quant.rs`
    PASS;
  - `cargo check -q -p ferrum-kernels --tests` PASS;
  - `cargo test -q -p ferrum-kernels cuda_quant_runtime_config_parses_marlin_and_moe_knobs --lib`
    PASS,0 tests matched in the non-CUDA local build.
- Evidence caveat:
  - 本机没有做 CUDA/vLLM feature build;该 checkpoint 必须通过下一轮
    4090 release CUDA build and diagnostic run 才能证明 dense vLLM Marlin
    path可加载、可正确生成、并有可比较性能。
- Next step:
  - 复用 1x4090 cache-retained instance 做 first-fail 小样本:
    release build with `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`,
    再跑 `FERRUM_VLLM_MARLIN=1` 的 `ferrum run` smoke 和
    c16/c32 diagnostic;如果 load/correctness 失败,立刻拷回失败 artifact
    并停机。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXVII — W2 projection-level Marlin profile: gate/up kernel is the dominant dense GPTQ target

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_projection_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T06:02:00Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA projection-level dense Marlin profile diagnostic`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,projection-level dense Marlin profile c16/c32 small sample complete
    and copied,or 45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset with
    `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1`,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS in `3m 28s`,binary SHA256:
    `0991e89489c205f6fdffec5dbf138923367e51c02f0356ddd8828c276003a950`;
  - remote git HEAD was `5fac46d8a45b99932d06c462d7be50d8825d9d55`;
  - remote git status had `337` lines because local `docs/.../artifacts/`
    was excluded from source rsync to avoid copying old evidence directories,
    so this remains profiling evidence only,not final clean performance
    evidence;
  - `FERRUM_MARLIN_PROFILE=1` adds per-Marlin-call syncs,so throughput in
    this artifact is profiling overhead and not a product performance claim.
- Build/bench result:
  - `run.status=PASS`,`nohup.rc=0`,`bench-serve.rc=0`;
  - server readiness poll: `29`;
  - c16:completed `[16]`,errored `[0]`,mean `290.042 tok/s`,p95 ITL
    `31.777 ms`,p95 TTFT `1567.437 ms`,output token source `usage`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `290.221 tok/s`,p95 ITL `31.855 ms`,p95 TTFT `1561.008 ms`,
    output token source `usage`.
- Op-profile result:
  - server log captured `264` `[batched-op-profile]` rows with projection
    fields populated and `marlin_other_* = 0`,so the profile labels covered
    all dense Marlin calls in this path;
  - for batch `m=16` (`118` rows),mean total per decode step
    `30063 us`; aggregate dense Marlin kernel was `55.0%` (`16548 us`);
    projection kernel split:
    - gate/up `29.0%` (`8728 us`);
    - down `14.5%` (`4352 us`);
    - qkv `7.1%` (`2132 us`);
    - o_proj `4.4%` (`1336 us`);
    aggregate workspace zero was `3.8%` (`1137 us`);
  - for batch `m=10` (`123` rows),aggregate dense Marlin kernel was
    `55.6%` (`16349 us`); projection kernel split:
    gate/up `29.2%` (`8593 us`),down `14.8%` (`4344 us`),
    qkv `7.1%` (`2099 us`),o_proj `4.5%` (`1313 us`);
    workspace zero was `3.8%` (`1120 us`).
- Interpretation:
  - gate/up dense Marlin kernel alone is the largest single measured decode
    cost and is bigger than any other dense GPTQ projection bucket;
  - workspace zero remains measurable but small relative to kernel time;
  - next useful checkpoint should compare/alter the gate/up dense GPTQ kernel
    path itself: Triton INT4 diagnostic viability, vLLM dense GPTQ repack/path
    comparison, or a shape-specific Marlin lever. Do not change product
    defaults from this artifact alone.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXVI — W2 profile instrumentation: split dense Marlin counters by projection

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_MARLIN_PROFILE=1` 的 dense Marlin nested counters 现在按
    projection label 细分输出:
    `qkv`,`o_proj`,`gate_up`,`down`,`lm_head`,`other`;
  - 每个 projection bucket 分别记录 `ws_zero` 与 `kernel` 的时间和调用数;
  - `[batched-op-profile]` 保留原有 aggregate
    `marlin_ws_zero`/`marlin_kernel`,并新增
    `marlin_qkv_*`,`marlin_o_*`,`marlin_gate_up_*`,
    `marlin_down_*`,`marlin_lm_head_*`,`marlin_other_*`;
  - 给 batched decode 的 `o_proj` 补上 CUDA alloc label,避免它在
    projection profile 中落入 `other`;
  - 默认路径不变;新增分桶只在 `FERRUM_MARLIN_PROFILE=1` 的诊断路径累加。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/marlin.rs crates/ferrum-models/src/models/llama_family_forward_batched.rs`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Evidence caveat:
  - 本机未做 CUDA feature 编译;该 checkpoint 需要下一轮 4090 release
    build/profile artifact 来验证 CUDA profile fields 的实际日志输出。
- Next step:
  - 复用 1x4090 cache-retained instance 做一个小样本
    `FERRUM_DECODE_OP_PROFILE=1` + `FERRUM_MARLIN_PROFILE=1` profile,
    确认 gate/up Marlin kernel 是否确实主导 dense GPTQ time;如果成立,
    下一步再比较 Triton INT4 或 vLLM dense GPTQ packing path。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade。

## 2026-06-15 XXV — W2 dense-Marlin nested profile: kernel dominates, workspace zero is not first lever

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_nested_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T05:38:17Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA dense-Marlin nested profile diagnostic`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,dense Marlin nested profile c16/c32 small sample complete and
    copied,or 45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset with
    `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1`,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS,binary SHA256:
    `3c503a7cabcc0acba90fd35ac40704c19f631f4f2a6f206f8a8374758b20a280`;
  - remote git HEAD was `95a27d7738d4834fa09b52ee5a86cf084c16de75`;
  - remote git status is dirty because local `docs/.../artifacts/` was
    excluded from source rsync to avoid copying old evidence directories,so
    this remains profiling evidence only,not final clean performance evidence;
  - `FERRUM_MARLIN_PROFILE=1` adds per-Marlin-call syncs,so the lower
    throughput in this artifact is profiling overhead and not a product
    performance claim.
- Build/bench result:
  - `run.status=PASS`,`nohup.rc=0`,`bench-serve.rc=0`;
  - build completed in `3m 29s`;
  - c16:completed `[16]`,errored `[0]`,mean `288.575 tok/s`,p95 ITL
    `32.181 ms`,p95 TTFT `1570.866 ms`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `289.619 tok/s`,p95 ITL `32.077 ms`,p95 TTFT `1560.208 ms`.
- Op-profile result:
  - server log captured `264` `[batched-op-profile]` rows;
  - for batch `m=16` (`118` rows),mean total per decode step
    `30134 us`; nested dense Marlin kernel aggregate was `54.9%`
    (`16550 us`),workspace zero aggregate was `3.9%` (`1181 us`);
    `tail_gate_up` was `31.6%` (`9526 us`),`tail_down` `17.6%`
    (`5299 us`),combined `tail_mlp` `49.2%` (`14825 us`);
  - for batch `m=10` (`123` rows),nested dense Marlin kernel aggregate was
    `55.5%` (`16350 us`),workspace zero aggregate `3.9%` (`1145 us`);
    `tail_gate_up` was `31.8%` (`9379 us`),`tail_down` `17.7%`
    (`5212 us`).
- Interpretation:
  - workspace zero is measurable but not the first lever; it is roughly
    `1.1-1.2 ms` per decode step across all dense Marlin calls;
  - most of the remaining time is dense Marlin kernel work,with the fused
    Gemma3 gate/up projection still the largest projection-level bucket;
  - next useful checkpoint should focus on the gate/up Marlin shape itself:
    kernel launch/shape behavior, existing Triton INT4 diagnostic viability,
    or a source comparison against the vLLM dense GPTQ path. Avoid changing
    product defaults until correctness and same-dataset diagnostics support it.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XXIV — W2 profile instrumentation: add dense Marlin nested counters

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - 新增 `FERRUM_MARLIN_PROFILE=1` profile-only 开关;
  - 在 dense Marlin `marlin_gemm_chunk` 内部新增 nested counters:
    `marlin_ws_zero` 和 `marlin_kernel`;
  - batched decode `[batched-op-profile]` 日志会输出这两个 nested
    字段,但不会把它们加入 `wrapped_us`,避免和 `tail_gate_up` /
    `tail_down` 双计;
  - 默认路径不变,`FERRUM_MARLIN_PROFILE` 未设置时不增加同步计时;
  - 未改变 `FERRUM_MARLIN_SKIP_WS_ZERO` 行为。该开关仍只用于已有的
    strided path,没有扩展到 dense path。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-kernels/src/backend/cuda/marlin.rs crates/ferrum-models/src/models/llama_family_forward_batched.rs`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Evidence caveat:
  - local non-CUDA checks passed, but the new CUDA-feature path still needs a
    4090 release build in the next diagnostic checkpoint before relying on the
    new Marlin nested fields.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - rerun the c16/c32 CUDA profile diagnostic with both
    `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_MARLIN_PROFILE=1` to split
    gate/up projection time into workspace-zero and Marlin kernel work.

## 2026-06-15 XXIII — W2 tail-gate/down profile: gate/up projection is the largest single decode bucket

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_gate_down_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T05:20:40Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA tail-gate/down profile diagnostic`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,tail gate/down profile c16/c32 small sample complete and copied,
    or 45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS,binary SHA256:
    `fb45a77d328c90233ffeb19cb4576bc12ef7079c2096632fe145431c83fcfe2a`;
  - remote git HEAD was `ccc58aba9f5333f1ecd258d841de0fd5ab40a379`;
  - remote git status is dirty because local `docs/.../artifacts/` was
    excluded from source rsync to avoid copying old evidence directories,so
    this remains profiling evidence only,not final clean performance evidence;
  - first remote tmux attempt failed before build because Rust was not on PATH
    in the non-login shell; runner was fixed to source `/root/.cargo/env`,
    the remote output directory was removed, and the corrected rerun produced
    the copied artifact.
- Build/bench result:
  - `run.status=PASS`,`nohup.rc=0`,`bench-serve.rc=0`;
  - build completed in `3m 25s`;
  - c16:completed `[16]`,errored `[0]`,mean `304.328 tok/s`,p95 ITL
    `30.068 ms`,p95 TTFT `1523.492 ms`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `305.245 tok/s`,p95 ITL `29.929 ms`,p95 TTFT `1512.247 ms`.
- Op-profile result:
  - server log captured `264` `[batched-op-profile]` rows;
  - for batch `m=16` (`118` rows),mean total per decode step
    `28076 us`; `tail_gate_up` was `32.2%` (`9039 us`),`tail_down`
    `16.8%` (`4709 us`),combined `tail_mlp` `49.0%` (`13748 us`);
    remaining unwrapped was `2.3%` (`658 us`);
  - for batch `m=10` (`123` rows),`tail_gate_up` was `32.5%`
    (`8901 us`),`tail_down` `16.8%` (`4621 us`),combined `tail_mlp`
    `49.3%` (`13522 us`).
- Interpretation:
  - the largest single decode bucket is the fused Gemma3 gate/up GPTQ
    projection,not down projection,attention/QKR,or logits readback;
  - current next target is the dense GPTQ Marlin path for the fused
    `gate_up_proj` shape. A useful next checkpoint is to measure Marlin
    fixed overhead/workspace-zero and vLLM-Marlin/Triton alternatives as
    explicit diagnostics before changing product defaults;
  - do not run a full `--require-ci` release sweep until this gate/up
    projection bottleneck is reduced.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XXII — W2 profile instrumentation: split tail MLP into gate/up and down

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_DECODE_OP_PROFILE` 的 batched decode 日志继续保留
    `tail_mlp` 聚合字段;
  - 新增 `tail_gate_up` 和 `tail_down` 子字段,分别计时 Gemma3 tail
    的 fused gate/up projection 和 down projection;
  - `unwrapped` 现在扣除 `tail_gate_up + tail_down`,避免双计
    `tail_mlp` 聚合值;
  - 非 profile 路径不改变。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check -- crates/ferrum-models/src/models/llama_family.rs crates/ferrum-models/src/models/llama_family_forward_batched.rs`
    PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - rerun a small c16/c32 CUDA profile diagnostic to quantify
    `tail_gate_up` vs `tail_down` before choosing the MLP projection
    optimization target.

## 2026-06-15 XXI — W2 tail-profile bucket validation: Gemma3 MLP projections dominate decode

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_profile_buckets_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=04 cur_state=stopped actual_status=exited intended_status=stopped`;
  - stop timestamp: `2026-06-15T05:00:20Z`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA tail-profile bucket validation`;
  - expected runtime/cost:15-35min,hard cap 45min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/source sync/build/server readiness first
    failure,tail bucket profile c16/c32 small sample complete and copied,or
    45min cap;
  - correctness gate:release build plus server readiness and
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - release build PASS,binary SHA256:
    `c32bb67c8ee9aee90b0054bcb0fb0eca0e1d127fad5c99929941b714bdf741ed`;
  - remote git HEAD remained `c51002b793f00c8345e160b99b6b74217ca273d9`
    with the profiling source files dirty-synced from the local checkpoint,so
    this is profiling evidence only,not final clean performance evidence;
  - decision trace selected `attention_decode_backend=legacy_paged_decode`
    and `sampling_readback_path=gpu_greedy_argmax`;
  - `bench-serve` rc=0,run status `PASS`,server log captured `264`
    `[batched-op-profile]` rows.
- Bench/profile result:
  - c16:completed `[16]`,errored `[0]`,mean `305.182 tok/s`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `305.181 tok/s`;
  - for batch `m=16` (`118` rows),mean total per decode step
    `28037 us`; mean shares:tail_mlp `49.0%` (`13744 us`),matmul
    `24.9%` (`6971 us`),attention `8.6%` (`2406 us`),tail_norm `2.9%`,
    tail_resid `2.0%`,tail_act `1.5%`,QKR `2.3%`,norm `1.6%`,
    remaining unwrapped `2.3%` (`649 us`);
  - for batch `m=10` (`122` rows),tail_mlp was again the largest bucket:
    `49.3%` (`13516 us`),with remaining unwrapped down to `2.4%`
    (`663 us`).
- Interpretation:
  - the previous `unwrapped` bucket was mostly Gemma3 tail MLP projection work
    rather than attention/QKR or logits readback;
  - current top target is the Gemma3 tail gate/up/down GPTQ linear path. The
    next useful profiling checkpoint is to split `tail_mlp` into gate/up and
    down projection buckets before choosing an optimization patch;
  - do not run a full `--require-ci` release sweep until this c16/c32 decode
    bottleneck is reduced.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XX — W2 profile instrumentation: split Gemma3 tail buckets

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `FERRUM_DECODE_OP_PROFILE` 的 batched decode 输出新增 tail bucket:
    `tail_norm`,`tail_mlp`,`tail_act`,`tail_resid`;
  - 新 bucket 细分 Gemma3 sandwich tail 的 post-attn/post-ffn norms、
    gate/up/down projections、GeGLU/SwiGLU activation、residual add;
  - `unwrapped` 现在会扣除这些 tail bucket,用于下一轮 GPU diagnostic
    定位 2026-06-15 XIX 里约 `55.6%` 的未拆分 decode-step 时间;
  - 非 profile 路径不改变。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models llama_batched_runtime_config_parses_startup_knobs --lib`
    PASS,1 test passed。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - rerun a small c16/c32 decode-op-profile diagnostic on CUDA to quantify the
    new tail buckets before choosing an optimization target.

## 2026-06-15 XIX — W2 decode-op profile: bottleneck is mostly unwrapped decode work

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_decode_op_profile_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA decode-op-profile diagnostic`;
  - expected runtime/cost:10-25min,hard cap 40min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:start/SSH/CUDA/server readiness first failure,profile log
    captures c16/c32 small sample,or 40min cap;
  - correctness gate:server readiness plus first
    `bench-serve --fail-on-error` zero-error diagnostic;
  - performance command:diagnostic-only natural ASCII ShareGPT dataset,
    `num_prompts=16`,`n_repeats=1`,`random-output-len=64`,`seed=9271`,
    c16/c32.
- Evidence scope:
  - reused existing release binary SHA256:
    `a942a2e79880bbc821c26a1c60720fa753d6b8e66a62a73900a4592d123abb0e`;
  - remote git HEAD remained `c51002b793f00c8345e160b99b6b74217ca273d9`
    with `crates/ferrum-types/src/auto_config.rs` dirty-synced from the
    current checkpoint,so this is profiling evidence only,not final clean
    performance evidence;
  - decision trace still selected `attention_decode_backend=legacy_paged_decode`
    and `sampling_readback_path=gpu_greedy_argmax`.
- Bench/profile result:
  - `bench-serve` rc=0,run status `PASS`;
  - c16:completed `[16]`,errored `[0]`,mean `313.252 tok/s`;
  - c32 client / active cap16:completed `[16]`,errored `[0]`,mean
    `315.362 tok/s`;
  - server log captured `264` `[batched-op-profile]` rows.
- Op-profile summary:
  - for batch `m=16` (`117` rows),mean total per decode step
    `26785 us`,p95 `27077 us`;
  - mean shares:unwrapped `55.6%` (`14884 us`),matmul `26.1%`
    (`6980 us`),attention `9.0%` (`2414 us`),QKR `2.4%`
    (`636 us`),norm `1.9%` (`500 us`),other `5.1%` (`1371 us`);
  - for batch `m=10` (`119` rows),shares were similar:unwrapped `55.8%`,
    matmul `26.4%`,attention `8.3%`.
- Interpretation:
  - the current profile does not point first at attention kernels; more than
    half of the measured decode-step time is outside the existing op counters;
  - the next useful checkpoint is to split the `unwrapped` bucket into concrete
    sections,likely Gemma3 sandwich tail/GeGLU/projector glue, device-shadow
    handling, sync/copy, or uninstrumented linear/activation work;
  - avoid a full `--require-ci` performance sweep until that unwrapped bucket
    is explained and reduced.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.

## 2026-06-15 XVIII — W2 greedy-argmax default diagnostic: product default confirmed, performance still below 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_greedy_argmax_default_diag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=03 cur_state=stopped actual_status=exited intended_status=stopped`。
- Source checkpoint:
  - `9a338235 fix(types): enable greedy argmax for accelerator defaults`;
  - `FERRUM_GREEDY_ARGMAX` now auto-resolves to true on CUDA/Metal when the
    compiled accelerator supports greedy argmax, unless explicitly disabled.
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA greedy-argmax default validation`;
  - expected runtime/cost:20-45min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/build/`ferrum run`/serve smoke first
    failure,decision trace missing `gpu_greedy_argmax`,c16/c32 diagnostic
    complete and artifact copied,or 60min cap;
  - correctness gate:`ferrum run` plus
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --fail-on-error --seed 9271`
    for c16/c32 small sample first.
- Build/correctness:
  - release build PASS,binary SHA256:
    `a942a2e79880bbc821c26a1c60720fa753d6b8e66a62a73900a4592d123abb0e`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke rc=0,PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Runtime default evidence:
  - both `ferrum run` and diagnostic `ferrum serve` decision traces selected
    `sampling_readback_path=gpu_greedy_argmax`;
  - selected source:`hardware_capability`;
  - `diagnostic_summary.json` reports
    `sampling_trace_has_gpu_greedy_argmax=true`.
- Diagnostic result(same natural ASCII ShareGPT dataset as the vLLM baseline
  and previous Ferrum diagnostic; `num_prompts=32`,`n_repeats=1`,zero errors,
  `output_token_count_source=usage`):
  - c16:completed `[32]`,errored `[0]`,mean `347.880 tok/s`,
    ratio vs vLLM natural baseline LCB `0.708`,ratio vs baseline mean
    `0.655`,required 80% of baseline LCB `392.920 tok/s`,p95 ITL
    `109.763 ms`;
  - c32 client / Ferrum active cap16:completed `[32]`,errored `[0]`,
    mean `356.835 tok/s`,ratio vs vLLM natural baseline LCB `0.650`,
    ratio vs baseline mean `0.634`,required 80% of baseline LCB
    `439.514 tok/s`,p95 ITL `109.657 ms`.
- Interpretation:
  - the typed default fix is product-visible and no hidden env var is needed;
  - performance did not materially improve versus the previous Ferrum natural
    diagnostic(c16 `350.868 tok/s`,c32 `354.291 tok/s`),so the remaining W2
    blocker is not an accidental logits-readback default;
  - continue with targeted decode/attention/batching evidence before any
    full `--require-ci` release sweep.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - inspect current CUDA decode path against the decision trace: notably the
    earlier diagnostic still selected `legacy_paged_decode`;
  - choose one targeted optimization/profiler step that can move c16/c32 tail
    ITL before rerunning the same natural dataset diagnostic.

## 2026-06-15 XVII — W2 Ferrum natural-prompt diagnostic: correctness clean, c16/c32 below 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_ferrum_natural_prompt_diag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA natural-prompt Ferrum diagnostic`;
  - expected runtime/cost:20-45min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/build/product-smoke first failure,c16/c32
    diagnostic complete and artifact copied,or 60min cap;
  - correctness gate:`ferrum run` plus
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --fail-on-error --seed 9271`
    for c16/c32 small sample first.
- Build/correctness:
  - release build PASS,binary SHA256:
    `90a30cafef8ea1fe9f1edf3ea326d04dd2f0ca1b8226923ffec559d61d8c5d78`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke rc=0,PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Dataset:
  - exact same `ascii_sharegpt.jsonl` as
    `w2_natural_prompt_baseline_probe_2026-06-15`;
  - diagnostic run used `num_prompts=32`,`n_repeats=1`,so it is not final
    release evidence and has no CI lower bound.
- Diagnostic result(all usage token counts,zero errors,bad_output `[0]`):
  - c16:completed `[32]`,errored `[0]`,mean `350.868 tok/s`,
    ratio vs vLLM natural baseline LCB `0.714`,ratio vs baseline mean `0.661`,
    required 80% of baseline LCB `392.920 tok/s`,p95 ITL `109.550 ms`
    vs baseline `28.130 ms`;
  - c32 client / Ferrum active cap16:completed `[32]`,errored `[0]`,
    mean `354.291 tok/s`,ratio vs vLLM natural baseline LCB `0.645`,
    ratio vs baseline mean `0.630`,required 80% of baseline LCB
    `439.514 tok/s`,p95 ITL `109.782 ms` vs baseline `27.716 ms`.
- Interpretation:
  - product-path correctness remains clean on the current build;
  - same natural prompt dataset removes the baseline correctness ambiguity and
    shows the remaining W2 blocker is performance,especially tail ITL and c32
    throughput under active cap16;
  - do not expand to a full `--require-ci` release sweep until a targeted
    optimization/profiler step moves c16/c32 close to the 80% thresholds.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - profile/optimize the decode path under natural prompts,with emphasis on
    high p95 ITL and the remaining c16/c32 throughput gap;
  - after a targeted fix, rerun the same natural dataset diagnostic before any
    full release-grade CI sweep.

## 2026-06-15 XVI — W2 natural-prompt baseline probe: vLLM c16/c32 zero-error

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA natural-prompt baseline safety probe`;
  - expected runtime/cost:20-50min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/vLLM smoke failure,ShareGPT-style ASCII
    dataset c16 nonzero,c16 pass then c32/cap16 nonzero,probe complete and
    artifact copied,or 60min cap;
  - correctness gate:torch CUDA smoke plus vLLM OpenAI smoke;
  - performance command:diagnostic-only
    `ferrum bench-serve --dataset sharegpt --sharegpt-path <artifact>/ascii_sharegpt.jsonl --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100`
    for c16 then c32.
- Dataset:
  - generated artifact-local JSONL:
    `dataset/ascii_sharegpt.jsonl`;
  - actual tokenizer-counted input length:requested `256`,min/max/mean `112`.
- vLLM setup:
  - venv:`/workspace/vllm-venv-0101-cu126`;
  - server:vLLM OpenAI API,`v0.10.1.1`,`transformers==4.55.4`,
    `--max-model-len 512 --max-num-seqs 16 --gpu-memory-utilization 0.92`;
  - smoke request returned content `"5\n"` with usage.
- Probe result(all `n_repeats=3`, `num_prompts=100`,
  `output_token_count_source=usage`,zero errors,bad_output `[0,0,0]`):
  - c16:completed `[100,100,100]`,errored `[0,0,0]`,
    mean `530.829 tok/s`,ci95 half-width `39.679`,LCB `491.150`,
    p95 ITL `28.130 ms`;
  - c32 client / vLLM `--max-num-seqs 16`:completed `[100,100,100]`,
    errored `[0,0,0]`,mean `562.685 tok/s`,ci95 half-width `13.292`,
    LCB `549.393`,p95 ITL `27.716 ms`.
- Interpretation:
  - natural ASCII ShareGPT-style prompts avoid the vLLM invalid-UTF8 failure
    seen on random-token prompts,so this is a viable correctness-clean
    baseline dataset candidate;
  - the baseline is substantially faster than current Ferrum random-matrix
    c16/c32,so the final W2 80% line would be about c16 `392.9 tok/s`
    and c32 `439.5 tok/s` if this dataset is adopted;
  - no final claim yet: Ferrum must be rerun on the exact same JSONL dataset
    and c32 effective concurrency must be published as 16.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - use this dataset for targeted Ferrum diagnostics or optimization;
  - avoid another full release sweep until there is evidence that Ferrum c16/c32
    can approach the natural-prompt baseline 80% thresholds.

## 2026-06-15 XV — W2 baseline safety probe: vLLM c16 invalid-UTF8 reproduces

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_baseline_safety_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA mainstream baseline safety probe`;
  - expected runtime/cost:20-50min,hard cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/vLLM server smoke failure,any probe cell
    nonzero,invalid-UTF8 reproduction,probe complete and artifact copied,or
    60min cap;
  - correctness gate:torch CUDA smoke plus vLLM OpenAI smoke;
  - performance command:
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100`
    for vLLM c16 first,then c32/cap16 only if c16 passes.
- vLLM setup:
  - venv:`/workspace/vllm-venv-0101-cu126`;
  - server:vLLM OpenAI API,`v0.10.1.1`,`transformers==4.55.4`,
    `--max-model-len 512 --max-num-seqs 16 --gpu-memory-utilization 0.92`;
  - smoke request returned content `"5\n"` with usage.
- Probe result:
  - c16 release-shape rerun reproduced the exact blocker:
    `[err] bad output invalid-utf8: �\"`;
  - repeats completed `[100,100,99]`,errored `[0,0,1]`,bad_output
    `[0,0,1]`,rc=`1`;
  - output token count source:`usage`;
  - diagnostic throughput mean:`385.332 tok/s`,ci95 half-width:`7.385`,LCB
    `377.947`,p95 ITL `27.353 ms`.
- Interpretation:
  - vLLM c16 is fast but not zero-error under the release-shape
    `bench-serve --fail-on-error --require-ci` contract;
  - this confirms the earlier `w2_vllm0101_cuda12_baseline_probe_2026-06-15`
    failure and means vLLM c16 cannot be used in the final W2 manifest as-is;
  - c32 was intentionally not run after the c16 first-fail stop.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - choose a final baseline strategy that can produce zero-error c16/c32
    evidence: either an alternate mainstream engine/config allowed by
    `RELEASE_GRADE_GOAL.md`,or a same-dataset rerun path where both Ferrum and
    baseline are release-clean.

## 2026-06-15 XIV — W2 sentinel-fix Ferrum release-shape matrix PASS;baseline still blocks release-grade

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_sentinel_fix_release_shape_ferrum_ci_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `poll=01 cur_state=stopped actual_status=running intended_status=stopped`;
  - `poll=02 cur_state=stopped actual_status=exited intended_status=stopped`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA sentinel-fix release-shape Ferrum CI matrix`;
  - expected runtime/cost:1.5-3h,hard cap 3h,1x RTX 4090 instance
    `40826362` at about USD 0.425/hr;
  - stop condition:startup/SSH/CUDA/build/correctness first failure,any bench
    cell nonzero or blocker warning,full matrix artifact copied,or 3h cap;
  - correctness gate:CUDA `argmax_rows` test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100`
    for c=1/4/16/32。
- Build/correctness:
  - CUDA `argmax_rows` test PASS,including sentinel case;
  - release build PASS/cache-hit,binary SHA256:
    `6883cc81f3c0a9e16c6c8d374cc98d5c154309e75bd1d7cac7cad832902cbcfb`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Ferrum release-shape matrix result(all `n_repeats=3`, `num_prompts=100`,
  `output_token_count_source=usage`,zero errors,blocker warning count `0`):
  - c=1:`39.152 tok/s`,ci95 half-width `0.053`,LCB `39.099`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `24.618 ms`;
  - c=4:`125.981 tok/s`,ci95 half-width `2.397`,LCB `123.584`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `26.236 ms`;
  - c=16:`259.130 tok/s`,ci95 half-width `80.145`,LCB `178.985`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `38.334 ms`;
  - c=32 client / typed active cap16 (`--kv-capacity 400 --max-num-seqs 16`):
    `281.525 tok/s`,ci95 half-width `15.552`,LCB `265.973`,
    completed `[100,100,100]`,errored `[0,0,0]`,p95 ITL `39.561 ms`。
- Interpretation:
  - Ferrum product-path correctness and full release-shape matrix now pass on
    the sentinel-fix build;
  - c4 release-shape LCB `123.584` clears same-hardware vLLM c4 80% threshold
    `123.335` by a narrow margin, replacing the earlier 32-request pre-gate as
    better c4 evidence;
  - c16 remains release-grade risk:LCB `178.985` is far below 80% of the
    previous vLLM diagnostic c16 mean (`381.5 * 0.8 = 305.2`), though that vLLM
    c16 run itself had invalid UTF-8 and cannot be final baseline evidence;
  - c32 must be represented as requested c=32 with effective/published
    concurrency 16 unless true active c=32 is implemented.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - create a checkpoint commit for the sentinel-fix/full-matrix evidence;
  - resolve release-grade baseline coverage for c=16/c=32, then assemble
    `model_release_grade_manifest.json` and run
    `python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>`。

## 2026-06-15 XIII — W2 c4 CI pre-gate: c4 lower bound clears 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c4_ci_pregate_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `10:38:02 cur_state=stopped actual_status=running`;
  - `10:38:13 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA c4 release-grade confidence pre-gate`;
  - expected runtime/cost:25-55min,stop cap 75min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,CUDA argmax test failure,
    `ferrum run` failure,serve smoke failure,c4 `--require-ci --n-repeats 3`
    nonzero/error/warning,c4 CI evidence completes,or 75min cap;
  - correctness gate:CUDA `argmax_rows` test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
    at c=4;
  - baseline:same-hardware vLLM 0.10.1.1 c4 baseline `154.169 tok/s`,
    80% threshold `123.335 tok/s`。
- Correctness:
  - CUDA `argmax_rows` test PASS,including sentinel case;
  - release build cache-hit PASS;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- c4 CI result:
  - repeats: `32/32/32 completed`, `0/0/0 errored`;
  - output token count source:`usage`;
  - greedy argmax warning count:`0`;
  - throughput mean:`128.988 tok/s`,stddev:`0.501`,ci95 half-width:`1.246`;
  - lower bound:`127.742 tok/s`,ratio to baseline:`0.829`;
  - mean ratio:`128.988 / 154.169 = 0.837`;
  - c4 p95 ITL mean:`25.582 ms`,ci95 half-width:`0.004 ms`;
  - c4 p95 TTFT mean:`814.284 ms`,ci95 half-width:`61.656 ms`;
  - c4 p95 TPOT mean:`27.212 ms`,ci95 half-width:`4.918 ms`。
- Interpretation:
  - c4 now has CI evidence clearing the 80% throughput line;
  - this is still a pre-gate,not final W2 release-grade, because required W2 cells
    c=1/16/32 and final manifest/validator are still missing, and c16/c32
    mainstream baseline handling must be resolved.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - collect Ferrum release-shape CI evidence for c=1/16/32 on the sentinel-fix
    build, then assemble/validate the W2 manifest;
  - in parallel, resolve release-grade baseline coverage for c=16/c=32, because
    previous vLLM c16 failed invalid UTF-8 and cannot be used as final baseline
    evidence as-is.

## 2026-06-15 XII — W2 sentinel-fix c4/c16 diagnostic: c4 mean crosses 80% line

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_sentinel_fix_c4_c16_diag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `10:26:19 cur_state=stopped actual_status=running`;
  - `10:26:30 cur_state=stopped actual_status=running`;
  - `10:26:43 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA sentinel-fix c4/c16 performance diagnostic`;
  - expected runtime/cost:20-50min,stop cap 70min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,CUDA argmax test failure,
    `ferrum run` failure,serve smoke failure,any bench cell nonzero error,
    greedy-argmax warning reproduces,c4/c16 diagnostic completes,or 70min cap;
  - correctness gate:CUDA `argmax_rows` test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1`
    for c=4 and c=16;
  - baseline:reuse previously saved same-hardware vLLM 0.10.1.1 c4 baseline
    `154.169 tok/s`;baseline not rerun in this diagnostic.
- Correctness:
  - CUDA `argmax_rows` test PASS,including sentinel case;
  - release build cache-hit PASS;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench result(非 release evidence,N=1,无 CI):
  - c4:`32 completed / 0 errored`, `125.057 tok/s`;
  - c16:`32 completed / 0 errored`, `305.287 tok/s`;
  - output token count source:`usage`;
  - greedy argmax warning count:`0`;
  - c4 p95 ITL:`25.562 ms`,p95 TTFT:`828.434 ms`,p95 TPOT:`30.289 ms`;
  - c16 p95 ITL:`28.539 ms`,p95 TTFT:`3251.742 ms`,p95 TPOT:`47.267 ms`;
  - health after c16:`force_full_logits_calls=0`,`calls=1805`,
    `total_items=10570`,`avg_items_per_call=5.856`,`max_items=16`,
    buckets `m3_4=1271`,`m9_16=379`。
- Interpretation:
  - c4 now crosses the 80% mean line versus same-hardware vLLM baseline:
    `125.057 / 154.169 = 0.811`;80% threshold is `123.335 tok/s`;
  - this is still only N=1 diagnostic evidence,not release-grade performance
    evidence under `RELEASE_GRADE_GOAL.md`;
  - next step should be release-grade performance collection with
    `--fail-on-error --require-ci --seed 9271 --n-repeats 3` for the required
    cells, plus manifest/validator wiring.
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade until CI/variance evidence and the final
    validator pass.

## 2026-06-15 XI — W2 masked-argmax sentinel fix: CUDA c16 diagnostic clean

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_sentinel_fix_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `10:13:50 cur_state=stopped actual_status=running`;
  - `10:14:01 cur_state=stopped actual_status=running`;
  - `10:14:13 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA masked-argmax sentinel-fix validation`;
  - expected runtime/cost:20-45min,stop cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,CUDA argmax test failure,
    `ferrum run` failure,serve smoke failure,c16 repeats forbidden-token diagnostic,
    c16 diagnostic clean completes,or 60min cap;
  - correctness gate:CUDA `argmax_rows` masked test including sentinel case,
    `ferrum run`,`scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1` at c=16;
  - baseline:reuse previously saved same-hardware vLLM 0.10.1.1 c4 baseline
    `154.169 tok/s`;baseline not rerun in this diagnostic.
- CUDA validation:
  - `argmax_rows_f16_masked_skips_invalid_tokens` PASS;
  - `argmax_rows_f16_masked_returns_sentinel_without_finite_valid_token` PASS;
  - release build PASS:`cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench result(非 release evidence,N=1,无 CI):
  - c16:`32 completed / 0 errored`, `305.275 tok/s`;
  - output token count source:`usage`;
  - greedy argmax warning count:`0`;
  - decode stats:`force_full_logits_calls=0`,`calls=392`,`total_items=5334`,
    `avg_items_per_call=13.607`,`max_items=16`,bucket `m9_16=379`;
  - run status:`diagnostic_clean`,run rc:`0`。
- Interpretation:
  - sentinel fix removed the reproduced c16 forbidden-token failure in this
    diagnostic shape;
  - c16 diagnostic throughput improved relative to masked-argmax retry
    (`300.242 -> 305.275 tok/s`),but this is still diagnostic and not a
    release-grade performance claim;
  - c4 remains the known release-grade bottleneck:latest valid diagnostic is
    still `120.056 tok/s` vs same-hardware vLLM c4 baseline `154.169 tok/s`,
    ratio `0.779`,below 80% (`123.335 tok/s`).
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - run a targeted c4/c16 diagnostic on the sentinel-fix build,then either:
    c4 clears the 80% mean line and we move to release-grade N>=3/CI evidence,
    or c4 remains below target and we move to the next performance lever;
  - likely next lever remains model hot path/kernel profiling,not scheduler
    formation, because c16 batches are already reaching `avg_m≈13.6`.

## 2026-06-15 X — W2 masked-argmax mask diagnostic:定位到 GPU masked argmax 返回被 mask token

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_maskdiag_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。artifact 已同步
  回本地并停机;stop poll 记录:
  - `09:37:08 cur_state=stopped actual_status=running`;
  - `09:37:19 cur_state=stopped actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA masked-argmax forbidden-token diagnostic`;
  - expected runtime/cost:20-45min,stop cap 60min,1x RTX 4090 instance
    `40826362` at about USD 0.402/hr;
  - stop condition:startup/SSH/CUDA/build failure,`ferrum run` failure,serve
    smoke failure,c16 diagnostic completes,forbidden-token diagnostic is captured,or
    60min cap;
  - correctness gate:CUDA `argmax_rows` masked test,`ferrum run`,
    `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:diagnostic-only
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1` at c=16;
  - baseline:reuse previously saved same-hardware vLLM 0.10.1.1 c4 baseline
    `154.169 tok/s`;baseline not rerun in this diagnostic.
- Correctness before diagnostic bench:
  - CUDA `argmax_rows` masked test PASS;
  - release build PASS:`cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic result:
  - c16 diagnostic was manually stopped after the first forbidden-token warning;
  - server warning:
    `token_id=0`, `token_text="<pad>"`, `generated_tokens=126`,
    `forbidden_count=6380`, `base_vocab_size=Some(262144)`,
    `argmax_mask=...len=262144,value=0`;
  - health at stop confirms this was the typed masked-argmax path:
    `force_full_logits_calls=0`, `calls=391`, `total_items=5336`,
    `avg_items_per_call=13.647`, `max_items=16`, bucket `m9_16=379`;
  - GPU memory returned to 1 MiB after manual stop.
- Conclusion:
  - engine-side mask construction was correct for the returned token (`value=0`);
  - the remaining correctness bug is in the CUDA/model masked-argmax path returning
    a token that should have been excluded, specifically the no-finite-valid-token
    fallback/default behavior returning index 0.
- Follow-up source fix:
  - CUDA `argmax_rows_f16_masked` now ignores non-finite logits and returns
    sentinel `u32::MAX` (`-1` as i32) when no finite valid token exists;
  - `LlamaFamilyModel` falls back to full logits if any row returns the sentinel,
    preserving correctness instead of emitting a masked token id;
  - CUDA test added:
    `argmax_rows_f16_masked_returns_sentinel_without_finite_valid_token`;
  - local validation:
    `cargo fmt --all -- --check`,
    `cargo check -q -p ferrum-models --tests`,
    `cargo check -q -p ferrum-kernels --tests`,
    `cargo test -q -p ferrum-engine model_greedy_argmax_sentinel -- --nocapture`,
    `cargo test -q -p ferrum-engine model_decode_logits_policy -- --nocapture`,
    `git diff --check`。
- Release-grade status:
  - no `model_release_grade_manifest.json`,no
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 remains not release-grade.
- Next step:
  - run a minimal CUDA validation of the sentinel fix: CUDA `argmax_rows` test,
    `ferrum run`,serve smoke,and c16 diagnostic;
  - only after c16 stays clean should we resume c4/c16 performance work toward
    the 80% line.

## 2026-06-15 IX — W2 masked GPU argmax probe: c4 小幅改善,但 c16 暴露 forbidden-token 风险

- 本轮 artifacts:
  - 首次探针:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_probe_2026-06-15/`;
  - sentinel 修正后重试:
    `docs/goals/model-coverage-2026-06-12/artifacts/w2_masked_argmax_retry_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。两轮结束后均已同步
  artifact 并停机;stop poll 分别记录:
  - `08:53:13 cur_state=stopped actual_status=exited`;
  - `09:09:09 cur_state=stopped actual_status=exited`。
- 改动:
  - 引入 typed `LogitsReturnPolicy::GreedyArgmax` 与 `TokenSelectionMask`;
  - CUDA `argmax_rows_f16_masked` 支持 GPU 侧 masked greedy argmax;
  - Gemma/Llama-family batched decode 在确定性 greedy/text 输出路径上可避免
    full-logits readback;
  - engine 仍保留 product-side forbidden/initial/extended-vocab/output-quality 校验,
    不允许模型侧 argmax 绕过采样 mask。
- 本地验证:
  - `cargo fmt --all -- --check`;
  - `cargo check -q -p ferrum-interfaces --tests`;
  - `cargo check -q -p ferrum-engine --tests`;
  - `cargo check -q -p ferrum-models --tests`;
  - `cargo check -q -p ferrum-kernels --tests`;
  - targeted engine/model tests for logits policy, sentinel acceptance, decode stats;
  - `git diff --check`。
- GPU correctness:
  - CUDA `argmax_rows` masked test PASS;
  - CUDA `flash_attn_batched_eq` tests PASS;
  - retry build:`CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - retry `ferrum run` rc=0,content `"5"`;
  - retry serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- 首次探针结果:
  - `ferrum run` 与 serve smoke 通过后,c=4 bench 启动即出现
    `model returned greedy token sentinel for request requiring full logits`;
  - 随后 server log 记录 CUDA illegal-address failure,本轮停止并复制 artifact;
  - 结论是 sentinel 接受条件仍按旧 `requires_full_logits_for_sampling()` 判断,
    未识别 typed masked-greedy product policy。
- 重试结果(非 release evidence,N=1,无 CI):
  - c=4:`32 completed / 0 errored`, `120.056 tok/s`,
    output token count source:`usage`;
  - c=4 health stats:`force_full_logits_calls=0`,
    `avg_items_per_call=3.595`,`max_items=4`;
  - 相比 sliding-window probe c=4 `117.172 tok/s`,增量约 `+2.5%`;
  - 相比 same-hardware vLLM c=4 baseline `154.169 tok/s`,ratio 约 `0.779`,
    仍低于 W2 80% 目标。
- c16 风险:
  - retry 进入 c=16 后 server log 出现 124 次
    `model greedy argmax returned a forbidden token`;
  - 虽然 artifact 中有 c=16 诊断 JSON,本轮已按 first-triage 原则停止,不把
    c=16 作为有效性能证据;
  - 下一步必须先定位 forbidden token id/mask 来源,再继续性能验证。
- 本地 follow-up:
  - `accept_model_greedy_argmax_token` 错误已补充 token id、token text、
    decoded delta、generated token 数、forbidden/initial-forbidden 数量、
    base vocab size 和 allowed-extended 数量;
  - 新增 targeted test 断言 forbidden-token 错误包含关键诊断字段;
  - 验证命令:
    `cargo fmt --all -- --check`,
    `cargo check -q -p ferrum-engine --tests`,
    `cargo test -q -p ferrum-engine model_greedy_argmax_sentinel -- --nocapture`,
    `cargo test -q -p ferrum-engine model_decode_logits_policy -- --nocapture`,
    `git diff --check`。
- 发布级判定:
  - 未生成 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未完成。
- 下一步:
  - 用新增诊断跑一次小 CUDA 探针,确认 masked argmax 是否仍返回被 mask token,
    或是 per-sequence mask/fingerprint 选择错误;
  - 正确性修复后还需要继续找至少约 `3.3 tok/s` 的 c=4 增量,才能越过
    `0.80 * 154.169 = 123.335 tok/s`。

## 2026-06-15 VIII — W2 sliding-window batched attention: c16 明显改善,c4 仍未达 80%

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_sliding_batched_attn_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。结束后已同步
  artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。停止前 `nvidia-smi` 显示无运行进程,GPU memory 1 MiB。
- 改动:
  - CUDA head-major single decode attention 增加 `sliding_window` 实现;
  - CUDA `flash_attention_batched_per_cache` 增加 common `sliding_window` 参数;
  - Gemma3 local-window 层不再强制 per-item attention fallback,而是通过 batched
    attention kernel 处理 local window;
  - 新增 CUDA test:
    `flash_attn_batched_sliding_window_one_selects_latest_v`。
- 本地验证:
  - `cargo fmt --all -- --check`;
  - `cargo check -q -p ferrum-models --tests`;
  - `cargo test -q -p ferrum-models decode_batch_stats_snapshot_records_shape_and_fallbacks -- --nocapture`;
  - `cargo check -q -p ferrum-kernels --tests`;
  - `git diff --check`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA sliding-window batched-attn probe`;
  - expected runtime/cost:45-75min,stop cap 90min / 约 USD 0.65;
  - stop condition:CUDA kernel test 失败、`ferrum run`/serve smoke 失败、c=4/c=16
    诊断完成,或达到 90min;
  - correctness gate:
    `cargo test --release -p ferrum-kernels --features cuda --test flash_attn_batched_eq -- --nocapture`,
    再跑 `ferrum run` 与 `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:诊断型
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1`
    分别跑 c=4 与 c=16。
- CUDA kernel test:
  - `flash_attn_batched_matches_per_item`:max diff `3.052e-5`;
  - `flash_attn_batched_sliding_window_one_selects_latest_v`:max diff `1.206e-4`;
  - `test result: ok. 2 passed`。
- Product correctness:
  - release build:`CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0,content `"5"`;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench 结果(非 release evidence,N=1,无 CI):
  - c=4:`32 completed / 0 errored`, `117.172 tok/s`;
  - c=16:`32 completed / 0 errored`, `245.801 tok/s`;
  - output token count source:`usage`。
- 对比上一轮 stats probe:
  - c=4:`105.050 -> 117.172 tok/s`,约 `+11.5%`;
  - c=16:`177.364 -> 245.801 tok/s`,约 `+38.6%`;
  - c=16 段增量仍形成大 batch:`avg_m=13.677`,bucket `m9_16=379/390`。
- 发布级判定:
  - 同硬件 vLLM c=4 baseline 为 `154.169 tok/s`;本轮 Ferrum c=4 诊断 ratio
    约 `0.760`,仍低于 80%;
  - c=16 vLLM baseline 仍因 invalid UTF-8 不能作为 release-grade evidence,但
    诊断 ratio 约 `245.8 / 381.4 = 0.645`;
  - 未生成 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 仍未完成。
- 下一步:
  - 继续一个窄性能 lever,优先用 `FERRUM_DECODE_OP_PROFILE` 或等价可记录 artifact
    确认剩余 c=4 gap 是否主要在 full-logits/lm_head readback、qkr/kv append、
    MLP/GEMM 小 m 效率,或 attention 本身;
  - 不重复 full sweep,直到 c=4 diagnostic ratio 明确越过 80% 或定位出下一个
    高收益修复。

## 2026-06-15 VII — W2 decode batch stats probe: c16 已形成大 batch,瓶颈转向模型/kernel 路径

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_decode_batch_stats_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。结束后已同步
  artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。停止前 `nvidia-smi` 显示无运行进程,GPU memory 1 MiB。
- 本轮新增 source instrumentation:
  - `LlamaFamilyModel` 通过 `/health.cache.prefix_cache.decode_batch` 暴露
    decode_batch 调用数、total rows、max m、m bucket、fallback 计数;
  - 只记录 metrics,不改变 scheduler/model/kernel/sampling 行为;
  - 本地验证:
    `cargo test -q -p ferrum-models decode_batch_stats_snapshot_records_shape_and_fallbacks -- --nocapture`,
    `cargo fmt --all -- --check`,
    `cargo check -q -p ferrum-models --tests`,
    `git diff --check`。
- GPU 执行合同:
  - lane:`W2 Gemma3 CUDA batched-decode stats probe`;
  - expected runtime/cost:35-60min,stop cap 90min / 约 USD 0.60;
  - stop condition:启动/SSH/构建失败、`ferrum run`/serve smoke 失败、c=16
    bench 出错、或采集完 health stats;
  - correctness gate:`ferrum run` + `scripts/model_coverage_smoke.sh gemma3:27b-gptq`;
  - performance command:诊断型
    `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1`
    分别跑 c=4 与 c=16。
- Correctness:
  - release build:`CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - `ferrum run` rc=0;
  - serve smoke PASS line:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Diagnostic bench 结果(非 release evidence,N=1,无 CI):
  - c=4:`32 completed / 0 errored`, `105.050 tok/s`;
  - c=16:`32 completed / 0 errored`, `177.364 tok/s`;
  - output token count source:`usage`。
- Decode batch stats:
  - c=4 段后累计:`calls=1422,total_items=5334,avg_m=3.751,max_m=4`;
    bucket:`m1=13,m2=151,m3_4=1258`;
  - c=16 段增量:`calls=391,total_items=5334,avg_m=13.642,max_m=16`;
    bucket:`m1=1,m3_4=4,m5_8=8,m9_16=378`;
  - fallback:`unsupported_fallback_calls=0,lora_fallback_calls=0`;
  - server log 首个 batched decode:`m=4 use_batched_qkr=true`,
    `batched-kv-append ok=true`, `batched-attn ok=true`。
- 结论:
  - c=16 扩展性差不是 scheduler 没形成大 batch;闭环 c=16 段实际平均 m≈13.6,
    绝大多数调用在 m=9..16;
  - 下一步应转向模型/kernel hot path,尤其是 Gemma local-window 层是否仍大量
    per-item attention、full-logits readback/sampling、以及 per-layer qkr/attention/MLP
    profile;
  - W2 仍未完成:没有 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。

## 2026-06-15 VI — W2 CUDA12 vLLM baseline probe:server 可用,但 release-grade 失败

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm0101_cuda12_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。结束后已同步
  artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA vLLM 0.10.1.1/CUDA12 baseline probe`;
  - expected runtime/cost:45-120min,约 USD 0.32-0.85;
  - stop condition:torch CUDA smoke 失败、vLLM Gemma3/GPTQ 不支持并保存日志、
    vLLM OpenAI smoke + baseline 完成,或任一 baseline cell 非零错误;
  - correctness gate:`torch.cuda` smoke,再走 vLLM OpenAI
    `/v1/chat/completions` 非空内容 + usage;
  - performance command:`ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- 环境与依赖:
  - GPU:`NVIDIA GeForce RTX 4090,24564 MiB,driver 565.77`;
  - `vllm==0.10.1.1`, `torch==2.7.1+cu126`;
  - 初始 pip 解析到 `transformers 5.12.0`,导致 Gemma3TextConfig 的 nested
    rope_scaling 与 vLLM 0.10.1.1 不兼容:
    `rope_scaling should have a 'rope_type' key`;
  - pin `transformers==4.55.4` 后模型 config smoke 通过;
  - 初始 pip 解析到 `fastapi 0.137.0` / `starlette 1.3.1` /
    `prometheus-fastapi-instrumentator 8.0.0`, `/v1/models` 触发
    `'_IncludedRouter' object has no attribute 'path'`;
  - pin `fastapi==0.116.1`, `starlette==0.47.2`,
    `prometheus-fastapi-instrumentator==7.1.0` 后 `pip check` clean。
- vLLM product-path smoke:
  - server 成功加载同一 HF/safetensors GPTQ model,日志显示
    `Resolved architecture: Gemma3ForCausalLM` 与
    `Using gptq_marlin kernel`;
  - `/v1/models` rc=0;
  - `/v1/chat/completions` 非流式 smoke rc=0,返回非空 content 且 usage 含
    completion tokens。
- Baseline 结果:
  - c=1: vLLM `43.486 tok/s`,Ferrum `40.021 tok/s`,mean ratio `0.920`;
  - c=4: vLLM `154.169 tok/s`,Ferrum `105.158 tok/s`,mean ratio `0.682`;
  - c=16: vLLM 两次 N=3 rerun 都在第三轮产生
    `bad output invalid-utf8: �"`;`--fail-on-error` 非零,因此 c=16 不能作为
    release-grade baseline evidence。诊断均值约 `381.4 tok/s`,Ferrum c=16
    `165.469 tok/s`,mean ratio 约 `0.434`。
- 发布级判定:
  - 有效 c=4 same-hardware mainstream baseline 已证明当前 Ferrum 低于 80%;
  - vLLM c=16 baseline 自身没有零错误,仍不能进入 final manifest;
  - 未生成 `model_release_grade_manifest.json`,没有
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 release-grade 仍未完成。
- 下一步:
  - 继续 W2-P2 性能修复,重点缩小 c=4/c=16 gap,不能用 hidden env 或 sampler
    参数绕过输出问题;
  - 优先审计 Gemma3 batched decode 是否实际形成 m=4 以上的 decode batch、
    local/sliding attention 是否仍强制小 batch fallback、以及 per-layer residual/
    norm/GeGLU 是否还有 host sync 或重复 materialize;
  - 下一轮 CUDA 只跑 targeted A/B 或 smoke,不要重复 full sweep,直到有明确
    高收益改动。

## 2026-06-15 V — W2 baseline probe:latest vLLM 0.23.0 安装成功但 CUDA13/driver565 不可用

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_baseline_probe_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。验证结束后已
  复制 artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA vLLM/GPTQ baseline probe`;
  - expected runtime/cost:30-90min,约 USD 0.21-0.64;
  - stop condition:vLLM 明确不支持该模型/量化并保存日志、vLLM smoke 通过并完成
    baseline、安装/启动超过 90min 无进展,或任一 baseline cell 非零错误;
  - correctness gate:vLLM OpenAI `/v1/chat/completions` 简单问题返回有效非空内容;
  - performance command:通过 smoke 后用
    `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- 环境:
  - GPU:`NVIDIA GeForce RTX 4090,24564 MiB,driver 565.77`;
  - CUDA compiler:`cuda_12.4.r12.4`;
  - Python:`3.10.12`;
  - 初始环境无 `vllm`/`torch`,Gemma3 GPTQ cache 存在。
- 安装过程:
  - 初次 `python3 -m venv /workspace/vllm-venv` 失败,因为镜像缺
    `python3.10-venv`;
  - 安装 `python3.10-venv` 后重建 venv;
  - `pip install vllm` 成功,解析到 `vllm 0.23.0` 与 `torch 2.11.0`;
  - 该 torch wheel 依赖 CUDA13 运行时包,包括 `cuda-toolkit 13.0.2`,
    `nvidia-cublas 13.1.0.3`, `nvidia-cudnn-cu13 9.19.0.56`,
    `nvidia-nccl-cu13 2.28.9` 等。
- CUDA smoke:
  - 命令:`/workspace/vllm-venv/bin/python import_smoke`;
  - 失败位置:`torch.cuda.get_device_name(0)` 触发 `_cuda_init()`;
  - 错误:
    `RuntimeError: The NVIDIA driver on your system is too old (found version 12070)`;
  - 因此 latest vLLM 0.23.0/CUDA13 wheel 栈不能在当前 driver 565.77 机器上作为
    W2 same-hardware baseline。
- 发布级判定:
  - 本轮没有生成 baseline throughput,没有 `model_release_grade_manifest.json`,
    没有 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - W2 release-grade 仍未完成。
- 下一步:
  - 不再继续 latest vLLM/CUDA13 路线,除非先更换/升级同硬件 driver;
  - 在同一 4090/cache-retained instance 上尝试 CUDA12 兼容的 vLLM 版本
    (独立 venv,先 `torch.cuda` smoke,再 vLLM server smoke);
  - 若 CUDA12-compatible vLLM 也不支持 Gemma3 GPTQ,保存明确模型/量化不支持证据,
    再选择 `RELEASE_GRADE_GOAL.md` 允许的最快同模型同格式 mainstream engine。

## 2026-06-15 IV — W2 Ferrum release-shape 全矩阵 PASS,release-grade 仍缺 mainstream baseline

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_release_shape_ferrum_cuda_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。验证结束后已
  复制 artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- GPU 执行合同:
  - lane:`W2 Gemma3-27B CUDA Ferrum release-shape matrix`;
  - expected runtime/cost:1.5-3h,约 USD 0.64-1.28;
  - stop condition:correctness 首败、任一 release-shape cell 非零错误、全矩阵
    完成并回收 artifact,或 3h;
  - correctness gate:`ferrum run` + `ferrum serve` smoke;
  - performance command:`bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
    覆盖 c=1/4/16/32。
- 远端源码/硬件:
  - git SHA `2656cc1a4c1b4f722f14700a5e50d4e0af37db14`;
  - 远端 dirty status 保存于 `remote_metadata.txt`;
  - GPU:`NVIDIA GeForce RTX 4090,24564 MiB,driver 565.77`;
  - CUDA compiler:`cuda_12.4.r12.4`;
  - Rust:`cargo 1.96.0`, `rustc 1.96.0`。
- CUDA release build:
  - `CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`
    PASS;
  - binary SHA256:
    `5fa06ab8dc93285bccca692702d5386bfbb39a8a6ba3e8e6b66a2467ee99c6b8`。
- Correctness/product path:
  - `ferrum run gemma3:27b-gptq --backend cuda ... --kv-capacity 2560 --max-num-seqs 2`
    rc=0,输出 `content:"5"`, `finish_reason:"stop"`;
  - `scripts/model_coverage_smoke.sh gemma3:27b-gptq --port 8401 --kv-capacity 2560 --max-seqs 2`
    rc=0,stdout 打印 `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`。
- Ferrum release-shape L5:
  - c=1:`40.0214 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=4:`105.1577 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=16:`165.4689 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=32 typed cap16 (`--kv-capacity 400 --max-num-seqs 16`):
    `169.4372 tok/s`,completed `[100,100,100]`,errored `[0,0,0]`,
    usage count,n_repeats=3;
  - c=32 server log 首个并发 decode 为
    `[batched-qkr] first batched_decode call: m=4 use_batched_qkr=true`,
    证明 release-shape 并发路径实际触发 legacy batched decode。
- 发布级判定:
  - Ferrum 侧 release-shape correctness/perf matrix 已干净,且相对上一轮
    flat 40 tok/s 有明确 c=4/16/32 提升;
  - 仍未补同硬件、同模型、同量化/格式的 mainstream baseline,也未生成
    `model_release_grade_manifest.json` 与最终
    `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - 因此 W2 仍不能宣称 release-grade。
- 下一步:
  - 在同一 RTX 4090 上优先尝试 vLLM/GPTQ baseline;若当前 vLLM 不支持
    Gemma3 27B GPTQ,必须保存不支持证据并选择目标文档允许的最快同模型同格式
    mainstream engine;
  - baseline 必须用同一 prompt/cell、同一 effective active cap(c=32 cap16 时),
    并保存 engine version/build/runtime config;
  - 之后生成 `model_release_grade_manifest.json` 并运行
    `python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>`。

## 2026-06-15 III — W2-P2 legacy batched decode CUDA 诊断:correctness PASS,并发路径已触发

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_legacy_batched_cuda_2026-06-15/`。
- 复用 Vast/cache-retained CUDA instance `40826362`,1x RTX 4090。验证结束后已
  复制 artifact 并停机;Vast API poll 记录 `cur_state=stopped`,
  `actual_status=exited`。
- 远端源码状态:
  - git SHA `2656cc1a4c1b4f722f14700a5e50d4e0af37db14`;
  - 远端 `git status --short` 保存于 `remote_metadata.txt`,包含本轮源码改动与既有
    未跟踪 W2 artifacts;
  - 为减少付费 GPU 空转,第二次 rsync 排除了历史 `docs/.../artifacts/`;
    本轮是 diagnostic,不是最终 release-grade artifact collection。
- CUDA release build:
  - `CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`
    PASS;
  - binary SHA256:
    `5fa06ab8dc93285bccca692702d5386bfbb39a8a6ba3e8e6b66a2467ee99c6b8`。
- Correctness/product path:
  - `ferrum run gemma3:27b-gptq --backend cuda ... --kv-capacity 2560 --max-num-seqs 2`
    rc=0,输出 `content:"5"`, `finish_reason:"stop"`;
  - `scripts/model_coverage_smoke.sh gemma3:27b-gptq --port 8401 --kv-capacity 2560 --max-seqs 2`
    rc=0,stdout 打印 `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`,覆盖
    known-answer 10/10,multi-turn,stream==non-stream,custom stop,tool-call,
    strict json_schema;
  - run/serve 日志均显示 `Gemma3 family: legacy batched_decode=true varlen_unified=false`。
- Diagnostic perf/correctness:
  - typed serve:`--kv-capacity 512 --max-num-seqs 16`;
  - `bench-serve --random-input-len 256 --random-output-len 128 --concurrency-sweep 4,16 --num-prompts 32 --n-repeats 1 --fail-on-error --seed 9271`;
  - c=4:completed `[32]`,errored `[0]`,usage count,105.5 tok/s;
  - c=16:completed `[32]`,errored `[0]`,usage count,177.3 tok/s;
  - server log 首个并发 decode 为
    `[batched-qkr] first batched_decode call: m=4 use_batched_qkr=true`,证明本轮
    legacy batched decode 窄开关在 CUDA 并发路径上实际触发。
- 发布级判定:
  - 未运行 `--require-ci --n-repeats 3` 全矩阵,未跑 c=32,未补同硬件主流引擎
    80% baseline,未生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`;
  - 因此 W2 仍是 functional/diagnostic 进展,不能宣称 release-grade。
- 下一步:
  - 在相同路径上跑 release-grade 形状的 c=1/4/16/32 correctness/perf,并补 baseline;
  - 若 c=32 仍需 active admission cap,release-grade manifest 必须把 effective
    concurrency/cap 与 baseline 对齐;
  - 继续审计 `batched-attn m=4 ok=true` 是否在 local-window 层走了预期 per-item
    fallback,以及是否还有 Gemma3 tail/attention 可融合热点。

## 2026-06-15 II — W2-P2 legacy batched decode 窄开关:CUDA 候选,待 GPU 验证

- 本轮仍未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - Gemma/sandwich family 的 `supports_batched_decode` 不再无条件禁用;
  - 窄开关条件为: sandwich norms + 非零 `sliding_window_pattern` + 后端支持
    device-side F32 residual shadow。当前目标是只让 CUDA Gemma3 进入 legacy
    contiguous batched decode 候选路径;
  - `supports_varlen_qkv` 对 sandwich 仍禁用,因为 paged/unified attention 还没有
    per-layer local-window 语义;
  - active LoRA cache 继续 fallback 到 per-item decode,因为 legacy batched qkv/o/gate/down
    仍不携带 per-cache LoRA adapter;
  - layer-split pipeline 的 batch stage 对 sandwich family 继续 fallback,避免 full-model
    decode_batch capability 影响未验证的 pipeline hidden path。
- 新增/更新测试:
  - `sandwich_legacy_batched_decode_requires_device_shadow_and_layer_schedule` 锁定
    capability 条件,避免后续扩大到无 device shadow 或无 Gemma layer schedule。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `git diff --check` PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models sandwich_legacy_batched_decode_requires_device_shadow_and_layer_schedule -- --nocapture` PASS;
  - `cargo test -q -p ferrum-models --lib` PASS,123 tests passed;
  - `cargo check -q -p ferrum-cli --all-targets` PASS.
- 下一步:
  - 必须在 1x RTX 4090 上跑 CUDA correctness smoke,确认 `ferrum run` 和
    `ferrum serve` 在新 legacy batched decode 路径下无 `<unk>`/`[PAD]`/
    NaN/stream DONE 问题;
  - 通过 correctness 后再跑短 c=4/16 diagnostic,观察吞吐是否不再完全平坦。

## 2026-06-15 — W2-P2 batched decode 语义铺垫:共享 Gemma tail,fast path 仍禁用

- 本轮未启动 GPU,没有新增 release-grade artifact,也没有生成
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 代码侧推进:
  - `LlamaFamilyModel` 抽出 `forward_layer_post_o_proj_with_residual_shadow`,
    让单序列 post-attn tail 与 legacy batched decode 在 o_proj 后共享同一套
    sandwich norm / GeGLU / device-F32 residual shadow 语义;
  - legacy batched decode layer 现在可从 host/device residual shadow 做 input
    RMSNorm,并把 Gemma embedding scale、shadow 初始化、final norm、shadow 归还
    接入 `decode_batch_internal_with_full_logits`;
  - legacy contig batched decode attention 现在按 source layer 选择 Gemma3
    local/global rope 与 `layer_window`;由于 single-launch batched attention
    kernel 尚无 sliding-window 参数,local-window 层会强制走 per-item attention
    fallback,保留正确性语义;
  - batched CUDA graph 在 host/device residual shadow 路径下保持禁用,避免 graph
    replay 在未验证的 Gemma shadow 状态上成为隐式产品路径。
- 仍未完成:
  - `supports_batched_decode` / `supports_varlen_qkv` 对 `sandwich_norms` 仍保持
    false,所以用户路径仍走已验证的 per-item decode;
  - paged/unified attention kernel API 仍未完整支持 Gemma3 per-layer local-window
    语义;unified tail 仍需单独接入 sandwich/shadow 语义;
  - 因此这只是 W2-P2 的语义地基,不是性能通过声明。
- 本地验证:
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-models --tests` PASS;
  - `cargo test -q -p ferrum-models --lib` PASS,122 tests passed。

## 2026-06-14(发布级 II)— W2 device-side F32 shadow:正确性 PASS,诊断 L5 提升,仍非 release-grade

- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_release_grade_device_shadow_cuda_2026-06-14/`。
- 复用 Vast/cache-retained CUDA instance `40826362`。验证结束后已复制 artifact 并
  停机;Vast API poll 记录 `cur_state=stopped`,`actual_status=exited`。
- 代码修正范围:
  - Gemma3 CUDA sandwich-norm 路径改为 device-side F32 residual shadow,避免每层
    host F32 shadow readback/copy;
  - `common.cuh` 的 block reduce helper 修正 `blockDim < 32` 时的
    `num_warps=(blockDim.x+31)/32`,否则小尺寸 CUDA precision tests 会把 variance
    归零并放大到 `rsqrt(eps)`;
  - `cuda/quant.rs` 的空 env-var test fixture 加显式类型,只影响测试编译。
- CUDA feature build/test:
  - `CUDA_COMPUTE_CAP=89 cargo check -q -p ferrum-kernels --tests --features cuda`
    PASS;
  - `CUDA_COMPUTE_CAP=89 cargo test -q -p ferrum-kernels --test cuda_activation_precision --features cuda`
    PASS,4 tests passed;
  - release binary command:
    `CUDA_COMPUTE_CAP=89 cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2`;
  - binary SHA256:
    `3af53becc860a5e038cda486da69de4fc5aa6e8d81543d04aba0dbebbe6a393f`.
- 产品 correctness:
  - `ferrum run gemma3:27b-gptq --backend cuda --prompt ... --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl`
    PASS:assistant content `5`,finish_reason `stop`,n_tokens `3`;
  - `ferrum serve` smoke PASS:
    `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`,覆盖 known-answer 10/10,
    natural EOS,multi-turn,stream/non-stream,custom stop,max_tokens,
    tool-call 10/10,strict json_schema 20/20。
- Ferrum post-device-shadow diagnostic L5:
  - c=1:`40.5367 tok/s`,100/100/100 completed,0 errors;
  - c=4:`40.4754 tok/s`,100/100/100 completed,0 errors;
  - c=16:`40.3856 tok/s`,100/100/100 completed,0 errors;
  - c=32 typed cap16 (`--kv-capacity 400 --max-num-seqs 16`):
    `40.3111 tok/s`,100/100/100 completed,0 errors.
- 对旧同卡 llama.cpp GGUF sanity baseline `50.478 tok/s` 的诊断 ratio:
  c=1 `0.8031`,c=4 `0.8018`,c=16 `0.8001`,c=32 `0.7986`。
  这些数字只能说明 device-side F32 shadow 把旧 host-shadow 的约 `25 tok/s`
  提升到约 `40 tok/s`;按 `RELEASE_GRADE_GOAL.md`,跨格式 llama.cpp 不能作为
  CUDA GPTQ 正式 80% baseline。
- 当前结论:
  - W2 Gemma3 CUDA GPTQ correctness/product path 已保持 PASS;
  - c=1/4/16/32 Ferrum diagnostic L5 已干净,但并发吞吐仍基本平坦,说明
    batched/varlen Gemma3 fast path 仍是 release-grade 性能工作项;
  - 未生成 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`,W2 仍不得宣称
    release-grade。

## 2026-06-14(发布级 I)— release-grade gate 与发布口径收窄

- 新增发布级目标文档 `RELEASE_GRADE_GOAL.md`:明确 W2 coverage PASS 不等于
  release-grade,后续 README/release notes/性能宣传以 80% 主流引擎 baseline
  为硬门。
- 新增 validator:
  `scripts/release/model_release_grade_goal_gate.py`。
  - 命令:`python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>`
    或 `w3 <out_dir>`;
  - 输入:`<out_dir>/model_release_grade_manifest.json`;
  - 输出:`model_release_grade_goal_gate.manifest.json`;
  - 只有 stdout 打印 `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` 或
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>` 才能宣称本文目标完成。
- validator self-test 已覆盖:
  - 合格 W2 manifest 可 PASS;
  - ratio `0.79` 会 FAIL;
  - `runtime_config.hidden_env` 非空会 FAIL。
- README/README_zh 已把 Gemma 3 27B CUDA 口径从 "certified/release-like"
  收窄为 "functional / known-gap":当前同卡 llama.cpp ratio `0.500260x`,
  低于 release-grade `0.8x`,因此不能写成 release-grade 支持。
- W2-P2 预备修正:
  - CUDA `Backend::rms_norm` 现在按 buffer dtype 分派 `rms_norm_f16` /
    `rms_norm_f32`,不再无条件调用 F16 kernel;
  - 新增 `cuda_activation_precision::f32_rms_norm_uses_f32_kernel`,为后续
    device-side F32 residual shadow / sandwich-norm 路径铺路;
  - 这不是 release-grade 性能修复,也没有启用 Gemma3 batched/varlen fast path。
- 本轮本地验证:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS;
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS:`MODEL RELEASE GRADE GOAL SELFTEST PASS`;
  - `python3 scripts/release/selftest_g0_validators.py`
    PASS:`G0 VALIDATOR SELFTEST PASS`;
  - `cargo fmt --all -- --check` PASS;
  - `cargo check -q -p ferrum-kernels --tests` PASS;
  - `cargo test -q -p ferrum-kernels --test cuda_activation_precision`
    PASS locally with 0 tests because CUDA feature is not enabled on this host.

## 2026-06-14(早 VI)— W2 L5/perf 打通:客户端 c=32 + admission cap 16 PASS

- 继续复用 stopped/cache-retained native CUDA instance `40826362`,没有重新租
  pod 或重装环境。全部验证结束后已复制 artifact 并停机,Vast API 确认
  `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_admission16_l5_pass_2026-06-14/`。
- L5 c=32 最终产品合同:
  - 客户端并发仍是 W2 要求的 `bench-serve --concurrency-sweep 32`;
  - 服务端使用 typed product CLI `--max-num-seqs 16 --kv-capacity 400`;
  - 这是产品入口参数,不是隐藏 env。health/auto_config 记录
    `selected_admission_limit=16`。
- 先跑小探针 `num_prompts=32,n_repeats=1`:32/32 completed,0 errored,
  throughput 24.9 tok/s。随后跑正式门:
  `num_prompts=100,n_repeats=3,--fail-on-error,--require-ci,--seed 9271`。
- 正式 c=32 结果:
  - repeat 1:`100 completed / 0 errored / 521.3s`
  - repeat 2:`100 completed / 0 errored / 515.1s`
  - repeat 3:`100 completed / 0 errored / 514.0s`
  - JSON: `full/full/l5_gemma3-27b-gptq_cuda_c32_admission16.json`;
  - merged L5 JSON:
    `l5_gemma3-27b-gptq_cuda.json`,覆盖 c=1/4/16/32。
- llama.cpp same-card perf:
  - 远端缺 `llama-bench` binary,保留源码/缓存;为避免全架构 CUDA 编译浪费,
    中断了误启动的全架构 build,重新配置 `build-sm89` 只编译
    `CMAKE_CUDA_ARCHITECTURES=89`;
  - 命令:`llama-bench -m <Gemma3-27B-Q4_K_M.gguf> -ngl 999 -p 0 -n 128 -r 3 -o json`;
  - llama.cpp tg128:50.478285 tok/s;
  - Ferrum c=1 decode:25.252275 tok/s;
  - ratio:0.500260 PASS,刚过 0.5 floor,应记录为 known-gap,不是性能优化完成声明。
- 当前结论:
  - W2 矩阵 8/8 可满足;待 validator exact PASS 作为最终 W2 完成证据。
  - W3 仍只交付立项合同,不在本目标内宣称 W3 完成。

## 2026-06-14(凌晨 V)— W2 c=32 admission cap 31/30 验证:排队不足以解除 24GB OOM

- 按用户建议继续复用 stopped/cache-retained native CUDA instance
  `40826362`,没有重新开机器或重装环境。验证/诊断完成后已复制 artifact 并停机,
  Vast API 确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_admission_cap_diagnostics_2026-06-14/`。
- 产品路径 admission-cap 复测:
  - 客户端仍保持 W2 L5 要求的 `bench-serve random 256/128 c=32`
    三轮命令形态;
  - 服务端分别使用 typed CLI 参数 `--max-num-seqs 31` 和
    `--max-num-seqs 30`,并保持 `--kv-capacity 400`;
  - 两轮均 health 第 60 秒通过,进入
    `closed_loop c=32 — repeat 1/3`,随后 OOM。
- 两个 cap 的 OOM 同形:
  `CudaBackend::alloc failed: dtype=F16 elements=804864 bytes=1609728 free=17498112 total=25278087168`.
  也就是 unified product-path KV hint 已把单 cache 物理容量降到
  `393 * 16 * 128`,但 24GB 总显存仍不足。
- 附加 scheduler 诊断(非 PASS 证据):开启 `FERRUM_SCHED_NONE_PROF=1`
  做短 c=32 trace,拿到第一条
  `[sched-some] n=0 returning_batch=4 | decode_queue=0 prefill_queue=6 waiting_queue=0`,
  说明首个 iteration 不是直接一次性塞满 30/32 个请求;当前 blocker 更像是
  活跃请求累计 + Gemma3-27B fp16 contiguous KV 生命周期/总峰值问题,不是简单的
  首批 admission 数过大。
- 当前结论:
  - W2 仍不是 PASS。矩阵保持 l0/l1/l2_gptq/l3/l4 pass,
    l2_gguf waived,l5_concurrency fail,perf_vs_llamacpp pending。
  - `--max-num-seqs 31/30` 不能作为 L5 绕过;继续往下盲调 cap 价值低。
  - 下一步应在本地先定位 KV 生命周期/释放/复用或 Gemma3 paged-window 结构方案,
    再用 GPU 做最小验证;否则需要目标合同明确修订 L5。

## 2026-06-14(凌晨 IV)— W2 c=32 KV-hint 产品路径验证:hint 生效但 24GB 仍不足

- 按用户建议继续复用同一台 stopped/cache-retained native CUDA instance
  `40826362`,没有重新开机器或重装环境。验证结束后已复制 artifact 并停机,
  Vast API 确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_kv_hint_diagnostics_2026-06-14/`。
- 代码侧保留的产品路径改动:
  - engine 在请求 metadata 中写入 `ferrum_kv_capacity_hint`;
  - `LlmExecutor` 的 `prefill`/`batch_prefill`/`unified_decode` 在 fresh
    prefill 前调用 `prepare_kv_capacity`;
  - contiguous fp16 KV 支持按 hint 分配物理容量,逻辑 `kv_capacity`
    仍用于上下文校验;
  - hint 使用实际会写入 KV 的上限:`input_tokens + max_tokens - 1`
    (prefill 后采样出的首 token 不写入 KV)。
- native CUDA c=32 结果:
  1. 初版只覆盖 `prefill`/`batch_prefill`,但 W2 serve 走
     `unified_decode`;c=32 仍按 400 分配并 OOM:
     `elements=819200 = 400 * 16 * 128`。
  2. 补上 unified prefill hook 后,hint 生效;失败分配降为
     `elements=806912 = 394 * 16 * 128`,但仍 OOM。
  3. 修正 off-by-one 后,失败分配继续降为
     `elements=804864 = 393 * 16 * 128`,但仍 OOM;OOM 时 `free`
     仍约 17.5MiB,说明不是 prompt inflation 或 hook 缺失,而是 fp16
     contiguous KV c=32 峰值在 24GB 4090 上仍压线失败。
- 当前结论:
  - W2 仍不是 PASS。矩阵保持 l0/l1/l2_gptq/l3/l4 pass,
    l2_gguf waived,l5_concurrency fail,perf_vs_llamacpp pending。
  - `--kv-dtype int8` 已在上一轮证明 correctness 不可接受;不能作为绕过。
  - 后续若继续推进,需要结构性内存方案(例如可验证的 Gemma3 paged-window
    支持、正确的 KV quant、或经目标文档批准的 L5 合同调整),不是继续重复
    同一 c=32 fp16 KV 命令。

### 本轮验证

- Local:
  - `cargo fmt --all`
  - `cargo test -q -p ferrum-engine model_decode_metadata_marks_structured_requests_for_full_logits -- --nocapture`
  - `cargo test -q -p ferrum-models unified_decode_prepares_fresh_prefill_kv_capacity_hint -- --nocapture`
  - `cargo test -q -p ferrum-models unified_decode_full_logits_prefill_prepares_kv_capacity_hint -- --nocapture`
  - `cargo test -q -p ferrum-models contiguous_kv_capacity_hint_sizes_physical_cache -- --nocapture`
  - `cargo check -q -p ferrum-models --tests`
  - `git diff --check`
- Remote CUDA:
  - `kv_hint_c32_initial/build.log`: release CUDA build PASS in 3m26s.
  - `kv_hint_unified_c32/build.log`: release CUDA build PASS in 3m26s.
  - `kv_hint_actual_c32/build.log`: release CUDA build PASS in 3m12s.

## 2026-06-14(凌晨 III)— W2 c=32 根因收敛:Gemma3-27B fp16 KV 在 24GB 4090 上无法满足 c=32/400

- 按用户建议继续复用 stopped/cache-retained native CUDA instance `40826362`,
  未新租机器、未重装环境。全部诊断完成后已复制 artifact 并停机,Vast API
  确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_kv_allocator_diagnostics_2026-06-14/`。
- 代码侧做了两个诊断/减峰尝试并本地验证:
  - CUDA allocator OOM 日志增加 thread-local allocation label;
  - `batch_logits` 从 prefill scratch 中拆成 lazy grow,避免无谓保留
    `seq_len * vocab` logits;本地
    `prefill_scratch_keeps_batch_logits_lazy` PASS。
- native CUDA c=32 复测结论:
  1. `kv-capacity=400/max-num-seqs=32` 仍 OOM,失败 allocation
     `elements=819200 = 400 * 16 kv_heads * 128 head_dim`,确认是
     Gemma3 contiguous KV cache 单个 K/V layer buffer,不是 scratch/logits。
  2. `kv-capacity=396` 仍 OOM,失败 allocation
     `elements=811008 = 396 * 16 * 128`;`kv=384` 已在上一轮证明会
     context validation fail,所以单靠继续压 KV capacity 不可行。
  3. prefill scratch 收缩实验仍在同一 KV allocation OOM,已撤回,不作为产品改动保留。
  4. 短窗口 paged fp16 实验把失败转移为单个 paged pool allocation
     `elements=26214400`/约 52MiB,仍因 free 约 23MiB OOM;该实验也已撤回。
  5. `--kv-dtype int8` 能启动 CUDA paged INT8 KV,但 sanity known-answer
     第一题输出乱码(`'ㄝ Task sera exquisiteFolrbatovski'`),correctness 不可接受,
     不能作为 W2 L5 绕过方案。
- 当前结论:
  - W2 仍不是 PASS。矩阵保持 l0/l1/l2_gptq/l3/l4 pass,
    l2_gguf waived,l5_concurrency fail,perf_vs_llamacpp pending。
  - c=32 blocker 已从"未知 OOM"收敛为"Gemma3-27B GPTQ + fp16 KV +
    c=32/400 在 24GB 4090 上总显存不足";可选后续是结构性 KV 内存方案
    (正确的 Gemma3 paged-window 支持、可验证的 KV quant correctness,或
    合同调整),不是继续重跑同一 L5。

### 本轮验证

- Local:
  - `cargo fmt --all`
  - `cargo check -q -p ferrum-models --tests`
  - `cargo check -q -p ferrum-cli --tests`
  - `cargo test -q -p ferrum-models prefill_scratch_keeps_batch_logits_lazy -- --nocapture`
  - `git diff --check`
- Remote CUDA:
  - `build.log`: release CUDA build PASS in 3m26s.
  - `build_shrink.log`: release CUDA build PASS in 3m25s (failed experiment, not retained).
  - `build_paged_window.log`: release CUDA build PASS in 3m24s (failed experiment, not retained).

## 2026-06-14(凌晨 II)— W2 c=32 allocator 诊断:OOM 是整体显存峰值贴满,不是单个大分配

- 继续按 native CUDA 最小验证策略复用 stopped/cache-retained instance
  `40826362`。没有新租机器;诊断完成后已停机,Vast API 确认
  `cur_state=stopped`,`actual_status=exited`。
- 本轮 paid GPU lane 合同:
  - lane: W2 Gemma3-27B CUDA L5 c=32 allocator-diagnostic rerun,existing 1x RTX 4090;
  - expected runtime/cost:15-45min,约 $0.10-$0.30 at ~$0.402/hr plus storage;
  - stop condition:first OOM backtrace artifact collected / c=32 unexpected PASS / 45min no progress;
  - correctness/performance command:同一条 c=32
    `bench-serve random 256/128 --concurrency-sweep 32 --fail-on-error --require-ci --seed 9271`。
- 为定位 OOM,增强 CUDA allocator 失败日志:
  `CudaBackend::alloc` / `alloc_typed` 现在在失败时打印 dtype、元素数、
  字节数、`cuMemGetInfo` free/total,以及强制 backtrace。该改动只改善失败
  诊断,不改变产品行为。
- 远端增量 release build 通过:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_alloc_diag_oom_2026-06-14/build.log`
  记录 `Finished release profile [optimized] target(s) in 28m 55s`。
- 第一次诊断运行 `c32/` 未进入模型请求:脚本把 `--tokenizer` 误传成
  `tokenizer.json` 文件路径,CLI 又追加 `/tokenizer.json`,因此 bench 立即报
  `Not a directory`。该运行只保留为脚本错误证据,不计入 W2。
- 第二次诊断运行 `c32_retry/` 使用 tokenizer snapshot 目录,服务成功启动并进入
  `closed_loop c=32 — repeat 1/3`。关键 OOM 行:
  `CudaBackend::alloc failed: dtype=F16 elements=819200 bytes=1638400 free=17498112 total=25278087168: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`。
- bench 侧写出失败 JSON
  `c32_retry/l5_gemma3-27b-gptq_cuda_c32_alloc_diag.json`;
  `bench.log` 记录 `0 completed / 100 errored` 和
  `bench-serve error rate 1.0000 exceeds max 0.0000`。
- 最新结论:
  1. c=32 OOM 不是某个巨型 allocation;失败 allocation 只有约 1.56MiB;
  2. OOM 时整卡只剩约 16.7MiB free,说明 c=32 的稳态/瞬态总峰值贴满 24GB;
  3. backtrace 因 release strip 显示 `<unknown>`,下一步要加 shape/caller label
     或做非 strip diagnostic,而不是继续盲目调 `kv-capacity`;
  4. W2 仍是 l0/l1/l2_gptq/l3/l4 pass,l2_gguf waived,
     l5_concurrency fail,perf_vs_llamacpp pending;**不宣称 W2 PASS**。

### 下一步

- 本地先沿 `elements=819200` 反查可能 shape/callsite,优先看 Gemma3
  c=32 单序 forward 下每步 activation/logits/scratch 分配。
- 下一次 GPU 只跑更小诊断:给 allocator callsite 加标签或禁用 strip 的 backtrace,
  目标是拿到"哪个张量/阶段"把 free 压到 17MB,而不是重复完整 L5。
- 修复方向要减少 c=32 总峰值,例如缩小可配置 batch token/scratch 峰值、释放可复用
  scratch、或修正 per-request retained buffer 生命周期;若改变 gate 合同必须写入
  W2 文档/矩阵,不能靠隐藏 env。

## 2026-06-14(凌晨 I)— W2 c=32 最小复测:prompt 长度问题已排除,运行期 OOM 仍阻塞

- 按用户建议复用现有 native CUDA GPU 机器,没有重新租新 pod/重装环境:
  instance `40826362`,1x RTX 4090,Iceland host 1647。复测完成后立即停机,
  Vast API 已确认 `cur_state=stopped`,`actual_status=exited`。
- 本轮 paid GPU lane 合同:
  - lane: W2 Gemma3-27B CUDA L5 c=32 minimal regression,existing 1x RTX 4090;
  - expected runtime/cost:30-90min,约 $0.20-$0.60 at $0.402/hr;
  - stop condition:c=32 PASS artifact / c=32 OOM or context failure artifact / 90min no progress;
  - correctness/performance command:`bench-serve random 256/128 --concurrency-sweep 32 --fail-on-error --require-ci --seed 9271`。
- 先修正并本地验证 `bench-serve` random prompt 生成:
  `--random-input-len 256` 现在按 tokenizer 重新编码后的实际 token 数逼近目标,
  避免 Gemma tokenizer 把随机 token 文本重编码成 270+ tokens。
  本地用 `unsloth/gemma-3-1b-it` tokenizer fixture 跑
  `random_prompt_generation_targets_reencoded_length_when_fixture_is_set` PASS。
- 远端只同步最小改动并 release build:
  `target/release/ferrum` build finished in 3m26s;远端 `.env.local` 在一次误同步后已立即删除,
  本轮正式运行目录的 `c32/ferrum_env.txt` 为空,没有隐藏 `FERRUM_` env 修复。
- c=32 复测命令使用产品 CLI 参数:
  `ferrum serve --model gemma3:27b-gptq --port 8402 --kv-capacity 400 --max-num-seqs 32`
  加 `ferrum bench-serve --random-input-len 256 --random-output-len 128 --concurrency-sweep 32 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271`。
- 新证据:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_c32_prompt_exact_oom_2026-06-14/`。
  `serve.log` 证明 `kv-capacity=400` 能启动 800 blocks/32 seqs 并进入服务;
  `bench.log` 进入 `closed_loop c=32 — repeat 1/3`;
  随后服务端在 `crates/ferrum-kernels/src/backend/cuda/mod.rs:774`
  报 `CUDA_ERROR_OUT_OF_MEMORY`。
- 因此最新结论是:
  1. 之前 `kv=400` 的"上下文不足"分支已被 prompt 精确化排除;
  2. c=32 仍在运行期 OOM,不是 KV pool 启动失败;
  3. W2 L5 仍 FAIL,`perf_vs_llamacpp` 仍 pending,**不宣称 W2 PASS**。

### 下一步

- 保持同一台 stopped/cache-retained 4090 作为后续最小验证目标。
- 下一轮不要再跑完整 L5。先定位运行期 CUDA transient/scratch/请求调度内存峰值:
  优先用 c=32、kv=400、100 prompts 的同命令加最小显存 instrumentation,
  看 OOM 前最后一次 allocator 请求大小和调用路径。
- 如果需要改变 batch/scratch 上限,必须走 typed CLI/config/default 或目标合同修订,
  不能靠用户不可见的隐藏 env 当作通过证据。

## 2026-06-13(晚 XIX)— W2 correctness 打通,L5 c=32 首个内存阻塞点落证

- 采用当前 1x RTX 4090 pod 原生 CUDA 最小验证,不重建环境:
  instance `40826362`,Iceland host 1647,`/workspace/ferrum-infer-rs`。
  本轮目标从"反复开机装环境"改为"同机快速定位,拿到证据后停机"。
- 根因定位:Gemma3 sandwich-norm CUDA 路径把 residual/activation 存成 f16,
  在中层出现合法的大幅值 norm 输出与 residual 相加后超过 f16 上限
  65504,导致后续 logits NaN。Gemma3 27B 的 post-ffn norm 权重可到
  700 级,该路径必须保留 f32/bf16 语义,不能把 residual shadow 降成 f16。
- 修复方式:CUDA f16 activation 后端为 Gemma3 sandwich-norm 路径启用
  host/F32 residual shadow;prefill/decode 的 norm、residual add、final norm
  通过 f32 shadow 保持有限值。该行为走产品默认路径,不是隐藏 env 修复。
- 最小 CUDA 验证:
  - `cargo test -p ferrum-kernels --features cuda --release --test cuda_activation_precision -- --nocapture`
    PASS,2 tests;
  - `ferrum run` 一 token smoke 输出 `content:"5"`,layer dump/logits 全 finite;
  - layer dump summary:64 entries,`first_nonfinite=None`,logits
    `262208/262208` finite,`maxabs=41.84375`。
- W2 L2/L3/L4 smoke 已过:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_gemma3_cuda_host_shadow_l5_fail_2026-06-13/gates/smoke_gemma3-27b-gptq.log`
  记录 `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`:
  known-answer 10/10,natural EOS,multi-turn Bob,stream==non-stream,
  custom stop,max_tokens,required tool-call 10/10,strict json_schema 20/20。
- W2 L5 未过,所以**不宣称 W2 PASS**:
  - `gates/l5_gemma3-27b-gptq_cuda_1_4_16.json` 已记录
    c=1/4/16 三档均 `completed_per_run=[100,100,100]`,
    `errored_per_run=[0,0,0]`,`output_token_count_source=usage`,
    decode throughput 约 25.1 tok/s;
  - required c=32 在 `--kv-capacity 448 --max-num-seqs 32` 触发
    `CUDA_ERROR_OUT_OF_MEMORY`,日志见
    `gates/serve_l5_gemma3-27b-gptq_32.log`;
  - targeted diagnostics:kv-capacity 400 可启动但上下文不足
    (`278 input + 128 output > 400`),kv-capacity 408/416 因 KV block
    rounding 仍 OOM;
  - 因 L5 首败即停,未进入 llama.cpp same-card ratio,`perf_vs_llamacpp`
    仍 pending。
- 当前矩阵结论:
  W1 PASS;W2 现在是 l0/l1/l2_gptq/l3/l4 pass,l2_gguf waived,
  l5_concurrency fail,perf_vs_llamacpp pending;W3 仍只有 charter 草案。

### 下一步

- 不再跑完整重装 pod。优先在现有 CUDA 内存模型上做一个小改动:
  让 c=32 gate 真正降低 per-seq/context 占用或改为可解释的 W2 合同修订;
  不能只把 kv-capacity 降到 400,因为该设置已被实测证明不满足
  256/128 bench prompt 的上下文需求。

## 2026-06-13(晚 XVIII)— W2 pod 脚本固化 parity-first:失败即停,通过才跑 Gemma3 early-smoke

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- `scripts/pod_w2_gemma3.sh` 已把下一次 1x4090 执行顺序固化:
  - `build.ok` 后立即运行 synthetic CUDA desc_act parity,不等待 27B 权重下载:
    1. `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`;
    2. `cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference`;
  - parity 日志写入 `$W/diagnostics/desc_act_parity.log`,
    return code 写入 `$W/diagnostics/desc_act_parity.rc`;
  - parity fail 会写 `$W/desc_act_parity.fail`,停止 HF 下载/llama.cpp build,
    touch `$W/w2_session.done`,不进入 Gemma3 smoke/L5/perf;
  - parity pass 才写 `$W/desc_act_parity.ok`,随后 early-smoke 等待
    `build.ok + dl_gptq.ok + desc_act_parity.ok`。
- early-smoke 默认诊断也更新:
  - `FERRUM_SMOKE_NAN_TRACE="${FERRUM_W2_NAN_TRACE:-layer0}"`;
  - `FERRUM_SMOKE_OP_DUMP_DIR="${FERRUM_W2_OP_DUMP_DIR:-/workspace/w2/gates_early/op_dump_layer0}"`;
  - 因晚 XVII 已支持 `layer0` 选择语法,下一次默认只写首层 op dump,
    而不是全模型 `all` dump。
- 下一次 paid GPU lane 合同(尚未启动):
  - lane: W2 Gemma3-27B CUDA GPTQ correctness micro-diagnostic,1x RTX 4090;
  - expected runtime/cost:约 1-2h,按最近 Vast $0.35-$0.45/hr 估算约 $0.35-$0.90,
    若需要继续 early-smoke/op-dump 可能到 3h/$1.35;
  - stop condition: parity fail / early-smoke fail artifact collected / first
    correctness PASS artifact collected / 3h 无进展;
  - correctness gate:两个 CUDA desc_act parity 先 PASS;若 PASS,再跑
    Gemma3 early-smoke L2 known-answer;
  - performance command:不跑性能。只有 L2/L3/L4 correctness PASS 后,才恢复
    `pod_w2_gates.sh` 的 L5 `bench-serve` 和 llama.cpp ratio。
- 本地验证:
  - `bash -n scripts/pod_w2_gemma3.sh scripts/pod_w2_gates.sh scripts/model_coverage_smoke.sh` PASS(本机 locale warning only)。
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists` PASS。
  - `python3 -m py_compile scripts/analyze_layer_dump.py scripts/inspect_hf_gptq_tensor.py scripts/w1_goal_validator.py scripts/w2_goal_validator.py` PASS。

### 下一步

- 下一次 GPU 只跑 `scripts/pod_w2_gemma3.sh` 的新 parity-first 流程;
  若 `desc_act_parity.fail`,直接修 CUDA Marlin desc_act/repack/scale layout;
  若 parity 过而 early-smoke 仍失败,读取
  `gates_early/op_dump_layer0/summary.jsonl` 的 `maxabs_row/maxabs_col`
  定位首个产生 `row=0,col=104` 爆炸的 op。

## 2026-06-13(晚 XVII)— W2 nan-trace 坐标化:下次 op 日志直接报 maxabs row/col

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 基于晚 XVI 的 `layer_00 row=0,col=104` 爆炸证据,增强
  `crates/ferrum-models/src/models/llama_family.rs` 的诊断输出:
  - `DumpStats` 现在记录 `maxabs_index`;
  - `FERRUM_NAN_TRACE` 日志现在在知道 row width 时打印
    `maxidx,row_width,row,col`;
  - `FERRUM_OP_DUMP` 的 `summary.jsonl` 同步写入 `maxabs_index`,
    且对有 row width 的 op 写入 `maxabs_row/maxabs_col`;
  - `FERRUM_NAN_TRACE` 保留 `0` 作为关闭,新增显式 `layer0`/`l0`/
    `source0` 语法,允许只抓首层 op 而不是 `all` 全模型 dump;
  - 新增 `down_proj` 和 `resid_ffn` 两个 trace 点,补齐 MLP 输出与最终
    residual add,避免下一次只看到 layer dump 爆炸却不知道 down-proj
    之前/之后的边界。
- row width 绑定:
  - token-major hidden/residual/norm/o_proj/down_proj/resid_ffn → `hidden_size`;
  - `qkv_proj` → `q_dim + 2*kv_dim`;
  - head-major q/attention → `head_dim`;
  - `gate_up` → `2*intermediate_size`;
  - `act_mul` → `intermediate_size`。
  这让下一次 Gemma3 early-smoke 日志能直接回答首层异常是否已经出现在
  `qkv_proj`,还是 attention/o_proj/MLP 后才出现。
- 本地验证:
  - `python3 -m py_compile scripts/analyze_layer_dump.py scripts/inspect_hf_gptq_tensor.py` PASS。
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-models --tests` PASS。
  - `cargo test -q -p ferrum-models dump_stats_counts_nonfinite_values` PASS。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 default tests;CUDA ignored tests未在本机执行)。
  - `git diff --check` PASS。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次可靠 1x4090 不跑完整 W2。先跑 CUDA desc_act parity;若过,再跑
  Gemma3 early-smoke with `FERRUM_W2_NAN_TRACE=layer0` +
  `FERRUM_W2_OP_DUMP_DIR=<artifact-dir>`,日志里重点看 layer0 每个 op 的
  `row=0,col=104` 是否在 qkv/o_proj/MLP 哪一步首次出现。

## 2026-06-13(晚 XVI)— W2 NaN artifact 机器化复盘:异常从 layer_00 内部开始,不是 logits-only

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 新增 `scripts/analyze_layer_dump.py`:无第三方依赖读取
  `FERRUM_LAYER_DUMP` f32 `.bin`,统计 finite/nonfinite、max_abs、first
  threshold crossing,并可用 `--last-dim` 把 flat index 标注为
  `[row,col]` 坐标。
- 用该脚本复盘既有
  `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/gates_early/logit_dump_smoke`
  并生成:
  `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/gates_early/layer_dump_summary.json`。
- 机器化 summary 结论:
  - `embed.bin`:shape `[24,5376]`,all finite,`max_abs=17.65625`;
  - `layer_00.bin`:all finite,但 `max_abs=23056.0` 出现在
    `row=0,col=104`;首个 `abs>100` 在 `row=0,col=61`,首个
    `abs>1000/10000` 同在 `row=0,col=104`;
  - `layer_01..layer_07`:仍 all finite,但异常值持续存在,`layer_07`
    `max_abs=42432.0` 仍在 `row=0,col=104`;
  - `logits.bin`:262208/262208 全 NaN,首个 nonfinite index 0。
- 这把定位从"最终 logits 全 NaN"收紧为"首层输出已经爆炸,后续层仍 finite,
  到 lm_head/logits 阶段才变全 NaN"。下一次 GPU op dump 不需要先扫全模型,
  应优先捕获 layer0 的 qkv、attention score/output、o_proj、post-attn norm、
  gate/up/down/activation/down_proj 输入输出,特别关注 token row 0、hidden col
  61/104 附近。
- 本轮新证据与晚 XV 的真实 GPTQ CPU dequant 结合后,当前最高概率分支仍是
  CUDA Marlin desc_act/scale layout/act-order 或 Gemma3 layer0 CUDA forward
  内某个 op,不是 HF 源权重本身、qzeros 形态、g_idx balance 或最终 logits
  读回单点问题。

### 下一步

- 下一次可靠 1x4090 的执行顺序:
  1. `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`;
  2. 若过,跑 `cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference`;
  3. 若仍过,跑 Gemma3 early-smoke + `FERRUM_W2_OP_DUMP_DIR`,优先 layer0
     op dump,用本轮 `row=0,col=61/104` 作为检查坐标。

## 2026-06-13(晚 XV)— W2 真实 layer0 attention 全投影 + MLP 采样:源权重继续排除,下轮直指 CUDA parity/op dump

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 新增可复用诊断脚本 `scripts/inspect_hf_gptq_tensor.py`:通过 HF
  `model.safetensors.index.json` 定位 shard,用 HTTP Range 读取单个 GPTQ
  tensor prefix 的 `qweight/scales/qzeros/g_idx`,输出 JSON summary。脚本默认
  8MiB 分块读取,避免 50MB+ range 被远端断连;无第三方依赖。
- 继续读取真实 `circulus/gemma-3-27b-it-gptq`
  commit `70d89a3a6b401b5f56558cb5d4c0f1fd158980b2` 的 layer0 权重,生成:
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_kproj_cpu_dequant_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_vproj_cpu_dequant_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_oproj_cpu_dequant_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_gateproj_cpu_dequant_sample_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_upproj_cpu_dequant_sample_report.json`
  - `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_downproj_cpu_dequant_sample_report.json`
- layer0 self-attn 的 k/v/o 三个 projection 做完整列 CPU dequant + deterministic
  matmul probe(晚 XIV 已覆盖 q_proj):
  - k_proj: `K=5376,N=2048`,all finite,`max_abs=0.1303`;
  - v_proj: `K=5376,N=2048`,all finite,`max_abs=0.2098`;
  - o_proj: `K=4096,N=5376`,all finite,`max_abs=0.1708`;
  - 三者均 `g_idx_balanced_full_groups=true`,
    `g_idx_sequential_non_desc_act=false`,`qzeros_all_code7=true`,
    `scales_all_finite=true`。
- layer0 MLP 的 gate/up/down 三个 projection 做 512 个均匀输出列采样
  (不是完整 MLP 证明,但覆盖真实 qweight/scales/qzeros/g_idx 读取和每个 K row):
  - gate_proj: sampled all finite,`max_abs=0.1321`;
  - up_proj: sampled all finite,`max_abs=0.2784`;
  - down_proj: sampled all finite,`max_abs=0.0411`;
  - 三者同样满足 balanced non-trivial `g_idx`、全 code7 `qzeros`、finite scales。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 新增 ignored CUDA
  micro-diagnostic:
  `cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference`。它顺序覆盖真实
  layer0 attention 的 `k_proj K5376/N2048`,`v_proj K5376/N2048`,
  `o_proj K4096/N5376`,补足 q_proj-only parity 可能漏掉 tile/shape 问题。
- 结合晚 XII-XIV,真实 Gemma3 GPTQ 源权重的基本形态、scale 有限性、
  qzero 对称性、desc_act group balance、layer0 attention 全投影 dequant
  都不像首层 2e4 级爆炸和最终全 NaN logits 的源头。当前优先级进一步收敛到:
  1. CUDA Marlin desc_act/Gemma3 真实形状 parity;
  2. 若 parity 过,用 `FERRUM_W2_OP_DUMP_DIR` 捕获 Gemma3 early-smoke
     首层 qkv/attn/o_proj/MLP op 输入输出,定位 CUDA forward 内首个爆炸点。
- 本地验证:
  - `python3 -m py_compile scripts/inspect_hf_gptq_tensor.py scripts/w1_goal_validator.py scripts/w2_goal_validator.py` PASS。
  - `git diff --check` PASS。
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 default tests;CUDA ignored tests未在本机执行)。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 不跑完整 W2。下一次可靠 1x4090 先跑:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference -- --ignored --nocapture`。
- 若 q_proj shape 过,同一 pod 继续跑:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference -- --ignored --nocapture`。
- 若 Gemma3-shape parity 失败,修 CUDA Marlin repack/scale layout/act-order;
  若通过,再跑 Gemma3 early-smoke + op dump。只有首步 logits finite 且 smoke
  L2 过,才恢复 L3/L4/L5/perf。

## 2026-06-13(晚 XIV)— W2 真实 layer0 q_proj CPU dequant:权重合成有限,不像首层爆炸源

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 用 HF Range 读取真实 `circulus/gemma-3-27b-it-gptq`
  layer0 `self_attn.q_proj` 的完整 GPTQ 四件套:
  `qweight [672,4096]`, `scales [42,4096]`, `qzeros [42,512]`,
  `g_idx [5376]`,生成:
  `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/layer0_qproj_cpu_dequant_report.json`。
- 本地 CPU 按 `g_idx` scale lookup 完整 dequant 该 projection:
  - `g_idx_balanced_full_groups=true`;
  - `qzeros_all_code7=true`;
  - `scales_all_finite=true`;
  - `dequant_weight_all_finite=true`;
  - `dequant_weight_max_abs=0.1151123046875`。
- 额外做两个 deterministic matmul probe:
  - `x[k]=f16(sin(k*0.0041))` 时输出 `max_abs≈2.00`;
  - 同一输入乘以 17.7(上一轮 embed dump maxabs 量级)时输出
    `max_abs≈35.41`。
  这不能替代 CUDA parity,但能排除"真实 layer0 q_proj 的 GPTQ
  dequant 结果本身非 finite 或自然产生 2e4 级输出"这个分支。
- 因此当前最高价值 GPU micro-diagnostic 仍是 CUDA Marlin vs CPU
  reference,尤其是 Gemma3 q_proj 真实形状;若它过,再从 q_proj 之外的
  首层 op dump 查爆炸边界。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 default tests;CUDA ignored tests未在本机执行)。
  - artifact sanity PASS:报告 summary 七项均符合预期。

### 下一步

- 下一次可靠 1x4090 先跑:
  `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`。若失败,优先修
  CUDA Marlin repack/kernel/scale layout;若通过,用 `FERRUM_W2_OP_DUMP_DIR`
  捕获 Gemma3 early-smoke 首层 qkv/attn/mlp op 输入输出,定位 q_proj
  之外的首个爆炸点。

## 2026-06-13(晚 XIII)— W2 真实 Gemma3 scales 采样:源 scales 有限且量级正常

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 延续晚 XII 的 HF Range 采样,对同一 layers 0 / 31 / 61、同一 7 个
  projection(`q/k/v/o/gate/up/down`)读取 `*.scales` 小样本,生成:
  `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/gptq_scale_sample_report.json`。
- 第一次 scales 读取误按 F32 解码,发现异常小量级后已废弃并用
  safetensors header 的 dtype-aware 解码覆盖报告。有效报告显示:
  - `dtypes=["F16"]`;
  - `sampled_scales_count=21`;
  - `all_finite=true`;
  - `bad_nonfinite_or_gt10_count=0`;
  - `max_abs_overall=0.0723876953125`。
  这排除了"真实 Gemma3 GPTQ 源 scales 本身非 finite 或异常大"这一分支。
- 结合晚 XII:
  - sampled `g_idx` 均 balanced full-group 且 non-trivial desc_act;
  - sampled `qzeros` 均 code7;
  - sampled `scales` 为有限 F16 且量级正常。
  因此下一步仍应集中在 CUDA Marlin repack/kernel/scale layout 与
  Gemma3 层内数值路径,不是权重 metadata 基本形态。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 default tests;CUDA ignored tests未在本机执行)。
  - artifact sanity PASS:layout summary 两项 true,scale summary all_finite
    true且 dtype 为 F16。

### 下一步

- 下一次可靠 1x4090 的第一条仍是两条 CUDA ignored parity:
  `cuda_desc_act_vs_cpu_reference` 与
  `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`。
- 若 Gemma3-shape parity 仍过,就把 focus 从 GPTQ metadata/repack 前置条件
  移到 Gemma3 forward 内的首层 op dump:全层 nan-trace 已默认打开,
  必要时显式 `FERRUM_W2_OP_DUMP_DIR` 捕获首个爆炸算子输入输出。

## 2026-06-13(晚 XII)— W2 真实 Gemma3 GPTQ layout 采样:guard 预计不拦截,下一步查 CUDA Marlin 数值

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 用 Hugging Face resolve HTTP Range 只读 `circulus/gemma-3-27b-it-gptq`
  commit `70d89a3a6b401b5f56558cb5d4c0f1fd158980b2` 的 safetensors
  header 与小 tensor 样本,未下载整模型权重:
  `artifacts/w2_gemma3_hf_gptq_layout_2026-06-13/gptq_layout_sample_report.json`。
- 采样范围:layers 0 / 31 / 61 的 7 个 projection
  (`q/k/v/o/gate/up/down`),共 21 个 `g_idx` + 21 个 `qzeros`,
  payload 约 5.47MB。一次全量 `g_idx` Range 尝试因远端 HTTP 408 停止,
  没有作为证据使用。
- 采样结论:
  - `sampled_g_idx_all_balanced_full_groups=true`;
  - `sampled_g_idx_sequential_non_desc_act_count=0`,确认样本确实是
    non-trivial desc_act,不是顺序 g_idx;
  - `sampled_qzeros_all_code7=true`。
  因此晚 X/XI 新增的 qzeros / balanced-g_idx guard 在这些真实
  Gemma3 GPTQ 样本上预计不会拦截;当前 L2 NaN blocker 更可能还在
  CUDA Marlin repack/kernel/scale layout 或 Gemma3 层内数值路径。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 新增 ignored
  CUDA micro-diagnostic:
  `cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference`。它使用真实
  Gemma3 q_proj 形状 `K=5376,N=4096,group_size=128` 的 synthetic
  desc_act/sym GPTQ,对比 CUDA Marlin 与 CPU `g_idx` reference;用于补足
  旧 `K=512,N=256` 小形状 parity 不能覆盖真实 tile/scale 布局的缺口。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 default tests;CUDA ignored tests未在本机执行)。
  - `cargo test -q -p ferrum-quantization validate_cuda_marlin_desc_act`
    PASS(3 tests)。
  - artifact sanity PASS:`sampled_g_idx_count=21`,
    `sampled_qzeros_count=21`,两项兼容性 summary 均为 true。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次可靠 1x4090 仍不跑完整 W2。先跑:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`
  和
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference -- --ignored --nocapture`。
- 若 small 过而 Gemma3-shape 失败,优先查 Marlin repack/scale layout 的
  shape-specific 假设;若两者都过,再跑 Gemma3 early-smoke,使用全层
  nan-trace 定位首个爆炸/非 finite 算子。

## 2026-06-13(晚 XI)— W2 desc_act 前提硬化:CUDA Marlin 只接受 balanced full-group g_idx

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 继续审计 `desc_act=true/static_groups=false` GPTQ 与当前 CUDA Marlin
  策略的等价前提。Ferrum 现策略是 load-time `argsort(g_idx)` 重排
  qweight row,运行时按同一 perm gather A,再交给固定 group-boundary 的
  IST-DASLab Marlin kernel。因此它只在每个 quant group 恰好有
  `group_size` 个 K row 时成立;若真实 `g_idx` 某 group 多/少,row 排序后
  的固定 `j/group_size` scale lookup 会错位。
- `crates/ferrum-quantization/src/native_safetensors.rs` 新增
  `validate_cuda_marlin_desc_act_g_idx()`:
  - CUDA build 下,普通 GPTQ linear、fused qkv/gate_up linear、stacked GPTQ
    experts 都会在 load 阶段校验 balanced full-group;
  - `quantize_config desc_act=true` 但缺 `g_idx` 的 stacked GPTQ 现在也会
    和普通 linear 一样显式报错,不再把 `None` 交给 backend 静默错跑;
  - 非 CUDA 的 CPU/Metal desc_act dequant 仍保留按 `g_idx` lookup 的通用
    fallback,不套 Marlin 前提。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `bash -n scripts/pod_w2_gemma3.sh scripts/model_coverage_smoke.sh scripts/pod_w2_gates.sh`
    PASS(本机 locale warning only)。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo check -q -p ferrum-kernels --tests` PASS。
  - `cargo test -q -p ferrum-quantization validate_cuda_marlin_desc_act`
    PASS(3 tests)。
  - `cargo test -q -p ferrum-quantization validate_gptq_g_idx` PASS
    (4 tests)。
  - `cargo test -q -p ferrum-quantization qzero_stats` PASS(2 tests)。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 tests)。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次可靠 1x4090 仍只跑 micro-diagnostic first:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- 若 parity 过,再跑 Gemma3 early-smoke。新的 loader guard 会把真实
  Gemma3 GPTQ 的 `g_idx`/`qzeros` 兼容性问题转成 load-stage 明确错误;
  若两项兼容性 guard 都过但仍 NaN,全层 nan-trace 才是下一步定位依据。

## 2026-06-13(晚 X)— W2 CUDA GPTQ guard 收紧:拒绝不安全 Marlin 路径,避免假诊断

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- `crates/ferrum-kernels/src/backend/cuda/quant.rs` 收紧默认 CUDA Marlin:
  - dense `FERRUM_VLLM_MARLIN=1` 现在显式 `unsupported`。原因:dense
    `load_gptq` 保存的是 IST-DASLab Marlin tile,而 vLLM Marlin kernel
    需要 vLLM-repacked weights;此前这个实验 env 可能让同一权重走错
    kernel,污染 W2 诊断。vLLM-repacked Marlin 仍只保留在 stacked MoE
    的 `FERRUM_VLLM_MOE` 路径。
  - 默认 dense/stacked Marlin 在忽略 `qzeros` 前,现在要求所有 GPTQ
    `qzeros` nibble 都是 code 7(GPTQ zero-1 编码下的对称 zero point 8)。
    若真实 Gemma3 GPTQ 不是该形态,下一次 early-smoke 会在 load 阶段
    明确失败并给出首个 bad code 位置,而不是继续进入可能全 NaN 的推理。
- 同步 `docs/runtime-env-registry.tsv`:`FERRUM_VLLM_MARLIN` 标记为
  dense GPTQ rejected,除非未来补 dense vLLM repack 证据,否则不能作为
  产品验证或诊断开关。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-kernels --tests` PASS(默认 feature;CUDA 模块
    不会在本机编译进来)。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test` PASS
    (5 tests)。
  - `python3 scripts/w1_goal_validator.py` PASS:
    `MODEL_COVERAGE_W1 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
  - `python3 scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。
  - `cargo check -q -p ferrum-kernels --features cuda --tests` 未跑到 Rust
    语义编译:本机缺 `nvcc`/`nvidia-smi`,失败在 CUDA build scripts。

### 下一步

- 下一次可靠 1x4090 的第一项仍是最小 CUDA parity:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- parity 通过后再跑 Gemma3 early-smoke。若新 qzeros guard 拦截真实权重,
  W2 的 L2 blocker 变成"当前 Marlin 不支持该 GPTQ zero-point 形态";
  若 qzeros 通过但仍 NaN,读取全层 nan-trace 定位首个非 finite op。

## 2026-06-13(晚 IX)— W2 early-smoke 诊断默认升级:下轮直接收全层 nan-trace

- 没有新开 GPU。W2 仍是 Gemma3-27B GPTQ CUDA correctness blocking,
  不进入 L5/perf,不宣称 W2 PASS。
- 复核上轮 NaN artifact:`FERRUM_LAYER_DUMP` 只给出 embed、layer_00..07
  和最终 logits;虽然 layer_00 起 maxabs 已爆炸,但缺少每层关键算子
  的 non-finite 边界,下一轮若仍失败会继续需要二次上卡定位。
- `scripts/pod_w2_gemma3.sh` 的 early-smoke 默认改为
  `FERRUM_SMOKE_NAN_TRACE="${FERRUM_W2_NAN_TRACE:-all}"`。这会让
  `ferrum serve` 日志在每一层的 qkv、attn、o_proj、post_attn_norm、
  gate_up、activation、down_proj 等关键点打印 finite/nan/inf/maxabs。
  `FERRUM_W2_NAN_TRACE` 仍可覆盖成单层列表;`FERRUM_W2_OP_DUMP_DIR`
  仍保持显式 opt-in,避免默认写出巨量 op dump。
- 本地验证:
  - `bash -n scripts/pod_w2_gemma3.sh` PASS(本机 locale warning only)。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists`
    PASS。
  - `cargo test -q -p ferrum-models dump_stats_counts_nonfinite_values` PASS。

### 下一步

- 下一次可靠 1x4090 仍先跑最小 CUDA parity;若需要直接跑 Gemma3
  early-smoke,它现在会一次性产出 g_idx/qzeros trace + 全层 nan-trace,
  足以定位首个非 finite op 或确认只是数值爆炸但未 NaN。

## 2026-06-13(晚 VIII)— W2 本地 qzeros/sym 审计:补真实模型 trace,CUDA parity fixture 收紧

- 没有新开 GPU。W2 仍停在 Gemma3-27B GPTQ CUDA correctness:
  首步 logits 全 NaN,不得进入 L5/perf,也不得宣称 W2 PASS。
- 本地审计确认 Ferrum 默认 CUDA Marlin 路径仍不读取 `qzeros`:
  Marlin no-zp/sym 路径隐含 int4 zero point = 8;GPTQ `qzeros`
  按 zero-1 存储时,对称量化应表现为 nibble code 7。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 的 CUDA ignored
  parity fixture 已改为 `sym=true` 代表性 qzeros:
  `make_synthetic_symmetric()` 将所有 qzeros word 置为 `0x77777777`,
  避免随机 qzeros 让 Marlin no-zp 路径产生非代表性失败。
- `crates/ferrum-quantization/src/native_safetensors.rs` 增加
  `FERRUM_GPTQ_GIDX_TRACE=1` 下的 qzeros 统计:
  每个 GPTQ linear / fused GPTQ load 会打印 qzero nibble histogram、
  `min_code/max_code`、`code7/total` 与 `all_code7`。下一次 W2
  early-smoke artifact 可直接回答真实 Gemma3 GPTQ 是否满足
  sym=true/no-zp Marlin 假设。
- 复核加载路径:`ferrum run/serve` 的主 GPTQ product path 由
  `NativeSafetensorsLoader::<B>::open()` + `WeightLoader::load_linear()`
  承载;旧日志里的 `ferrum_models::loader::gptq_loader` 来自 registry
  的兼容 `QuantizeConfig` probe,不能单独代表实际 linear loader。
  因此本轮 trace 加在 `NativeSafetensorsLoader` 上,覆盖下一次 W2
  early-smoke 的真实 GPTQ linear/fused-linear load。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test`
    PASS(5 tests)。
  - `cargo test -q -p ferrum-quantization qzero_stats` PASS(2 tests)。
  - `cargo test -q -p ferrum-quantization validate_gptq_g_idx` PASS(4 tests)。
  - `cargo check -q -p ferrum-quantization --tests` PASS。

### 下一步

- 仍不跑完整 W2。下一次有可靠 1x4090 通道时,先跑最小 CUDA parity:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- 若 parity 通过,再跑 Gemma3 early-smoke,读取新增 qzeros/g_idx trace 与
  layer/op dump 来定位首个非 finite 算子;若 parity 失败,先修 Marlin
  repack/perm/scales/qzeros 假设,不要进入 L5/perf。

## 2026-06-13(晚 VII)— W2 CUDA parity 租机未跑到:Vast SSH/proxy 失败,实例归零

- 目标只是一条 W2 micro-diagnostic,不是完整 W2 gate:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
  correctness-only,无性能命令、无 W2 pass 声明。
- Vast 四次尝试均未跑到测试:
  - `40812709`(Vietnam host 55116,$0.356/hr):卡在
    `actual_status=loading`,未 SSH,已销毁。
  - `40813345`(Iceland host 1647,$0.402/hr):Vast reported running/onstart
    success,但 SSH publickey 认证失败,已销毁。
  - `40813765`(Iceland host 1647,$0.402/hr):按 Vast 文档改用
    `runtype=ssh` + attach-key API 后进入 running;但 container log 显示
    `remote port forwarding failed for listen port 13764`,proxy SSH 被关;
    direct SSH 也 rejected attached key;已销毁。
  - `40814425`(Ukraine host 103274,$0.401/hr):按下一步改用
    `runtype=ssh_direct`,实例进入 running 且 direct port 打开;但
    `root/ubuntu/vastai/user` 均 rejected associated public key;已销毁。
- Artifact/证据:
  `artifacts/w2_desc_act_cuda_parity_2026-06-13/` 保存 offer/创建响应
  (instance key 已脱敏)、Vast logs、destroy responses。未产生
  `cuda_desc_act_vs_cpu_reference` 输出。
- Vast API 已确认 `instances_found: 0`;`ACTIVE_PODS.md` 已标记四台均
  DESTROYED/ZERO-VERIFIED。

### 下一步

- Vast 通道暂停。下一步回到本地源码审计,优先核查 CUDA GPTQ Marlin
  对 `qzeros`/sym 的假设与 CPU reference 是否一致;GPU 通道恢复前不再
  循环租 offers。

## 2026-06-13(晚 VI)— W2 desc_act 诊断收窄:补本地 parity,下一轮只测 CUDA repack/kernel

- 没有新开 GPU。当前本机是 `Darwin arm64` 且无 `nvcc`,因此
  `cuda_desc_act_vs_cpu_reference` 只能作为下一轮 4090 ignored test,
  不能在本地执行。
- `crates/ferrum-quantization/tests/gptq_parity_test.rs` 新增/修正
  desc_act 合成诊断:
  - 合成 `g_idx` 改为 balanced full-K 形态:每个 group 正好
    `group_size` 个元素,但 K 轴交错,避免非现实非均匀 group 误报。
  - `desc_act_reference_uses_g_idx_for_scale_lookup` 证明该 fixture
    确实会区分顺序 group lookup 与 `g_idx[k]` lookup。
  - `desc_act_perm_gather_is_equivalent_to_g_idx_reference` 证明在
    balanced full-K 前提下,Ferrum 当前 host-level
    `argsort(g_idx)` qweight 重排 + activation gather 的代数结果等价于
    `g_idx` CPU reference。也就是说,晚 V 的“vLLM wrapper 未传 g_idx
    必然不等价”表述过强;vendored vLLM 在 `has_act_order && is_k_full`
    时同样会 permute A 后把 `has_act_order` 降为 false。
- 新增可选诊断 `FERRUM_GPTQ_GIDX_TRACE=1`:GPTQ loader 会打印真实
  `g_idx` 的 group count min/max、nonzero group 数、unbalanced group 数
  和前 16 项。`scripts/pod_w2_gemma3.sh` 的 early-smoke 已打开该开关,
  下一轮 artifact 能直接确认 Gemma3 GPTQ 的真实 g_idx 分布是否满足
  balanced full-K 前提。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q -p ferrum-quantization --tests` PASS。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test desc_act`
    PASS(2 tests)。
  - `cargo test -q -p ferrum-quantization --test gptq_parity_test cpu_selfcheck`
    PASS。
  - `bash -n scripts/pod_w2_gemma3.sh` PASS(本机 locale warning only)。
  - `scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次付费 GPU 不跑完整 W2。先跑最小 CUDA parity:
  `cargo test -p ferrum-quantization --features cuda --release --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture`。
- 若 parity 失败,优先查 CUDA qweight repack/perm 方向、Marlin scale
  repack、qzeros/sym 假设和 real `g_idx` 分布;若 parity 通过,再用
  `FERRUM_GPTQ_GIDX_TRACE=1` + early-smoke 的 layer/op dump 定位
  Gemma3 层内首个爆炸算子。

## 2026-06-13(晚 V)— W2 本地定位:desc_act GPTQ/Marlin 成为主嫌,固化 early-smoke 止损

- 基于 `w2_gemma3_cuda_nan_logits_2026-06-13` artifact 继续本地定位:
  已回收层 dump 的 `embed.bin` 与 `layer_00..07.bin` 全 finite,但数值幅度
  从 `embed maxabs=17.7` 到 `layer_00 maxabs=23056` 已明显爆炸,
  `layer_07 maxabs=42432`,最终 `logits.bin` 为 262208/262208 NaN。
  这说明不是最终 tokenizer/stop/template 问题,也不是 final softcap 缺失;
  数值从首层 GPTQ transformer 路径开始异常。
- HF config 落证:
  - `circulus/gemma-3-27b-it-gptq`: `desc_act=true`, `sym=true`,
    `group_size=128`, `static_groups=false`, `lm_head=false`;
    `final_logit_softcapping=null`。
  - `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g` 是
    `compressed-tensors/pack-quantized`,不是当前 GPTQ loader 可用的
    Marlin-clean 格式,不能直接替换成 W2 GPTQ 载体。
  - `orvp/gemma-3-27b-it-gptq` 与 `circulus` 同样是
    `desc_act=true/static_groups=false` GPTQModel 路径。
- 对照 vLLM current path:其 GPTQ-Marlin apply 路径把 `g_idx` 与
  `g_idx_sort_indices` 传入 Marlin op;ferrum 当前 CUDA act-order 路径是
  load-time qweight permute + runtime gather A,并在 vendored vLLM Marlin
  wrapper 中把 `g_idx/perm/a_tmp` 传 null、`has_act_order=false`。这与
  vLLM current path 不等价,是当前 NaN 的最高优先级嫌疑。
- 代码/脚本加固(不改变默认产品路径):
  - `FERRUM_LAYER_DUMP` 现在写 `summary.jsonl`,记录每个 dump 的
    finite/nan/inf/maxabs;smoke 会打印首个 non-finite entry。
  - `FERRUM_NAN_TRACE` 从只跟 layer 0 改为支持 `all` 或逗号分隔层号;
    `FERRUM_OP_DUMP` 输出文件名带 `layer_NN_` 前缀。
  - `scripts/model_coverage_smoke.sh` 新增
    `FERRUM_SMOKE_NAN_TRACE` / `FERRUM_SMOKE_OP_DUMP_DIR` 可选透传。
  - `scripts/pod_w2_gemma3.sh` 固化 early-smoke:build+GPTQ 下载完成后
    先跑 correctness smoke;若失败,写 `early_smoke.fail` 并停止 GGUF /
    llama.cpp 后台工作,不再等待 perf 前置项。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `bash -n scripts/model_coverage_smoke.sh scripts/pod_w2_gates.sh scripts/pod_w2_gemma3.sh`
    PASS。
  - `cargo test -q -p ferrum-models dump_stats_counts_nonfinite_values` PASS。
  - `cargo test -q -p ferrum-models nan_trace_selector_accepts_all_or_layer_lists` PASS。
  - `cargo test -q -p ferrum-models gemma3` PASS。

### 下一步

- 不开完整 W2 gate。下一次 GPU 只做小诊断:
  1. 先跑合成 `desc_act=true` GPTQ CUDA parity(补/跑 ignored test),证明
     ferrum act-order Marlin 是否已偏离 CPU reference;
  2. 若 parity 失败,修 `MarlinWeight` 保存 `g_idx`/sort indices,让
     vendored vLLM Marlin wrapper 使用原生 act-order(`a_tmp` scratch +
     `has_act_order=true`),或实现等价 kernel-side g_idx 路径;
  3. 只有合成 parity 过、Gemma3 early-smoke 首步 logits finite 后,才恢复
     W2 L2-L5 正式 gate。

## 2026-06-13(晚 IV)— W2 CUDA 早停:首步 logits 全 NaN,实例销毁

- Vast 实例 `40806710`(Iceland host 1647,1×RTX 4090,120GB,$0.402/hr)
  用于 W2 Gemma3-27B CUDA retry。Ferrum release CUDA build 完成,
  GPTQ/GGUF 下载完成;在 llama.cpp 仍编译时提前并行启动同参数
  `model_coverage_smoke` early smoke,避免等待 perf 前置项。
- 结果:Gemma3-27B GPTQ 仍不能转绿。服务能加载并进入首个 known-answer
  prefill,但 `logit_dump_smoke/logits.bin` 为 **262208/262208 全 NaN**:
  `finite=0, nan=262208, posinf=0, neginf=0`。`early_smoke.log` 记录
  `logits-topk known-answer-0-prefill: n=262208 finite=0 nonfinite=262208`,
  随后 known-answer 输出仍为空。
- 按 correctness first-stop,未进入 L5/perf。artifact 已回收到
  `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/`;核心证据包括
  `early_smoke.log`、`gates_early/nan_logits_summary.json`、
  `gates_early/logit_dump_smoke/logits.bin` 与 serve/build/download 日志。
- 已回收的 partial layer dump 显示 `embed.bin` 与 `layer_00.bin` 到
  `layer_07.bin` 全部 finite,但最终 `logits.bin` 全 NaN。NaN 不是
  tokenizer/embedding 入口即炸,下一步应定位 layer 8+ 或 final norm /
  tied lm_head / logit softcap 边界。
- 实例已销毁;Vast API 验证 `count: 0`。`w2_matrix.json` 仅更新
  `l2_gptq_cuda` 的失败证据指向本轮 NaN artifact;`l3_behavior`/
  `l4_agent` 仍引用上一轮完整 smoke 失败,`l5_concurrency` 与
  `perf_vs_llamacpp` 继续 pending。

### 下一步

- 不再重跑整条 W2 gate。先本地/小远端定位 Gemma3 GPTQ CUDA NaN:
  1. 用已有 layer dump 找首个 NaN 层/算子边界;
  2. 优先查 GPTQ desc_act permutation、Gemma3 tied lm_head / embed scale、
     `final_logit_softcapping` 和 CUDA dtype/scale 路径;
  3. 修到首步 logits finite 后,再开 1×4090 smoke;只有 smoke PASS 才跑
     L5/perf。

## 2026-06-13(晚 III)— W2 本地诊断加固:修一个采样 mask 缺口,下轮收 top-k

- 没有重开 GPU,先基于 `w2_gemma3_cuda_failure_2026-06-13` 失败证据做本地收口。
  W2 仍是 correctness blocking,不得转 pass。
- 代码修复:`SequenceState::requires_full_logits_for_sampling()` 现在会在 tokenizer
  暴露 base vocab 之外的可生成 control token 时强制 full logits。此前若
  `FERRUM_GREEDY_ARGMAX=1` 且模型 argmax 落在扩展/保留区,可能绕过
  `sample_with_processors` 的 extended-vocab mask。Gemma3 的
  model vocab/tokenizer base vocab 形态正好需要防这类风险。
- 诊断加固:`scripts/model_coverage_smoke.sh` 新增可选
  `FERRUM_SMOKE_LOGIT_DUMP_DIR`。设置后复用既有 `FERRUM_LAYER_DUMP`,
  在首个 known-answer 请求后打印 prefill `logits.bin` 的 finite/nonfinite
  统计与 top10 token id/logit。`scripts/pod_w2_gates.sh` 已为 W2 smoke
  开启该 dump,并在 smoke 首败时复制 `/tmp/ferrum_w1_smoke_8400.log`
  到 gate artifact 目录。
- 本地验证:
  - `cargo test -q -p ferrum-engine sample_allows_generated_control_tokens_above_base_vocab`
    PASS。
  - `cargo test -q -p ferrum-engine continuous_engine` PASS(18 tests)。
  - `bash -n scripts/model_coverage_smoke.sh scripts/pod_w2_gates.sh scripts/pod_w2_gemma3.sh`
    PASS。
  - `cargo fmt --all -- --check` PASS。
  - `scripts/w1_goal_validator.py` 仍 `MODEL_COVERAGE_W1 GOAL PASS`。
  - `scripts/w2_goal_validator.py` 仍 3/8,5 blocking cells(预期)。

### 下一步

- 下一次 1×4090 W2 重跑前先看 smoke top-k:
  - 若 top-k 集中在 `>= tokenizer_base_vocab_size` 或控制 token,优先验证本次
    full-logits/extended-mask 修复是否已把输出拉回正常文本。
  - 若 top-k 已是正常文本 token 但 decode 仍空,继续查 tokenizer decode /
    streaming delta。
  - 若 top-k 本身全异常/非数/同一保留 token,转向 GPTQ loader 或 Gemma3
    lm_head/logit softcap/量化 scale 路径。

## 2026-06-13(晚 II)— W2 Gemma3 CUDA gate 首败:加载成功,正确性失败,实例销毁

- Vast 实例 `40798977`(Iceland host 1647,1×RTX 4090,120GB,$0.402/hr)完成
  W2 重试并已销毁;API 验证 0 实例。artifact 已回收到
  `artifacts/w2_gemma3_cuda_failure_2026-06-13/`。
- 远端证据:
  - `build.ok`:Ferrum release CUDA build 完成。
  - `dl_gptq.ok` / `dl_gguf.ok`:circulus GPTQ 与 unsloth Q4_K_M 下载完成。
  - `llamacpp.ok`:llama.cpp `llama-bench` 构建完成。
  - `gates/session_metadata.json`:git SHA `86633c2d...`,dirty files,RTX 4090
    24GB,driver 565.77,nvcc 12.4 已记录。
- 环境修复:复现 W1 的 GeForce forward-compat `libcuda` 问题
  (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`);已在远端移走 `/usr/local/cuda/compat/libcuda*`,
  并固化进 `scripts/pod_w2_gemma3.sh`。
- Gate 结果:Gemma3-27B GPTQ **能加载并服务**,但 smoke 正确性首败:
  - known-answer **0/10**,所有答案为空字符串,`finish_reason=length`。
  - natural EOS / custom stop / multi-turn 失败;stream identity 与 max_tokens
    mechanics 本身通过。
  - required tool-call **0/10**(HTTP 400),strict json_schema **0/20**(HTTP 500)。
  - 按 GOAL 首败即停,**未进入 L5/perf**。
- 矩阵更新:`w2_matrix.json` 将 `l2_gptq_cuda`、`l3_behavior`、`l4_agent`
  标为 fail 并引用 smoke artifact;`l5_concurrency` 与 `perf_vs_llamacpp`
  仍 pending(正确性未过,不得 bench)。

### 下一步

- 不再重复整条 W2 gate。先本地/小远端诊断 Gemma3 GPTQ 空输出:
  1. dump prompt token ids + first-step logits/top-k/EOS/PAD ids,判断是模板/EOS
     还是 logits/quant 退化;
  2. 用 1B BF16/GGUF 已绿路径对比同 prompt,确认 CUDA GPTQ-only 问题;
  3. 重点查 desc_act GPTQ permutation、Gemma3 `final_logit_softcapping`、
     tied lm_head / embed scale、logit mask/EOS stop。

## 2026-06-13(晚)— W2 pod 停止后本地收口:证据仍缺 5 格,gate 脚本加固

- 已按 `ACTIVE_PODS.md` 记录:W2 Gemma3 pod `40770078` 在 run 4 中途按用户
  "pod 销毁停止" 指令销毁,API 归零验证 0 实例。该 pod 已把三个修复提交回
  当前分支:smoke 阶梯不再钉死 KV、27B 16-kv-head capacity math、CUDA
  `scale_inplace` 保持 device dtype。
- 当前权威验证状态:
  - `scripts/w1_goal_validator.py` → `MODEL_COVERAGE_W1 GOAL PASS`
    (72/72)。
  - `scripts/w2_goal_validator.py` → 3/8 satisfied,仍阻塞在
    `l2_gptq_cuda`、`l3_behavior`、`l4_agent`、`l5_concurrency`、
    `perf_vs_llamacpp`。27B GPTQ 已能加载并服务单请求,但没有完整 L2-L5
    gate artifact,不得转 pass。
- 本地加固:W2 pod gate 脚本改为原始判据对齐:
  - `bench-serve` 强制 `--fail-on-error --require-ci --seed 9271`
    且显式 `random 256/128`。
  - c=32 不再 best-effort;`l5_gemma3-27b-gptq_cuda.json` 必须包含
    c=1/4/16/32 全 cell,每 run 100/100 完成、零错误。
  - llama.cpp 同卡比值 `<0.5x` 会让脚本失败,不只打印 FAIL。
- 本地加固:W2 validator 不再只检查 artifact 存在。若矩阵 cell 标为 pass,
  validator 会检查 smoke log 的 `SMOKE PASS`,L5 JSON 的
  c=1/4/16/32、N=3、usage token count、零错误,以及 perf ratio ≥0.5。

### 下一步

- 若继续 W2,重新开 1×4090 pod 前沿用下方 W2 pod 执行合同;跑
  `scripts/pod_w2_gemma3.sh`,回收 `/workspace/w2/gates/` 后只按
  validator 可证明的 artifact 更新 `w2_matrix.json`。
- 不得在缺少 `MODEL_COVERAGE_W2 GOAL PASS` 前启动 W3 实现;W3 当前只保持
  `W3_CHARTER.md` 草案交付物。

## 2026-06-13(凌晨)— CUDA pod 批次收官:实例归零,9 个产品 bug,4 模型 CUDA 认证

**pod 全部销毁,API 验证 0 实例**(单卡 ~9h + 双卡 ~5h + 两台坏机即弃,
总花费 ≈ $7)。夜班战果:

- **CUDA 侧转绿**:R1-8B BF16 smoke 12/12 + L5(54.5/163.7/382.5);
  R1-Distill-70B 双卡 L2/L3 + L5(21.4/67.3/68.1);R1-Distill-32B GPTQ
  L2/L3(known-answer 10/10 @ kv8192×1);Qwen3-32B GPTQ L2(10/10 +
  tools 10/10);Qwen2.5-Coder-32B GPTQ 全梯一次过(含 schema 20/20)。
  M3 同 pod 基线锚点 c=32 556.5±84。
- **修复并验证的引擎 bug**(夜班新增,均已推送):流式 think 泄漏
  (distill 全家,70B 上 E2E 验证);Marlin n%256≠0 在 m>16 崩溃 →
  m≤16 分块复用已证 128×128 路径(dense-dequant 首版方案引入新问题已
  撤);GGUF 多卡 layers=auto 未物化;serve/pull 过期 alias 副本;
  pod 构建缺 vllm-paged-attn-v2 的硬报错;smoke harness 两处
  (异常计为 miss、KV 钉死与 autosizer 叠加)。
- **L1 终局(等批提案 #3)**:20 分叉点中位 logit 间距 0.75/min 0 →
  BF16 平票翻转,非数值缺陷;跨实现逐位一致原则上不可达。
- **OPEN ISSUES(下一会话,按正确性优先排序)**:
  1. strict json_schema 在 32B-GPTQ 上间歇 500(~25-30% 请求,
     R1-32B 15/20、Qwen3-32B 14/20;R1-8B-BF16 与 Qwen2.5-Coder-32B
     同路径 20/20)+ 500 不落引擎日志的可观测性缺口。
  2. Coder-30B jart25-GPTQ 的 CUDA chat 首 token 即 EOS(Metal GGUF
     同 prompt 正常、CUDA 随机上下文 L5 正常);prime suspect =
     仓库自带魔改模板(sha 30b8ba8f ≠ 官方 5a38bfa0);待 prompt-ids
     双端 dump 对照。
  3. 同构 perf:Coder-30B c=32 411.6 vs M3 556.5(0.74×,宿主机噪声
     stddev 15% 需复核)— 超 ≤10% 判据,按 GOAL 记接入问题待查。
  4. CUDA autosizer:reasoning/32B-GPTQ 的 (seqs×capacity) 联合推导
     (512/seq 400、0-blocks 报错两态)。
- distill 系 tools-in-think 行为差异落档:32B 注入式 tools 10/10 可用,
  70B 把调用写进未闭合 think(0528 系 10/10)→ README 按"agent 分级"
  如实标注。

## 2026-06-12(夜)— 用户决策落地:32B Metal 收束 + CUDA pod 批次启动

- **用户指令**:32B 稠密不再在 32GB Mac 上折腾("同架构已证即可"——
  Qwen3-14B/R1-8B 的 Metal pass 即同架构证明);Vast 已充值,批准开
  GPU;**严格要求高效利用 + 异步并行 + 空闲即毁 + 结束全毁**(用户刚
  手动清理了数台未销毁实例)。API 已核实当前 0 实例。
- 矩阵落实:R1-Distill-32B 与 Qwen2.5-Coder-32B 的 `l2_gguf_metal`
  waived(同架构证明 + 部署无场景);Qwen3 dense 行按 14B 证据 pass。
- L5 Metal 批次进行中:Coder-30B ✅(c1/4/16/32 全零错)、R1-8B ✅
  (22.9/23.4/54.2 tok/s)、14B/Mistral-Small/Magistral 排队自动跑。

### CUDA pod 批量执行合同(开 pod 前置,GOAL 模板)

```text
Lever: W1 CUDA gate 批量 —— 单卡 4090:L1 代表(R1-8B BF16 byte-equal
  N≥20 vs transformers)+ L2-GPTQ smoke(R1-Distill-32B/OPEA、
  Qwen3-32B/JunHowie、Qwen2.5-Coder-32B/官方、Qwen3-Coder-30B/jart25)
  + R1-8B CUDA BF16 smoke + 各模型 L5(c=1/4/16,30B 级补 32)
  + C7/G0 存量回归(M2 Llama-8B-INT4、M3 Qwen3-30B-A3B-GPTQ floor)。
  双卡 2×4090:R1-Distill-Llama-70B GGUF 4bit layer-split smoke + L5。
Expected gain: ~18-20 个 gate cell 转绿,W1 除 README 外收口
Files: scripts/model_coverage_smoke.sh(复用)+ pod 上逐步驱动
Correctness gate: 每模型 smoke 全绿;首败即停该模型并记录
Benchmark gate: L5 全 cell 100%/零错误;同构 ≤10%;C7/G0 不回退
Budget cap: ≤2 pod-day;预计单卡 ~$0.35-0.5/hr + 双卡 ~$0.7-1/hr,
  目标一晚收口(~$10-25)
Stop condition: 单模型卡壳 >4h 降级记录;pod 空闲即毁;
  结束后 API 验证实例数 = 0(用户硬性要求)
```

## 2026-06-12(晚 II)— Mistral 线 2/3 收口;Devstral 2 降级(mistral3);[THINK] 修复

- ✅ **Mistral-Small-3.2 全过**(10/10,首个满足 L4 schema 20/20 新判据的
  模型);✅ **Magistral 12/12 全过**——其 reasoning 走 `[THINK]` 特殊
  token,暴露并修复了一个普适 bug:**skip-special 解码会吞掉标 special
  的 think 标记**,思考文本漏进 content(Qwen3 标 special 的 `<think>`
  同样潜伏)。修复:tokenizer 解码按标记 id 分段、规范化为
  `<think>/</think>` 再拼接,下游零改动,带单测。
- 🔻 **Devstral 2 按 GOAL 卡壳规则降级到 W2 末尾**:GGUF arch 是
  **mistral3**(YaRN factor 48 / 原窗 8192 / `attention.temperature_scale
  0.1`,全在 `mistral3.*` 命名空间)。loader 此前静默走 llama-family
  路径 → 退化输出(known-answer 3/10、重复循环)。已加**未知架构硬报错**
  守卫(带单测)——明确不支持好过悄悄输出垃圾。实现 mistral3 = 新
  rope/注意力数学,超出 W1 SMALL 预算;W2 与 Gemma 3 异构注意力地基
  一并评估。
- 验证器 19/63 → **30/72**(Devstral 拆分出独立降级行)。
- Mistral 线剩余 cell:L5 并发 + perf(pod)+ README。

## 2026-06-12(傍晚)— 修订批准落实;32B 稠密 Metal 诊断(需重启)

- **两个 GOAL 修订经用户批准并写入 GOAL.md 修订记录**:L1 按代码路径
  代表执行(5 个不可行 cell 转 waived,R1-8B 是 dense 路径代表);
  L3/L4 逐模型载体改为 smoke 阶梯(判据数字不变,schema 升 20/20)。
  R1-8B 与 Coder-30B 的 L3 cell 凭既有扩展 smoke 证据转 pass。
  验证器 12/63 → **19/63**。
- **Qwen3-14B Metal smoke 10/10 过**(cell 与 32B 同行,等 32B)。
- **32B 稠密 Metal 诊断**:R1-Distill-32B smoke 两次超时后实测解码
  **0.14 tok/s**(TTFT 5.2s 正常)——每 token 把被驱逐的 18GB mmap 权重
  从 SSD 重读(~2.6GB/s = SSD 速度)。llama.cpp 同文件对照**同样卡死**
  (22 CPU 分钟未完成加载)→ 非 ferrum 接入 bug。根因:早上 KV 池
  thrash 事故在压缩器里留下 ~9GB 系统级残留,可用内存 < 模型工作集。
  **需要用户重启后公平复测**;若干净 32GB Mac 仍装不下 32B 稠密 + 服务
  开销,则 32B 级稠密(R1-32B / Qwen3-32B / Qwen2.5-Coder-32B)的
  Metal cell 按修订精神 waive 给 CUDA GPTQ lane。
- 教训入库:32GB 机器一次只跑一个重负载(13GB 下载 + 18GB 常驻模型的
  page-cache 互相驱逐就是第一次超时的原因)。

## 2026-06-12(午后 II)— R1-8B L2-Metal cell 转绿;HF_ENDPOINT 落地

- ✅ **R1-8B 扩展阶梯 12/12 全过**(known-answer 10/10 语义正确 + stop
  不漏 + max_tokens 守预算 + reasoning/stream/tools/schema 机制),
  `l2_gguf_metal` cell 转 pass,验证器 11/63。
- **`HF_ENDPOINT` 支持落地**(huggingface_hub 同约定):本网络实测
  hf-mirror 直连 2.08MB/s vs 代理 0.156MB/s(**13×**)。Coder-30B 下载
  已切镜像直连续传(ETag 与 hub 一致,blob 无缝续);预计 ~1h 内落盘。
- R1-Distill-Llama-70B 模板与 R1-Distill-Qwen-32B fixture 同 hash 同
  EOS(`56a1447ad31926fd`),L0 模板面由现有 fixture 覆盖。

## 2026-06-12(午后)— gate 矩阵 + 验证器落地;两个 GOAL 修订提案待批

- **`w1_matrix.json` + `scripts/w1_goal_validator.py` 落地**:7 个模型 ×
  9 个 cell = 63 cell,当前 10/63 满足(L0 ×6 + waived ×4)。验证器是唯一
  允许打印 `MODEL_COVERAGE_W1 GOAL PASS` 的程序;cell 必须 pass(带
  artifact)或 waived(带理由),引用的 artifact 必须存在。
- **L0 完成度**:43/43 golden 全过(9 个 fixture 模型,新增 Mistral 线
  ×3 + Llama-3.1;`strftime_now` 时钟注入 + `tojson(indent=N)` 两个真实
  渲染缺口由 L0 抓出并已修,commit `c8f3703e`)。
- smoke 阶梯扩充:known-answer 1x→10x(对齐 L2 判据),新增自定义 stop
  机制断言 + max_tokens 截断断言(L3 缺口)。
- **修订提案 #1(L1,需用户决定)**:L1 BF16 byte-equal 对 14B+ 在现有
  硬件上物理不可行(14B BF16=28GB>24GB 单卡;32B=64GB;70B=140GB)。
  提案:L1 按"代码路径代表"执行——每条代码路径取硬件放得下的最大代表
  (Qwen3 dense → 8B/0.6B 已有 reference_match;Qwen3-MoE → 30B-A3B 需
  pod 上 BF16?同样放不下,24GB 单卡上 MoE BF16 60GB 也不可行 → MoE 路径
  L1 只能 waive 到"Mac/CPU 逐层激活对照"或双卡)。大尺寸模型靠"同代码
  路径 + L2 行为对照"传递。**未批前 5 个 l1_bf16 cell 保持 pending。**
- **修订提案 #2(L3,需用户决定)**:blast-radius 套件断言对 0.6B 哨兵
  模型定制(canonical id、即答行为),对 8B-32B reasoning 模型强行参数化
  会又重又脆。提案:L3 判据改为"model_coverage_smoke 的 L3 段全绿"
  (多轮/stream/自然 EOS/自定义 stop/max_tokens/reasoning 提取,行为
  断言与套件同源),blast-radius 套件保持小模型哨兵职责(引擎级回归)。
  **未批前 L3 cell 不以 smoke 结果记 pass。**

## 2026-06-12(午前)— L0 扩面:模板同一性 + Mistral/Llama golden

- **模板同一性(HF raw tokenizer_config,sha256 前 16 位)**:
  Qwen3-0.6B / 14B / 32B 模板逐字节同一份(`a55ee1b1660128b7`,EOS
  `<|im_end|>`);R1-Distill-Qwen-14B / 32B 同一份(`56a1447ad31926fd`,
  EOS `<｜end▁of▁sentence｜>`)。**结论:Qwen3-14B/32B 与 R1-Distill-14B
  的 L0 由现有 golden fixture 直接覆盖**,各自只剩 per-model
  EOS/generation_config 断言(T4 机制已通用)。
- Mistral 24B 线 + Llama-3.1 golden fixture 生成中(来源 = serve 实际用的
  tokenizer 仓库:unsloth 镜像 ×2 + mistralai 上游 ×1 + unsloth Llama)。
  环境坑:huggingface_hub 新版走 httpx,SOCKS 代理需要 `httpx[socks]`
  (socksio),`pysocks` 只管 requests。
- Coder-30B GGUF(17.28GB)断点续传循环推进中(代理频繁断流,每次
  尝试落 1–3GB,`.incomplete` blob 在涨)。

## 2026-06-12(深夜 III)— ✅ R1-0528-Qwen3-8B GGUF Metal smoke 全绿

**W1 第一个模型过本地阶梯**:`FERRUM W1 SMOKE PASS: deepseek-r1:8b-q4_k_m`
(8/8:known-answer、自然 EOS、reasoning 提取、think 不漏入 content、
多轮记忆、stream==non-stream、required tool 10/10、strict json_schema
10/10)。证据:`artifacts/smoke_deepseek-r1-8b-q4_k_m_metal_2026-06-12.txt`。
serve 参数:`--kv-capacity 8192 --max-num-seqs 4`(见下条 thrash 诊断)。
注意:这是 L2/L3/L4 的可跑子集;最终认证仍需完整套件
(json_schema 20/20 走 server_structured_output)+ CUDA 侧 gate。

## 2026-06-12(深夜 II)— GGUF pull 产品缺口修复 + KV 池 thrash 诊断

R1-8B GGUF smoke 调试中钉死三个真实产品问题(全部影响 W1 每个 GGUF alias):

1. **pull sidecar 全量下载 bug(已修)**:GGUF 仓库缺 tokenizer.json 时,
   兜底走 `HfDownloader::download(sibling)` —— 会把 sibling 的 **safetensors
   权重(8B≈16GB)整库拉下来**,只为拿 tokenizer。磁盘紧张时必死,这就是
   此前需要手工拷 tokenizer 的根因。新增
   `HfDownloader::download_sidecar_files`(只拉指定小文件),pull 改用之,
   清单补上 `generation_config.json`(EOS 解析第一优先级)+
   `chat_template.jinja`。
2. **bartowski 系 sibling 映射全断(已修)**:HF API 实测 9 个 W1 GGUF 仓库
   **全部不带 tokenizer.json**,sibling 兜底是必经之路;而 strip `-GGUF`
   约定对 bartowski/*(无 safetensors 镜像)全部失效。
   `tokenizer_sibling_repo` 加显式映射(2026-06-12 HF API 逐个核实
   tokenizer.json 存在):Qwen2.5-Coder→Qwen 官方;Mistral-Small-3.2 /
   Magistral→unsloth 镜像(**mistralai 上游只有 tekken 格式,无 HF
   tokenizer.json**);Devstral 2→mistralai 上游;Llama 系→unsloth 镜像
   (meta-llama 上游 gated)。
3. **`--kv-capacity` 单独抬高 = 32GB Mac 内存灾难(smoke 已加防护)**:
   KV 池 = `max_num_seqs × kv_capacity`。autosizer server 档默认
   (32, 512)≈2GB;只把 capacity 提到 8192 会得到 32×8192≈36GB 池
   (8B/36 层/8KV头/128hd),Metal 分配直接把机器打进内存压缩 thrash
   (实测:health 能过、首个请求触发 `ensure_kv` 后 600s 超时,压缩器
   存页 38GB)。smoke 的 reasoning 档改为
   `--kv-capacity 8192 --max-num-seqs 4`(池 32K token,与默认同量级)。
   **autosizer 产品缺口升级**:reasoning 模型需要的不是"调大 capacity",
   而是 (seqs × capacity) 在显存预算内的联合推导 + 长上下文低并发档位;
   `--kv-capacity` 作为独立产品 flag 缺少联动护栏。

## 2026-06-12(深夜)— 本地验证推进与环境修正

- **修正**:HF 缓存里的 R1-0528-8B / R1-Distill-32B / Qwen3-Coder-30B /
  Qwen2.5-Coder-32B 仅为 6–11MB 元数据壳(config/tokenizer),**无权重**。
  W1 端到端一律需要下载。
- 磁盘:删除 target/debug(15GB)后约 16GB 可用;R1-8B Q4_K_M GGUF(~5GB)
  下载中(第一次因网络/代理 "error decoding response body" 失败,重试中);
  Qwen3-Coder-30B Q4_K_M(~18.6GB)需要更多空间——待用户清理或换机。
- 新增 `scripts/model_coverage_smoke.sh <alias> [--reasoning]`:
  L2/L3/L4 阶梯(known-answer + 自然 EOS / 多轮 / stream==non-stream /
  reasoning 提取 / required tool 10x / strict schema 10x),所有 W1 模型复用。
- 下一步(按序):R1-8B GGUF smoke(--reasoning)→ 视磁盘跑
  qwen3-coder:30b-q4_k_m → W1 收尾(README 矩阵 + 验证器)→ pod 合同。

## 2026-06-12(深夜)— blast-radius 存量回归结果

T3/T4/T5 处于 EOS/stop/模板爆炸半径,全套件(release + Metal,真模型)结果:

- ✅ chat_smoke 13 / server_smoke 10 / chat_pty 3 / chat_stress 2 / server_stress 2
- ✅ server_openai_compat 7/7 — 其中两处修复:
  - `test_python_openai_sdk_*`:本机环境缺 `openai`/`socksio`(SOCKS 代理),
    已 pip --user 安装,非代码问题。
  - `test_openai_client_tools_stream_*`:模板修正后 prompt 与 transformers
    字节一致(差 1 token),0.6B 贪心解码改为真的调用工具——服务器输出了
    规范的 tool_calls delta + finish=tool_calls + usage。测试断言改为
    "文本 XOR 合法工具调用"(7c69e2a7),钉住流式机制而非模型选择。
- ⏸ reference_match:1 行 drift **等用户审核后 re-baseline**(分类器按
  CLAUDE.md 拦截了自动重置,正确):case `qwen3-0.6b-arith-2-plus-3`
  内容与 token 数完全一致,仅 `finish_reason: length → stop` ——
  这是 EOS 修复的直接证据(此前 tokenizer 探测不到 Qwen EOS,自然停止
  被误归因为 budget 耗尽)。审核通过后执行:
  `FERRUM_UPDATE_FIXTURES=1 cargo test --release -p ferrum-cli --features metal --test reference_match -- --ignored --test-threads=1`

## 2026-06-12(晚)

- **T5 完成:L0 golden 基建落地并修出 7 处真实偏差**(PR #234,auto-merge):
  - `scripts/gen_chat_template_goldens.py` + 5 模型 23 用例 fixture 入库,
    `chat_template_golden` 测试 23/23 与 transformers 字节级一致。
  - 修复项:trim_blocks/lstrip_blocks 对齐 transformers;tojson 改 Python
    json.dumps 风格(自定义 filter);minijinja+serde_json 双 preserve_order
    (minijinja 对 Rust struct 字段强制字母序,tools 改为有序 JSON 值进模板);
    `PromptMessage::new` 不再急切剥离 assistant 历史的 `<think>`
    (剥不剥是模板的政策:DeepSeek 剥、Qwen3-Coder 保留)。
- **W1 全模型 alias 配齐**(均经 HF API 核实文件名):safetensors/GPTQ/GGUF
  三组,含 deepseek-r1:8b/14b/32b、qwen3-coder:30b、qwen3:14b/32b、
  qwen2.5-coder、mistral-small/devstral/magistral 24b 线。
- **YaRN clamp 落地**:不支持的 rope_scaling → `max_seq_len` clamp 到
  `original_max_position_embeddings` + 启动警告(R1-0528 由 131072 clamp 到
  32768),含单测。
- **环境约束发现**:本机磁盘 100%(HF 缓存 42GB);已清理 target/debug/
  incremental 释放 7.3GB。**新模型权重无法下载**,但缓存中已有
  R1-0528-Qwen3-8B、R1-Distill-32B、Qwen3-Coder-30B、Qwen2.5-Coder-32B 的
  safetensors + blast-radius 三小模型 → 本地验证用缓存模型推进。
- blast-radius 套件(chat_smoke/pty/stress + server 三件 + reference_match)
  在后台执行中——T3/T4/T5 改动处于 EOS/stop/模板爆炸半径,存量回归必须绿。

## 2026-06-12(下午)

- **T3 完成并提交(`778082a6`)**:minijinja-contrib pycompat 接入;模板渲染失败/
  渲染为空改为硬错误(消灭静默 fallback);tools-unaware 模板改为"注入工具 spec 后
  仍走模型模板渲染"(此前会静默丢弃工具定义)。新增 5 条防回归测试,
  workspace 测试全绿。
- **T2 完成(外部核查,逐仓库 config 原文)**,三个重要修正:
  - **GLM-4.7-Flash 是 MLA 注意力**(q_lora 768/kv_lora 512)+ noaux_tc 路由 + MTP,
    接入成本 MEDIUM→LARGE,已从 W2 移出(W2 只剩 Gemma 3 27B)。
  - **R1-0528-Qwen3-8B 无 Marlin-clean GPTQ**(QuantTrio 版 sym=false+4/8 混合)
    → CUDA 走 BF16 + GGUF。**Devstral 2 同样无 Marlin GPTQ** → GGUF/BF16。
  - W1 各模型的 Marlin-clean GPTQ 仓库已逐一锁定(jart25 / OPEA / JunHowie /
    Qwen 官方 / Intel AutoRound),写入 GOAL.md UNVERIFIED #4。
  - Qwen3.6 无官方 GPTQ-Int4(仅 FP8);官方 GPTQ 停在 Qwen3.5 代。

## 2026-06-12

- GOAL.md 建立并提交(分支 `goal/model-coverage-20260612`)。
- 验收 gate 定义(L0–L5 正确性 + 分类性能门槛)写入 GOAL.md。
- UNVERIFIED 落证(本地 4 项,全部完成):
  - #1 YaRN:不支持(仅 Llama3 变体);发现 max_seq_len 不 clamp 的隐患,
    已追加为 W1 公共工程项。
  - #2 AWQ:无 loader,纯 Future 注释;维持 defer。
  - #3 gguf arch 白名单:`qwen3|qwen3moe|qwen2|qwen|llama|mistral`,
    W1 够用,GLM(W2)需新增。
  - #8(新增)模板渲染失败静默 fallback 实锤(`chat_template.rs:226/488`),
    待 T3 消灭。
- UNVERIFIED #4/#5/#7(GPTQ group size / GLM config / Qwen3.6 官方 GPTQ)
  由后台 web 核查进行中。
- 任务分解:12 个任务建于会话任务系统(T1–T12),T1 完成。

### 下一步

- T3:模板引擎改造(minijinja pycompat 路线 + 渲染失败显式报错 + 防回归测试)。
- T4:EOS/BOS generation_config 审计。
- T5:L0 golden 测试基建(需本机 Python transformers 生成 fixture)。

### 阻塞项(预先声明)

- CUDA 侧 gate(L2-GPTQ / L5 / C7 回归)需要 4090 pod:开 pod 前按 GOAL
  执行合同填表并征得用户预算批准(CLAUDE.md 要求)。当前无可用 pod
  (上一台 38237968 已失;见 memory)。本地(Metal/CPU)可推进项先行。

## 2026-06-13 15:25 — W1 GOAL PASS

- `scripts/w1_goal_validator.py`: **72/72 cells satisfied →
  `MODEL_COVERAGE_W1 GOAL PASS`**。
- 最后 6 cell(32B 三连 l5_concurrency + perf_same_arch)由冰岛 pod
  40751023 一次干净会话收齐:
  - `l5f_r1-32b_cuda.json` c=1/4/16/32 = 40.8/116.6/248.9/300.6 tok/s,
    1200 请求 0 错误。
  - `l5f_qwen3-32b_cuda.json` c=32 = 273.6 tok/s,0 错误。
  - `l5f_qwen25-coder-32b_cuda.json` c=32 = 257.1 tok/s,0 错误。
  - perf_same_arch(修订 #4 判据):三方互校最差偏差 8.5% ≤ 10% →
    `W1_PERF_SPREAD PASS`。
- GPU 纪律:三台问题宿主(台湾 docker_build 坏 / 阿根廷不开机 /
  冰岛 cuInit=804)处置后,**API 归零验证 0 实例**。本夜累计 GPU 支出
  约 $9。
- 新宿主病理学(已固化进 `pod_w1_final_armored.sh`):
  - cuInit=804 = 容器 compat libcuda(550)压住宿主驱动,GeForce 不在
    compat 支持表;删 compat so + ldconfig 即愈。
  - rsproxy.cn 在部分欧洲宿主被 TLS 劫持;脚本现在先探测 crates.io
    再选镜像。
  - hf xet/hf_transfer 在该宿主网络下饿死(0 MB/s);关 xet + 关
    hf_transfer 的普通 HTTP 路径反而跑满 3.6 Gbps。
- 待办移交(不阻塞 W1,记录在 GOAL.md 开放问题):schema-500、
  Coder jart25 CUDA chat、CUDA autosizer、Metal L5 复跑(等本机恢复)。

### 下一步

- W2:Gemma 3 27B 家族接入(SWA 5:1 / 双 rope / GeGLU / 三明治 norm /
  query_pre_attn_scalar),本地 Mac/CPU dump 对照先行,CUDA 验证晚开 pod。
- W3:DeltaNet 调查(W1+W2 后解锁)。

## 2026-06-13 — W2 Gemma3 接入(本地段完成)

- W2-1 实现:Gemma3 经 config 门控并入 LlamaFamilyModel(5:1 SWA 逐层
  调度 / 双 rope 表 + Linear scaling / GeGLU / 三明治 norm / (1+w) 与
  q_scalar 载入期折叠 / embed×√h)。batched/varlen/paged 快路对
  sandwich 家族构造期禁用(防静默错误),W2-3 再接。
- W2-2 验证:L0 golden 4 例字节相等;L1 dump 对照 CPU+Metal 双 PASS;
  greedy 18/20 byte-equal,2 例 HF top1-top2 gap=0.25(一个 bf16 ulp)
  平局翻转(修订 #3 方法)。证据 `artifacts/gemma3_l1/`。
- 顺带修复两个 Metal 内核潜伏 bug(Gemma3 首次踩出):
  flash_attn 简单核 acc[4] 在 head_dim=256 寄存器越界(→acc[8]);
  gelu_tanh fast-math 溢出 NaN(→clamp)。微基准 5 项钉死
  (`gemma3_metal_ops_test.rs`)。
- 27B 量化落证:ISTA-DASLab 是 compressed-tensors(不可直载);
  **circulus/gemma-3-27b-it-gptq = 经典 GPTQ 4b/g128/sym/desc_act=true,
  纯文本导出 `model.*` 命名** — ferrum perm-aware Marlin 已支持
  desc_act(quant.rs:151)。GOAL 的"ISTA GPTQ"假设据此修正。
- W2 矩阵 + 验证器就位(`w2_matrix.json` + `w2_goal_validator.py`),
  当前 2/8(l0/l1 pass)。

### W2 pod 执行合同(开 pod 前按 CLAUDE.md 填)

```text
Lever: Gemma3-27B CUDA gates(L2 GPTQ known-answer → L3 行为 → L4 agent
  → L5 bench)+ 同卡 llama.cpp Q4_K_M decode ≥0.5× 对照
Expected gain: w2_matrix 6 个 pending cell 出 pass/fail 结论
Files: scripts/pod_w2_gemma3.sh(armored 模式复用 W1 套件)、
  model_coverage_smoke.sh、bench-serve、pod 端构建 llama.cpp
Correctness gate: known-answer 10/10 + smoke 机制全绿,任一 rung 失败
  即停(不进 bench)
Benchmark gate: bench-serve c=1/4/16/32 零错误;llama.cpp 同卡比 ≥0.5
  (0.5–0.8 记 known-gap 不阻塞)
Budget cap: 1 pod-day 硬顶;目标 ≤6h(约 $2.5,单卡 4090)
Stop condition: 正确性 gate 失败 → 停手出报告;8h 无进展 → 销毁重估
```

## 2026-06-14 — 发布级推进 II:Gemma3 CUDA device F32 residual shadow

- 实现:Gemma3 sandwich-norm 残差流在 CUDA 上改为设备侧 F32 shadow,只把
  norm 后的投影输入物化回常规 activation dtype。覆盖 `prefill_internal`、
  `decode_internal`、speculative `forward_verify` 和 layer-split stage helper;
  CPU/Metal 保持原 host/default fallback。batched/varlen 快路仍在构造期对
  sandwich 家族禁用,避免未实现 Gemma 语义时静默走错路径。
- CUDA backend 增加 `sandwich_norm.cu` 三个 helper kernel:
  activation→F32 shadow、activation RMSNorm→F32 branch、F32 shadow
  RMSNorm→activation;同时 `rms_norm` 支持 F32 typed buffer,`copy_slice`
  支持 F32→F32。
- 目标意义:移除上一轮 W2 c=32 证据里的每层 D2H/F32 host shadow sync/copy
  热路径,为后续同卡 A/B 重新测 `Ferrum / llama.cpp >= 0.8x` 做源代码准备。
  这仍不是发布级通过声明;W2 release-grade 需要
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -q --workspace --all-targets` PASS。
  - `cargo test -q -p ferrum-kernels --tests` PASS。
  - `cargo test -q -p ferrum-kernels --test cuda_activation_precision` PASS
    (本机默认特性下 0 CUDA tests)。
  - `cargo test -q -p ferrum-models --tests` PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS:`MODEL RELEASE GRADE GOAL SELFTEST PASS`。
  - `python3 scripts/release/selftest_g0_validators.py`
    PASS:`G0 VALIDATOR SELFTEST PASS`。
  - `python3 scripts/w2_goal_validator.py`
    PASS:`MODEL_COVERAGE_W2 GOAL PASS: docs/goals/model-coverage-2026-06-12`。
- 本机 CUDA feature check 受环境阻塞:
  `cargo check -q -p ferrum-kernels --tests --features cuda` 在 build script
  阶段因缺少 `nvcc`/`nvidia-smi` 失败,未能在本机编译 CUDA 源码。下一步需
  4090 CUDA pod 运行 feature build、CUDA precision tests、Gemma3 smoke 和
  release-grade W2 manifest/gate。

## 2026-06-18 — W3 Qwen3.5 reference product path + validator hardening

- Qwen3.5/Qwen3.6 走当前抽象接入: `Architecture::Qwen35/Qwen35Moe`、
  `Qwen35TextConfig`、`ModelDefinition` 的 `ferrum_qwen35_text_config`、
  `Qwen35W3Executor`、`RecurrentStateSpec/Manager`、以及 `ferrum run` /
  `ferrum serve` 产品入口。当前执行路径是显式 `--qwen35-reference` 的
  CPU/FP32 reference runtime,不是 CUDA/Metal 发布执行路径。
- 已提交并 push:
  - `99ddd18b test(w3): add Qwen3.5 reference run smoke`
  - `e3976153 test(w3): add Qwen3.5 reference serve smoke`
  - `38605a9c test(w3): add Qwen3.5 product smoke artifact`
- 新增 artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_product_smoke_local_20260617T222748Z/`
  记录 toy Qwen3.5 safetensors/tokenizer/config,真实 `ferrum run`,
  真实 `ferrum serve` non-stream + stream SSE,以及
  `w3_s2_whole_model_product_path.json`。
- 本 checkpoint 强化 `scripts/release/model_release_grade_goal_gate.py`:
  W3 final gate 不再只检查 S2 product artifact 是否存在,而是校验
  `runtime_surface`、空 `hidden_env`、`ferrum_run` assistant 输出、
  `ferrum_serve` non-stream 输出、stream usage、以及 exactly-one `[DONE]`。
- 本地验证:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py`
    PASS。
  - `git diff --check -- scripts/release/model_release_grade_goal_gate.py`
    PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS:`MODEL RELEASE GRADE GOAL SELFTEST PASS`。
  - 既有 W3 S2 product artifact 结构探针 PASS:
    `W3 S2 PRODUCT ARTIFACT STRUCTURE PASS`。
- 限制:这不是 W3 release-grade 完成声明;仍缺真实模型/真实后端的 W3 L0-L5
  正确性矩阵、同硬件性能矩阵、以及最终
  `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-18 — W3 Qwen3.5 backend weight materialization boundary

- Qwen3.5/Qwen3.6 继续按当前架构抽象推进,本 checkpoint 新增
  `Qwen35BackendModel<B>` 作为 backend-native 权重 materialization 边界:
  从 `ModelDefinition` 解析 `Qwen35TextConfig`/`LlmRuntimeConfig`,校验
  `Qwen35ResolvedWeightPlan`,再通过通用 `WeightLoader<B>` 生成
  `Qwen35ModelWeights<B>`。
- 新入口保留两层边界:
  - `from_definition_with_loader(...)` 用于测试/自定义 loader/CPU 最小验证。
  - `from_definition_with_native_safetensors(...)` 用现有
    `NativeSafetensorsLoader` 和 safetensors inventory,为后续 CUDA/Metal
    backend executor 接入复用当前 quantization/loader 抽象。
- 本 checkpoint 没有打开默认产品路径。`ferrum run` / `ferrum serve`
  对 Qwen3.5/Qwen3.6 仍保持显式 `--qwen35-reference` CPU/FP32 reference
  guard;原因是 backend prefill/decode、linear/full attention state cache、
  MoE/shared expert forward 尚未完成,不能把只有权重加载的模型注册成
  release backend executor。
- 本地验证:
  - `cargo fmt --all` PASS。
  - `cargo test -p ferrum-models qwen35_backend_model -- --nocapture` PASS:
    2 passed,0 failed。
- 限制:这不是 W3 release-grade 完成声明;仍缺真实 Qwen3.5 GPTQ/full-size
  L2 known-answer、L3/L4/L5 产品正确性和性能证据,以及最终
  `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-18 — Vast cleanup and W3 runtime status clarification

- 用户询问为什么 GPU 还显示一台、为什么 W3 没有整模型跑起来。复核结果:
  最近的 CUDA 有效证据是 W3 S0 native CUDA/PTX delta-rule microbench,不是
  W3 整模型 `ferrum run`/`ferrum serve` 执行。
- 已有 CUDA S0 artifact:
  `artifacts/w3_delta_rule_s0_cuda_20260617T203149Z_c8b8da1f/`,
  PASS line:
  `W3 DELTA RULE S0 MICROBENCH PASS: /workspace/w3_delta_rule_s0_cuda_20260617T203149Z_c8b8da1f`;
  manifest 记录 `ptx_arch=sm_89`,CUDA max_abs error 约 `3.0e-9`。
- 没有继续在 GPU 上跑 W3 整模型的原因:当前 product registry 仍显式 guard
  Qwen3.5/Qwen3.6 默认执行路径,只允许 `--qwen35-reference` CPU/FP32;
  backend prefill/decode、recurrent state cache 写回、linear/full attention
  state cache、MoE/shared expert forward 尚未接入。此时开 CUDA 跑整模型
  只会得到已知的 unsupported guard,不是有效 release evidence。
- Vast 清理:
  - API 首次复核显示 `41287720` 为 `cur_state=stopped` /
    `actual_status=exited`,不是 running GPU。
  - 已执行 `DELETE /api/v0/instances/41287720/`,返回 `success=true`。
  - 随后 `GET /api/v0/instances/` 返回 `INSTANCE_COUNT 0`。
- 结论:当前没有 Vast GPU 实例在 running/stopped/loading。下一步继续本地把
  W3 backend executor 的 recurrent-state handle/manager 边界补齐;只有能产出
  新的最小 CUDA 正确性证据时再按 paid GPU contract 开机器。

## 2026-06-18 — W3 Qwen3.5 backend recurrent-state handle boundary

- 实现 `Qwen35RecurrentStateHandle<B>` 和
  `Qwen35RecurrentStateManager<B>`,把 Qwen3.5 recurrent state 从模型内部
  cache 结构推进到通用 `RecurrentStateHandle/Manager` 抽象。ContinuousBatch
  后续可以持有 trait object,executor 可以 downcast 回 Qwen35 typed handle
  访问 backend-native state cache。
- `Qwen35RecurrentStateCache::from_spec` 改为使用 backend typed allocator
  `B::alloc_typed(...)`,不再用 `B::from_slice(&zeros_f32)` 作为 CPU-friendly
  占位。当前支持 FP32 和 FP16/BF16(按现有 backend dtype tag 落到 F16 storage);
  FP8 等未实现 dtype 会硬失败,避免 release path 静默用错状态格式。
- Manager 覆盖 request-id duplicate guard、memory/slot capacity accounting、
  deallocate/reset invalidation、clone_handle 共享 cache 而非深拷贝 backend
  buffer。这是 W3-S0/S2 之间的 executor 状态生命周期边界,不是整模型
  CUDA/Metal forward 接入。
- 本地验证:
  - `cargo fmt --all` PASS。
  - `git diff --check -- crates/ferrum-models/src/models/qwen35.rs crates/ferrum-models/src/models/mod.rs` PASS。
  - `cargo test -p ferrum-models recurrent_state -- --nocapture` PASS:
    7 passed,0 failed。
  - `cargo test -p ferrum-models qwen35_backend_model -- --nocapture` PASS:
    2 passed,0 failed。
- 限制:这不是 W3 release-grade 完成声明;仍缺 backend prefill/decode 写回、
  linear/full attention state cache、MoE/shared expert forward、真实模型
  run/serve L2-L5、性能 baseline 和最终
  `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-20 — W3 Qwen3.5 CUDA prefill profiler checkpoint

- 源码 checkpoint:
  - `050d73f3 perf(qwen35): add prefill stage profiler`
  - 在 `Qwen35BackendModel::forward_stateful_chunk_taken` 增加诊断开关
    `FERRUM_QWEN35_PREFILL_PROFILE=1`,按 backend timer 记录 embedding、
    每层 linear/full attention、final norm、final token gather、lm_head、
    readback,并写入 `qwen35_prefill_prof` profile JSONL 事件。
  - 默认产品路径不变;该 env 只用于诊断,不是用户必须设置的行为开关。
- 本地验证:
  - `cargo check -p ferrum-models --all-targets` PASS。
  - `cargo fmt --all -- --check` PASS。
  - `git diff --check` PASS。
  - `cargo test -p ferrum-models qwen35 -- --nocapture` PASS:
    80 lib tests + `qwen35_config_test` 1 test passed。
  - `cargo test -p ferrum-models linear_attention_prefill_backend_matches_reference_core -- --nocapture`
    PASS。
  - `cargo test -p ferrum-models linear_attention_decode_backend_matches_full_reference_last_token -- --nocapture`
    PASS。
- GPU diagnostic lane:
  - Vast instance `41422823`,1x RTX 4090,49140 MiB,CUDA 12.4,driver
    580.126.09,`$0.662962962962963/hr`。
  - Remote clean SHA:
    `050d73f3a11cea757a53fb4e91d9cd236a4a62e0`。
  - Artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_prefill_profile_050d73f3_20260619T230656Z/`
  - PASS line:
    `W3 QWEN35 PREFILL PROFILE SMOKE PASS: /workspace/artifacts/w3_qwen35_prefill_profile_050d73f3_20260619T230656Z`
  - Binary SHA256:
    `a09fdd10e1ab0d5152546c76b66d43526731701e077e7b473af9a8cd7aebe675`
  - Product smoke covered `ferrum run` plus `ferrum serve` non-stream,
    stream with usage, tool call, and strict JSON schema.
  - Diagnostic bench,profile overhead enabled,not release evidence:
    - c=1:8 completed / 0 errored,TTFT p50 `2694.9ms`,
      TPOT p50 `16.0ms`,output throughput `13.09 tok/s`。
    - c=32:8 completed / 0 errored,TTFT p50 `21642.5ms`,
      TPOT p50 `36.0ms`,output throughput `15.18 tok/s`。
- Profile finding:
  - `profile_aggregate.json` captured 17 events,including 5 fresh-prefill
    rows and 12 decode rows.
  - Fresh prefill layer sum median `2.696s`;linear-attention layer sum median
    `2.684s`;full-attention layer sum median only `6.324ms`。
  - Long ShareGPT fresh prefill row (`tokens=282`) still spends
    `3.013s` in linear-attention layers vs `29.3ms` in full-attention layers。
  - Decode rows are much smaller:layer sum median `14.27ms`,linear median
    `11.13ms`,full median `3.13ms`。
  - Conclusion:the current W3 performance gap is dominated by Qwen35
    GatedDelta/linear-attention prefill,not full-attention paged KV,final
    lm_head,or readback.
- vLLM comparison note:
  - Local vLLM source uses Qwen GDN prefill as fused post-conv prep followed by
    `chunk_gated_delta_rule(..., cu_seqlens=...)`,writing final recurrent
    state back to per-sequence state slots.
  - Ferrum currently has single-sequence `linear_attention_prepare_f32` +
    `recurrent_gated_delta_rule_f32` for prefill and one-token batch decode
    kernels,but no varlen/equal-length batched GDN prefill API with
    `cu_seqlens` and per-sequence final-state writeback.
  - Next source work should target native Qwen35 batched/chunked GDN prefill
    rather than more decode-only tuning.
- Caveat:
  - `qwen35_prefill_profile.jsonl` has valid stage timing fields,but the first
    wrapper run populated profile metadata `commit_sha`/`model` incorrectly due
    to a shell interpolation bug. Authoritative SHA/model/hardware are recorded
    in `environment.log`,`lane_contract.json`,`ferrum.sha256`,and this status
    entry. The copied `run_smoke.sh` has been corrected for reruns.
- Cleanup:
  - Artifact secret scan for `HF_TOKEN`,`VAST_API_KEY`,and
    `Authorization: Bearer` returned no matches.
  - Vast stop check saved in `vast_stop_check.json`;API reported
    `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`。
- 限制:这不是 W3 release-grade 完成声明;仍缺真实 Qwen3.5 GPTQ L0-L5
  release correctness矩阵、release-grade `--require-ci --n-repeats 3`
  性能矩阵、80% vLLM ratio report,以及最终
  `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-22 — W3 scheduler prefill-step chunk control

- 背景:
  - 2026-06-21 scheduler trace 已定位:当前 c32 主要风险不是 scheduler
    微秒级开销,也不是纯 `decode=32` hot path;去掉 cohort prefill 后,
    mixed prefill+decode 会退化到 `22.7 output tok/s`。
  - 保留 cohort prefill 后,c32 diagnostic 能到约 `651 output tok/s`,
    但仍有大 prefill-only stall,例如 `prefill=21` 约 `735ms`、
    `prefill=19` 约 `636ms`。
  - vLLM 调度参考点是统一 token budget/`num_computed_tokens` 语义;
    Ferrum 需要在 scheduler 层约束 prefill token step,而不是用
    engine 级 `FERRUM_CHUNKED_PREFILL` 绕开 unified batch prefill。
- 源码变更:
  - 新增 typed scheduler 字段 `SchedulerConfig::prefill_step_chunk`,
    对应 runtime key `FERRUM_SCHED_PREFILL_STEP_CHUNK`。
  - `ferrum serve` 新增 CLI:
    `--scheduler-prefill-step-chunk <N>`。
  - `ContinuousBatchScheduler` 在计算 `tokens_to_process` 时应用
    `prefill_step_chunk`;当 active-decode chunk 同时存在时取更严格的
    per-request cap。
  - `FerrumConfigBuilder` 在 accelerator/default scheduler 策略下自动
    materialize `FERRUM_SCHED_PREFILL_STEP_CHUNK =
    ceil(max_batched_tokens / max_sequences)`;c1 不变,c32 会得到每请求公平
    token budget。若用户显式设置 `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE`,不会
    偷偷注入该默认值;显式 CLI/env 仍可覆盖。
  - Effective config 和 decision trace 会显示
    `prefill_first_until_active:<N>+prefill_step_chunk:<M>` 或组合
    `+active_decode_prefill_chunk:<K>`。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -p ferrum-types auto_config::tests:: -- --nocapture` PASS:
    46 passed。
  - `cargo test -p ferrum-scheduler -- --nocapture` PASS:57 passed。
  - `cargo test -p ferrum-types --test config_tests engine_config_applies_runtime_snapshot -- --nocapture`
    PASS。
  - `cargo test -p ferrum-cli serve_cli_runtime_entries_are_cli_sourced_and_classified -- --nocapture`
    PASS。
  - `cargo check -p ferrum-types -p ferrum-scheduler -p ferrum-engine -p ferrum-cli`
    PASS。
- Vast/GPU 状态:
  - 复用实例 `41422823` 的 start request 返回
    `Required resources are currently unavailable, state change queued`。
  - 最新 API 复核仍为 `cur_state=stopped`, `actual_status=exited`;没有
    running GPU 成本。
- 下一步 GPU 最小验证:
  - 同步当前 PR head 到 Vast 后,用 current binary 启动
    `ferrum serve`。
  - 不显式传 `--scheduler-prefill-first-until-active`;确认 effective config
    自动包含 `FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE=32` 和
    `FERRUM_SCHED_PREFILL_STEP_CHUNK=<ceil(max_batched_tokens/max_sequences)>`。
  - 跑非流式、stream+usage smoke,再跑同一 c32 64x1
    `bench-serve --fail-on-error --seed 9271` diagnostic,对比 trace 中
    prefill-only stall 和总 throughput。
- 限制:
  - 这是 source/control-plane 进展,不是性能提升声明;GPU 同硬件 A/B 尚未
    跑到。
  - 不是 W3 release-grade 完成声明;仍缺最终
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-22 — W3 scheduler trace request-detail checkpoint

- 背景:
  - Vast 4090 instance `41422823` 再次 start 失败:
    `resources_unavailable`, `Required resources are currently unavailable,
    state change queued`。
  - 最新 API 复核仍为 `cur_state=stopped`, `actual_status=exited`;
    没有 running GPU 成本。
  - GPU 未起来时继续推进本地证据采集能力,避免下一次 c32 A/B 仍靠手工
    日志推断 scheduler/engine 行为。
- 源码变更:
  - `scheduler_trace_jsonl` 的 `plan` 结构新增 per-request 明细列表
    `requests`。
  - 每个 request 记录:
    `request_id`, `phase`, `scheduled_tokens`, `tokens_to_process_missing`,
    `prompt_tokens`, `generated_tokens`, `prefill_tokens_processed`,
    `prefill_tokens_remaining_before`, `is_final_prefill_chunk`。
  - 该采集只在显式 `--scheduler-trace-jsonl` / `FERRUM_SCHEDULER_TRACE_JSONL`
    开启时执行;默认产品路径不写 trace。
- 目的:
  - 下一次 GPU artifact 可以直接回答每个 scheduler iteration 中哪些请求在
    prefill/decode、每个 prefill chunk 是否 final、decode cohort 是否保持接近
    c32、以及 prefill-step cap 是否真的减少大 prefill-only stall。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -p ferrum-engine scheduler_trace_plan_stats_reports_request_details -- --nocapture`
    PASS。
  - `cargo check -p ferrum-types -p ferrum-scheduler -p ferrum-engine -p ferrum-cli`
    PASS。
- 限制:
  - 这是 trace/evidence 进展,不是性能提升声明。
  - 仍需 1x4090 同 pod 跑 smoke + c32 diagnostic,然后再判断
    `FERRUM_SCHED_PREFILL_STEP_CHUNK` 是否正向。

## 2026-06-22 — Scheduler trace analyzer helper

- 新增 `scripts/release/analyze_scheduler_trace.py`。
- 作用:
  - 读取 `--scheduler-trace-jsonl` 产物。
  - 输出稳定 JSON summary,包括 scheduler iteration 数量、Some/None/error
    计数、phase mix、batch/decode/prefill 分布、request-detail 是否存在、
    prefill chunk/final chunk 分布、decode generated-token 分布、最慢
    process iterations。
  - 打印 `SCHEDULER TRACE ANALYSIS PASS: <out>` 作为 diagnostic artifact
    完成标记。
- 自测:
  - `python3 scripts/release/analyze_scheduler_trace.py --self-test` PASS:
    `SCHEDULER TRACE ANALYSIS SELFTEST PASS`。
  - `python3 -m py_compile scripts/release/analyze_scheduler_trace.py` PASS。
- 使用建议:
  - 下一次 Vast c32 diagnostic 后运行:
    `python3 scripts/release/analyze_scheduler_trace.py <run>/scheduler_trace_c32.jsonl --out <run>/scheduler_trace_summary.json`
  - 该 summary 用于判断 prefill-step cap 是否降低大 prefill-only stall,以及
    decode cohort 是否保持接近 c32。
- 限制:
  - 这是分析工具,不是 release gate,也不是性能通过证据。

## 2026-06-22 — Vast 1x4090 retry stopped by insufficient credit

- 目标 lane:
  - W3 Qwen3.5 GPTQ scheduler prefill-step diagnostic。
  - 硬件限定:exact 1x RTX 4090。
  - correctness gate:`ferrum serve` 非流式 smoke + stream usage smoke。
  - performance command:c32 64x1 `bench-serve --fail-on-error --seed 9271`
    diagnostic,不是 release evidence。
- 旧实例:
  - `41422823` 仍为 `cur_state=stopped`, `actual_status=exited`。
- 新 offer 尝试:
  - 首选 offer `30872861` 约 `$0.16888888888888887/hr`,但创建时返回
    `no_such_ask`;未创建实例。
  - 重新筛选 1x RTX 4090 offers 后选择 `39797598`,约
    `$0.29555555555555557/hr`,24GB VRAM,467GB disk,约 1.0Gbps down,
    reliability `0.987461`。
  - `PUT /api/v0/asks/39797598/` 返回 `insufficient_credit`:
    `Your account lacks credit; see the billing page.`。
- cleanup/status:
  - 按 GPU policy 停止所有继续租赁尝试,没有继续循环 offers。
  - `GET /api/v0/instances/` 只显示旧 stopped/exited 实例 `41422823`;
    没有新 running/loading 实例。
  - 本地 artifact:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_prefill_step_diag_20260621T180847Z_8b14507d/local_vast/`
  - Artifact secret scan for `HF_TOKEN`, `VAST_API_KEY`, `Authorization:
    Bearer`, and `hf_...` returned no matches。
- 限制:
  - GPU validation is externally blocked by Vast account credit, so the
    prefill-step code still has no same-hardware performance verdict。
  - 这不是 W3 release-grade 完成声明;仍缺最终
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-22 — W3 varlen GDN prefill token-row metadata

- 背景:
  - 之前 profile 已定位 Qwen35 仍主要 prefill/linear-attention bound。
  - vLLM GDN prefill 路径把 batch metadata 作为 attention metadata 传入;
    Ferrum 已有 `cu_seqlens`,但 CUDA
    `linear_attention_prepare_varlen_*` kernel 内部每个
    token/channel 仍通过 while 线性扫描 `cu_seqlens` 找 seq。
  - 在 c32/长 prefill 下,这个扫描发生在
    `total_tokens * conv_channels` 维度上,属于 GPU-side metadata 开销。
- 源码变更:
  - `Backend::linear_attention_prepare_varlen_f32` 新增
    `token_seq_indices: [total_tokens]` typed metadata。
  - Qwen35 product fresh batch prefill 在构造 `cu_seqlens` 时同步构造
    token->seq row 映射,并在所有 linear-attention layers 复用该 buffer。
  - CUDA `linear_attention_prepare_varlen_*` kernel 用
    `token_seq_indices[token]` O(1) 定位 seq,移除每个 token/channel 的
    `while token >= cu_seqlens[seq + 1]` 扫描。
  - CPU backend 校验 `token_seq_indices` 必须和 `cu_seqlens` 一致;错误
    metadata 会直接报错,不会静默产生错误边界。
- 本地验证:
  - `cargo fmt --all` PASS。
  - `cargo test -p ferrum-kernels --test linear_attention_cpu linear_attention_prepare_varlen_cpu_rejects_mismatched_token_seq_indices -- --nocapture`
    PASS。
  - `cargo test -p ferrum-models linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture`
    PASS。
  - `cargo check -p ferrum-kernels -p ferrum-models` PASS。
- 限制:
  - 这是源码层 prefill hot-path 优化,不是性能通过证据。
  - 本机 `nvcc` not found,所以 CUDA feature build / `.cu` 编译仍需在
    1x4090 CUDA lane 恢复后验证。
  - 仍需 1x4090 同硬件跑 Qwen35 smoke + c32 diagnostic,比较
    `qwen35_linear_prefill_core_prepare`、`qwen35_linear_prefill_core_recurrent`
    和总体 output tok/s。
  - 不是 W3 release-grade 完成声明;仍缺最终
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-22 — W3 Qwen35 full-attention varlen token-row metadata

- 背景:
  - clean c32 profile 显示 decode batch=32 hot path 折算已接近目标量级,
    但端到端仍受 cohort prefill/full+linear attention work 拖累。
  - 上一轮只把 `token_seq_indices` 接入了 Qwen35 GDN/linear-attention
    prepare;Qwen35 full-attention paged QKV varlen CUDA writer 仍在每个
    token 上通过 `while tok >= cu_seqlens_q[seq + 1]` 线性扫描 seq。
  - 这和 vLLM attention metadata 方向不一致,也让同一个 varlen batch
    metadata 只被部分 kernels 复用。
- 源码变更:
  - `BackendPagedKv::qwen35_split_qkv_norm_rope_into_paged_cache_varlen{,_vllm}`
    新增 `token_seq_indices: [total_q_tokens]` 参数。
  - `Qwen35PagedScratch` 新增 token-row scratch buffer。
  - Qwen35 batch prefill 复用入口已经构造好的 `token_seq_indices`。
  - Qwen35 batch decode 为 `cu_seqlens=[0,1,2,...]` 构造
    `token_seq_indices=[0,1,2,...]`。
  - Qwen35 single/stateful paged full-attention prefill 构造全 0
    token-row buffer。
  - CUDA `qwen35_split_qkv_norm_rope_into_paged_cache_varlen{,_vllm}_f16`
    改为 O(1) 读取 `token_seq_indices[tok]`,移除 per-token seq scan。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo check -p ferrum-kernels -p ferrum-models` PASS。
  - `cargo test -p ferrum-models linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture`
    PASS。
  - `cargo test -p ferrum-models dense_full_attention_layer_accepts_qwen35_gate_shape_with_hidden_not_q_total -- --nocapture`
    PASS。
  - `git diff --check` PASS。
- 限制:
  - 本机 `nvcc not found`;CUDA feature build and `.cu` compile still require
    the 1x4090 CUDA lane.
  - 这是源码层 prefill metadata hot-path 优化,不是性能通过证据。
  - 仍需同硬件跑 Qwen35 correctness smoke + c32 diagnostic,再比较
    `qwen35_full_attention_prefill`/paged-QKV stage 和总体 output tok/s。
  - 不是 W3 release-grade 完成声明;仍缺最终
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-22 — W3 bench-serve fixed-output ignore-eos contract

- 背景:
  - 现有 W3 vLLM/Ferrum perf artifacts 在输出长度上不等价:
    vLLM baseline `w3_vllm_sharegpt_baseline_20260619` 的
    c=1/4/16/32 每个请求均为 128 output tokens;Ferrum
    `w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9` 的 mean output
    tokens/request 约为 c1 `47`,c4 `45.627`,c16 `45.243`,c32
    `45.013`。
  - Ferrum server/engine 产品路径已经支持 OpenAI/vLLM 兼容的
    `ignore_eos`,但 canonical `ferrum bench-serve` 没有 typed CLI
    开关,导致 release-shape perf 命令不能显式声明 fixed-output
    stop 语义。
- 源码变更:
  - `crates/ferrum-cli/src/commands/bench_serve.rs` 新增
    `--ignore-eos`。
  - 默认行为不变:未设置该 flag 时请求体不包含 `ignore_eos`。
  - 设置该 flag 时,`bench-serve` 对 `/v1/chat/completions` 发送
    `"ignore_eos": true`,让 fixed-output benchmark 请求跑到
    `max_tokens`。
  - closed-loop/open-loop 的 warmup 和 measurement 路径都透传该参数。
- 本地验证:
  - `cargo fmt --all -- --check` PASS。
  - `cargo test -p ferrum-cli chat_completion_body -- --nocapture` PASS。
  - `cargo check -p ferrum-cli` PASS。
  - `cargo test -p ferrum-cli commands::bench_serve::tests -- --nocapture`
    PASS。
  - `cargo run -p ferrum-cli -- bench-serve --help` PASS and prints
    `--ignore-eos`。
  - `git diff --check` PASS。
- 交接文档:
  - `docs/goals/model-coverage-2026-06-12/HANDOFF_W3_QWEN35_20260622_2H.md`
    记录当前进度、性能差距、Vast blocker、下一次 1x4090 correctness
    smoke + ShareGPT `--ignore-eos` sweep 命令。
- 限制:
  - 这是 benchmark contract / product CLI source progress,不是性能通过
    证据。
  - 仍需 1x4090 同硬件 CUDA build、`ferrum run`/`ferrum serve`
    correctness smoke、stream usage smoke,再跑 ShareGPT
    `bench-serve --ignore-eos --fail-on-error --require-ci --seed 9271
    --n-repeats 3`。
  - 不是 W3 release-grade 完成声明;仍缺最终
    `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`。

## 2026-06-22 — W3 final gate fixed-output perf equivalence check

- 背景:
  - 上一条只把 `--ignore-eos` 接到 canonical `bench-serve` CLI;如果
    final validator 继续只看 throughput/ITL,旧的短输出 Ferrum artifact
    仍可能被拿来和 128-token vLLM baseline 做 80% 计算。
- 源码变更:
  - `scripts/release/model_release_grade_goal_gate.py` 的 W3 performance
    cell 现在要求:
    - Ferrum 和 baseline `bench_command_line` 显式包含 `--ignore-eos`。
    - Ferrum/baseline command 均包含相同的 `--random-output-len N`。
    - cell 携带 `output_tokens_per_request` 和
      `baseline_output_tokens_per_request`。
    - Ferrum/baseline 每个 repeat、每个 request 的 output tokens 都必须
      等于 `N`。
    - final validator 会回读 `performance.cells[].artifact` 和
      `performance.baseline.artifact` 指向的原始 `bench-serve` report,
      按 concurrency 交叉验证 output-token 矩阵和 manifest 一致。
  - `scripts/release/model_release_grade_manifest.py` 现在从
    `bench-serve` report 复制上述 output-token 矩阵进 final manifest,
    并校验矩阵维度等于 `n_repeats x n_requests_per_run`。
  - W2 final gate 不强制 `--ignore-eos`;该 fixed-output strictness 只在
    W3 performance cell 上启用。
- 本地验证:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py`
    PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS,覆盖 W3 Ferrum/baseline 缺 `--ignore-eos` 和 output token
    短输出负例。
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS。
- 旧 artifact probe:
  - 使用现有真实 W3 L0/L1/L2/L3/L4/L5 + 旧 Ferrum perf
    `w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9` + 历史 vLLM
    baseline 构造临时 manifest。
  - final validator 现在明确拒绝四个 cell:
    `performance.c{1,4,16,32} command missing --ignore-eos`,
    `performance.c{1,4,16,32}.baseline command missing --ignore-eos`,以及
    `output_tokens_per_request[...] must equal --random-output-len 128`。
  - artifact 回读也明确拒绝:
    `performance.c{1,4,16,32}.artifact.output_tokens_per_request[...] must
    equal --random-output-len 128`。
  - 旧 ratio/ITL failure 仍存在;新增拒绝点证明旧短输出 artifact
    不再能作为 W3 80% perf 证据。
- 限制:
  - 这仍不是 W3 release-grade 完成声明。
  - 下一次 1x4090 必须重新跑 Ferrum ShareGPT `--ignore-eos` sweep,
    首先确认 output token 矩阵全为 128,再评估 ratio/ITL。

## 2026-06-22 — W3 final gate bench-report count cross-check

- 背景:
  - W3 final gate 已回读原始 `bench-serve` report 的 output-token
    矩阵,但 completed/error/quality counts 仍可能只信 final manifest。
    如果 manifest 手写为 zero-error,但原始 report 有错误请求,final gate
    应该拒绝。
- 源码变更:
  - `scripts/release/model_release_grade_goal_gate.py` 现在回读 Ferrum 和
    baseline `bench-serve` report artifact 后,按 concurrency 交叉验证:
    - `n_repeats`;
    - `n_requests_per_run`;
    - `completed_per_run`;
    - `errored_per_run`;
    - `bad_output/malformed_stream/missing_done/duplicate_done/
      zero_output_tokens/stream_bulk_flush/http_500/panic` per-run counts。
  - 支持 report 直接携带 `*_per_run` 字段,也支持从
    `quality_issues_per_run` 提取质量计数。
  - W3 self-test 增加负例:原始 Ferrum perf artifact 的
    `errored_per_run=[0,1,0]`、manifest 仍写 `[0,0,0]` 时,final gate
    必须报 `performance.c1.artifact.errored_per_run must match manifest
    errored_per_run`。
- 本地验证:
  - `python3 -m py_compile scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py`
    PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS。
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS。
- 限制:
  - 这仍不是 W3 release-grade 完成声明;仍缺可用 1x4090 上的新
    correctness smoke 和 `--ignore-eos` ShareGPT perf sweep。

## 2026-06-22 — W3 L5 fixed-output evidence gate

- 背景:
  - W3 final performance gate 已经要求 Ferrum/vLLM perf report 使用
    `--ignore-eos` 和固定 `--random-output-len`,但 L5 concurrency gate
    仍只检查并发/错误计数和 release bench 命令形状。
  - 这会让旧 L5 artifact 在正确性链路里继续看起来可用,而新的 W3
    fixed-output 性能证据必须重跑同硬件 ShareGPT sweep。
- 源码变更:
  - `scripts/release/w3_l5_concurrency_gate.py` 现在要求 L5 打包命令显式
    包含 `--ignore-eos` 和 `--random-output-len 128`。
  - L5 打包现在从 `bench-serve` report 读取
    `output_tokens_per_request`,并要求每个 repeat、每个 request 都等于
    `--random-output-len`。
  - L5 artifact 的 `concurrency` 区块现在记录
    `expected_output_tokens_per_request` 和每个 cell 的
    `output_tokens_per_request` 矩阵。
  - `scripts/release/model_release_grade_goal_gate.py` 的最终 W3 validator
    现在复核 L5 命令、L5 fixed-output 长度、以及 L5 report token 矩阵。
  - `scripts/release/model_release_grade_manifest.py` 的 W3 self-test fixture
    已同步 fixed-output L5 字段。
- 本地验证:
  - `python3 -m py_compile scripts/release/w3_l5_concurrency_gate.py scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py`
    PASS。
  - `python3 scripts/release/w3_l5_concurrency_gate.py --self-test` PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS。
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS。
- 旧 artifact probe:
  - 直接复核
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9/l5_concurrency/w3_l5_concurrency.json`
    现在失败,问题为:
    - `correctness.l5_concurrency.commands[0] command missing --ignore-eos`;
    - 四个 c=1/4/16/32 cell 缺
      `output_tokens_per_request` 矩阵。
- GPU 状态:
  - SSH `ssh7.vast.ai:22822` 仍返回 `Connection refused`,未启动远端 CUDA
    任务。
- 限制:
  - 这是证据合同和防误用旧 artifact 的源码进展,不是性能通过证据。
  - 下一次 1x4090 必须重跑 L5/concurrency/perf fixed-output report,再跑
    W3 final validator。

## 2026-06-22 — W3 L2 output hygiene artifact

- 背景:
  - W3 L2 gate 已经要求真实模型、known-answer pass、`ferrum run` 和
    `ferrum serve` 命令证据,但它主要信任 known-answer report 的
    `passed=true`/计数字段。
  - W3 release blocker 还包括 `<unk>`、`[PAD]`、reserved special token、
    mojibake、panic、KV overflow 等坏输出;这些应在 L2 打包阶段被
    独立复核,不能只靠 runner 当场检查。
- 源码变更:
  - `scripts/release/w3_l2_quantized_gate.py` 现在要求
    `known_answer_cases` 实际存在,每个 case 有非空输出文本和 response
    artifact。
  - L2 gate 会扫描 case `content` 和可解析到的 response artifact 文本,
    拒绝 forbidden output patterns。
  - L2 artifact 新增 `output_hygiene`,记录:
    `known_answer_cases_checked`, `response_artifacts_checked`,
    `content_non_empty`, `forbidden_patterns_absent`,
    `artifact_text_scanned`。
  - `scripts/release/model_release_grade_goal_gate.py` 的最终 W3 validator
    现在要求 L2 artifact 携带上述 `output_hygiene`,并要求扫描数量覆盖
    `known_answer_total`。
  - `scripts/release/model_release_grade_manifest.py` 的 W3 self-test fixture
    已同步该字段。
- 新证据:
  - 用 tracked 原始报告
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_unified_prefill_cuda_smoke_20260620T021129Z_75ec7e6e/real_product_report/known_answer_report.json`
    重新打包出:
    `docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_hygiene_from_real_product_20260622_75ec7e6e/w3_l2_quantized.json`。
  - PASS line:
    `W3 L2 QUANTIZED PASS: docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_hygiene_from_real_product_20260622_75ec7e6e`。
  - 该 artifact 的 `output_hygiene` 为 11/11 cases 和 response artifacts
    scanned,`forbidden_patterns_absent=true`。
  - 旧 L2 artifact
    `w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e`
    现在被最终 validator 拒绝,原因是缺
    `correctness.l2_quantized.output_hygiene`。
- 本地验证:
  - `python3 -m py_compile scripts/release/w3_l2_quantized_gate.py scripts/release/model_release_grade_goal_gate.py scripts/release/model_release_grade_manifest.py scripts/release/w3_qwen35_real_product_report.py`
    PASS。
  - `python3 scripts/release/w3_l2_quantized_gate.py --self-test` PASS。
  - `python3 scripts/release/w3_qwen35_real_product_report.py --self-test`
    PASS。
  - `python3 scripts/release/model_release_grade_goal_gate.py --self-test`
    PASS。
  - `python3 scripts/release/model_release_grade_manifest.py --self-test`
    PASS。
  - Direct final-validator L2 probe of the new artifact returned
    `PROBLEM_COUNT=0`。
- 限制:
  - 这提升并更新了 W3 L2 正确性证据,但 W3 仍未完成;还缺当前 SHA 上
    fixed-output L5/performance 和最终 `MODEL_RELEASE_GRADE_W3 PASS`。
