# W2 Gemma3 Active-Decode Prefill Chunk CUDA Diagnostic

## Scope

- Diagnostic only. No release performance claim.
- No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
- Source checkpoint under test:
  `eed031e334c78cf181a4b1077c1ba2089d0d6d6f`.
- Model: `gemma3:27b-gptq`
  (`circulus/gemma-3-27b-it-gptq`).
- Backend/lane: CUDA, 1x RTX 4090.
- Artifact root:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_decode_chunk_c16_cuda_diag_2026-06-17/`.

## GPU Lifecycle

- Vast instance: `41230499`.
- Offer: `36368074`, 1x RTX 4090, quoted total `0.5766666666666667` USD/hour.
- Remote GPU evidence:
  NVIDIA GeForce RTX 4090, driver `590.48.01`, `24564 MiB`.
- CUDA compiler: `nvcc` 12.4.131.
- Artifact copied back locally after the run.
- Instance cleanup:
  `cur_state=stopped`, `actual_status=exited`.

## Build Evidence

- Remote git SHA:
  `eed031e334c78cf181a4b1077c1ba2089d0d6d6f`.
- Remote worktree before run: clean.
- CUDA release build command:
  `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.
- Build result: PASS, `29m25.806s`.
- Binary sha256:
  `426c9b029d08ede6edb986a7dd80e5330e2a9f7489ce7de6224a1b482361d4c7`.

## Correctness Evidence

- `ferrum run` smoke passed:
  stdout content `5`, finish reason `stop`, `n_tokens=3`.
- `ferrum serve` streaming smoke passed:
  - content: `5\n`;
  - `done_count=1`;
  - usage present: `prompt_tokens=23`, `completion_tokens=3`,
    `total_tokens=26`.
- No correctness issue was observed in this diagnostic artifact.

## Runtime Config Evidence

- Effective config materialized:
  `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`.
- Decision trace selected:
  `scheduler_admission_policy=active_decode_prefill_chunk:16`.
- Source for the scheduler decision:
  `model_metadata`.

## c16 Diagnostic Bench

- Command:
  `ferrum bench-serve --dataset random --random-input-len 256 --random-output-len 128 --concurrency-sweep 16 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`.
- Result:
  - completed `[16]`;
  - errored `[0]`;
  - malformed stream `[0]`;
  - missing `[DONE]` `[0]`;
  - duplicate `[DONE]` `[0]`;
  - zero output tokens `[0]`;
  - `output_token_count_source=usage`.
- Diagnostic-only throughput:
  `294.61885808275144` output tokens/s.
- This is `n_repeats=1` smoke evidence only, not release performance evidence.

## Profile Result

- Summary:
  `large_mixed_prefill_decode_lines=[]`.
- Previous target failure shape after logits-readback fix:
  `m_total=897 num_seqs=16 prefill=12 decode=4 total=383684us`.
- This run no longer shows a large mixed prefill+decode frame.
- The largest observed mixed prefill+decode frame was chunk-shaped:
  `m_total=151 num_seqs=16 prefill=9 decode=7`, with sampled examples around
  `79ms` to `107ms`.
- Large full-prompt prefill still occurs as pure prefill, for example
  `m_total=1866 prefill=7 decode=0`; this is expected and is not the
  scheduler regression targeted by this checkpoint.

## Interpretation

- The Gemma3 CUDA GPTQ typed default is wired into product entrypoints and
  survives both `ferrum run` and `ferrum serve`.
- The active-decode prefill chunk lever addressed the concrete scheduler
  symptom: full-prompt prefill is no longer mixed into active decode at the
  previously observed large `m_total` shape.
- Remaining work is still W2 release-grade validation, not another broad
  unscoped sweep:
  - run a same-shape A/B against the previous default if a quantitative delta
    is needed;
  - then run the W2 gate/final validator before any release-ready claim.
