# W2 Active Chunk ShareGPT c16 CI Diagnostic GPU Contract

- Date: 2026-06-17
- Lane: W2 Gemma3 CUDA active-chunk c16 ShareGPT CI diagnostic.
- Scope: minimal same-shape Ferrum validation after defaulting
  `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16` for CUDA Gemma3 GPTQ/int4.
- Hardware target: 1x RTX 4090, preferably cached Vast instance `41230499`.
- Expected runtime/cost: 35-75 minutes if cache is usable, approximately
  USD 0.35-0.75 at USD 0.5766666667/hour.
- Hard stop condition: stop after artifact copy on PASS/failure, or after
  90 minutes without a complete build/correctness/bench artifact.
- Correctness gate:
  - release CUDA build of `ferrum`
  - `ferrum run` numeric smoke
  - `ferrum serve` streaming smoke with exactly one `[DONE]` and usage tokens
- Performance command shape:
  - `ferrum bench-serve`
  - `--dataset sharegpt`
  - `--sharegpt-path ascii_sharegpt.jsonl`
  - `--random-output-len 128`
  - `--concurrency-sweep 16`
  - `--num-prompts 100`
  - `--n-repeats 3`
  - `--fail-on-error`
  - `--require-ci`
  - `--seed 9271`
- Dataset SHA256:
  `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`
- This diagnostic is not a W2 release-grade claim. The release-grade gate still
  requires c=1/4/16/32, same-hardware vLLM baseline, full correctness L0-L5,
  and `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
