# W2 vLLM Same-Hardware ShareGPT c16 Baseline GPU Contract

- Date: 2026-06-17
- Lane: W2 Gemma3 vLLM same-hardware c16 ShareGPT baseline diagnostic.
- Scope: minimal same-hardware baseline for the current Ferrum c16 diagnostic at
  commit `674c6678 perf(scheduler): admit prefills with remaining step budget`.
- Hardware target: cached Vast instance `41230499`, 1x RTX 4090.
- Expected runtime/cost: 20-45 minutes if the vLLM venv and model cache are
  present, approximately USD 0.20-0.45 at USD 0.5766666667/hour. If vLLM must be
  installed, hard stop at 90 minutes, approximately USD 0.90.
- Stop condition: copy back artifact logs after PASS, smoke failure, server
  startup failure, install failure, or benchmark failure, then stop the instance.
- Correctness gate:
  - vLLM `/v1/models` responds for served model `gemma3:27b-gptq`
  - streaming `/v1/chat/completions` smoke receives exactly one `[DONE]`
  - streaming smoke emits at least one output token
  - streaming smoke reports usage with positive completion tokens
- Performance command shape:
  - `ferrum bench-serve`
  - `--base-url http://127.0.0.1:<vllm-port>`
  - `--model gemma3:27b-gptq`
  - `--dataset sharegpt`
  - `--sharegpt-path ascii_sharegpt_w2_100.jsonl`
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
  requires c=1/4/16/32, full W2 correctness L0-L5, and
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
- Cached instance `41230499` returned `resources_unavailable` on restart, so
  the fallback lane is same-pod c16 A/B on a newly-created 1x RTX 4090:
  run vLLM c16 and Ferrum c16 on the same replacement instance before drawing
  any ratio conclusion.

## Actual Execution

- Replacement instance: Vast `41241013`, 1x RTX 4090, Netherlands, driver
  `580.95.05`, CUDA devel image `nvidia/cuda:12.4.0-devel-ubuntu22.04`,
  total quoted rate `0.47111111111111115 USD/h`.
- Remote source: clean git checkout
  `96d2df73e82ab4c0d643ced32d1f424b29dc5353`.
- Ferrum binary SHA256:
  `ca11f78f9e1be27a26bd12f50e377f3def602f14220cb10e1099eadb4f35ca93`.
- Same-pod result: diagnostic PASS for c16 throughput and FAIL for c16 p95 ITL;
  see `remote/summary.json` and `remote/summary.md`.
- Cleanup: artifacts copied back locally; sanitized Vast metadata records final
  `cur_state=stopped`, `actual_status=exited`.
