# S1 CUDA Qwen3.5-4B projected binding diagnostic

- Lane: `S1 CUDA / Qwen3.5-4B / ferrum run correctness diagnostic`
- Git SHA: `e9cc7640`
- Vast instance: `44965014` (reused stopped instance with warm model/build cache)
- Hardware: `1x NVIDIA GeForce RTX 4090, 24564 MiB`
- Hourly rate: `$0.4166666667`
- Expected runtime/cost: `10-20 minutes / <= $0.14`
- Correctness prerequisite: projected binding unit test `1/1 PASS`; device-operation batch contract `2/2 PASS`
- Stop condition: old binding error disappears and output validates, first new exact failure, build failure, or 90 seconds without a first token
- Model: `Qwen/Qwen3.5-4B`, cached HF snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Remote artifact: `/workspace/artifacts/s1_cuda_qwen35_4b_e9cc7640_20260715T121949Z`

## Prediction

The request-sized `value.input.token_ids` backing must pass generic dispatch validation because its immutable token projection and `DynamicResourceDemand::Tokens` agree on 4 bytes per token. The previous `differs from its value binding` failure must disappear. The accepted signal is non-empty assistant output or a new failure after provider encode begins.

## Result

- Status: `CUDA run/serve checkpoint PASS`; this is not an S1 or total Goal PASS.
- CUDA build: `PASS`, `2026-07-15T12:22:26Z` to `2026-07-15T12:43:21Z` (`20m55s`).
- Binary SHA256: `c5036c63720c7e798a224e07110bab9c44ff28880902cf5e915f5d4e083cdf8b`.
- `ferrum run`: exit `0`, assistant `Paris`, `n_tokens=2`, `finish_reason=stop`, `334.96 ms` generation row.
- Run validator: `QWEN35 CUDA RUN ARTIFACT VALIDATION PASS`.
- `ferrum serve`: `/health` and `/v1/models` succeeded; streaming chat returned HTTP `200`, assistant `Paris`, exactly one `[DONE]`, exactly one usage row with positive completion tokens.
- Serve validator: `QWEN35_CUDA_SERVE_STREAM_ARTIFACT_VALIDATION_PASS`.
- Forbidden scans: no panic, crash, invalid UTF-8/mojibake, `<unk>`, `[PAD]`, binding mismatch, CUDA OOM, or KV overflow.
- JSONL validation: all six run/serve profile, scheduler trace, stdout, and stream data files parse successfully.
- Previous failure: `CONFIRMED FIXED`; `value.input.token_ids differs from its value binding` did not recur.
- Paid instance cleanup: instance `44965014` is `stopped/exited`; conservative session cost upper bound `<= $0.25`.

## Remaining Gap

The successful run produced 12 product profile rows and 19 scheduler/resource trace rows; serve produced 13 product profile rows and 19 scheduler/resource trace rows. Resource RAII phases are visible, but operation/node dispatch spans are still absent. S1 remains open until the existing typed profile/trace path records operation identity and the S1 validator consumes the resulting artifact.
