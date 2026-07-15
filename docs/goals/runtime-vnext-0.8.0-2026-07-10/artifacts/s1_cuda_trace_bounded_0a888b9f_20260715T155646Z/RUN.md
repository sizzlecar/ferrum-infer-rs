# S1 CUDA bounded basic trace production evidence

## Status

- Product source: `0a888b9f21a7e2e540c3a11b3a12fcb08d688895` (clean remote checkout)
- Gate source: `2ba8d624cae954f5b52f07f0329ea46800167014` (clean local checkout)
- Model: `Qwen/Qwen3.5-4B`, snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Hardware: one NVIDIA GeForce RTX 4090, 24564 MiB, driver `580.119.02`
- Binary SHA256: `1f7e6592138e275e6ab24fe899495245509b057ad072f6fc0ab334afe5191dfe`
- Raw archive SHA256: `8672ec04c1d6e93ffce0ff006b295f1e9a66d949a731fc41a45f8d9111e54404`

The clean unified gate printed:

```text
FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS: /private/tmp/runtime-vnext-s1-cuda-2ba8d624-20260716
FERRUM GATE vnext-s1-cuda PASS: /private/tmp/runtime-vnext-s1-cuda-2ba8d624-20260716
```

This is the production-reference artifact that unlocks G01B. It does not prove the S1 milestone,
G01B, aggregate G01, full G06, model migration, or release readiness. S1 still requires the G01B
dynamic admission/backpressure and zero-legacy-runtime-fallback evidence.

## Correctness

- `ferrum run`: exit 0, assistant output `Paris`, `finish_reason=stop`, 2 output tokens.
- `ferrum serve`: HTTP 200, streamed `Paris`, exactly one `[DONE]`, exactly one usage row, clean stop.
- `ferrum bench-serve` smoke: 4/4 requests completed, 0 errors, usage token source.
- Run and serve typed traces have contiguous sequences, complete plan/node/operation/provider/device
  identity, balanced request/resource lifecycles, and no failure event.
- Log scan found no panic, OOM, KV overflow, reserved-token output, mojibake, or crash signature.

## Performance

The canonical client ran `off1 -> basic1 -> basic2 -> off2 -> basic3 -> off3 -> off4 -> basic4`
on the same binary and hardware at c=1, random 128/64, 4 measured requests, 1 warmup, 3 repeats per
slot, seed 9271, `--require-ci`, and `--fail-on-error`.

| Metric | Off | Basic | Limit |
|---|---:|---:|---:|
| Samples | 12 | 12 | 12 |
| Completed requests | 48 | 48 | 48 |
| Failed requests | 0 | 0 | 0 |
| Mean output throughput | 35.0580 tok/s | 35.3880 tok/s | basic overhead <= 2% |
| Median output throughput | 34.6805 tok/s | 35.2625 tok/s | basic overhead <= 2% |
| CV | 4.6400% | 3.6064% | <= 5% |

- Recomputed mean overhead: `-0.9413%`.
- Recomputed median overhead: `-1.6780%`.
- Each of four basic slots contains 15 typed requests and exactly one captured frame per request.
- Each request contains 399 typed events and 131 node/operation/retire triplets.
- Terminal counts reconcile with usage for every measured repeat.
- Maximum trace size is `854077.33 bytes/request`, below the S1 diagnostic limit of 1 MiB/request.
- The accepted serve trace is 4,435,055 bytes versus the earlier 54,843,732-byte checkpoint,
  a diagnostic reduction of about 91.9%.

The first four-slot ABBA sample was retained as REJECT because off CV was `5.1157%`, above the 5%
noise limit. The full ABBA-BAAB sample passed. A local mutation that forced basic throughput to
1 tok/s was rejected with `basic mean overhead 0.971476 exceeds 0.02`.

## Cost And Lifecycle

- Vast instance: `44965014`.
- Rate: `$0.4166666667/hour`; conservative session cost upper bound: `$0.42`.
- Final state: `cur_state=stopped`, `actual_status=exited`.
- No sibling paid instance or residual Ferrum/Cargo process remained after artifact collection.

The compressed archive contains the complete 60 MiB raw artifact. `gate/validation.json` is the
validator's recomputation, `gate/manifest.json` is the child checkpoint, and
`gate/gate.manifest.json` is the clean unified runner receipt.
