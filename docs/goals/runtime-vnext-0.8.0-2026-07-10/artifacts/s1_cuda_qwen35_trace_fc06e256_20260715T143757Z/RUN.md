# S1 CUDA Qwen3.5-4B typed trace checkpoint

- Status: `TRACE CHECKPOINT PASS`; this is not S1, G01B, G06, performance, or total Goal PASS.
- Product source SHA: `fc06e2568ab7dbe4bea7c19f348b8abd49ed450c`.
- Validator fixes: `94c92e39`, `64310b6a`.
- Evidence commit: `ddaeff599c91823c7c1df56699379bdf245dc183`.
- Binary SHA256: `e81c1c072cbf719e7d7827f3041962a3c72c8f24e5e09a91974c151eddc99b62`.
- Model: `Qwen/Qwen3.5-4B`, snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Hardware: `1x NVIDIA GeForce RTX 4090, 24564 MiB`, driver `580.119.02`, CUDA `12.4`.
- Vast instance: `44965014`, reused with warm model/build cache at `$0.4166666667/h`.
- Cleanup: `cur_state=stopped`, `actual_status=exited`; no second paid instance existed.
- Conservative session compute cost: `<= $0.38`.

## Commands and results

- CUDA release build used `cuda,vllm-moe-marlin,vllm-paged-attn-v2` and passed in `4m18s`.
- `ferrum run` used product CLI profile options at `profile-detail=basic` and the canonical
  scheduler trace path. It returned `Paris`, `finish_reason=stop`, and `2` output tokens.
- `ferrum serve` passed `/health`, `/v1/models`, and streaming chat with HTTP `200`, answer
  `Paris`, exactly one `[DONE]`, exactly one usage row, and positive completion tokens.
- `ferrum bench-serve` diagnostic smoke used c=1, random 128/16, four measured requests,
  one warmup, one repeat, `--fail-on-error`, and seed `9271`. It completed `4/4` with zero
  request, SSE, usage, output, HTTP 500, panic, or `[DONE]` errors.
- Diagnostic-only smoke measurements were `0.516492 req/s` and `8.263872 output tok/s`.
  The run was not repeated, did not provide a confidence interval, and makes no performance claim.

## Typed trace evidence

`run` produced `813` scheduler-trace rows, including `794` typed execution events:

- request/plan/sequence/request terminal events: `1/1/1/1`;
- frame started/completed: `2/2`;
- node started / operation submitted / node retired: `262/262/262`;
- operation rows missing plan, node, operation, provider, or device identity: `0`;
- balanced resource owners: `3`; resource rows: `19`.

After the explicit stream request and bench smoke, `serve` produced `32,414` typed events:

- typed requests: `6`, each with a contiguous event sequence and terminal completion;
- frame started/completed: `82/82`;
- node started / operation submitted / node retired: `10,742/10,742/10,742`;
- operation identity omissions: `0`;
- balanced resource owners: `19`; resource rows: `299`.

The first default-prompt diagnostic returned a correct answer containing `Paris` but reached
`finish_reason=length` at 16 tokens. It is retained under `run-attempt1` and was not used as the
accepted `run` result. The accepted retry changed only the prompt to require a one-word answer.

## Artifact integrity

- Raw artifact before compression: approximately `77 MiB`.
- Final serve scheduler trace: `54,843,732` bytes.
- Compressed raw artifact: `raw-artifact.tar.gz`, `1,757,429` bytes.
- Archive SHA256: `d09ac23c4b53a8af8da8f5a7b103ce313ee667768a2ac9975e65d6e2f5b55d49`.
- `validation.json` binds every raw file SHA256, source SHA, binary SHA, hardware, event counts,
  correctness, resource balance, and bench protocol results.
- The checked-in archive was unpacked and revalidated locally after GitHub transfer.

Validator output:

```text
FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT PASS: /workspace/artifacts/s1_cuda_qwen35_trace_fc06e256_20260715T143757Z
```

## Remaining S1 gap

The production identity chain and CUDA correctness slice now work, but the current basic sink writes
one verbose JSON event for every node transition on every frame. A four-request smoke generated a
`54.8 MB` serve trace. Before S1 can close, basic tracing must become bounded and its same-hardware
overhead must measure `<=2%`; the final S1 artifact must then be registered through `run_gate.py`.
