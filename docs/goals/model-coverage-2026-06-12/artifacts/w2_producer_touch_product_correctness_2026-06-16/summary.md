# W2 Producer-Touch Product Correctness Smoke

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_product_correctness_2026-06-16`
- Lane: W2 producer-touch product run/serve correctness smoke
- Instance: cached Vast 1x RTX 4090 instance `40826362`
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 565.77
- Remote base HEAD: `017300426514d62e8e50ac1546ff77d4d54fd6ce`
- Dirty source: product producer-touch source files were synced onto the
  remote working tree; the remote source diff is saved in `remote/source.diff`.
- Binary SHA256:
  `5078ea014ee5299a936de62f34475456f9a3c0500d34ab41a96ebcaf9c69fbd8`
- Vast cleanup: `cur_state=stopped actual_status=exited`

## Source Change Under Test

The product prototype adds a typed backend hint path:

- `Backend::fused_gelu_tanh_mul_split_with_down_hint(...)` defaults to the
  existing GeGLU behavior.
- CUDA overrides it and uses the hint only when the downstream projection is a
  `CudaMarlinLinear` backed by Marlin weights.
- Gemma unified and non-unified paths pass `layer.down_proj` as the hint for
  `Activation::GeluTanh`.
- The CUDA kernel computes the same GeGLU output and performs a small volatile
  read from `down_proj` qweight. This is the product analogue of the native
  `producer_touch_qweight_1x` signal.

## Correctness Evidence

`ferrum run`:

- Command shape:
  `ferrum run gemma3:27b-gptq --backend cuda --max-tokens 8 --temperature 0 --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl`
- rc: `0`
- assistant content: `5`
- finish_reason: `stop`
- n_tokens: `3`

`ferrum serve`:

- Command shape:
  `ferrum serve --model gemma3:27b-gptq --backend cuda --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512`
- readiness: `/v1/models` ready
- chat rc: `0`
- response content: `5`
- usage:
  `prompt_tokens=23`, `completion_tokens=1`, `total_tokens=24`

Log scan:

- `server/error_scan.txt`: `0` lines for panic/error/NaN/`<unk>`/`[PAD]`/
  invalid UTF/mojibake/illegal address/CUDA error patterns used in this
  artifact.
- `correctness_check.json`: `ok=true`

## Interpretation

The producer-touch product prototype preserves basic Gemma3 CUDA GPTQ product
correctness for both required entrypoints in this smoke: `ferrum run` and
`ferrum serve`.

This is not release-grade evidence and it is not performance evidence. The next
step is a focused same-dataset endpoint diagnostic against the existing vLLM
ShareGPT baseline to see whether the native 4-5% tail-MLP segment win survives
product overhead.

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
