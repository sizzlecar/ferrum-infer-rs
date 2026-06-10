# Ferrum 0.7.7 G0 Evidence

Status: source accelerator gates passed on git `22b8b7ada3fd586f018a95ba7c1d550d1c57001e`.
This is not the final release-complete packet; release asset, Homebrew, final
summary, and release-completion gates are still pending.

## Source Gates

| Lane | PASS line | Artifact |
| --- | --- | --- |
| Unit | `FERRUM GATE unit PASS: docs/release/g0/0.7.7/unit` | [`unit/`](unit/) |
| Metal | `FERRUM GATE metal PASS: docs/release/g0/0.7.7/metal` | [`metal/`](metal/) |
| CUDA Qwen full | `FERRUM GATE cuda-full PASS: docs/release/g0/0.7.7/cuda-full` | [`cuda-full/`](cuda-full/) |
| CUDA Llama dense | `FERRUM GATE cuda-llama-dense PASS: docs/release/g0/0.7.7/cuda-llama-dense` | [`cuda-llama-dense/`](cuda-llama-dense/) |

## Accelerator Evidence

- Metal covers `ferrum run` and `ferrum serve` correctness for Llama-3.1-8B,
  Qwen3-8B, and Qwen3-30B-A3B GGUF, plus tool-call, stream, stateful-loop,
  and throughput cells. Summary: [`metal/metal-readme/summary.md`](metal/metal-readme/summary.md).
- CUDA Qwen full covers `Qwen/Qwen3-30B-A3B-GPTQ-Int4` on one RTX 4090,
  random `256/128`, c=1/4/16/32, `n_repeats=3`, with zero request errors.
  Summary: [`cuda-full/summary.json`](cuda-full/summary.json).
- CUDA Llama dense covers
  `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` on one RTX 4090,
  including `ferrum run`, `ferrum serve`, tool-call, stream, and
  `bench-serve --fail-on-error --require-ci` c=1/4/16/32. Report:
  [`cuda-llama-dense/bench-serve.json`](cuda-llama-dense/bench-serve.json).

## Remote GPU Cleanup

Vast instance `40449138` was stopped after artifacts were copied back. The final
poll recorded `cur_state=stopped` and `actual_status=exited` in
[`cuda-remote/vast_instance_40449138_after_stop_poll1.json`](cuda-remote/vast_instance_40449138_after_stop_poll1.json).

## Pending Release Gates

```text
METAL TARBALL GATE PASS: docs/release/g0/0.7.7/metal-tarball
CUDA TARBALL GATE PASS: docs/release/g0/0.7.7/cuda-tarball
HOMEBREW METAL GATE PASS: docs/release/g0/0.7.7/homebrew-metal
HOMEBREW CUDA FETCH GATE PASS: docs/release/g0/0.7.7/homebrew-cuda-fetch
G0 RELEASE PASS: docs/release/g0/0.7.7
FERRUM RELEASE COMPLETION PASS: docs/release/g0/0.7.7/release-complete
```
