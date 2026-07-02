# W2 unified graph layers-only CUDA smoke (2026-06-16)

Status: diagnostic smoke PASS for product run/serve. This is not release-grade evidence and does not produce `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Scope

- Goal lane: W2 Gemma3 27B GPTQ CUDA unified graph layers-only smoke.
- Instance: Vast `40826362`, 1x RTX 4090.
- Source SHA: `2ab90fda518a85a317782abee300660598edab9f`.
- Remote source state: clean detached worktree at `/workspace/ferrum-clean-2ab90fda`.
- Binary: `/workspace/ferrum-infer-rs/target/release/ferrum`.
- Binary SHA256: `4d141487f5c6ad98c1b11a85e01752b291b5750e5a6a6d434058c12245d030f6`.
- Version: `ferrum 0.7.7`.
- GPU cleanup: stopped; `vast_shutdown/stopped.json` has `actual_status=exited`.

## Build

Command:

```bash
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

Rust/Cargo:

```text
rustc 1.96.0 (ac68faa20 2026-05-25)
cargo 1.96.0 (30a34c682 2026-05-25)
```

## Correctness Smoke

### ferrum run

Command recorded in `run/run.command.txt`.

- rc: `0`
- selected graph mode: `unified_decode_graph_layers_only`
- error scan: no `CUDA_ERROR`, illegal address, OOM, panic, `<unk>`, `[PAD]`, invalid UTF/mojibake hits in `run/`.

### ferrum serve

Command recorded in `server/serve.command.txt`.

- rc: `0`
- selected graph mode: `unified_decode_graph_layers_only`
- validation: `serve_smoke/chat_validation.json`
- response text: `The capital of France is Paris.`
- usage: `{"prompt_tokens":23,"completion_tokens":8,"total_tokens":31}`
- error scan: no `CUDA_ERROR`, illegal address, OOM, panic, `<unk>`, `[PAD]`, invalid UTF/mojibake hits in `server/` or `serve_smoke/`.

## Interpretation

The layers-only graph scope avoids the previous full unified graph instantiate/OOM failure in a product-path run/serve smoke. This supports the vLLM-source hypothesis that Ferrum's full unified graph captures too broad a region; the next performance question is whether replay is actually happening often enough and whether it moves c16 throughput.

## Limits

- Smoke only; no `bench-serve --require-ci`, no N=3, and no same-hardware performance ratio claim.
- Only Gemma3 27B GPTQ W2 product smoke; not W2 release-grade.
