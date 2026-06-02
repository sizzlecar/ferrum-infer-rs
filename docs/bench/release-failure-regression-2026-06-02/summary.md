# Release-failure regression evidence, 2026-06-02

Commit under test: `f51564c fix: enforce generation token policy`.

## What was fixed

- Metal `ferrum run` Qwen3-30B-A3B GGUF no longer samples `<unk>` / `[PAD...]` tokens.
- Sampling now masks forbidden generation tokens before decode instead of filtering decoded text.
- HF tokenizer now supports `token_text(id)` reverse lookup for token-policy construction.
- Server default chat/completions sampling now uses deterministic temperature when the request omits `temperature`, matching CLI stability.
- CUDA-only builds no longer reference the cfg-gated `Device::Metal` variant.

## Metal Qwen3-30B-A3B GGUF

- `run` JSONL multi-turn: passed = `True`.
- `run` text four-turn: passed = `True`.
- `run` invalid token text present: `False`.
- `run` panic present: `False`.
- `run` text turn tok/s: `[12.8, 41.5, 47.8, 46.0]`.
- `serve` default-sampling multi-turn: passed = `True`.
- `serve` default-sampling stream: passed = `True`, chunks = `9`.
- `serve` invalid token text present: `False`.
- `serve` panic present: `False`.

Known limitation: Metal GGUF MoE c16 serving benchmark is not part of this run-fix pass. Correctness-safe product default currently uses `FERRUM_PAGED_MAX_SEQS=1`; attempting the old c16 bench path exhausts the paged KV pool and is tracked separately from the `run`/single-session regression fixed here.

## CUDA Qwen3-30B-A3B GPTQ-Int4 on RTX 4090

- CUDA release build: finished successfully; build tail saved in `cuda-build-f51564c-tail.log`.
- `run` strict multi-turn: passed = `True`.
- `run` final answers: `['\n\n已记住。', '\n\n蓝色月亮']`.
- `run` tok/s: `[153.14, 170.08]`.
- `serve` default-sampling multi-turn: passed = `True`.
- `serve` default-sampling stream: passed = `True`, chunks = `99`.
- `serve` invalid token text present: `False`.
- `serve` panic present: `False`.
- c16 sync performance smoke: passed = `True`, completed = `16`, tokens = `1024`, tok/s = `980.5`.

## Validation commands

Local targeted tests passed:

```sh
cargo test -q -p ferrum-server chat_accepts_stop_string_and_max_completion_tokens --lib
cargo test -q -p ferrum-engine sample_
cargo test -q -p ferrum-tokenizer test_tokenizer_token_text_reverse_lookup
cargo test -q -p ferrum-cli source_resolver --lib
CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build -q -p ferrum-cli --features metal --release
```

Remote CUDA build command used release features:

```sh
cargo build --release -p ferrum-cli --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
```
