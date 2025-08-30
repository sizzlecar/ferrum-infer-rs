## Candle-based MVP Development Plan

### Status and Intent
- Target: minimal, end-to-end single-node inference path using Candle
- Scope: 1 model family (TinyLlama/7B-compatible), single GPU or CPU fallback, simple batching, basic sampling, streaming
- Out of scope (MVP): PagedAttention, preemption, multi-GPU, quantization, advanced schedulers

### Success Criteria
- Build succeeds with Candle backend (CPU and optionally CUDA)
- CLI can generate tokens for a prompt end-to-end
- Minimal OpenAI-compatible `/v1/chat/completions` works (non-stream + SSE)
- Throughput and latency are reasonable for a tiny model (sanity only)

### Architecture Mapping (existing crates → MVP responsibilities)
- `crates/ferrum-core`
  - Keep traits as-is (`Backend`, `Model`, `InferenceEngine`, `BatchManager`, `Scheduler`, etc.)
- `crates/ferrum-engine`
  - Wire model runner with Candle backend
  - Implement simple decode loop (greedy / top-k / top-p)
  - Use naive per-request KV cache (no paging)
  - Simplify batch manager to static/dynamic batching without preemption
- `crates/ferrum-models`
  - Provide Candle model configs and tokenizer plumbing
- `crates/ferrum-server`
  - Implement minimal Axum HTTP server with `/v1/chat/completions` (JSON + SSE)
- `crates/ferrum-runtime`
  - Optional for MVP: provide device selection and basic memory stats
- `crates/ferrum-cli`
  - `ferrum-cli generate --model <id> --prompt "..."` end-to-end demo

### Features (MVP)
- Model: TinyLlama (preferred for tests) or LLaMA-2/7B for GPU validation
- Tokenizer: `tokenizers` with model-specific tokenizer (e.g., LLaMA)
- Sampling: greedy, top-k, top-p, temperature, stop sequences
- Batching: simple dynamic batching with max batch size + small wait (no preemption)
- KV cache: per-request, contiguous tensors on device; reused across decode steps
- Streaming: SSE frames per token (delta text), end event when finished
- Logging/metrics: tracing spans and basic counters (requests, tokens, latency)

### Candle Backend Integration
- Create `ferrum-backend-candle` module inside `crates/ferrum-engine` or a new crate later
- Implement `ferrum_core::Backend` for `CandleBackend`
  - `initialize()` → select device (CPU/CUDA) and verify capability
  - `create_tensor(...)` → wrapper around Candle tensor creation on device
  - `load_weights(path, dtype, device)` → load model via `candle-transformers` and return a `CandleModel` (impl `ferrum_core::Model`)
- `CandleModel` (impl `Model`)
  - `info()` → model id, dtype, max seq len
  - `encode(text)` / `decode(tokens)` → via `tokenizers`
  - `forward(input)` → call Candle forward for prefill
  - `generate_next_token(input_ids, past_kv, sampling_params)` →
    - If `past_kv` is None → prefill forward; else decode step with kv reuse
    - Compute logits, apply sampling, update kv

### KV Cache (simple)
- Structure: `KVCache { keys: Tensor, values: Tensor, layout: [layers, heads, seq, head_dim] }`
- Prefill: allocate large-enough tensors for the sequence; copy K/V from layer outputs
- Decode: append new positions; index into last position to compute next token
- Device: CPU by default; CUDA if available (compile-time feature)

### Batch Manager (simple)
- Strategy: dynamic with parameters `{ max_batch_size, max_wait_ms }`
- Behavior (per tick): collect waiting requests → pad to max length → run prefill or one decode step per active request
- For MVP: no mid-batch join during decode; keep it simple

### Sampling
- Implement utility applying temperature, top-k, top-p
- Greedy path for deterministic tests
- Stop sequences and max tokens enforcement

### Server (Axum, minimal)
- `POST /v1/chat/completions`
  - Non-stream: return final text in one response
  - `stream=true`: SSE with per-token deltas, then `[DONE]`
- Simple auth off for MVP; add later if needed

### CLI
- `ferrum-cli generate --model <id> --device <cpu|cuda:0> --prompt "..." --max-tokens 64 --temperature 0.8`
- Prints streaming tokens to stdout

### Build & Features
- Workspace dependencies already include Candle crates
- Add features:
  - `candle-core = { version = "0.7", features = ["cuda"] }` for CUDA builds
  - Provide `--features cuda` build path; default to CPU if not set
- Example:
```bash
# CPU
cargo build -p ferrum-engine
# CUDA (if supported toolchain installed)
cargo build -p ferrum-engine --features cuda
```

### Implementation Steps (Checkpoints)
1) Backend and Model
- Implement `CandleBackend` and `CandleModel` with encode/decode/forward
- Wire tokenizer and minimal LLaMA/TinyLlama model loading

2) Decode Loop & Sampling
- Implement `generate_next_token` and sampling utils
- Sanity test: CLI generates text for tiny model on CPU

3) KV Cache v0
- Add per-request KV struct; prefill + decode reuse
- Unit tests for shape/indexing correctness

4) Batch Manager v0
- Aggregate requests with `{max_batch_size, max_wait_ms}`
- Run prefill/step decode across batch with padding

5) Server (SSE)
- Minimal `/v1/chat/completions` route
- Stream tokens via SSE; integrate with engine path

6) Observability
- Add tracing spans and counters (tokens generated, requests, errors)
- Basic latency measurement around forward/step

### Testing Strategy
- Unit tests: tokenization idempotency, sampling edge cases, KV append
- Integration tests:
  - CLI prompt → non-empty output (CPU TinyLlama) within time budget
  - Server non-stream and stream endpoints (against Tiny model)
- Optional regression: determinism under greedy sampling

### Models & Assets
- Prefer tiny models for CI/dev: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Local cache via `hf-hub` to avoid repeated downloads
- Document expected VRAM for 7B FP16 if validating CUDA locally

### Risks & Mitigations
- Missing Candle ops: avoid exotic layers; start with supported architectures
- CUDA environment: document toolchain setup; keep CPU path working
- Memory: Tiny model for tests; 7B only for manual validation

### Acceptance Checklist
- [ ] `cargo build` succeeds (CPU)
- [ ] CLI can generate tokens for given prompt
- [ ] Server returns a streamed completion for a short prompt
- [ ] Sampling/stop/max_tokens behave as configured
- [ ] Basic metrics/logs visible

### Next (post-MVP)
- PagedAttention block tables and allocation
- Prefill/decode separation with token-level continuous batching
- Prefix cache reuse
- Quantization and fused attention kernels via FFI
