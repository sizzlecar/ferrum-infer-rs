# Real-Model API Smoke Attempt - HF 401 Blocker

Purpose: advance Milestone F/G real-model OpenAI API smoke evidence.

Remote artifact attempt:

- `/workspace/m3-real-model-api-smoke-20260601`
- log: `/workspace/m3-real-model-api-smoke-20260601.tmux.log`

Attempted command environment:

```bash
OUT_ROOT=/workspace/m3-real-model-api-smoke-20260601
MODEL=qwen3:0.6b
CARGO_FEATURES=cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
FERRUM_BIN=target/release/ferrum
PULL_MODEL=1
RUN_PYTHON_CHECK=1
ASYNC_TESTS=1
PYTHON_TEST=1
bash scripts/m3_real_model_api_smoke.sh
```

Result:

- The first command, `target/release/ferrum pull qwen3:0.6b`, failed before
  any SDK smoke could run.
- Error:

```text
API error (401 Unauthorized): {"error":"Invalid username or password."}
```

Follow-up:

- The smoke tmux session was killed after the pull failure to avoid wasting GPU
  time on a known-bad model-cache state.
- This is not F/G completion evidence.
- The ignored real-model tests currently use `Qwen/Qwen3-0.6B` as a fixed
  smoke model, so the existing cached Qwen3-30B GPTQ model cannot be used as a
  drop-in replacement without changing the test fixture or pre-populating the
  expected 0.6B cache entry.
- Vast instance `38872161` was stopped through the Vast API after GPU work
  completed and no further GPU task was ready.
