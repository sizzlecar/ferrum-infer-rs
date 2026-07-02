# W2 Gemma3 CUDA Graph Capture Shape Diagnostic

Date: 2026-06-16

This is diagnostic evidence only. It is not release-grade evidence and does not satisfy
`MODEL_RELEASE_GRADE_W2 PASS`.

## Scope

- Model: `gemma3:27b-gptq` (`circulus/gemma-3-27b-it-gptq`)
- Hardware: Vast instance `41178475`, 1x RTX 4090
- Driver/CUDA: NVIDIA driver `570.133.07`, CUDA toolkit `12.4`
- Git SHA: `7e56451e8faf8f2be977c946a6f0cdf9b6aa6e90`
- Source status for tracked source paths on remote: clean
- Binary SHA256: `2bd1ceb703e9315711635c20ec72da1ffd08df309646dfec3e4123cd1c298ffc`
- Instance lifecycle: artifacts copied back; instance stopped and polled to `actual_status=exited`

## Commands

Full unified graph diagnostic:

```bash
ferrum run gemma3:27b-gptq \
  --backend cuda \
  --prompt "What is 2+3? Answer with only the number." \
  --max-tokens 8 \
  --temperature 0 \
  --kv-capacity 512 \
  --max-num-seqs 2 \
  --unified-graph \
  --output-format jsonl
```

Layers-only graph diagnostic:

```bash
ferrum run gemma3:27b-gptq \
  --backend cuda \
  --prompt "What is 2+3? Answer with only the number." \
  --max-tokens 8 \
  --temperature 0 \
  --kv-capacity 512 \
  --max-num-seqs 2 \
  --unified-graph \
  --unified-graph-layers-only \
  --output-format jsonl

ferrum serve \
  --model gemma3:27b-gptq \
  --backend cuda \
  --host 127.0.0.1 \
  --port 18142 \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --unified-graph \
  --unified-graph-layers-only

ferrum bench-serve \
  --base-url http://127.0.0.1:18142 \
  --model gemma3:27b-gptq \
  --tokenizer /workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2 \
  --dataset random \
  --random-input-len 128 \
  --random-output-len 128 \
  --concurrency 16 \
  --num-prompts 64 \
  --warmup-requests 0 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271
```

Default same-hardware A/B:

```bash
ferrum serve \
  --model gemma3:27b-gptq \
  --backend cuda \
  --host 127.0.0.1 \
  --port 18143 \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048

ferrum bench-serve \
  --base-url http://127.0.0.1:18143 \
  --model gemma3:27b-gptq \
  --tokenizer /workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2 \
  --dataset random \
  --random-input-len 128 \
  --random-output-len 128 \
  --concurrency 16 \
  --num-prompts 64 \
  --warmup-requests 0 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271
```

## Results

Full unified graph is intentionally blocked for this Gemma3 product path:

```text
Configuration error: invalid auto config: FERRUM_UNIFIED_GRAPH: invalid override: full unified decode graph is disabled for Gemma3 sandwich-norm models
```

Layers-only graph correctness:

```text
RUN_SMOKE_PASS content='5' tokens=3
SERVE_SMOKE_PASS content='5' completion_tokens=3
```

Layers-only graph c16 diagnostic:

```text
64 completed / 0 errored / 26.3s
throughput      307.0 tok/s
output_token_count_source=usage
```

Default c16 A/B:

```text
64 completed / 0 errored / 25.9s
throughput      311.1 tok/s
output_token_count_source=usage
```

Observed layers-only graph behavior:

```text
[unified-graph-capture] count=1 scope=layers_only key=9582445794656808421 attention_key=8316016994327462400 m_total=16 num_seqs=16 max_kv_len=144
[unified-graph-replay] origin=pure count=64 scope=layers_only key=9582445794656808421 attention_key=8316016994327462400 m_total=16 num_seqs=16 max_kv_len=207
[unified-graph-capture] count=2 scope=layers_only key=13141637050213626555 attention_key=8316016994327462400 m_total=11 num_seqs=11 max_kv_len=266
[unified-graph-capture] count=4 scope=layers_only key=12159953342962959819 attention_key=8316016994327462400 m_total=15 num_seqs=15 max_kv_len=197
[unified-graph-capture] count=8 scope=layers_only key=12216274641861386432 attention_key=8318267720516763781 m_total=4 num_seqs=4 max_kv_len=265
```

## Interpretation

- There is no correctness failure in this diagnostic: `ferrum run` and `ferrum serve` smoke both pass on layers-only graph.
- The full unified graph path is not a legal Gemma3 product path now; typed config rejects it before runtime.
- Layers-only graph no longer shows the earlier `max_kv_len` key explosion in this c16 random diagnostic: the same `m_total=16` graph replays while `max_kv_len` changes.
- Layers-only graph does not improve this same-hardware c16 run: `307.0 tok/s` vs default `311.1 tok/s`.
- Next performance work should not force layers-only graph as the default. The higher-return direction is vLLM source comparison around decode dispatch, persistent input buffers, scheduler/admission shape stability, and per-token launch/CPU overhead.

## Artifacts

- Full graph disabled artifact: `remote_full_graph_disabled/`
- Layers-only graph diagnostic artifact: `remote_layers_only_diag/`
- Default A/B artifact: `remote_default_c16_ab/`
- Remote runner scripts:
  - `run_remote_graph_capture_shape.sh`
  - `run_remote_graph_capture_shape_layers_only_retry.sh`
  - `run_remote_default_c16_ab.sh`
