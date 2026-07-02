#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/.cargo/bin:/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin:$PATH
SHA=b8116dc498e7fad50263d0dc580daf38194cb74e
SHORT=${SHA:0:8}
MODEL="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ART="/workspace/artifacts/w3_qwen35_capacity_defer_kv_release_c32_${SHORT}_${STAMP}"
PORT=55991
DATASET="/workspace/ferrum-infer-rs/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl"
export HF_HOME=/workspace/hf-cache
export CARGO_TERM_COLOR=never
mkdir -p "$ART"/{env,hardware,logs,run,server,perf,scripts}
cp "$0" "$ART/scripts/$(basename "$0")"
log(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$ART/logs/lane.log"; }
SERVER_PID=""
cleanup(){
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits > "$ART/logs/nvidia_after.csv" 2>/dev/null || true
}
trap cleanup EXIT
log "lane_start art=$ART sha=$SHA release_evidence=false no_live_vllm=true"
{
  hostname
  rustc --version
  cargo --version
  nvcc --version | tail -n 1
  nvidia-smi
} > "$ART/logs/preflight.log" 2>&1 || true
nvidia-smi > "$ART/hardware/nvidia_smi_before.txt" 2>&1 || true
cd /workspace/ferrum-infer-rs
log git_sync_start
git fetch origin goal/w2-w3-release-grade > "$ART/logs/git-fetch.log" 2>&1
git checkout goal/w2-w3-release-grade > "$ART/logs/git-checkout.log" 2>&1
git reset --hard "$SHA" > "$ART/logs/git-reset.log" 2>&1
git rev-parse HEAD > "$ART/env/git_sha.txt"
git status --short --untracked-files=no > "$ART/env/git_status_short.txt"
log git_sync_done
TOKENIZER=$(find /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots -name tokenizer.json -printf '%h\n' | head -1)
if [ -z "$TOKENIZER" ]; then echo "missing tokenizer snapshot" >&2; exit 11; fi
echo "$TOKENIZER" > "$ART/env/tokenizer_path.txt"
log cargo_check_start
echo 'cargo check -p ferrum-engine -p ferrum-scheduler' > "$ART/env/cargo-check.command.txt"
set +e
cargo check -p ferrum-engine -p ferrum-scheduler > "$ART/logs/cargo-check.stdout.log" 2> "$ART/logs/cargo-check.stderr.log"
CHECK_EXIT=$?
set -e
echo "$CHECK_EXIT" > "$ART/env/cargo-check.exit"
if [ "$CHECK_EXIT" -ne 0 ]; then log "cargo_check_failed exit=$CHECK_EXIT"; exit 20; fi
log cargo_check_done
log release_build_start
echo 'cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source' > "$ART/env/build.command.txt"
set +e
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source > "$ART/logs/build.stdout.log" 2> "$ART/logs/build.stderr.log"
BUILD_EXIT=$?
set -e
echo "$BUILD_EXIT" > "$ART/env/build.exit"
if [ "$BUILD_EXIT" -ne 0 ]; then log "release_build_failed exit=$BUILD_EXIT"; exit 21; fi
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
log release_build_done
log ferrum_run_smoke_start
RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --temperature 0 --max-tokens 8 --output-format jsonl --prompt "What is 2+3? Answer with only the number." --effective-config-json "$ART/run/effective_config.json" --decision-trace-jsonl "$ART/run/decision_trace.jsonl")
printf '%q ' "${RUN_CMD[@]}" > "$ART/run/run.command.txt"; echo >> "$ART/run/run.command.txt"
set +e
"${RUN_CMD[@]}" > "$ART/run/run.stdout.jsonl" 2> "$ART/run/run.stderr.log"
RUN_EXIT=$?
set -e
echo "$RUN_EXIT" > "$ART/run/run.exit"
if [ "$RUN_EXIT" -ne 0 ]; then log "ferrum_run_failed exit=$RUN_EXIT"; exit 30; fi
python3 - "$ART/run/run.stdout.jsonl" "$ART/run/effective_config.json" <<'PY'
import json, sys
stdout, cfgp = sys.argv[1], sys.argv[2]
lines=[json.loads(l) for l in open(stdout) if l.strip()]
assert any(str(o.get('content','')).strip() == '5' for o in lines), lines[-3:]
cfg=json.load(open(cfgp))
assert cfg.get('selected_max_sequences') == 32, cfg.get('selected_max_sequences')
assert cfg.get('selected_recurrent_state_max_slots') == 32, cfg.get('selected_recurrent_state_max_slots')
assert cfg.get('selected_admission_limit') == 32, cfg.get('selected_admission_limit')
PY
echo true > "$ART/run/effective_config.assert.txt"
log ferrum_run_smoke_done
log serve_start
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --scheduler-prefill-first-until-active 32 --scheduler-prefill-step-chunk 6 --effective-config-json "$ART/server/effective_config.json" --decision-trace-jsonl "$ART/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ART/server/scheduler_trace.jsonl")
printf '%q ' "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"; echo >> "$ART/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/serve.stdout.log" 2> "$ART/server/serve.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
    log serve_ready
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then log serve_exited_early; exit 31; fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then log serve_not_ready; exit 32; fi
python3 - "$ART/server/effective_config.json" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1]))
assert cfg.get('selected_max_sequences') == 32, cfg.get('selected_max_sequences')
assert cfg.get('selected_recurrent_state_max_slots') == 32, cfg.get('selected_recurrent_state_max_slots')
assert cfg.get('selected_admission_limit') == 32, cfg.get('selected_admission_limit')
PY
echo true > "$ART/server/effective_config.assert.txt"
CHAT_BODY='{"model":"Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"temperature":0,"max_tokens":8}'
curl -fsS -H 'Content-Type: application/json' -d "$CHAT_BODY" "http://127.0.0.1:$PORT/v1/chat/completions" > "$ART/server/chat_smoke_response.json" 2> "$ART/server/chat_smoke.curl.err"
python3 - "$ART/server/chat_smoke_response.json" <<'PY'
import json, sys
o=json.load(open(sys.argv[1]))
content=o['choices'][0]['message']['content'].strip()
assert content == '5', content
PY
log serve_chat_smoke_done
log bench_c32_start
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 32 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$ART/perf/bench_ferrum_c32_32x1.json")
printf '%q ' "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum-c32.command.txt"; echo >> "$ART/perf/bench-ferrum-c32.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench.stdout.log" 2> "$ART/perf/bench.stderr.log"
BENCH_EXIT=$?
set -e
echo "$BENCH_EXIT" > "$ART/perf/bench.exit"
if [ "$BENCH_EXIT" -ne 0 ]; then
  tail -n 200 "$ART/perf/bench.stderr.log" > "$ART/perf/bench.failure.tail.txt" || true
  log "bench_c32_failed exit=$BENCH_EXIT"
  exit 50
fi
log bench_c32_done
python3 - "$ART" <<'PY'
import gzip, json, pathlib, sys
art=pathlib.Path(sys.argv[1])
bench=json.load(open(art/'perf'/'bench_ferrum_c32_32x1.json'))
summary={'bench_keys': sorted(bench.keys())}
# Keep extraction defensive because bench schema has evolved.
for k in ['closed_loop','results','cells','summary','completed','errored','output_throughput_tok_s']:
    if k in bench:
        summary[k]=bench[k]
(art/'perf'/'bench_metrics_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
trace=art/'server'/'scheduler_trace.jsonl'
if trace.exists():
    last=None; first=None; count=0; max_cancelled=0; max_completed=0; max_admitted=0
    with open(trace, errors='replace') as f:
        for line in f:
            count += 1
            try: o=json.loads(line)
            except Exception: continue
            if first is None: first=o
            last=o
            for s in [o.get('scheduler_before') or {}, o.get('scheduler_after_schedule') or {}, o.get('scheduler_after_process') or {}]:
                max_cancelled=max(max_cancelled, s.get('cancelled_total') or 0)
                max_completed=max(max_completed, s.get('completed_total') or 0)
                max_admitted=max(max_admitted, s.get('admitted_total') or 0)
    def pick(o):
        if not o: return None
        s=o.get('scheduler_after_process') or o.get('scheduler_after_schedule') or {}
        p=o.get('plan') or {}
        e=o.get('engine_counters') or {}
        return {'iteration':o.get('iteration'),'result':o.get('result'),'completed_total':s.get('completed_total'),'cancelled_total':s.get('cancelled_total'),'admitted_total':s.get('admitted_total'),'waiting_queue_len':s.get('waiting_queue_len'),'active_len':s.get('active_len'),'prefill_items':p.get('prefill_items'),'decode_items':p.get('decode_items'),'scheduled_tokens_total':p.get('scheduled_tokens_total'),'prefill_delta':e.get('prefill_tokens_delta'),'decode_delta':e.get('decode_tokens_delta')}
    summary={'line_count':count,'max_cancelled_total':max_cancelled,'max_completed_total':max_completed,'max_admitted_total':max_admitted,'first':pick(first),'last':pick(last)}
    (art/'server'/'scheduler_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
PY
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
# Preserve useful tails and compress noisy logs.
if [ -f "$ART/server/serve.log" ]; then tail -n 400 "$ART/server/serve.log" > "$ART/server/serve.tail.400.log"; gzip -f "$ART/server/serve.log"; fi
if [ -f "$ART/server/scheduler_trace.jsonl" ]; then gzip -f "$ART/server/scheduler_trace.jsonl"; fi
if [ -f "$ART/logs/build.stderr.log" ]; then gzip -f "$ART/logs/build.stderr.log"; fi
log "FERRUM W3 QWEN35 CAPACITY DEFER KV RELEASE C32 DIAG PASS: $ART"
echo "PASS_ART=$ART"
