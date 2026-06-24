#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/root/.cargo/bin:/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin:$PATH
export HF_HOME=/workspace/hf-cache
export RUST_BACKTRACE=1
export FERRUM_SCHED_TRACE=1
export FERRUM_UNIFIED_PROF=1
SHA=1cdc98f70f2ff544784fd677a934094749db0e6f
MODEL='Qwen/Qwen3.5-35B-A3B-GPTQ-Int4'
TOKENIZER='/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b'
DATASET='/workspace/ferrum-infer-rs/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl'
PORT=55994
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ART=/workspace/artifacts/w3_qwen35_fp16_indexed_state_c32_1cdc98f7_${STAMP}
mkdir -p "$ART"/{env,hardware,logs,run,server,perf,scripts}
cp "$0" "$ART/scripts/$(basename "$0")"
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$ART/logs/lane.log"; }
cleanup(){
  set +e
  if [ -f "$ART/server/server.pid" ]; then kill "$(cat "$ART/server/server.pid")" 2>/dev/null || true; fi
  pkill -P $$ 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits > "$ART/logs/nvidia_after.csv" 2>/dev/null || true
}
trap cleanup EXIT
log "lane_start art=$ART sha=$SHA release_evidence=false no_live_vllm=true"
{
  hostname
  which rustc || true
  which cargo || true
  rustc --version || true
  cargo --version || true
  nvidia-smi
  nvcc --version
} > "$ART/logs/preflight.log" 2>&1
nvidia-smi > "$ART/hardware/nvidia_smi_before.txt" 2>&1 || true
cd /workspace/ferrum-infer-rs
log git_sync_start
git fetch origin goal/w2-w3-release-grade > "$ART/logs/git-fetch.log" 2>&1
git pull --rebase --autostash > "$ART/logs/git-pull.log" 2>&1
git rev-parse HEAD > "$ART/env/git_sha.txt"
if [ "$(cat "$ART/env/git_sha.txt")" != "$SHA" ]; then log git_sha_mismatch; exit 22; fi
git status --short > "$ART/env/git_status_short.txt"
log git_sync_done
log cargo_check_start
(cargo check -p ferrum-kernels -p ferrum-models -p ferrum-cli > "$ART/logs/cargo-check.stdout.log" 2> "$ART/logs/cargo-check.stderr.log"); echo $? > "$ART/env/cargo-check.exit"
if [ "$(cat "$ART/env/cargo-check.exit")" != 0 ]; then log cargo_check_failed; exit 20; fi
log cargo_check_done
log release_build_start
(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source > "$ART/logs/build.stdout.log" 2> "$ART/logs/build.stderr.log"); echo $? > "$ART/env/build.exit"
if [ "$(cat "$ART/env/build.exit")" != 0 ]; then log release_build_failed; exit 21; fi
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
log release_build_done
printf '%s\n' "$TOKENIZER" > "$ART/env/tokenizer_path.txt"
log ferrum_run_smoke_start
RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --temperature 0 --max-tokens 8 --output-format jsonl --prompt 'What is 2+3? Answer with only the number.' --effective-config-json "$ART/run/effective_config.json" --decision-trace-jsonl "$ART/run/decision_trace.jsonl")
printf '%q ' "${RUN_CMD[@]}" > "$ART/run/run.command.txt"; printf '\n' >> "$ART/run/run.command.txt"
("${RUN_CMD[@]}" > "$ART/run/run.stdout.jsonl" 2> "$ART/run/run.stderr.log"); echo $? > "$ART/run/run.exit"
if [ "$(cat "$ART/run/run.exit")" != 0 ]; then log ferrum_run_failed; exit 30; fi
python3 - "$ART/run/effective_config.json" <<'PY' > "$ART/run/effective_config.assert.txt"
import json, sys
p=sys.argv[1]
data=json.load(open(p))
assert data.get('selected_max_sequences') == 32, data.get('selected_max_sequences')
assert data.get('selected_recurrent_state_max_slots') == 32, data.get('selected_recurrent_state_max_slots')
assert data.get('selected_admission_limit') == 32, data.get('selected_admission_limit')
print('ok')
PY
log ferrum_run_smoke_done
log serve_start
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --scheduler-prefill-first-until-active 32 --scheduler-prefill-step-chunk 6 --effective-config-json "$ART/server/effective_config.json" --decision-trace-jsonl "$ART/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ART/server/scheduler_trace.jsonl")
printf '%q ' "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"; printf '\n' >> "$ART/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/serve.stdout.log" 2> "$ART/server/serve.log" &
echo $! > "$ART/server/server.pid"
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
    break
  fi
  if ! kill -0 "$(cat "$ART/server/server.pid")" 2>/dev/null; then log serve_exited_before_ready; exit 31; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"
python3 - "$ART/server/effective_config.json" <<'PY' > "$ART/server/effective_config.assert.txt"
import json, sys
data=json.load(open(sys.argv[1]))
assert data.get('selected_max_sequences') == 32, data.get('selected_max_sequences')
assert data.get('selected_recurrent_state_max_slots') == 32, data.get('selected_recurrent_state_max_slots')
assert data.get('selected_admission_limit') == 32, data.get('selected_admission_limit')
print('ok')
PY
log serve_ready
python3 - <<'PY' > "$ART/server/chat_smoke_payload.json"
import json
print(json.dumps({"model":"Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"temperature":0,"max_tokens":8,"stream":False}))
PY
curl -fsS -H 'Content-Type: application/json' --data-binary "@$ART/server/chat_smoke_payload.json" "http://127.0.0.1:${PORT}/v1/chat/completions" > "$ART/server/chat_smoke_response.json" 2> "$ART/server/chat_smoke.curl.err"
python3 - "$ART/server/chat_smoke_response.json" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
content=data['choices'][0]['message']['content'].strip()
assert content == '5', content
PY
log serve_chat_smoke_done
log bench_c32_start
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 32 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$ART/perf/bench_ferrum_c32_32x1.json")
printf '%q ' "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum-c32.command.txt"; printf '\n' >> "$ART/perf/bench-ferrum-c32.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench.stdout.log" 2> "$ART/perf/bench.stderr.log"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/perf/bench.exit"
python3 - "$ART" <<'PY'
import json, pathlib, sys
art=pathlib.Path(sys.argv[1])
summary={"artifact":str(art),"sha":"1cdc98f70f2ff544784fd677a934094749db0e6f","bench_exit":(art/'perf/bench.exit').read_text().strip(),"release_evidence":False}
bench=art/'perf/bench_ferrum_c32_32x1.json'
if bench.exists():
    try:
        data=json.load(open(bench))
        summary['bench_keys']=sorted(data.keys())[:50]
        text=json.dumps(data)
        for key in ['output_throughput_tok_s','completed','errors','request_errors','throughput']:
            if key in data: summary[key]=data[key]
    except Exception as e:
        summary['bench_parse_error']=str(e)
trace=art/'server/scheduler_trace.jsonl'
last=None
if trace.exists():
    for line in trace.read_text(errors='replace').splitlines()[-500:]:
        if line.startswith('{'):
            try: last=json.loads(line)
            except Exception: pass
if last:
    for key in ['iteration','result','error','engine_counters','timing_us']:
        summary['last_'+key]=last.get(key)
    for section in ['scheduler_before','scheduler_after_schedule','scheduler_after_process','plan']:
        value=last.get(section)
        if isinstance(value, dict):
            if section == 'plan':
                summary['last_plan']={k:value.get(k) for k in ['batch_size','prefill_items','decode_items','scheduled_tokens_total','prefill_tokens','decode_tokens']}
            else:
                summary['last_'+section]={k:value.get(k) for k in ['waiting_queue_len','prefill_queue_len','decode_queue_len','active_len','completed_total','failed_total','cancelled_total','admitted_total','capacity_deferred_total','capacity_backpressure_admit_limit']}
serve=art/'server/serve.log'
if serve.exists():
    text=serve.read_text(errors='replace')
    summary['no_victim_warning_count']=text.count('Unified prefill alloc deferred')
    summary['oom_mentions']=text.count('out of memory')+text.count('OutOfMemory')+text.count('OOM')
(art/'perf/bench_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
find "$ART" -type f \( -name '*.log' -o -name '*.jsonl' \) ! -name '*.gz' -size +1M -print0 | xargs -0 -r gzip -f
if [ "$bench_rc" != 0 ]; then
  log "bench_c32_failed exit=$bench_rc"
  exit 50
fi
log "FERRUM W3 QWEN35 FP16 INDEXED STATE C32 DIAG PASS: $ART"
