#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/root/.cargo/bin:/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin:$PATH
export HF_HOME=/workspace/hf-cache
export HF_XET_HIGH_PERFORMANCE=1
export RUST_BACKTRACE=1
export FERRUM_SCHED_TRACE=1
export FERRUM_UNIFIED_PROF=1
SHA=7476295db6f4c5510d83328e450440b2aec1d8cf
BRANCH=goal/w2-w3-release-grade
REPO_URL=https://github.com/sizzlecar/ferrum-infer-rs.git
REPO=/workspace/ferrum-infer-rs
MODEL='Qwen/Qwen3.5-35B-A3B-GPTQ-Int4'
DATASET='/workspace/ferrum-infer-rs/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl'
PORT=55994
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ART=/workspace/artifacts/w3_qwen35_capacity_wait_c32_7476295d_${STAMP}
mkdir -p "$ART"/{env,hardware,logs,run,server,perf,scripts,prefetch}
cp "$0" "$ART/scripts/$(basename "$0")"
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$ART/logs/lane.log"; }
cleanup(){
  set +e
  if [ -f "$ART/server/server.pid" ]; then kill "$(cat "$ART/server/server.pid")" 2>/dev/null || true; fi
  pkill -P $$ 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits > "$ART/logs/nvidia_after.csv" 2>/dev/null || true
}
trap cleanup EXIT
log "lane_start art=$ART sha=$SHA release_evidence=false no_live_vllm=true cost_per_hr_approx=0.583"
{
  date -u
  hostname
  uname -a
  nvidia-smi || true
  nvcc --version || true
  df -h /workspace || true
} > "$ART/logs/preflight.log" 2>&1
nvidia-smi > "$ART/hardware/nvidia_smi_before.txt" 2>&1 || true
log bootstrap_start
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > "$ART/logs/rustup-init.sh"
  sh "$ART/logs/rustup-init.sh" -y --default-toolchain 1.96.0 > "$ART/logs/rustup-install.log" 2>&1
else
  rustup toolchain install 1.96.0 > "$ART/logs/rustup-toolchain.log" 2>&1 || true
  rustup default 1.96.0 > "$ART/logs/rustup-default.log" 2>&1 || true
fi
export PATH=/root/.cargo/bin:/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin:$PATH
python3 -m pip install -q --upgrade huggingface_hub hf_xet > "$ART/logs/pip-hf.log" 2>&1 || true
rustc --version > "$ART/env/rustc.txt" 2>&1 || true
cargo --version > "$ART/env/cargo.txt" 2>&1 || true
log bootstrap_done
log git_sync_start
if [ ! -d "$REPO/.git" ]; then
  git clone "$REPO_URL" "$REPO" > "$ART/logs/git-clone.log" 2>&1
fi
cd "$REPO"
git remote set-url origin "$REPO_URL" || true
git fetch origin "$BRANCH" --prune > "$ART/logs/git-fetch.log" 2>&1
git checkout "$BRANCH" > "$ART/logs/git-checkout-branch.log" 2>&1 || git checkout -B "$BRANCH" "origin/$BRANCH" > "$ART/logs/git-checkout-branch.log" 2>&1
git pull --rebase --autostash origin "$BRANCH" > "$ART/logs/git-pull.log" 2>&1
git checkout --detach "$SHA" > "$ART/logs/git-checkout-detach.log" 2>&1
git rev-parse HEAD > "$ART/env/git_sha.txt"
if [ "$(cat "$ART/env/git_sha.txt")" != "$SHA" ]; then log git_sha_mismatch; exit 22; fi
git status --short > "$ART/env/git_status_short.txt"
log git_sync_done
if [ ! -f "$DATASET" ]; then log dataset_missing; exit 23; fi
log hf_prefetch_start
HF_HOME="$HF_HOME" HF_XET_HIGH_PERFORMANCE=1 python3 - <<'PY_HF' > "$ART/prefetch/hf-prefetch.log" 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", cache_dir="/workspace/hf-cache/hub")
PY_HF
PREFETCH_PID=$!
log cargo_check_start
(cargo check -p ferrum-kernels -p ferrum-models -p ferrum-cli > "$ART/logs/cargo-check.stdout.log" 2> "$ART/logs/cargo-check.stderr.log"); echo $? > "$ART/env/cargo-check.exit"
if [ "$(cat "$ART/env/cargo-check.exit")" != 0 ]; then log cargo_check_failed; wait "$PREFETCH_PID" || true; exit 20; fi
log cargo_check_done
log release_build_start
(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source > "$ART/logs/build.stdout.log" 2> "$ART/logs/build.stderr.log"); echo $? > "$ART/env/build.exit"
if [ "$(cat "$ART/env/build.exit")" != 0 ]; then log release_build_failed; wait "$PREFETCH_PID" || true; exit 21; fi
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
log release_build_done
log hf_prefetch_wait_start
set +e
wait "$PREFETCH_PID"
prefetch_rc=$?
set -e
echo "$prefetch_rc" > "$ART/prefetch/hf-prefetch.exit"
if [ "$prefetch_rc" != 0 ]; then log hf_prefetch_failed; exit 24; fi
TOKENIZER=$(find "$HF_HOME/hub" -path '*models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4*/snapshots/*/tokenizer.json' -print | head -n 1 | xargs -r dirname)
if [ -z "$TOKENIZER" ] || [ ! -f "$TOKENIZER/tokenizer.json" ]; then log tokenizer_missing; exit 25; fi
printf '%s\n' "$TOKENIZER" > "$ART/env/tokenizer_path.txt"
log hf_prefetch_done tokenizer="$TOKENIZER"
log ferrum_run_smoke_start
RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --temperature 0 --max-tokens 8 --output-format jsonl --prompt 'What is 2+3? Answer with only the number.' --effective-config-json "$ART/run/effective_config.json" --decision-trace-jsonl "$ART/run/decision_trace.jsonl")
printf '%q ' "${RUN_CMD[@]}" > "$ART/run/run.command.txt"; printf '\n' >> "$ART/run/run.command.txt"
("${RUN_CMD[@]}" > "$ART/run/run.stdout.jsonl" 2> "$ART/run/run.stderr.log"); echo $? > "$ART/run/run.exit"
if [ "$(cat "$ART/run/run.exit")" != 0 ]; then log ferrum_run_failed; exit 30; fi
python3 - "$ART/run/effective_config.json" <<'PY_RUN_ASSERT' > "$ART/run/effective_config.assert.txt"
import json, sys
p=sys.argv[1]
data=json.load(open(p))
assert data.get('selected_max_sequences') == 32, data.get('selected_max_sequences')
assert data.get('selected_recurrent_state_max_slots') == 32, data.get('selected_recurrent_state_max_slots')
assert data.get('selected_admission_limit') == 32, data.get('selected_admission_limit')
print('ok')
PY_RUN_ASSERT
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
python3 - "$ART/server/effective_config.json" <<'PY_SERVE_ASSERT' > "$ART/server/effective_config.assert.txt"
import json, sys
data=json.load(open(sys.argv[1]))
assert data.get('selected_max_sequences') == 32, data.get('selected_max_sequences')
assert data.get('selected_recurrent_state_max_slots') == 32, data.get('selected_recurrent_state_max_slots')
assert data.get('selected_admission_limit') == 32, data.get('selected_admission_limit')
print('ok')
PY_SERVE_ASSERT
log serve_ready
python3 - <<'PY_PAYLOAD' > "$ART/server/chat_smoke_payload.json"
import json
print(json.dumps({"model":"Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"temperature":0,"max_tokens":8,"stream":False}))
PY_PAYLOAD
curl -fsS -H 'Content-Type: application/json' --data-binary "@$ART/server/chat_smoke_payload.json" "http://127.0.0.1:${PORT}/v1/chat/completions" > "$ART/server/chat_smoke_response.json" 2> "$ART/server/chat_smoke.curl.err"
python3 - "$ART/server/chat_smoke_response.json" <<'PY_CHAT_ASSERT'
import json, sys
data=json.load(open(sys.argv[1]))
content=data['choices'][0]['message']['content'].strip()
assert content == '5', content
PY_CHAT_ASSERT
log serve_chat_smoke_done
log bench_c32_start
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 32 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$ART/perf/bench_ferrum_c32_32x1.json")
printf '%q ' "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum-c32.command.txt"; printf '\n' >> "$ART/perf/bench-ferrum-c32.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench.stdout.log" 2> "$ART/perf/bench.stderr.log"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/perf/bench.exit"
python3 - "$ART" "$SHA" <<'PY_SUMMARY'
import gzip, json, pathlib, re, sys
art=pathlib.Path(sys.argv[1]); sha=sys.argv[2]
summary={"artifact":str(art),"sha":sha,"bench_exit":(art/'perf/bench.exit').read_text().strip(),"release_evidence":False,"no_live_vllm":True}
bench=art/'perf/bench_ferrum_c32_32x1.json'
if bench.exists():
    try:
        data=json.load(open(bench))
        summary['bench_keys']=sorted(data.keys())[:80]
        def walk(o, prefix=''):
            if isinstance(o, dict):
                for k,v in o.items():
                    p=f'{prefix}.{k}' if prefix else k
                    if isinstance(v,(dict,list)):
                        yield from walk(v,p)
                    elif any(s in k.lower() for s in ['throughput','completed','error','token_count_source','request']):
                        yield p,v
            elif isinstance(o, list):
                for i,v in enumerate(o):
                    yield from walk(v,f'{prefix}[{i}]')
        summary['bench_selected_metrics']={k:v for k,v in walk(data)}
    except Exception as e:
        summary['bench_parse_error']=str(e)
trace=art/'server/scheduler_trace.jsonl'
if not trace.exists() and (art/'server/scheduler_trace.jsonl.gz').exists():
    lines=gzip.open(art/'server/scheduler_trace.jsonl.gz','rt',errors='replace').read().splitlines()
elif trace.exists():
    lines=trace.read_text(errors='replace').splitlines()
else:
    lines=[]
last=None
for line in lines[-2000:]:
    if line.startswith('{'):
        try:
            last=json.loads(line)
        except Exception:
            pass
if last:
    summary['last_iteration']=last.get('iteration')
    for section in ['scheduler_before','scheduler_after_schedule','scheduler_after_process']:
        value=last.get(section)
        if isinstance(value, dict):
            summary['last_'+section]={k:value.get(k) for k in ['waiting_queue_len','prefill_queue_len','decode_queue_len','active_len','completed_total','failed_total','cancelled_total','admitted_total','capacity_deferred_total','capacity_backpressure_admit_limit','capacity_blocked_waiting_len']}
    plan=last.get('plan')
    if isinstance(plan, dict):
        summary['last_plan']={k:plan.get(k) for k in ['batch_size','prefill_items','decode_items','scheduled_tokens_total','prefill_tokens','decode_tokens']}
serve=art/'server/serve.log'
text=serve.read_text(errors='replace') if serve.exists() else ''
summary['log_counts']={
    'unified_kv_admission_failed': text.count('Unified KV admission failed'),
    'cancelled_during_decode': text.count('cancelled during decode'),
    'preempting_request': text.count('Preempting request'),
    'oom_mentions': len(re.findall(r'out of memory|OutOfMemory|OOM', text)),
    'panic_mentions': text.lower().count('panic'),
    'block_pool_exhausted': text.count('Block pool exhausted'),
    'unified_prefill_alloc_deferred': text.count('Unified prefill alloc deferred'),
}
(art/'perf/bench_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY_SUMMARY
if [ "$bench_rc" != 0 ]; then
  log "bench_c32_failed exit=$bench_rc"
  find "$ART" -type f \( -name '*.log' -o -name '*.jsonl' \) ! -name '*.gz' -size +1M -print0 | xargs -0 -r gzip -f
  exit 50
fi
find "$ART" -type f \( -name '*.log' -o -name '*.jsonl' \) ! -name '*.gz' -size +1M -print0 | xargs -0 -r gzip -f
log "FERRUM W3 QWEN35 CAPACITY WAIT C32 DIAG PASS: $ART"
