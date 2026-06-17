#!/usr/bin/env bash
set -euo pipefail
ART=/workspace/w2_dynamic_kv_c32_serve_smoke_cuda_2026-06-17
REPO=/workspace/ferrum-infer-rs
PORT=18458
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi
mkdir -p "$ART/env" "$ART/build" "$ART/correctness/smoke"
exec > >(tee -a "$ART/run.log") 2>&1

date -u +%FT%TZ > "$ART/env/start_utc.txt"
cd "$REPO"
git rev-parse HEAD > "$ART/env/git_head.txt"
git status --short > "$ART/env/git_status_short.txt"
rustc --version > "$ART/env/rustc_version.txt"
cargo --version > "$ART/env/cargo_version.txt"
nvidia-smi > "$ART/env/nvidia_smi_before.txt"
nvcc --version > "$ART/env/nvcc_version.txt"
df -h /workspace > "$ART/env/df_before.txt"

BUILD_START=$(date +%s)
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source 2>&1 | tee "$ART/build/cargo_build_release.log"
BUILD_END=$(date +%s)
printf '%s\n' "$((BUILD_END-BUILD_START))" > "$ART/build/build_seconds.txt"
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"

export HF_HOME=/workspace/hf-cache
export FERRUM_MODEL_DIR=/workspace/hf-cache
export FERRUM_LOG=info
SERVE_LOG="$ART/correctness/serve.log"
EFFECTIVE="$ART/correctness/serve_effective_config.json"
TRACE="$ART/correctness/serve_decision_trace.jsonl"
CMD=(target/release/ferrum serve gemma3:27b-gptq --backend cuda --host 127.0.0.1 --port "$PORT" --max-num-seqs 32 --effective-config-json "$EFFECTIVE" --decision-trace-jsonl "$TRACE")
printf '%q ' "${CMD[@]}" > "$ART/correctness/serve.command.txt"
printf '\n' >> "$ART/correctness/serve.command.txt"
( "${CMD[@]}" ) > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/correctness/server.pid"
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 - <<'PY' "$PORT" "$ART/correctness/health_wait.json"
import json, sys, time, urllib.request
port=sys.argv[1]
out=sys.argv[2]
rows=[]
deadline=time.time()+900
ok=False
while time.time()<deadline:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as r:
            body=r.read().decode('utf-8','replace')
        rows.append({'ts':time.time(),'status':'ok','body':body[:500]})
        ok=True
        break
    except Exception as e:
        rows.append({'ts':time.time(),'status':'wait','error':repr(e)})
        time.sleep(5)
with open(out,'w') as f:
    json.dump({'ok':ok,'rows':rows}, f, indent=2)
if not ok:
    raise SystemExit('server did not become healthy')
PY

python3 - <<'PY' "$EFFECTIVE" "$ART/correctness/effective_config_keys.json"
import json, sys
path=sys.argv[1]
out=sys.argv[2]
with open(path) as f:
    data=json.load(f)
entries={e['key']: e for e in data.get('entries', [])}
keys=['FERRUM_KV_MAX_BLOCKS','FERRUM_KV_CAPACITY','FERRUM_PAGED_MAX_SEQS','FERRUM_MAX_BATCHED_TOKENS','FERRUM_ACTIVE_DECODE_PREFILL_CHUNK']
selected={k: entries.get(k) for k in keys if k in entries}
with open(out,'w') as f:
    json.dump(selected, f, indent=2)
max_seqs=entries.get('FERRUM_PAGED_MAX_SEQS',{}).get('effective_value')
if max_seqs != '32':
    raise SystemExit(f'FERRUM_PAGED_MAX_SEQS expected 32, got {max_seqs!r}')
PY

python3 - <<'PY' "$PORT" "$ART/correctness/smoke/stream_raw.sse" "$ART/correctness/smoke/stream_summary.json"
import json, sys, urllib.request, time
port=sys.argv[1]
raw_path=sys.argv[2]
summary_path=sys.argv[3]
payload={
  'model':'gemma3:27b-gptq',
  'messages':[{'role':'user','content':'Answer with only the digit 5.'}],
  'temperature':0,
  'max_tokens':16,
  'stream':True,
  'stream_options':{'include_usage':True},
}
req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
chunks=[]
content=[]
usage=None
done_count=0
json_errors=[]
start=time.time()
with urllib.request.urlopen(req, timeout=180) as resp, open(raw_path,'wb') as raw:
    for line in resp:
        raw.write(line)
        text=line.decode('utf-8','replace').strip()
        if not text or not text.startswith('data:'):
            continue
        data=text[5:].strip()
        if data == '[DONE]':
            done_count += 1
            continue
        try:
            obj=json.loads(data)
            chunks.append(obj)
            if obj.get('usage'):
                usage=obj['usage']
            for choice in obj.get('choices',[]):
                delta=choice.get('delta') or {}
                if isinstance(delta.get('content'), str):
                    content.append(delta['content'])
        except Exception as e:
            json_errors.append({'line':text[:500], 'error':repr(e)})
summary={
  'elapsed_s': time.time()-start,
  'done_count': done_count,
  'json_error_count': len(json_errors),
  'json_errors': json_errors[:3],
  'chunk_count': len(chunks),
  'content': ''.join(content),
  'usage': usage,
}
with open(summary_path,'w') as f:
    json.dump(summary, f, indent=2)
if done_count != 1 or json_errors or not ''.join(content) or not usage:
    raise SystemExit(f'stream smoke failed: {summary}')
PY

sleep 2
if grep -Ei 'out of memory|oom|panic|cuda_error_out_of_memory' "$SERVE_LOG" > "$ART/correctness/log_blocker_scan.txt"; then
  echo 'blocker scan hit' >&2
  cat "$ART/correctness/log_blocker_scan.txt" >&2
  exit 1
fi
nvidia-smi > "$ART/env/nvidia_smi_after.txt"
date -u +%FT%TZ > "$ART/env/end_utc.txt"
cat > "$ART/summary.json" <<JSON
{
  "pass_line": "W2 DYNAMIC KV C32 SERVE SMOKE PASS: $ART",
  "artifact_dir": "$ART",
  "git_head": "$(cat "$ART/env/git_head.txt")",
  "build_seconds": $(cat "$ART/build/build_seconds.txt"),
  "binary_sha256": "$(cut -d' ' -f1 "$ART/env/ferrum.sha256")"
}
JSON
echo "W2 DYNAMIC KV C32 SERVE SMOKE PASS: $ART"
