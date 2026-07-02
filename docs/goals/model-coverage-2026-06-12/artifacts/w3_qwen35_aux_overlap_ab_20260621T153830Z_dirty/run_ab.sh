#!/usr/bin/env bash
set -euo pipefail
ART="$(cat /workspace/artifacts/latest_w3_aux_overlap_artifact.txt)"
cd /workspace/ferrum-w3-clean-b6
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
DATASET=/workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl
run_case() {
  local name="$1"
  local aux_value="$2"
  local port="$3"
  local out="$ART/bench_${name}"
  mkdir -p "$out"
  echo "$aux_value" > "$out/aux_context_overlap.env"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$out/gpu_before_serve.csv"
  cat > "$out/serve.command.txt" <<EOF
FERRUM_CUDA_AUX_CONTEXT_OVERLAP=$aux_value HF_HOME=/workspace/hf-cache target/release/ferrum serve $MODEL --backend cuda --host 127.0.0.1 --port $port --gpu-devices 0 --gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 8192 --kv-capacity 2048 --scheduler-prefill-first-until-active 32 --scheduler-active-decode-prefill-chunk 8192 --greedy-argmax --effective-config-json $out/effective_config.json --decision-trace-jsonl $out/decision_trace.jsonl
EOF
  FERRUM_CUDA_AUX_CONTEXT_OVERLAP="$aux_value" HF_HOME=/workspace/hf-cache \
    target/release/ferrum serve "$MODEL" \
      --backend cuda --host 127.0.0.1 --port "$port" --gpu-devices 0 \
      --gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 32 \
      --max-num-batched-tokens 8192 --kv-capacity 2048 \
      --scheduler-prefill-first-until-active 32 \
      --scheduler-active-decode-prefill-chunk 8192 --greedy-argmax \
      --effective-config-json "$out/effective_config.json" \
      --decision-trace-jsonl "$out/decision_trace.jsonl" \
      > "$out/serve.log" 2>&1 &
  local spid=$!
  echo "$spid" > "$out/server.pid"
  for i in $(seq 1 240); do
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "server exited before health" > "$out/server_early_exit.txt"
      wait "$spid" || true
      return 20
    fi
    if curl -fsS "http://127.0.0.1:$port/health" > "$out/health.json" 2> "$out/health.err"; then
      break
    fi
    sleep 2
  done
  if [ ! -s "$out/health.json" ]; then
    echo "server health timeout" > "$out/server_health_timeout.txt"
    kill "$spid" 2>/dev/null || true
    wait "$spid" 2>/dev/null || true
    return 21
  fi
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$out/gpu_before_bench.csv"
  cat > "$out/bench.command.txt" <<EOF
HF_HOME=/workspace/hf-cache target/release/ferrum bench-serve --base-url http://127.0.0.1:$port --model $MODEL --tokenizer $MODEL --dataset sharegpt --sharegpt-path $DATASET --num-prompts 100 --warmup-requests 10 --random-output-len 128 --concurrency 32 --n-repeats 1 --seed 9271 --fail-on-error --output json --out $out/bench_c32_100x1.json
EOF
  HF_HOME=/workspace/hf-cache target/release/ferrum bench-serve \
    --base-url "http://127.0.0.1:$port" --model "$MODEL" --tokenizer "$MODEL" \
    --dataset sharegpt --sharegpt-path "$DATASET" --num-prompts 100 \
    --warmup-requests 10 --random-output-len 128 --concurrency 32 \
    --n-repeats 1 --seed 9271 --fail-on-error --output json \
    --out "$out/bench_c32_100x1.json" \
    > "$out/bench.stdout" 2> "$out/bench.stderr"
  echo $? > "$out/bench.exit"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$out/gpu_after_bench.csv"
  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
}
run_case off 0 58420
run_case on 1 58421
python3 - <<'PY'
import json, pathlib
art = pathlib.Path('/workspace/artifacts/latest_w3_aux_overlap_artifact.txt').read_text().strip()
root = pathlib.Path(art)
def load_case(name):
    p = root / f'bench_{name}' / 'bench_c32_100x1.json'
    data = json.loads(p.read_text())
    return {
        'path': str(p),
        'aux_context_overlap': (root / f'bench_{name}' / 'aux_context_overlap.env').read_text().strip(),
        'output_tps_mean': data['output_throughput_tps']['mean'],
        'completed_per_run': data.get('completed_per_run'),
        'errored_per_run': data.get('errored_per_run'),
        'bad_output_per_run': data.get('bad_output_per_run'),
        'missing_done_per_run': data.get('missing_done_per_run'),
        'duplicate_done_per_run': data.get('duplicate_done_per_run'),
        'zero_output_tokens_per_run': data.get('zero_output_tokens_per_run'),
        'output_token_count_source': data.get('output_token_count_source'),
    }
off = load_case('off')
on = load_case('on')
summary = {
    'schema_version': 1,
    'status': 'pass',
    'artifact_dir': art,
    'git_head': (root / 'git_head.txt').read_text().strip(),
    'dirty': True,
    'binary_sha256': (root / 'ferrum.sha256').read_text().split()[0] if (root / 'ferrum.sha256').exists() else None,
    'off': off,
    'on': on,
    'ratio_on_over_off': on['output_tps_mean'] / off['output_tps_mean'],
    'vllm_c32_mean': 1708.52785,
    'vllm_80pct_target_mean': 1366.82228,
    'on_vs_vllm_mean_ratio': on['output_tps_mean'] / 1708.52785,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print('QWEN35_AUX_OVERLAP_AB_DIAG_PASS', art)
PY
