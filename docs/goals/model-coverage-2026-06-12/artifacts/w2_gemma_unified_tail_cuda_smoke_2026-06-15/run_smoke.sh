#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/ferrum-infer-rs
OUT=/workspace/w2_gemma_unified_tail_cuda_smoke_2026-06-15
FERRUM="$REPO/target/release/ferrum"
MODEL=gemma3:27b-gptq
PORT=8491
cd "$REPO"
source /root/.cargo/env
export PATH="/root/.cargo/bin:${PATH}"
export CUDA_COMPUTE_CAP=89
mkdir -p "$OUT/build" "$OUT/run" "$OUT/serve" "$OUT/remote"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt"
cat > "$OUT/build/cargo_build.command.txt" <<TXT
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
TXT
set +e
timeout 1500 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source > "$OUT/build/cargo_build.stdout" 2> "$OUT/build/cargo_build.stderr"
BUILD_RC=$?
set -e
echo "$BUILD_RC" > "$OUT/build/cargo_build.rc"
if [ "$BUILD_RC" -ne 0 ]; then
  echo FAIL_BUILD > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit "$BUILD_RC"
fi
sha256sum "$FERRUM" > "$OUT/build/ferrum.sha256"
"$FERRUM" --version > "$OUT/build/ferrum.version" 2>&1 || true

cat > "$OUT/run/ferrum_run.command.json" <<JSON
{"cmd":["timeout","1200","target/release/ferrum","run","$MODEL","--backend","cuda","--prompt","What is 2+3? Answer with just the number.","--max-tokens","64","--temperature","0","--kv-capacity","2560","--max-num-seqs","2","--output-format","jsonl","--effective-config-json","$OUT/run/run_effective_config.json","--decision-trace-jsonl","$OUT/run/run_decision_trace.jsonl"]}
JSON
set +e
timeout 1200 "$FERRUM" run "$MODEL" \
  --backend cuda \
  --prompt "What is 2+3? Answer with just the number." \
  --max-tokens 64 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  --effective-config-json "$OUT/run/run_effective_config.json" \
  --decision-trace-jsonl "$OUT/run/run_decision_trace.jsonl" \
  > "$OUT/run/ferrum_run.stdout" 2> "$OUT/run/ferrum_run.stderr"
RUN_RC=$?
set -e
echo "$RUN_RC" > "$OUT/run/ferrum_run.rc"
python3 - "$RUN_RC" "$OUT/run/ferrum_run.stdout" "$OUT/run/ferrum_run_validation.json" <<PY
import json, pathlib, sys
rc = int(sys.argv[1])
rows = []
for line in pathlib.Path(sys.argv[2]).read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        rows.append({"event":"decode_error","raw":line})
assistant = next((r for r in rows if r.get("event") == "assistant"), {})
content = assistant.get("content") or ""
bad = any(x in content for x in ["<unk>", "[PAD", "�"])
ok = rc == 0 and "5" in content and not bad and (assistant.get("n_tokens") or 0) > 0
out = {"status":"pass" if ok else "fail", "run_rc":rc, "assistant":assistant, "row_count":len(rows), "bad_output":bad}
pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if ok else 1)
PY

pkill -f "target/release/ferrum serve" 2>/dev/null || true
sleep 2
cat > "$OUT/serve/serve_smoke.command.json" <<JSON
{"env":{"FERRUM_BIN":"target/release/ferrum","SMOKE_REQ_TIMEOUT":"240"},"cmd":["bash","scripts/model_coverage_smoke.sh","$MODEL","--port","$PORT","--kv-capacity","2560","--max-seqs","2"]}
JSON
set +e
FERRUM_BIN=target/release/ferrum SMOKE_REQ_TIMEOUT=240 timeout 1800 bash scripts/model_coverage_smoke.sh "$MODEL" --port "$PORT" --kv-capacity 2560 --max-seqs 2 > "$OUT/serve/serve_smoke.stdout" 2> "$OUT/serve/serve_smoke.stderr"
SMOKE_RC=$?
set -e
echo "$SMOKE_RC" > "$OUT/serve/serve_smoke.rc"
cp "/tmp/ferrum_w1_smoke_${PORT}.log" "$OUT/serve/serve_smoke.server.log" 2>/dev/null || true
python3 - "$RUN_RC" "$SMOKE_RC" "$OUT/serve/serve_smoke.stdout" "$OUT/serve/serve_smoke.server.log" "$OUT/serve/serve_smoke_validation.json" <<PY
import json, pathlib, sys
run_rc = int(sys.argv[1])
smoke_rc = int(sys.argv[2])
stdout = pathlib.Path(sys.argv[3]).read_text(errors="replace")
server = pathlib.Path(sys.argv[4]).read_text(errors="replace") if pathlib.Path(sys.argv[4]).exists() else ""
pass_line = "FERRUM W1 SMOKE PASS: gemma3:27b-gptq" in stdout
varlen_true = "varlen_unified=true" in server
bad_log = any(x in server for x in ["panic", "<unk>", "[PAD", "CUDA_ERROR"])
ok = run_rc == 0 and smoke_rc == 0 and pass_line and varlen_true and not bad_log
out = {"status":"pass" if ok else "fail", "run_rc":run_rc, "serve_smoke_rc":smoke_rc, "pass_line":pass_line, "server_varlen_unified_true":varlen_true, "bad_log_marker":bad_log}
pathlib.Path(sys.argv[5]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if ok else 1)
PY
nvidia-smi > "$OUT/remote/nvidia_smi_after.txt" || true
echo PASS > "$OUT/run.status"
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
echo "W2 GEMMA UNIFIED TAIL CUDA SMOKE PASS: $OUT"
