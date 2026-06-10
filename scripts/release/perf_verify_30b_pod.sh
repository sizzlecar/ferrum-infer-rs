#!/usr/bin/env bash
# End-to-end perf proof for the CUDA GPTQ-MoE fast-path fix: after a fresh
# CLI rebuild, `ferrum run` on Qwen3-30B-A3B-GPTQ-Int4 with NO explicit env
# must hit the vLLM-Marlin fast path (~59 tok/s on a 4090, not ~9.7) and stay
# coherent. Writes out/perf_verify.json. Run detached on a CUDA pod.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
M="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
P="用中文简单介绍一下你自己，再讲一个关于程序员的笑话。"
mkdir -p out

# wait for the rebuild kicked off alongside this script
for _ in $(seq 1 240); do
  grep -q "Finished" /root/cli_rebuild.log 2>/dev/null && break
  grep -q "error\[" /root/cli_rebuild.log 2>/dev/null && { echo '{"error":"build failed"}' > out/perf_verify.json; exit 1; }
  sleep 10
done

run() { # $1=label  $2..=env assignments
  local label="$1"; shift
  printf "%s\n" "$P" | env "$@" timeout 240 ./target/release/ferrum run "$M" --max-tokens 96 \
    > "out/perf_${label}.out" 2> "out/perf_${label}.err"
  local toks; toks=$(grep -oE "[0-9.]+ tok/s" "out/perf_${label}.err" | tail -1)
  echo "$toks"
}

DEF=$(run default)
EXP=$(run explicit FERRUM_VLLM_MOE=1 FERRUM_MOE_DEVICE_ROUTE=1)
printf '{"default_tok_s":"%s","explicit_tok_s":"%s","binary":"%s"}\n' \
  "$DEF" "$EXP" "$(git rev-parse --short HEAD)" > out/perf_verify.json
echo "PERF VERIFY DONE: default=$DEF explicit=$EXP"
