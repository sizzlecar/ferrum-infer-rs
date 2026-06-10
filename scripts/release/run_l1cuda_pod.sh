#!/usr/bin/env bash
# Drive lane_l1_cuda.sh 3x on a CUDA pod to collect the Gate C3/C5 artifacts:
#   - l1_cuda_cold_seconds  (run 1, includes the cold cargo build)
#   - l1_cuda_warm_seconds  (run 2, build cached)
#   - l1_cuda stability     (3 runs, all must PASS)
# Writes out/l1cuda_result.json. Designed to run detached (nohup over ssh);
# lane_l1_cuda.sh self-sources $HOME/.cargo/env so cargo is on PATH.
set -u
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/out"
mkdir -p "$OUT"

t0=$(date +%s)
bash scripts/release/lane_l1_cuda.sh "$OUT/l1cuda-cold" > "$OUT/cold.log" 2>&1
rc1=$?
cold=$(( $(date +%s) - t0 ))

t0=$(date +%s)
bash scripts/release/lane_l1_cuda.sh "$OUT/l1cuda-w1" > "$OUT/w1.log" 2>&1
rc2=$?
warm=$(( $(date +%s) - t0 ))

bash scripts/release/lane_l1_cuda.sh "$OUT/l1cuda-w2" > "$OUT/w2.log" 2>&1
rc3=$?

green=$(( (rc1 == 0) + (rc2 == 0) + (rc3 == 0) ))
printf '{"l1_cuda_cold_seconds": %d, "l1_cuda_warm_seconds": %d, "l1_cuda_runs": 3, "l1_cuda_green": %d, "rc": [%d, %d, %d]}\n' \
  "$cold" "$warm" "$green" "$rc1" "$rc2" "$rc3" > "$OUT/l1cuda_result.json"
echo "DONE cold=${cold}s warm=${warm}s green=${green}/3"
