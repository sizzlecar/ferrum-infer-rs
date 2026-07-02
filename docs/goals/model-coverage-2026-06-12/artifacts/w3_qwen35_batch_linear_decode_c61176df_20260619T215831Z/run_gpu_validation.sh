#!/usr/bin/env bash
set -euo pipefail
ART="/workspace/artifacts/w3_qwen35_batch_linear_decode_c61176df_20260619T215831Z"
REPO="/workspace/ferrum-clean"
cd "${REPO}"
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi
export PATH="/root/.cargo/bin:/usr/local/cuda/bin:${PATH}"
export HF_HOME=/workspace/hf-cache
export CARGO_TARGET_DIR=/workspace/ferrum-w3-qwen35-ac98207/target
export RUST_BACKTRACE=1
export CUDA_VISIBLE_DEVICES=0
export NO_COLOR=1
mkdir -p "${ART}"
while IFS="=" read -r name _; do
  case "${name}" in
    FERRUM_*) unset "${name}" ;;
  esac
done < <(env)
{
  echo "wrapper_started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=${REPO}"
  echo "artifact=${ART}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_status_short_begin"
  git status --short --branch
  echo "git_status_short_end"
  echo "gpu_begin"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader
  echo "gpu_end"
  rustc --version
  cargo --version
  nvcc --version
} | tee "${ART}/wrapper_environment.log"
LINEAR_CMD=(cargo test -p ferrum-kernels --features cuda --test linear_attention_cuda_eq -- --nocapture)
printf "%q " "${LINEAR_CMD[@]}" > "${ART}/cuda_linear_attention_test.command.txt"
printf "\n" >> "${ART}/cuda_linear_attention_test.command.txt"
"${LINEAR_CMD[@]}" 2>&1 | tee "${ART}/cuda_linear_attention_test.log"
GDR_CMD=(cargo test -p ferrum-kernels --features cuda --test gated_delta_rule_cuda_eq -- --nocapture)
printf "%q " "${GDR_CMD[@]}" > "${ART}/cuda_gated_delta_rule_test.command.txt"
printf "\n" >> "${ART}/cuda_gated_delta_rule_test.command.txt"
"${GDR_CMD[@]}" 2>&1 | tee "${ART}/cuda_gated_delta_rule_test.log"
ART="${ART}" bash "${ART}/run_smoke.sh" 2>&1 | tee "${ART}/wrapper_smoke.log"
