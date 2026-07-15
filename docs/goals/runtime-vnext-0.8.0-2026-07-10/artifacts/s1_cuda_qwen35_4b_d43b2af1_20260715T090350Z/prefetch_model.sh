#!/usr/bin/env bash
set -uo pipefail

artifact=/workspace/artifacts/s1_cuda_qwen35_4b_d43b2af1_20260715T090350Z
log="$artifact/model-prefetch.log"
export HF_HOME=/workspace/hf-cache
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$artifact" "$HF_HOME/hub"
{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  echo 'model_id=Qwen/Qwen3.5-4B'
  echo 'authentication=anonymous'
  echo "HF_HOME=$HF_HOME"
} >"$log"

set +e
python3 -m pip install -q --upgrade huggingface_hub hf_xet >>"$log" 2>&1 && \
  python3 - <<'PY' >>"$log" 2>&1
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3.5-4B",
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
)
print(f"snapshot_path={path}", flush=True)
PY
status=$?
set -e

date -u +finished_at=%Y-%m-%dT%H:%M:%SZ >>"$log"
printf '%s\n' "$status" >"$artifact/model-prefetch.exit"
exit "$status"
