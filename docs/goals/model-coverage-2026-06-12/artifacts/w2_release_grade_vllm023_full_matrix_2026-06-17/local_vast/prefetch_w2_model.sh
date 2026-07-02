#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/workspace/w2_model_prefetch_2026-06-17_a3f0ab9f}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
HF_REPO="${HF_REPO:-circulus/gemma-3-27b-it-gptq}"
HF_PYTHON="${HF_PYTHON:-/workspace/vllm-venv-0_23_0/bin/python}"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/prefetch.log") 2>&1

echo "[w2-prefetch] start $(date -u +%FT%TZ)"
echo "[w2-prefetch] hf_home=$HF_HOME repo=$HF_REPO python=$HF_PYTHON"

HF_HOME="$HF_HOME" HF_REPO="$HF_REPO" HF_XET_HIGH_PERFORMANCE=1 "$HF_PYTHON" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["HF_REPO"]
path = snapshot_download(repo_id=repo, resume_download=True)
print(path)
model_path = Path(path)
missing = [
    f"model-{i:05d}-of-00005.safetensors"
    for i in range(1, 6)
    if not (model_path / f"model-{i:05d}-of-00005.safetensors").exists()
]
if missing:
    raise SystemExit(f"missing shards after download: {missing}")
PY

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
printf '%s\n' "$MODEL_PATH" > "$OUT/model_path.txt"
ls -lh "$MODEL_PATH"/model-*.safetensors > "$OUT/model_shards.txt"
du -sh "$HF_HOME" > "$OUT/hf_home_du.txt" 2>&1 || true
date -u +%FT%TZ > "$OUT/end_utc.txt"
echo "[w2-prefetch] done $(date -u +%FT%TZ)"
