#!/usr/bin/env bash
set -euo pipefail

# Build and run the direct vLLM FlashAttention-2 FFI probe.
#
# Defaults are set for the M3 Vast pod. Override VLLM_PYTHON, FA_SRC_DIR,
# FA2_ITERS, or OUT_BIN if needed.

VLLM_PYTHON="${VLLM_PYTHON:-/workspace/vllm-venv/bin/python}"
FA_SRC_DIR="${FA_SRC_DIR:-/workspace/vllm-flash-attention-f5bc33c}"
FA_GIT_URL="${FA_GIT_URL:-https://github.com/vllm-project/flash-attention.git}"
FA_GIT_REV="${FA_GIT_REV:-f5bc33cfc02c744d24a2e9d50e6db656de40611c}"
SRC="${SRC:-scripts/microbenches/fa2_direct_ffi_probe.cpp}"
OUT_BIN="${OUT_BIN:-/tmp/fa2_direct_ffi_probe}"
CXX="${CXX:-g++}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "VLLM_PYTHON is not executable: $VLLM_PYTHON" >&2
  exit 1
fi

if [[ ! -d "$FA_SRC_DIR/.git" ]]; then
  echo "[microbench] cloning vllm flash-attention source to $FA_SRC_DIR"
  git clone --filter=blob:none "$FA_GIT_URL" "$FA_SRC_DIR"
fi
git -C "$FA_SRC_DIR" checkout -q "$FA_GIT_REV"

PY_INFO="$("$VLLM_PYTHON" - <<'PY'
from pathlib import Path
import importlib.util
import shlex
import torch
from torch.utils.cpp_extension import include_paths

spec = importlib.util.find_spec("vllm.vllm_flash_attn._vllm_fa2_C")
if spec is None or spec.origin is None:
    raise SystemExit("could not locate vllm.vllm_flash_attn._vllm_fa2_C")

print("FA2_SO=" + shlex.quote(spec.origin))
print("FA2_DIR=" + shlex.quote(str(Path(spec.origin).parent)))
print("TORCH_LIB=" + shlex.quote(str(Path(torch.__file__).parent / "lib")))
print("TORCH_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
print("TORCH_INCLUDE_FLAGS=" + shlex.quote(" ".join("-I" + p for p in include_paths())))
PY
)"
eval "$PY_INFO"

echo "[microbench] compiling $SRC"
# shellcheck disable=SC2086
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT_BIN" \
  -D_GLIBCXX_USE_CXX11_ABI="$TORCH_CXX11_ABI" \
  -I"$FA_SRC_DIR/csrc/flash_attn/src" \
  -I/usr/local/cuda/include \
  $TORCH_INCLUDE_FLAGS \
  "$FA2_SO" \
  -L"$TORCH_LIB" \
  -L/usr/local/cuda/lib64 \
  -Wl,-rpath,"$TORCH_LIB" \
  -Wl,-rpath,"$FA2_DIR" \
  -Wl,-rpath,/usr/local/cuda/lib64 \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart -ldl -pthread

echo "[microbench] running $OUT_BIN"
LD_LIBRARY_PATH="$TORCH_LIB:$FA2_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
  "$OUT_BIN"
