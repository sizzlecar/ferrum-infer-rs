#!/usr/bin/env bash
set -euo pipefail

# Build the out-of-tree C ABI shim used by FERRUM_FA2_DIRECT_FFI=1.
#
# This legacy diagnostic is not used by product or release builds. VLLM_PYTHON
# and FA_SRC_DIR must be explicit when the vLLM virtualenv / FlashAttention
# checkout lives outside the defaults below.

VLLM_PYTHON="${VLLM_PYTHON:-/workspace/vllm-venv/bin/python}"
FA_SRC_DIR="${FA_SRC_DIR:-}"
FA_GIT_URL="${FA_GIT_URL:-https://github.com/vllm-project/flash-attention.git}"
FA_GIT_REV="${FA_GIT_REV:-f5bc33cfc02c744d24a2e9d50e6db656de40611c}"
SRC="${SRC:-scripts/microbenches/fa2_ferrum_shim.cpp}"
OUT_SO="${OUT_SO:-/workspace/libferrum_fa2_shim.so}"
CXX="${CXX:-g++}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "VLLM_PYTHON is not executable: $VLLM_PYTHON" >&2
  exit 1
fi

if [[ -z "$FA_SRC_DIR" ]]; then
  echo "FA_SRC_DIR must point at a FlashAttention source checkout for this legacy diagnostic" >&2
  exit 1
fi

if [[ ! -d "$FA_SRC_DIR/.git" ]]; then
  echo "[fa2-shim] cloning vllm flash-attention source to $FA_SRC_DIR"
  git clone --filter=blob:none "$FA_GIT_URL" "$FA_SRC_DIR"
fi
git -C "$FA_SRC_DIR" checkout -q "$FA_GIT_REV"

PY_INFO="$("$VLLM_PYTHON" - <<'PY'
from pathlib import Path
import importlib.util
import shlex
import sysconfig
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
libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""
libs = sysconfig.get_config_var("LIBS") or ""
syslibs = sysconfig.get_config_var("SYSLIBS") or ""
py_flags = []
if libdir:
    py_flags.append("-L" + libdir)
if ldlibrary.startswith("lib") and ldlibrary.endswith(".so"):
    py_flags.append("-l:" + ldlibrary)
elif ldlibrary:
    py_flags.append("-l" + ldlibrary.removeprefix("lib").removesuffix(".so"))
py_flags.extend(libs.split())
py_flags.extend(syslibs.split())
print("PY_LDFLAGS=" + shlex.quote(" ".join(py_flags)))
PY
)"
eval "$PY_INFO"

mkdir -p "$(dirname "$OUT_SO")"
echo "[fa2-shim] compiling $OUT_SO"
# shellcheck disable=SC2086
"$CXX" -O3 -std=c++17 -shared -fPIC "$SRC" -o "$OUT_SO" \
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
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart -ldl -pthread \
  $PY_LDFLAGS

echo "[fa2-shim] wrote $OUT_SO"
ldd "$OUT_SO" | sed 's/^/[fa2-shim] /'
