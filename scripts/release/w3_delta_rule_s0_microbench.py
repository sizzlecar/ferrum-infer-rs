#!/usr/bin/env python3
"""W3-S0 delta-rule microbench harness.

Self-test mode validates the deterministic Python reference and chunking logic.
CUDA mode compiles a minimal native CUDA kernel with nvcc and compares it
against the same reference. Only CUDA mode is evidence for the W3-S0 native
CUDA/PTX microbench contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_SELFTEST = "W3 DELTA RULE S0 SELFTEST PASS"
PASS_CUDA = "W3 DELTA RULE S0 MICROBENCH PASS"
MANIFEST_NAME = "w3_delta_rule_s0_microbench_manifest.json"
INPUT_DISTRIBUTION = {
    "generator": "lcg_u32_centered_uniform",
    "q_range": [-0.25, 0.25],
    "k_range": [-0.20, 0.20],
    "v_range": [-0.30, 0.30],
    "beta_range": [0.50, 0.75],
}


class MicrobenchError(Exception):
    pass


@dataclass(frozen=True)
class Shape:
    batch: int
    heads: int
    tokens: int
    key_dim: int
    value_dim: int

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise MicrobenchError(f"{name} must be positive, got {value}")
        if self.key_dim * self.value_dim > 4096:
            raise MicrobenchError("key_dim * value_dim must be <= 4096 for the minimal CUDA kernel")


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def run_command(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def best_effort_command(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = run_command(cmd)
    except OSError as exc:
        return {
            "command": cmd,
            "available": False,
            "error": str(exc),
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    return {
        "command": cmd,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = run_command(["git", *args])
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()


def git_summary() -> dict[str, Any]:
    status = git_output(["status", "--short", "--untracked-files=no"], default="")
    tracked_lines = [line for line in status.splitlines() if line.strip()]
    untracked = git_output(["ls-files", "--others", "--exclude-standard"], default="")
    untracked_lines = [line for line in untracked.splitlines() if line.strip()]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked_lines or untracked_lines),
        "tracked_status_short": tracked_lines,
        "untracked_count": len(untracked_lines),
        "untracked_sample": untracked_lines[:20],
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lcg(seed: int) -> Any:
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state


def rand_float(gen: Any, scale: float) -> float:
    raw = next(gen)
    centered = (raw / 0xFFFFFFFF) * 2.0 - 1.0
    return centered * scale


def make_inputs(shape: Shape, seed: int) -> dict[str, list[float]]:
    gen = lcg(seed)
    q_count = shape.batch * shape.heads * shape.tokens * shape.key_dim
    k_count = q_count
    v_count = shape.batch * shape.heads * shape.tokens * shape.value_dim
    beta_count = shape.batch * shape.heads * shape.tokens
    return {
        "q": [rand_float(gen, 0.25) for _ in range(q_count)],
        "k": [rand_float(gen, 0.20) for _ in range(k_count)],
        "v": [rand_float(gen, 0.30) for _ in range(v_count)],
        "beta": [0.5 + abs(rand_float(gen, 0.25)) for _ in range(beta_count)],
    }


def qk_index(shape: Shape, b: int, h: int, t: int, kk: int) -> int:
    return (((b * shape.heads + h) * shape.tokens + t) * shape.key_dim) + kk


def v_index(shape: Shape, b: int, h: int, t: int, vv: int) -> int:
    return (((b * shape.heads + h) * shape.tokens + t) * shape.value_dim) + vv


def beta_index(shape: Shape, b: int, h: int, t: int) -> int:
    return (b * shape.heads + h) * shape.tokens + t


def delta_rule_reference(
    shape: Shape,
    inputs: dict[str, list[float]],
    *,
    chunk_size: int | None = None,
) -> list[float]:
    out = [0.0] * (shape.batch * shape.heads * shape.tokens * shape.value_dim)
    for b in range(shape.batch):
        for h in range(shape.heads):
            state = [0.0] * (shape.key_dim * shape.value_dim)
            chunk = chunk_size or shape.tokens
            for chunk_start in range(0, shape.tokens, chunk):
                chunk_end = min(shape.tokens, chunk_start + chunk)
                for t in range(chunk_start, chunk_end):
                    beta = inputs["beta"][beta_index(shape, b, h, t)]
                    for vv in range(shape.value_dim):
                        pred = 0.0
                        for kk in range(shape.key_dim):
                            pred += inputs["k"][qk_index(shape, b, h, t, kk)] * state[kk * shape.value_dim + vv]
                        delta = beta * (inputs["v"][v_index(shape, b, h, t, vv)] - pred)
                        for kk in range(shape.key_dim):
                            state[kk * shape.value_dim + vv] += (
                                inputs["k"][qk_index(shape, b, h, t, kk)] * delta
                            )
                    for vv in range(shape.value_dim):
                        acc = 0.0
                        for kk in range(shape.key_dim):
                            acc += inputs["q"][qk_index(shape, b, h, t, kk)] * state[kk * shape.value_dim + vv]
                        out[v_index(shape, b, h, t, vv)] = acc
    return out


def error_stats(expected: list[float], actual: list[float]) -> dict[str, float]:
    if len(expected) != len(actual):
        raise MicrobenchError(f"output length mismatch: expected {len(expected)}, got {len(actual)}")
    max_abs = 0.0
    max_rel = 0.0
    rmse_acc = 0.0
    for exp, got in zip(expected, actual):
        diff = abs(exp - got)
        max_abs = max(max_abs, diff)
        denom = max(abs(exp), 1e-12)
        max_rel = max(max_rel, diff / denom)
        rmse_acc += diff * diff
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rmse": math.sqrt(rmse_acc / max(len(expected), 1)),
    }


CUDA_SOURCE = r"""
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error " << cudaGetErrorString(err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
      return 10;                                                            \
    }                                                                       \
  } while (0)

__global__ void delta_rule_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* beta,
    float* out,
    int B,
    int H,
    int T,
    int K,
    int V) {
  int bh = blockIdx.x;
  if (threadIdx.x != 0) return;
  int b = bh / H;
  int h = bh % H;
  float state[4096];
  for (int i = 0; i < K * V; ++i) state[i] = 0.0f;
  for (int t = 0; t < T; ++t) {
    float bt = beta[(b * H + h) * T + t];
    for (int vv = 0; vv < V; ++vv) {
      float pred = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        int qki = (((b * H + h) * T + t) * K) + kk;
        pred += k[qki] * state[kk * V + vv];
      }
      int vi = (((b * H + h) * T + t) * V) + vv;
      float delta = bt * (v[vi] - pred);
      for (int kk = 0; kk < K; ++kk) {
        int qki = (((b * H + h) * T + t) * K) + kk;
        state[kk * V + vv] += k[qki] * delta;
      }
    }
    for (int vv = 0; vv < V; ++vv) {
      float acc = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        int qki = (((b * H + h) * T + t) * K) + kk;
        acc += q[qki] * state[kk * V + vv];
      }
      int oi = (((b * H + h) * T + t) * V) + vv;
      out[oi] = acc;
    }
  }
}

static bool read_floats(const char* path, std::vector<float>& data) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  return static_cast<size_t>(in.gcount()) == data.size() * sizeof(float);
}

static bool write_floats(const char* path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  return out.good();
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: " << argv[0] << " <input.bin> <output.bin> B H T K V\n";
    return 2;
  }
  const char* input_path = argv[1];
  const char* output_path = argv[2];
  int B = std::atoi(argv[3]);
  int H = std::atoi(argv[4]);
  int T = std::atoi(argv[5]);
  int K = std::atoi(argv[6]);
  int V = std::atoi(argv[7]);
  size_t qk_count = static_cast<size_t>(B) * H * T * K;
  size_t value_count = static_cast<size_t>(B) * H * T * V;
  size_t beta_count = static_cast<size_t>(B) * H * T;
  std::vector<float> input(qk_count * 2 + value_count + beta_count);
  if (!read_floats(input_path, input)) {
    std::cerr << "failed to read input\n";
    return 3;
  }
  const float* q_host = input.data();
  const float* k_host = q_host + qk_count;
  const float* v_host = k_host + qk_count;
  const float* beta_host = v_host + value_count;
  std::vector<float> output(value_count, 0.0f);

  float *q, *k, *v, *beta, *out;
  CUDA_CHECK(cudaMalloc(&q, qk_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&k, qk_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&v, value_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&beta, beta_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, value_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(q, q_host, qk_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(k, k_host, qk_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v, v_host, value_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(beta, beta_host, beta_count * sizeof(float), cudaMemcpyHostToDevice));
  delta_rule_kernel<<<B * H, 1>>>(q, k, v, beta, out, B, H, T, K, V);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), out, value_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(q));
  CUDA_CHECK(cudaFree(k));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(beta));
  CUDA_CHECK(cudaFree(out));
  if (!write_floats(output_path, output)) {
    std::cerr << "failed to write output\n";
    return 4;
  }
  return 0;
}
"""


def write_float_file(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def read_float_file(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise MicrobenchError(f"{path} size is not a float32 multiple")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def write_process_logs(prefix: Path, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout_path = prefix.with_suffix(".stdout.txt")
    stderr_path = prefix.with_suffix(".stderr.txt")
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "returncode": proc.returncode,
        "stdout_path": stdout_path.name,
        "stderr_path": stderr_path.name,
    }


def cuda_metadata(nvcc: str) -> dict[str, Any]:
    return {
        "nvcc_path": nvcc,
        "nvcc_version": best_effort_command([nvcc, "--version"]),
        "nvidia_smi": best_effort_command(["nvidia-smi", "-q"]),
        "compute_cap": best_effort_command(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def resolve_cuda_arch(requested: str) -> str:
    if requested != "auto":
        return requested
    proc = run_command(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
    if proc.returncode == 0:
        first = proc.stdout.splitlines()[0].strip() if proc.stdout.splitlines() else ""
        digits = "".join(ch for ch in first if ch.isdigit())
        if digits:
            return f"sm_{digits}"
    # RTX 4090 fallback. The manifest records the requested value and metadata.
    return "sm_89"


def run_cuda(
    shape: Shape,
    inputs: dict[str, list[float]],
    out_dir: Path,
    *,
    cuda_arch: str,
) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise MicrobenchError("nvcc not found; cannot run native CUDA microbench")
    resolved_arch = resolve_cuda_arch(cuda_arch)
    build_dir = out_dir / "cuda_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "delta_rule_s0.cu"
    binary = build_dir / "delta_rule_s0"
    input_bin = build_dir / "input.bin"
    output_bin = build_dir / "output.bin"
    source.write_text(CUDA_SOURCE, encoding="utf-8")
    packed_inputs = inputs["q"] + inputs["k"] + inputs["v"] + inputs["beta"]
    write_float_file(input_bin, packed_inputs)
    compile_cmd = [
        nvcc,
        "-O2",
        "--generate-line-info",
        f"-arch={resolved_arch}",
        str(source),
        "-o",
        str(binary),
    ]
    compile_proc = run_command(compile_cmd)
    compile_logs = write_process_logs(build_dir / "nvcc_compile", compile_proc)
    if compile_proc.returncode != 0:
        raise MicrobenchError(f"nvcc failed rc={compile_proc.returncode}\n{compile_proc.stderr}")
    run_cmd = [
        str(binary),
        str(input_bin),
        str(output_bin),
        str(shape.batch),
        str(shape.heads),
        str(shape.tokens),
        str(shape.key_dim),
        str(shape.value_dim),
    ]
    run_proc = run_command(run_cmd)
    run_logs = write_process_logs(build_dir / "cuda_run", run_proc)
    if run_proc.returncode != 0:
        raise MicrobenchError(f"cuda microbench failed rc={run_proc.returncode}\n{run_proc.stderr}")
    return {
        "nvcc": nvcc,
        "requested_arch": cuda_arch,
        "resolved_arch": resolved_arch,
        "cuda_metadata": cuda_metadata(nvcc),
        "compile_command": compile_cmd,
        "compile_logs": compile_logs,
        "run_command": run_cmd,
        "run_logs": run_logs,
        "binary_sha256": sha256(binary),
        "source": str(source),
        "input": str(input_bin),
        "output_path": str(output_bin),
        "output": read_float_file(output_bin),
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    shape = Shape(args.batch, args.heads, args.tokens, args.key_dim, args.value_dim)
    shape.validate()
    if args.chunk_size <= 0:
        raise MicrobenchError("--chunk-size must be positive")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs(shape, args.seed)
    reference = delta_rule_reference(shape, inputs)
    chunked = delta_rule_reference(shape, inputs, chunk_size=args.chunk_size)
    chunk_stats = error_stats(reference, chunked)
    cuda_stats = None
    cuda_meta = None
    if args.cuda:
        cuda_meta = run_cuda(shape, inputs, out_dir, cuda_arch=args.cuda_arch)
        cuda_stats = error_stats(reference, cuda_meta["output"])
        if cuda_stats["max_abs"] > args.atol:
            raise MicrobenchError(f"CUDA max_abs {cuda_stats['max_abs']} exceeds {args.atol}")
    if chunk_stats["max_abs"] > 0.0:
        raise MicrobenchError(f"chunked reference mismatch: {chunk_stats}")
    pass_line = f"{PASS_CUDA if args.cuda else PASS_SELFTEST}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "mode": "cuda" if args.cuda else "self-test",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "shape": shape.__dict__,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "input_distribution": INPUT_DISTRIBUTION,
        "tolerance": {"max_abs": args.atol},
        "reference": {
            "name": args.reference_name,
            "revision": args.reference_rev,
            "formula": "S_t = S_{t-1} + beta_t * k_t^T * (v_t - k_t @ S_{t-1}); o_t = q_t @ S_t",
        },
        "chunked_reference_error": chunk_stats,
        "cuda_error": cuda_stats,
        "error_stats": cuda_stats if args.cuda else chunk_stats,
        "ptx_arch": cuda_meta["resolved_arch"] if cuda_meta else None,
        "cuda_binary_sha256": cuda_meta["binary_sha256"] if cuda_meta else None,
        "cuda": {
            "nvcc_path": cuda_meta["nvcc"],
            "requested_arch": cuda_meta["requested_arch"],
            "resolved_arch": cuda_meta["resolved_arch"],
            "metadata": cuda_meta["cuda_metadata"],
            "compile_command": cuda_meta["compile_command"],
            "compile_logs": cuda_meta["compile_logs"],
            "run_command": cuda_meta["run_command"],
            "run_logs": cuda_meta["run_logs"],
            "binary_sha256": cuda_meta["binary_sha256"],
            "source": cuda_meta["source"],
            "input": cuda_meta["input"],
            "output": cuda_meta["output_path"],
        }
        if cuda_meta
        else None,
        "note": (
            "self-test is not W3-S0 CUDA evidence"
            if not args.cuda
            else "native CUDA/PTX minimal correctness evidence"
        ),
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/w3_delta_rule_s0_selftest")
    parser.add_argument("--cuda", action="store_true", help="require nvcc and run native CUDA")
    parser.add_argument("--self-test", action="store_true", help="alias for default CPU self-test mode")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=4)
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9271)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--reference-name", default="internal-python-delta-rule-reference")
    parser.add_argument("--reference-rev", default="self-test")
    parser.add_argument(
        "--cuda-arch",
        default="auto",
        help="nvcc -arch value, or auto to query nvidia-smi compute capability",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except MicrobenchError as exc:
        print(f"W3 DELTA RULE S0 MICROBENCH FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
