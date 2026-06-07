#!/usr/bin/env python3
"""CUDA source gate scaffold for Llama 3.3 70B 4bit on 2x RTX 4090.

This runner intentionally refuses to PASS until the product path can execute
the typed 2-GPU layer_split request and the rest of the correctness/perf matrix
is implemented. It still writes failure artifacts for the first product command
so CUDA development can move from a concrete failing point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LANE = "g0_cuda2x4090_llama33_70b_4bit"
PASS_LINE_PREFIX = f"G0 SOURCE {LANE} PASS"
REQUIRED_GPU_DEVICES = [0, 1]
BAD_PATTERNS = [
    "panic",
    "CUDA illegal memory access",
    "NCCL error",
    "OOM",
    "KV cache overflow",
    "missing tokenizer",
    "chat template render failure",
    "CPU fallback",
    "single-GPU fallback",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def run(
    cmd: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: list[str], repo: Path) -> str:
    proc = run(["git", *args], cwd=repo, timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def capture_nvidia_smi(root: Path, label: str) -> None:
    proc = run(["nvidia-smi"], cwd=root, timeout=30)
    body = proc.stdout if proc.returncode == 0 else proc.stderr
    write_text(root / f"nvidia-smi.{label}.txt", body or f"nvidia-smi rc={proc.returncode}\n")


def query_gpu_names(repo: Path) -> list[str]:
    proc = run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        cwd=repo,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("gpu_devices") != REQUIRED_GPU_DEVICES:
        raise RuntimeError(f"config gpu_devices must be {REQUIRED_GPU_DEVICES}")
    if cfg.get("distributed_strategy") != "layer_split":
        raise RuntimeError("config distributed_strategy must be layer_split")
    model = cfg.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError("config model must be a non-empty model id/path")
    quant = str(cfg.get("quant_format", "")).lower()
    if not any(marker in quant for marker in ("gptq", "awq", "q4")):
        raise RuntimeError("config quant_format must identify a 4bit format")


def build_run_command(ferrum_bin: Path, model: str, cfg: dict[str, Any], root: Path) -> list[str]:
    cmd = [
        str(ferrum_bin),
        "run",
        model,
        "--backend",
        "cuda",
        "--gpu-devices",
        ",".join(str(device) for device in REQUIRED_GPU_DEVICES),
        "--max-tokens",
        "256",
        "--output-format",
        "jsonl",
        "--effective-config-json",
        str(root / "run.effective_config.json"),
        "--decision-trace-jsonl",
        str(root / "run.decision_trace.jsonl"),
    ]
    if cfg.get("max_model_len") is not None:
        # `ferrum run` does not yet expose --max-model-len; preserve the
        # intended value in run.command.json for the 70B lane artifact.
        pass
    return cmd


def write_metadata(root: Path, repo: Path, ferrum_bin: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    gpu_names = query_gpu_names(repo)
    metadata = {
        "schema_version": 1,
        "lane": LANE,
        "status": "running",
        "created_at": iso_now(),
        "git_sha": git_output(["rev-parse", "HEAD"], repo),
        "dirty_status": {
            "is_dirty": bool(git_output(["status", "--short"], repo)),
            "status_short": git_output(["status", "--short"], repo).splitlines(),
        },
        "binary_sha256": sha256(ferrum_bin),
        "build_features": ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2", "fa2-source"],
        "gpu_names": gpu_names,
        "requested_gpu_devices": REQUIRED_GPU_DEVICES,
        "selected_gpu_devices": REQUIRED_GPU_DEVICES,
        "model_id": cfg["model"],
        "quant_format": cfg["quant_format"],
        "distributed_strategy": "layer_split",
        "layer_split_plan": "stage0:cuda:0:layers=auto;stage1:cuda:1:layers=auto",
        "sanitized_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("FERRUM_") or key in {"CUDA_VISIBLE_DEVICES", "HF_HOME"}
        },
    }
    write_json(root / "metadata.json", metadata)
    return metadata


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pattern!r} in {label}")


def run_cli_probe(root: Path, repo: Path, ferrum_bin: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    input_text = "\n".join(
        [
            "请记住短语 ferrum-blue。只回答 OK。",
            "第一条用户消息里的 ferrum 开头短语是什么？只输出短语。",
            "/bye",
            "",
        ]
    )
    cmd = build_run_command(ferrum_bin, cfg["model"], cfg, root)
    write_json(
        root / "run.command.json",
        {
            "cmd": cmd,
            "expected_gpu_devices": REQUIRED_GPU_DEVICES,
            "expected_distributed_strategy": "layer_split",
            "config": {
                "max_model_len": cfg.get("max_model_len"),
                "kv_capacity": cfg.get("kv_capacity"),
                "max_num_seqs": cfg.get("max_num_seqs"),
                "max_num_batched_tokens": cfg.get("max_num_batched_tokens"),
            },
        },
    )
    write_text(root / "run.stdin", input_text)
    proc = run(cmd, cwd=repo, input_text=input_text, timeout=1800)
    write_text(root / "run.stdout", proc.stdout)
    write_text(root / "run.stderr", proc.stderr)
    assert_no_bad_patterns("ferrum run stdout/stderr", proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"ferrum run layer_split probe failed rc={proc.returncode}")
    return {"status": "pass", "returncode": proc.returncode}


def gate_fail(root: Path, error: str, checks: dict[str, Any]) -> int:
    write_json(
        root / "gate.json",
        {
            "schema_version": 1,
            "status": "fail",
            "lane": LANE,
            "error": error,
            "checks": checks,
            "pass_line": None,
        },
    )
    print(f"G0 CUDA LLAMA33 70B 4BIT 2X4090 GATE FAIL: {error}", file=sys.stderr)
    return 1


def gate_pass(root: Path, checks: dict[str, Any]) -> int:
    pass_line = f"{PASS_LINE_PREFIX}: {root}"
    write_json(
        root / "gate.json",
        {
            "schema_version": 1,
            "status": "pass",
            "lane": LANE,
            "checks": checks,
            "pass_line": pass_line,
        },
    )
    print(pass_line)
    return 0


def self_test() -> int:
    cfg = {
        "model": "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
        "quant_format": "gptq_int4",
        "gpu_devices": [0, 1],
        "distributed_strategy": "layer_split",
    }
    validate_config(cfg)
    cmd = build_run_command(Path("./target/release/ferrum"), cfg["model"], cfg, Path("/tmp/out"))
    assert "--gpu-devices" in cmd
    assert "0,1" in cmd
    assert "--effective-config-json" in cmd
    print("G0 CUDA LLAMA33 70B 4BIT 2X4090 GATE SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path, default=Path("./target/release/ferrum"))
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.config is None or args.out is None:
        parser.error("--config and --out are required")

    repo = Path(__file__).resolve().parents[2]
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    try:
        cfg = load_json(args.config)
        validate_config(cfg)
        checks["config"] = {"status": "pass", "path": str(args.config)}
        capture_nvidia_smi(root, "before")
        write_metadata(root, repo, args.ferrum_bin, cfg)
        checks["run"] = run_cli_probe(root, repo, args.ferrum_bin, cfg)

        raise RuntimeError(
            "serve correctness, concurrency quality, bench-serve, and vLLM baseline checks are not implemented for this lane yet"
        )
    except Exception as exc:
        capture_nvidia_smi(root, "after")
        return gate_fail(root, str(exc), checks)
    finally:
        capture_nvidia_smi(root, "during")

    return gate_pass(root, checks)


if __name__ == "__main__":
    raise SystemExit(main())
