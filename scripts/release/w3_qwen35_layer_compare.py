#!/usr/bin/env python3
"""Compare W3 Qwen3.5 HF and Ferrum first-layer dumps."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_COMPARE = "W3 QWEN35 LAYER COMPARE PASS"
PASS_SELFTEST = "W3 QWEN35 LAYER COMPARE SELFTEST PASS"
HF_MANIFEST = "w3_qwen35_hf_layer_dump_manifest.json"
FERRUM_MANIFEST = "w3_qwen35_ferrum_layer_dump_manifest.json"
COMPARE_MANIFEST = "w3_qwen35_layer_compare_manifest.json"

TENSORS = [
    "layer_input",
    "input_norm",
    "mixed_qkv_raw",
    "z_raw",
    "b_raw",
    "a_raw",
    "mixed_qkv_conv",
    "delta_q",
    "delta_k",
    "delta_v",
    "delta_beta",
    "delta_g",
    "delta_core",
    "delta_norm",
    "delta_output",
    "residual_after_mixer",
    "post_attention_norm",
    "mlp_output",
    "layer_output",
]


class CompareError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = run_command(["git", *args])
    except OSError:
        return default
    return proc.stdout.strip() if proc.returncode == 0 else default


def git_summary() -> dict[str, Any]:
    tracked = [
        line
        for line in git_output(["status", "--short", "--untracked-files=no"], default="").splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"], default="").splitlines()
        if line.strip()
    ]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked or untracked),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompareError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CompareError(f"invalid JSON in {path}: {exc}") from exc


def read_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise CompareError(f"{path} length is not a float32 multiple")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def tensor_entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or name not in tensors:
        raise CompareError(f"manifest missing tensor entry {name}")
    entry = tensors[name]
    if not isinstance(entry, dict):
        raise CompareError(f"tensor entry {name} must be an object")
    return entry


def read_tensor(root: Path, manifest: dict[str, Any], name: str) -> tuple[list[float], list[int]]:
    entry = tensor_entry(manifest, name)
    rel = entry.get("file")
    shape = entry.get("shape")
    if not isinstance(rel, str):
        raise CompareError(f"{name}.file must be a string")
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise CompareError(f"{name}.shape must be an integer list")
    values = read_f32(root / "tensors" / rel)
    expected = math.prod(shape)
    if len(values) != expected:
        raise CompareError(f"{name} length {len(values)} != shape count {expected}")
    return values, shape


def error_stats(expected: list[float], actual: list[float]) -> dict[str, float]:
    if len(expected) != len(actual):
        raise CompareError(f"length mismatch {len(expected)} != {len(actual)}")
    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    rmse_acc = 0.0
    for exp, got in zip(expected, actual):
        diff = abs(exp - got)
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, diff / max(abs(exp), 1e-12))
        sum_abs += diff
        rmse_acc += diff * diff
    n = max(len(expected), 1)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": sum_abs / n,
        "rmse": math.sqrt(rmse_acc / n),
    }


def compare(hf_dir: Path, ferrum_dir: Path, *, atol: float) -> dict[str, Any]:
    hf_manifest = load_json(hf_dir / HF_MANIFEST)
    ferrum_manifest = load_json(ferrum_dir / FERRUM_MANIFEST)
    if hf_manifest.get("prompt_input_ids") != ferrum_manifest.get("prompt_input_ids"):
        raise CompareError("prompt_input_ids differ")
    if hf_manifest.get("selected_layer_idx") != ferrum_manifest.get("selected_layer_idx"):
        raise CompareError("selected_layer_idx differs")
    if hf_manifest.get("selected_layer_type") != ferrum_manifest.get("selected_layer_type"):
        raise CompareError("selected_layer_type differs")

    comparisons = {}
    failures = []
    for name in TENSORS:
        hf_values, hf_shape = read_tensor(hf_dir, hf_manifest, name)
        ferrum_values, ferrum_shape = read_tensor(ferrum_dir, ferrum_manifest, name)
        if hf_shape != ferrum_shape:
            raise CompareError(f"{name} shape mismatch {hf_shape} != {ferrum_shape}")
        stats = error_stats(hf_values, ferrum_values)
        passed = stats["max_abs"] <= atol
        comparisons[name] = {
            **stats,
            "atol": atol,
            "shape": hf_shape,
            "status": "pass" if passed else "fail",
        }
        if not passed:
            failures.append(f"{name} max_abs {stats['max_abs']} > {atol}")
    if failures:
        raise CompareError("; ".join(failures))
    return {
        "hf_manifest": hf_manifest,
        "ferrum_manifest": ferrum_manifest,
        "comparisons": comparisons,
    }


def run_compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = compare(Path(args.hf_dump).resolve(), Path(args.ferrum_dump).resolve(), atol=args.atol)
    pass_line = f"{PASS_COMPARE}: {out_dir}"
    write_json(
        out_dir / COMPARE_MANIFEST,
        {
            "schema_version": 1,
            "status": "pass",
            "mode": "compare",
            "pass_line": pass_line,
            "created_at": iso_now(),
            "command_line": command_line(),
            "git": git_summary(),
            "hf_dump": str(Path(args.hf_dump).resolve()),
            "ferrum_dump": str(Path(args.ferrum_dump).resolve()),
            "tolerance": {"max_abs": args.atol},
            "comparisons": result["comparisons"],
            "hf_git": result["hf_manifest"].get("git"),
            "ferrum_git": result["ferrum_manifest"].get("git"),
        },
    )
    print(pass_line)
    return 0


def make_synthetic_dump(root: Path, manifest_name: str, *, offset: float) -> None:
    tensors = {}
    for idx, name in enumerate(TENSORS):
        values = [idx + offset, idx + 0.25 + offset, idx + 0.5 + offset, idx + 0.75 + offset]
        write_f32(root / "tensors" / f"{name}.bin", values)
        tensors[name] = {
            "file": f"{name}.bin",
            "shape": [1, 4],
            "dtype": "float32",
            "numel": len(values),
        }
    write_json(
        root / manifest_name,
        {
            "schema_version": 1,
            "status": "pass",
            "model_id": "self-test/qwen35",
            "selected_layer_idx": 0,
            "selected_layer_type": "linear_attention",
            "prompt_input_ids": [[1, 2, 3, 4]],
            "git": git_summary(),
            "tensors": tensors,
        },
    )


def run_self_test(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    hf_dir = out_dir / "hf"
    ferrum_dir = out_dir / "ferrum"
    make_synthetic_dump(hf_dir, HF_MANIFEST, offset=0.0)
    make_synthetic_dump(ferrum_dir, FERRUM_MANIFEST, offset=0.0)
    result = compare(hf_dir, ferrum_dir, atol=args.atol)
    pass_line = f"{PASS_SELFTEST}: {out_dir}"
    write_json(
        out_dir / COMPARE_MANIFEST,
        {
            "schema_version": 1,
            "status": "pass",
            "mode": "self-test",
            "pass_line": pass_line,
            "created_at": iso_now(),
            "command_line": command_line(),
            "git": git_summary(),
            "comparisons": result["comparisons"],
            "note": "self-test validates Qwen3.5 dump comparator schema only",
        },
    )
    print(pass_line)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/w3_qwen35_layer_compare")
    parser.add_argument("--hf-dump")
    parser.add_argument("--ferrum-dump")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and (not args.hf_dump or not args.ferrum_dump):
        parser.error("--hf-dump and --ferrum-dump are required unless --self-test is used")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            return run_self_test(args)
        return run_compare(args)
    except CompareError as exc:
        print(f"W3 QWEN35 LAYER COMPARE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
