#!/usr/bin/env python3
"""Validate manifests emitted by scripts/m3_cuda_build_boundary_probe.py."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_keys(where: str, value: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(f"{where}: missing keys: {', '.join(missing)}")


def require_number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{where} must be numeric")
    return float(value)


def validate_summary_validation(where: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ValidationError(f"{where} must be an object")
    require_keys(where, value, {"ok", "row_count", "status_counts", "artifacts"})
    if value["ok"] is not True:
        raise ValidationError(f"{where}.ok must be true")
    if not isinstance(value["row_count"], int) or value["row_count"] <= 0:
        raise ValidationError(f"{where}.row_count must be positive integer")
    if not isinstance(value["status_counts"], dict):
        raise ValidationError(f"{where}.status_counts must be object")
    if not isinstance(value["artifacts"], list) or not all(
        isinstance(item, str) and item for item in value["artifacts"]
    ):
        raise ValidationError(f"{where}.artifacts must be non-empty strings")


def validate_manifest(path: Path, *, require_limits_pass: bool = False) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValidationError("manifest root must be an object")
    require_keys(
        "manifest",
        manifest,
        {
            "schema_version",
            "probe",
            "repo",
            "git",
            "kernel",
            "mutation",
            "iterations",
            "features",
            "package",
            "command",
            "timing",
            "required_cache_hit",
            "required_built",
            "runs",
        },
    )
    if manifest["schema_version"] != 1:
        raise ValidationError("manifest.schema_version must be 1")
    if manifest["probe"] != "m3_cuda_build_boundary":
        raise ValidationError("manifest.probe must be m3_cuda_build_boundary")
    if manifest["mutation"] not in {"touch", "content-change"}:
        raise ValidationError("manifest.mutation invalid")
    iterations = manifest["iterations"]
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValidationError("manifest.iterations must be positive integer")
    if not isinstance(manifest["command"], list) or not manifest["command"]:
        raise ValidationError("manifest.command must be non-empty list")

    timing = manifest["timing"]
    if not isinstance(timing, dict):
        raise ValidationError("manifest.timing must be object")
    require_keys(
        "manifest.timing",
        timing,
        {
            "elapsed_sec",
            "p50_sec_nearest_rank",
            "p95_sec_nearest_rank",
            "p50_limit_sec",
            "p95_limit_sec",
            "limits_pass",
        },
    )
    elapsed = timing["elapsed_sec"]
    if not isinstance(elapsed, list) or len(elapsed) != iterations:
        raise ValidationError("manifest.timing.elapsed_sec length must equal iterations")
    elapsed_values = [require_number("manifest.timing.elapsed_sec[]", value) for value in elapsed]
    if any(value < 0 for value in elapsed_values):
        raise ValidationError("manifest.timing.elapsed_sec values must be non-negative")
    p50 = require_number("manifest.timing.p50_sec_nearest_rank", timing["p50_sec_nearest_rank"])
    p95 = require_number("manifest.timing.p95_sec_nearest_rank", timing["p95_sec_nearest_rank"])
    p50_limit = require_number("manifest.timing.p50_limit_sec", timing["p50_limit_sec"])
    p95_limit = require_number("manifest.timing.p95_limit_sec", timing["p95_limit_sec"])
    if not isinstance(timing["limits_pass"], bool):
        raise ValidationError("manifest.timing.limits_pass must be boolean")
    expected_limits_pass = p50 <= p50_limit and p95 <= p95_limit
    if timing["limits_pass"] != expected_limits_pass:
        raise ValidationError("manifest.timing.limits_pass does not match p50/p95 limits")
    if require_limits_pass and not timing["limits_pass"]:
        raise ValidationError("manifest.timing.limits_pass is false")

    for key in ("required_cache_hit", "required_built"):
        if not isinstance(manifest[key], list) or not all(
            isinstance(item, str) and item for item in manifest[key]
        ):
            raise ValidationError(f"manifest.{key} must be string list")

    runs = manifest["runs"]
    if not isinstance(runs, list) or len(runs) != iterations:
        raise ValidationError("manifest.runs length must equal iterations")
    for expected_index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValidationError(f"manifest.runs[{expected_index}] must be object")
        require_keys(
            f"manifest.runs[{expected_index}]",
            run,
            {
                "index",
                "command",
                "exit_code",
                "elapsed_sec",
                "build_log",
                "summary_validation",
                "summary_validation_error",
            },
        )
        if run["index"] != expected_index:
            raise ValidationError(f"manifest.runs[{expected_index}].index mismatch")
        if run["exit_code"] != 0:
            raise ValidationError(f"manifest.runs[{expected_index}].exit_code is non-zero")
        if run["summary_validation_error"] is not None:
            raise ValidationError(
                f"manifest.runs[{expected_index}].summary_validation_error is set"
            )
        require_number(f"manifest.runs[{expected_index}].elapsed_sec", run["elapsed_sec"])
        validate_summary_validation(
            f"manifest.runs[{expected_index}].summary_validation",
            run["summary_validation"],
        )

    return {
        "ok": True,
        "iterations": iterations,
        "p50_sec": p50,
        "p95_sec": p95,
        "limits_pass": timing["limits_pass"],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "build_boundary_manifest.json"
        summary_validation = {
            "ok": True,
            "row_count": 2,
            "status_counts": {"cache_hit": 2},
            "artifacts": ["core-ptx:kernels/paged_varlen_attention_vllm.cu", "marlin"],
        }
        write_json(
            path,
            {
                "schema_version": 1,
                "probe": "m3_cuda_build_boundary",
                "repo": "/repo",
                "git": {"head": "abc", "status_short": ""},
                "kernel": "crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu",
                "mutation": "touch",
                "iterations": 2,
                "features": "cuda",
                "package": "ferrum-cli",
                "command": ["cargo", "build"],
                "timing": {
                    "elapsed_sec": [1.0, 2.0],
                    "p50_sec_nearest_rank": 1.0,
                    "p95_sec_nearest_rank": 2.0,
                    "p50_limit_sec": 75.0,
                    "p95_limit_sec": 90.0,
                    "limits_pass": True,
                },
                "required_cache_hit": ["marlin"],
                "required_built": [],
                "runs": [
                    {
                        "index": 1,
                        "command": ["cargo", "build"],
                        "exit_code": 0,
                        "elapsed_sec": 1.0,
                        "build_log": "/tmp/run1.log",
                        "summary_validation": summary_validation,
                        "summary_validation_error": None,
                    },
                    {
                        "index": 2,
                        "command": ["cargo", "build"],
                        "exit_code": 0,
                        "elapsed_sec": 2.0,
                        "build_log": "/tmp/run2.log",
                        "summary_validation": summary_validation,
                        "summary_validation_error": None,
                    },
                ],
            },
        )
        result = validate_manifest(path, require_limits_pass=True)
        assert result["ok"]
        assert result["p95_sec"] == 2.0

        bad = load_json(path)
        bad["timing"]["limits_pass"] = False
        write_json(path, bad)
        try:
            validate_manifest(path)
        except ValidationError as exc:
            assert "limits_pass does not match" in str(exc)
        else:
            raise AssertionError("mismatched limits_pass unexpectedly validated")

        bad["timing"]["limits_pass"] = True
        bad["runs"][1]["summary_validation_error"] = "missing cache hit"
        write_json(path, bad)
        try:
            validate_manifest(path)
        except ValidationError as exc:
            assert "summary_validation_error is set" in str(exc)
        else:
            raise AssertionError("failed run summary unexpectedly validated")

    print("validate_cuda_build_boundary_manifest self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--require-limits-pass", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.manifest is None:
        raise SystemExit("manifest path is required unless --self-test is used")
    result = validate_manifest(args.manifest, require_limits_pass=args.require_limits_pass)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "cuda build boundary manifest ok: "
            f"iterations={result['iterations']} "
            f"p50={result['p50_sec']}s p95={result['p95_sec']}s "
            f"limits_pass={result['limits_pass']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
