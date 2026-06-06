#!/usr/bin/env python3
"""Self-test the G0 release gate validators with tiny synthetic artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METAL_VALIDATOR = REPO_ROOT / "scripts/release/validate_metal_readme_regression.py"
SUMMARY_VALIDATOR = REPO_ROOT / "scripts/release/g0_release_summary.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from m3_validate_runner_artifact import (  # noqa: E402
    ValidationError,
    validate_concurrency_quality_gate,
    validate_tool_call_gate,
)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def make_metal_artifact(root: Path) -> None:
    write_json(
        root / "summary.json",
        {
            "models": [
                {
                    "key": "qwen3_0_6b",
                    "default_startup": {
                        "passed": True,
                        "max_sequences": 4,
                        "min_required_max_sequences": 2,
                        "max_allowed_max_sequences": 4,
                    },
                    "server_ready": True,
                    "serve_startup": {
                        "passed": True,
                        "max_sequences": 4,
                        "min_required_max_sequences": 1,
                    },
                    "chat": {
                        "paris": {"passed": True},
                        "multiturn": {"passed": True},
                        "stream": {"passed": True},
                    },
                    "tool_call": {
                        "status": "pass",
                        "checks": {
                            "omitted_tool_choice": {"passed": True},
                            "explicit_auto_tool_choice": {"passed": True},
                            "required_tool_choice": {"passed": True},
                            "tool_result_fill": {"passed": True},
                        },
                    },
                    "run": {"passed": True},
                    "cells": [
                        {
                            "concurrency": 1,
                            "prompts": 2,
                            "completed": 2,
                            "failed": 0,
                            "quality": {
                                "passed": True,
                                "requests": 1,
                                "status_200": 1,
                                "marker_ok": 1,
                                "square_ok": 1,
                                "format_ok": 1,
                                "crosstalk": 0,
                                "length_finishes": 0,
                            },
                            "output_throughput_tok_s": 42.0,
                            "ratio_to_readme": 1.0,
                            "not_regressed_90pct": True,
                        }
                    ],
                }
            ]
        },
    )
    (root / "qwen3_0_6b.server.stdout").write_text("server ready\n")
    (root / "qwen3_0_6b.run.stdout").write_text("model answered normally\n")


def make_summary_artifact(root: Path) -> None:
    for rel in [
        "source-unit/unit.gate.json",
        "source-metal/metal.gate.json",
        "source-cuda-full/g0_cuda4090_full.gate.json",
        "source-cuda-llama-dense/g0_cuda4090_llama_dense.gate.json",
        "metal-tarball/gate.json",
        "cuda-tarball/gate.json",
        "homebrew-metal/gate.json",
        "homebrew-cuda-fetch/gate.json",
    ]:
        write_json(root / rel, {"status": "pass"})


def test_metal_validator() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-metal-gate-") as tmp:
        root = Path(tmp)
        make_metal_artifact(root)
        ok = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(ok.returncode == 0, ok.stderr or ok.stdout)
        require("METAL README GATE PASS" in ok.stdout, ok.stdout)

        data = json.loads((root / "summary.json").read_text())
        data["models"][0]["default_startup"]["max_allowed_max_sequences"] = 3
        write_json(root / "summary.json", data)
        bad_default = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(bad_default.returncode != 0, "unsafe default max_sequences unexpectedly passed")
        require("default max_sequences 4 > allowed 3" in bad_default.stderr, bad_default.stderr)

        data["models"][0]["default_startup"]["max_allowed_max_sequences"] = 4
        write_json(root / "summary.json", data)
        (root / "qwen3_0_6b.run.stderr").write_text("thread panicked\n")
        bad = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(bad.returncode != 0, "bad metal artifact unexpectedly passed")
        require("METAL README GATE FAIL" in bad.stderr, bad.stderr)


def test_summary_validator() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-summary-gate-") as tmp:
        root = Path(tmp)
        make_summary_artifact(root)
        ok = run([sys.executable, str(SUMMARY_VALIDATOR), str(root)])
        require(ok.returncode == 0, ok.stderr or ok.stdout)
        require("G0 RELEASE PASS" in ok.stdout, ok.stdout)
        require((root / "g0_release_summary.json").is_file(), "missing summary output")

        write_json(root / "cuda-tarball/gate.json", {"status": "fail"})
        bad = run([sys.executable, str(SUMMARY_VALIDATOR), str(root)])
        require(bad.returncode != 0, "bad release summary unexpectedly passed")
        require("G0 RELEASE FAIL" in bad.stderr, bad.stderr)


def test_m3_quality_gate_artifact_validators() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-m3-gates-") as tmp:
        root = Path(tmp)
        tool = root / "tool_call_regression.json"
        write_json(
            tool,
            {
                "status": "pass",
                "checks": {
                    "omitted_tool_choice": {"passed": True},
                    "explicit_auto_tool_choice": {"passed": True},
                    "required_tool_choice": {"passed": True},
                    "tool_result_fill": {"passed": True},
                },
            },
        )
        validate_tool_call_gate("case", tool)

        bad_tool = root / "bad_tool_call_regression.json"
        write_json(
            bad_tool,
            {
                "status": "pass",
                "checks": {
                    "omitted_tool_choice": {"passed": True},
                    "explicit_auto_tool_choice": {"passed": True},
                    "required_tool_choice": {"passed": True},
                    "tool_result_fill": {"passed": False},
                },
            },
        )
        try:
            validate_tool_call_gate("case", bad_tool)
            raise AssertionError("bad tool-call gate unexpectedly passed")
        except ValidationError:
            pass

        quality = root / "concurrency_quality_regression.json"
        write_json(
            quality,
            {
                "status": "pass",
                "cells": [
                    {
                        "concurrency": 4,
                        "requests": 4,
                        "status_200": 4,
                        "json_ok": 4,
                        "marker_ok": 4,
                        "square_ok": 4,
                        "format_ok": 4,
                        "crosstalk": 0,
                        "length_finishes": 0,
                        "forbidden_count": 0,
                        "passed": True,
                    }
                ],
            },
        )
        validate_concurrency_quality_gate("case", quality)

        bad_quality = root / "bad_concurrency_quality_regression.json"
        bad = json.loads(quality.read_text())
        bad["cells"][0]["format_ok"] = 3
        write_json(bad_quality, bad)
        try:
            validate_concurrency_quality_gate("case", bad_quality)
            raise AssertionError("bad concurrency-quality gate unexpectedly passed")
        except ValidationError:
            pass


def main() -> int:
    test_metal_validator()
    test_summary_validator()
    test_m3_quality_gate_artifact_validators()
    print("G0 VALIDATOR SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
