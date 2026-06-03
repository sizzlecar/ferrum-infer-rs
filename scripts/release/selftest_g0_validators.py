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
                    "server_ready": True,
                    "chat": {
                        "paris": {"passed": True},
                        "multiturn": {"passed": True},
                        "stream": {"passed": True},
                    },
                    "run": {"passed": True},
                    "cells": [
                        {
                            "concurrency": 1,
                            "prompts": 2,
                            "completed": 2,
                            "failed": 0,
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
        "unit.gate.json",
        "source/metal.gate.json",
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


def main() -> int:
    test_metal_validator()
    test_summary_validator()
    print("G0 VALIDATOR SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
