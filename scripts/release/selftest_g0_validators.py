#!/usr/bin/env python3
"""Self-test the G0 release gate validators with tiny synthetic artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METAL_VALIDATOR = REPO_ROOT / "scripts/release/validate_metal_readme_regression.py"
SUMMARY_VALIDATOR = REPO_ROOT / "scripts/release/g0_release_summary.py"
RELEASE_BINARY_GATE = REPO_ROOT / "scripts/release/release_binary_gate.py"
RUN_GATE = REPO_ROOT / "scripts/release/run_gate.py"
RUN_SCENARIOS = REPO_ROOT / "scripts/release/run_scenarios.py"
BACKEND_RUNTIME_GOAL_GATE = REPO_ROOT / "scripts/release/backend_runtime_preset_goal_gate.py"
LLAMA33_GOAL_GATE = REPO_ROOT / "scripts/release/llama33_70b_4bit_2x4090_goal_gate.py"
LAYER_SPLIT_PERF_GOAL_GATE = REPO_ROOT / "scripts/release/layer_split_perf_goal_gate.py"
LAYER_SPLIT_PERF_ORCHESTRATOR = REPO_ROOT / "scripts/release/run_layer_split_perf_goal.py"
LLAMA33_SOURCE_GATE = REPO_ROOT / "scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py"
MODEL_RELEASE_GRADE_GOAL_GATE = REPO_ROOT / "scripts/release/model_release_grade_goal_gate.py"
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
                        "stateful_loop": {
                            "passed": True,
                            "length_finishes": 0,
                            "repeated_prefixes": 0,
                        },
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
                                "format_ok": 0,
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


def load_release_binary_gate():
    spec = importlib.util.spec_from_file_location("release_binary_gate", RELEASE_BINARY_GATE)
    require(spec is not None and spec.loader is not None, "failed to load release binary gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        data["models"][0]["chat"]["stateful_loop"]["repeated_prefixes"] = 1
        write_json(root / "summary.json", data)
        bad_loop = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(bad_loop.returncode != 0, "stateful loop regression unexpectedly passed")
        require("stateful_loop repeated_prefixes != 0" in bad_loop.stderr, bad_loop.stderr)

        data["models"][0]["chat"]["stateful_loop"]["repeated_prefixes"] = 0
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


def test_release_binary_gate_staged_asset_path() -> None:
    gate = load_release_binary_gate()
    with tempfile.TemporaryDirectory(prefix="ferrum-release-binary-gate-") as tmp:
        root = Path(tmp)
        payload = root / "payload"
        payload.mkdir()
        (payload / "ferrum").write_text("#!/bin/sh\necho ferrum 0.7.6\n")
        asset = root / "ferrum-macos-aarch64.tar.gz"
        with tarfile.open(asset, "w:gz") as tf:
            tf.add(payload / "ferrum", arcname="ferrum")
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        (root / f"{asset.name}.sha256").write_text(f"{digest}  {asset.name}\n")

        out = root / "out"
        bin_path = gate.prepare_tarball("0.7.6", asset.name, out, None, asset)
        require(bin_path.is_file(), "staged asset extraction did not produce ferrum binary")
        require((out / asset.name).is_file(), "staged asset tarball was not copied")
        require((out / f"{asset.name}.sha256").is_file(), "staged asset sha256 was not copied")

        no_sha = root / "no-sha.tar.gz"
        no_sha.write_bytes(asset.read_bytes())
        try:
            gate.prepare_tarball("0.7.6", no_sha.name, root / "no-sha-out", None, no_sha)
            raise AssertionError("local staged asset without sha256 unexpectedly passed")
        except RuntimeError as e:
            require("missing sha256 for local asset" in str(e), str(e))

        try:
            gate.prepare_tarball("0.7.6", asset.name, root / "bad-sha-out", "0" * 64, asset)
            raise AssertionError("local staged asset with bad sha256 unexpectedly passed")
        except RuntimeError as e:
            require("sha256 mismatch" in str(e), str(e))


def test_run_gate_selftest() -> None:
    ok = run([sys.executable, str(RUN_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUN GATE SELFTEST PASS" in ok.stdout, ok.stdout)


def test_run_scenarios_selftest() -> None:
    ok = run([sys.executable, str(RUN_SCENARIOS), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("BACKEND SCENARIO RUNNER SELFTEST PASS" in ok.stdout, ok.stdout)


def test_backend_runtime_goal_gate_selftest() -> None:
    ok = run([sys.executable, str(BACKEND_RUNTIME_GOAL_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("BACKEND RUNTIME PRESET GOAL SELFTEST PASS" in ok.stdout, ok.stdout)


def test_llama33_goal_gate_selftest() -> None:
    ok = run([sys.executable, str(LLAMA33_GOAL_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("LLAMA33_70B_4BIT_2X4090 GOAL SELFTEST PASS" in ok.stdout, ok.stdout)


def test_layer_split_perf_goal_gate_selftest() -> None:
    ok = run([sys.executable, str(LAYER_SPLIT_PERF_GOAL_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("LAYER_SPLIT_PERF GOAL SELFTEST PASS" in ok.stdout, ok.stdout)


def test_layer_split_perf_orchestrator_selftest() -> None:
    ok = run([sys.executable, str(LAYER_SPLIT_PERF_ORCHESTRATOR), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("LAYER_SPLIT_PERF ORCHESTRATOR SELFTEST PASS" in ok.stdout, ok.stdout)


def test_llama33_source_gate_selftest() -> None:
    ok = run([sys.executable, str(LLAMA33_SOURCE_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "G0 CUDA LLAMA33 70B 4BIT 2X4090 GATE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_model_release_grade_goal_gate_selftest() -> None:
    ok = run([sys.executable, str(MODEL_RELEASE_GRADE_GOAL_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("MODEL RELEASE GRADE GOAL SELFTEST PASS" in ok.stdout, ok.stdout)


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
    test_release_binary_gate_staged_asset_path()
    test_run_gate_selftest()
    test_run_scenarios_selftest()
    test_backend_runtime_goal_gate_selftest()
    test_llama33_goal_gate_selftest()
    test_layer_split_perf_goal_gate_selftest()
    test_layer_split_perf_orchestrator_selftest()
    test_llama33_source_gate_selftest()
    test_model_release_grade_goal_gate_selftest()
    test_m3_quality_gate_artifact_validators()
    print("G0 VALIDATOR SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
