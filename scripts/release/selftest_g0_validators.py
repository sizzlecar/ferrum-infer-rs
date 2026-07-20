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
RUNTIME_VNEXT_BASELINE_GATE = REPO_ROOT / "scripts/release/runtime_vnext_baseline_gate.py"
RUNTIME_VNEXT_INVENTORY = REPO_ROOT / "scripts/release/runtime_vnext_inventory.py"
RUNTIME_VNEXT_MODEL_RESOLVER = REPO_ROOT / "scripts/release/runtime_vnext_model_resolver.py"
RUNTIME_VNEXT_HARDWARE_PROBE = REPO_ROOT / "scripts/release/runtime_vnext_hardware_probe.py"
RUNTIME_VNEXT_BUILD_TIMING = REPO_ROOT / "scripts/release/runtime_vnext_build_timing.py"
RUNTIME_VNEXT_BASELINE_SCENARIOS = REPO_ROOT / "scripts/release/runtime_vnext_baseline_scenarios.py"
RUNTIME_VNEXT_EXPECTATION_AMENDMENT = REPO_ROOT / "scripts/release/runtime_vnext_expectation_amendment.py"
RUNTIME_VNEXT_BLOCKED_LANE = REPO_ROOT / "scripts/release/runtime_vnext_blocked_lane.py"
RUNTIME_VNEXT_RESOURCE_SAMPLER = REPO_ROOT / "scripts/release/runtime_vnext_resource_sampler.py"
RUNTIME_VNEXT_PERFORMANCE_COLLECTOR = REPO_ROOT / "scripts/release/runtime_vnext_performance_collector.py"
RUNTIME_VNEXT_G00A_CHECKPOINT = REPO_ROOT / "scripts/release/runtime_vnext_g00a_checkpoint.py"
RUNTIME_VNEXT_G00_ORCHESTRATOR = REPO_ROOT / "scripts/release/runtime_vnext_g00_orchestrator.py"
RUNTIME_VNEXT_HISTORICAL_CORPUS = REPO_ROOT / "scripts/release/runtime_vnext_historical_corpus.py"
RUNTIME_VNEXT_HISTORICAL_REPLAY = REPO_ROOT / "scripts/release/runtime_vnext_historical_replay.py"
RUNTIME_VNEXT_HISTORICAL_CAPTURE = REPO_ROOT / "scripts/release/runtime_vnext_historical_capture.py"
RUNTIME_VNEXT_G01A_CHECKPOINT = REPO_ROOT / "scripts/release/runtime_vnext_g01a_checkpoint.py"
RUNTIME_VNEXT_NUMERICAL_TOLERANCES = (
    REPO_ROOT / "scripts/release/runtime_vnext_numerical_tolerances.py"
)
RUNTIME_VNEXT_CHECKPOINT_ARTIFACT = (
    REPO_ROOT / "scripts/release/runtime_vnext_checkpoint_artifact.py"
)
QWEN35_GGUF_LINEAR_ATTENTION_REFERENCE = (
    REPO_ROOT / "scripts/release/qwen35_gguf_linear_attention_reference.py"
)
RUNTIME_VNEXT_QWEN35_LAYER_REFERENCE_GATE = (
    REPO_ROOT / "scripts/release/runtime_vnext_qwen35_layer_reference_gate.py"
)
RUNTIME_VNEXT_S1_CUDA_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_s1_cuda_checkpoint.py"
)
RUNTIME_VNEXT_S1_CUDA_BASIC_COLLECTOR = (
    REPO_ROOT / "scripts/release/runtime_vnext_s1_cuda_basic_collector.py"
)
RUNTIME_VNEXT_S1_CUDA_CAPACITY = REPO_ROOT / "scripts/release/runtime_vnext_s1_cuda_capacity.py"
RUNTIME_VNEXT_S1_CUDA_DECODE_CAPACITY = (
    REPO_ROOT / "scripts/release/runtime_vnext_s1_cuda_decode_capacity.py"
)
BOUNDED_COMMAND = REPO_ROOT / "scripts/release/bounded_command.py"
RUN_SCENARIOS = REPO_ROOT / "scripts/release/run_scenarios.py"
OPENAI_TOOL_CALL_REGRESSION = REPO_ROOT / "scripts/release/openai_tool_call_regression.py"
RUNTIME_VNEXT_S2_RESPONSE_FORMAT_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_s2_response_format_checkpoint.py"
)
RUNTIME_VNEXT_S2_API_MODALITY_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_s2_api_modality_checkpoint.py"
)
RUNTIME_VNEXT_S2_STREAM_DISCONNECT_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_s2_stream_disconnect_checkpoint.py"
)
PRODUCT_BACKEND_SENTINEL_GATE = REPO_ROOT / "scripts/release/product_backend_sentinel_gate.py"
PRODUCT_OBSERVABILITY_L1_SMOKE = REPO_ROOT / "scripts/release/product_observability_l1_smoke.py"
BACKEND_RUNTIME_GOAL_GATE = REPO_ROOT / "scripts/release/backend_runtime_preset_goal_gate.py"
LLAMA33_GOAL_GATE = REPO_ROOT / "scripts/release/llama33_70b_4bit_2x4090_goal_gate.py"
LAYER_SPLIT_PERF_GOAL_GATE = REPO_ROOT / "scripts/release/layer_split_perf_goal_gate.py"
LAYER_SPLIT_PERF_ORCHESTRATOR = REPO_ROOT / "scripts/release/run_layer_split_perf_goal.py"
LLAMA33_SOURCE_GATE = REPO_ROOT / "scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py"
MODEL_RELEASE_GRADE_GOAL_GATE = REPO_ROOT / "scripts/release/model_release_grade_goal_gate.py"
MODEL_RELEASE_GRADE_MANIFEST = REPO_ROOT / "scripts/release/model_release_grade_manifest.py"
MODEL_ONBOARDING_CONTRACT_GATE = REPO_ROOT / "scripts/release/model_onboarding_contract_gate.py"
RELEASE_REGRESSION_HARDENING_GOAL_GATE = REPO_ROOT / "scripts/release/release_regression_hardening_goal_gate.py"
ACTUAL_MODEL_REGRESSION_SUMMARY_GATE = REPO_ROOT / "scripts/release/actual_model_regression_summary_gate.py"
L2_ACTUAL_MODEL_ARTIFACT_GATE = REPO_ROOT / "scripts/release/l2_actual_model_artifact_gate.py"
SUPPORT_MATRIX_CONTRACT_GATE = REPO_ROOT / "scripts/release/support_matrix_contract_gate.py"
RUNTIME_VNEXT_BASELINE_FAST_SELFTEST_PASS = (
    "FERRUM RUNTIME VNEXT G00 BASELINE FAST SELFTEST PASS"
)
RUNTIME_VNEXT_BASELINE_SELFTEST_SUMMARY_PREFIX = (
    "FERRUM RUNTIME VNEXT G00 BASELINE SELFTEST SUMMARY:"
)
RUNTIME_VNEXT_BASELINE_MUTATION_COUNT = 115
RUNTIME_VNEXT_BASELINE_MUTATION_MATRIX_SHA256 = (
    "54a1cb0ffd4742f26c416b1c40f13803840d65fe7c7ba51c4866725fca9db3eb"
)
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


def load_openai_tool_call_regression():
    spec = importlib.util.spec_from_file_location(
        "openai_tool_call_regression", OPENAI_TOOL_CALL_REGRESSION
    )
    assert spec is not None and spec.loader is not None
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


def test_runtime_vnext_baseline_gate_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_BASELINE_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    lines = ok.stdout.splitlines()
    require(lines.count(RUNTIME_VNEXT_BASELINE_FAST_SELFTEST_PASS) == 1, ok.stdout)
    summaries = [
        line.removeprefix(RUNTIME_VNEXT_BASELINE_SELFTEST_SUMMARY_PREFIX).strip()
        for line in lines
        if line.startswith(RUNTIME_VNEXT_BASELINE_SELFTEST_SUMMARY_PREFIX)
    ]
    require(len(summaries) == 1, ok.stdout)
    try:
        summary = json.loads(summaries[0])
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid Runtime vNext FAST self-test summary: {exc}") from exc
    require(summary.get("schema_version") == 1, str(summary))
    require(summary.get("mode") == "fast", str(summary))
    require(
        summary.get("mutation_assertion_count")
        == summary.get("expected_mutation_assertion_count")
        == RUNTIME_VNEXT_BASELINE_MUTATION_COUNT,
        str(summary),
    )
    mutation_names = summary.get("mutation_names")
    require(
        isinstance(mutation_names, list)
        and len(mutation_names) == RUNTIME_VNEXT_BASELINE_MUTATION_COUNT
        and all(isinstance(name, str) and name for name in mutation_names)
        and len(set(mutation_names)) == len(mutation_names),
        str(summary),
    )
    mutation_matrix_sha256 = hashlib.sha256(
        json.dumps(
            mutation_names,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    require(
        mutation_matrix_sha256 == RUNTIME_VNEXT_BASELINE_MUTATION_MATRIX_SHA256,
        str(summary),
    )
    validator_counts = summary.get("validator_counts")
    require(
        isinstance(validator_counts, dict)
        and validator_counts.get("root-integration", 0) > 0
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in validator_counts.values())
        and sum(validator_counts.values()) == RUNTIME_VNEXT_BASELINE_MUTATION_COUNT,
        str(summary),
    )


def test_runtime_vnext_inventory_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_INVENTORY), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("RUNTIME VNEXT INVENTORY SELF-TEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_model_resolver_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_MODEL_RESOLVER), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("RUNTIME VNEXT MODEL RESOLUTION SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_hardware_probe_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_HARDWARE_PROBE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("RUNTIME VNEXT HARDWARE PROBE SELF-TEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_build_timing_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_BUILD_TIMING), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("RUNTIME VNEXT BUILD TIMING SELF-TEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_baseline_scenarios_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_BASELINE_SCENARIOS), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUNTIME VNEXT G00 SCENARIOS SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_expectation_amendment_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_EXPECTATION_AMENDMENT), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("RUNTIME VNEXT EXPECTATION AMENDMENT SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_blocked_lane_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_BLOCKED_LANE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUNTIME VNEXT G00 BLOCKED LANE SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_resource_sampler_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_RESOURCE_SAMPLER), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUNTIME VNEXT RESOURCE SAMPLER SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_performance_collector_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_PERFORMANCE_COLLECTOR), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUNTIME VNEXT PERFORMANCE COLLECTOR SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_g00a_checkpoint_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_G00A_CHECKPOINT), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_g00_orchestrator_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_G00_ORCHESTRATOR), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G00 ORCHESTRATOR SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_historical_corpus_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_HISTORICAL_CORPUS), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G00 HISTORICAL CORPUS SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_historical_replay_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_HISTORICAL_REPLAY), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT HISTORICAL REPLAY SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_historical_capture_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_HISTORICAL_CAPTURE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G00 HISTORICAL CAPTURE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g01a_checkpoint_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_G01A_CHECKPOINT), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G01A CONTRACT CHECKPOINT SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_numerical_tolerances_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_NUMERICAL_TOLERANCES), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT NUMERICAL TOLERANCE SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )
    catalog = run(
        [sys.executable, str(RUNTIME_VNEXT_NUMERICAL_TOLERANCES), "--working-tree"]
    )
    require(catalog.returncode == 0, catalog.stderr or catalog.stdout)
    require(
        "RUNTIME VNEXT NUMERICAL TOLERANCE WORKTREE VALID" in catalog.stdout,
        catalog.stdout,
    )


def test_runtime_vnext_checkpoint_artifact_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_CHECKPOINT_ARTIFACT), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT CHECKPOINT ARTIFACT SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_qwen35_gguf_linear_attention_reference_selftest() -> None:
    ok = run(
        [sys.executable, str(QWEN35_GGUF_LINEAR_ATTENTION_REFERENCE), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "QWEN35 GGUF LINEAR ATTENTION REFERENCE SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_qwen35_layer_reference_gate_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_QWEN35_LAYER_REFERENCE_GATE), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT QWEN35 LINEAR ATTENTION NUMERICS SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s1_cuda_capacity_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_S1_CUDA_CAPACITY), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S1 CUDA CAPACITY SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s1_cuda_checkpoint_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_S1_CUDA_CHECKPOINT), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s1_cuda_basic_collector_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_S1_CUDA_BASIC_COLLECTOR), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S1 CUDA BASIC COLLECTOR SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s1_cuda_decode_capacity_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_S1_CUDA_DECODE_CAPACITY), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_bounded_command_selftest() -> None:
    ok = run([sys.executable, str(BOUNDED_COMMAND), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("BOUNDED COMMAND SELFTEST PASS" in ok.stdout, ok.stdout)


def test_run_scenarios_selftest() -> None:
    ok = run([sys.executable, str(RUN_SCENARIOS), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("BACKEND SCENARIO RUNNER SELFTEST PASS" in ok.stdout, ok.stdout)


def test_openai_tool_call_auto_choice_semantics() -> None:
    module = load_openai_tool_call_regression()
    content = {
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "I can answer without a tool."},
    }
    require(
        module.assert_auto_tool_choice_response("content", content)["outcome"] == "content",
        "auto content outcome was not accepted",
    )

    tool_call = {
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "北京", "unit": "celsius"}),
                    },
                }
            ],
        },
    }
    require(
        module.assert_auto_tool_choice_response("tool", tool_call)["outcome"] == "tool_call",
        "auto tool-call outcome was not accepted",
    )

    invalid = [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": ""}},
        {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "<tool_call>broken"},
        },
    ]
    for index, choice in enumerate(invalid):
        try:
            module.assert_auto_tool_choice_response(f"invalid-{index}", choice)
            raise AssertionError(f"invalid auto tool outcome {index} unexpectedly passed")
        except RuntimeError:
            pass


def test_runtime_vnext_s2_response_format_checkpoint_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_S2_RESPONSE_FORMAT_CHECKPOINT), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 RESPONSE FORMAT SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_api_modality_checkpoint_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_S2_API_MODALITY_CHECKPOINT), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 API MODALITY SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_stream_disconnect_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_S2_STREAM_DISCONNECT_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 STREAM DISCONNECT SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_product_backend_sentinel_selftest() -> None:
    ok = run([sys.executable, str(PRODUCT_BACKEND_SENTINEL_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("PRODUCT BACKEND SENTINEL SELFTEST PASS" in ok.stdout, ok.stdout)


def test_product_observability_l1_smoke_selftest() -> None:
    ok = run([sys.executable, str(PRODUCT_OBSERVABILITY_L1_SMOKE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("PRODUCT OBSERVABILITY L1 SMOKE SELFTEST PASS" in ok.stdout, ok.stdout)


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


def test_model_release_grade_manifest_selftest() -> None:
    ok = run([sys.executable, str(MODEL_RELEASE_GRADE_MANIFEST), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("MODEL RELEASE GRADE MANIFEST SELFTEST PASS" in ok.stdout, ok.stdout)


def test_model_onboarding_contract_gate_selftest() -> None:
    ok = run([sys.executable, str(MODEL_ONBOARDING_CONTRACT_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("MODEL ONBOARDING CONTRACT SELFTEST PASS" in ok.stdout, ok.stdout)


def test_release_regression_hardening_goal_gate_selftest() -> None:
    ok = run([sys.executable, str(RELEASE_REGRESSION_HARDENING_GOAL_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("RELEASE_REGRESSION_HARDENING GOAL SELFTEST PASS" in ok.stdout, ok.stdout)


def test_actual_model_regression_summary_gate_selftest() -> None:
    ok = run([sys.executable, str(ACTUAL_MODEL_REGRESSION_SUMMARY_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("ACTUAL MODEL REGRESSION SUMMARY SELFTEST PASS" in ok.stdout, ok.stdout)


def test_l2_actual_model_artifact_gate_selftest() -> None:
    ok = run([sys.executable, str(L2_ACTUAL_MODEL_ARTIFACT_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("L2 ACTUAL MODEL ARTIFACT SELFTEST PASS" in ok.stdout, ok.stdout)


def test_support_matrix_contract_gate_selftest() -> None:
    ok = run([sys.executable, str(SUPPORT_MATRIX_CONTRACT_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("SUPPORT MATRIX CONTRACT SELFTEST PASS" in ok.stdout, ok.stdout)


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
    test_runtime_vnext_baseline_gate_selftest()
    test_runtime_vnext_inventory_selftest()
    test_runtime_vnext_model_resolver_selftest()
    test_runtime_vnext_hardware_probe_selftest()
    test_runtime_vnext_build_timing_selftest()
    test_runtime_vnext_baseline_scenarios_selftest()
    test_runtime_vnext_expectation_amendment_selftest()
    test_runtime_vnext_blocked_lane_selftest()
    test_runtime_vnext_resource_sampler_selftest()
    test_runtime_vnext_performance_collector_selftest()
    test_runtime_vnext_g00a_checkpoint_selftest()
    test_runtime_vnext_g00_orchestrator_selftest()
    test_runtime_vnext_historical_corpus_selftest()
    test_runtime_vnext_historical_replay_selftest()
    test_runtime_vnext_historical_capture_selftest()
    test_runtime_vnext_g01a_checkpoint_selftest()
    test_runtime_vnext_numerical_tolerances_selftest()
    test_runtime_vnext_checkpoint_artifact_selftest()
    test_qwen35_gguf_linear_attention_reference_selftest()
    test_runtime_vnext_qwen35_layer_reference_gate_selftest()
    test_runtime_vnext_s1_cuda_checkpoint_selftest()
    test_runtime_vnext_s1_cuda_basic_collector_selftest()
    test_runtime_vnext_s1_cuda_capacity_selftest()
    test_runtime_vnext_s1_cuda_decode_capacity_selftest()
    test_bounded_command_selftest()
    test_run_gate_selftest()
    test_run_scenarios_selftest()
    test_openai_tool_call_auto_choice_semantics()
    test_runtime_vnext_s2_response_format_checkpoint_selftest()
    test_runtime_vnext_s2_api_modality_checkpoint_selftest()
    test_runtime_vnext_s2_stream_disconnect_checkpoint_selftest()
    test_product_backend_sentinel_selftest()
    test_product_observability_l1_smoke_selftest()
    test_backend_runtime_goal_gate_selftest()
    test_llama33_goal_gate_selftest()
    test_layer_split_perf_goal_gate_selftest()
    test_layer_split_perf_orchestrator_selftest()
    test_llama33_source_gate_selftest()
    test_model_release_grade_goal_gate_selftest()
    test_model_release_grade_manifest_selftest()
    test_model_onboarding_contract_gate_selftest()
    test_l2_actual_model_artifact_gate_selftest()
    test_actual_model_regression_summary_gate_selftest()
    test_support_matrix_contract_gate_selftest()
    test_release_regression_hardening_goal_gate_selftest()
    test_m3_quality_gate_artifact_validators()
    print("G0 VALIDATOR SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
