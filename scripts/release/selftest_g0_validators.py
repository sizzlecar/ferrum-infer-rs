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
NATIVE_OPERATOR_SOURCE_BUNDLE = (
    REPO_ROOT / "scripts/release/native_operator_source_bundle.py"
)
RUNTIME_VNEXT_MODEL_RESOLVER = REPO_ROOT / "scripts/release/runtime_vnext_model_resolver.py"
RUNTIME_VNEXT_HARDWARE_PROBE = REPO_ROOT / "scripts/release/runtime_vnext_hardware_probe.py"
RUNTIME_VNEXT_BUILD_TIMING = REPO_ROOT / "scripts/release/runtime_vnext_build_timing.py"
RUNTIME_VNEXT_CUDA_CORRECTNESS_BUILD = (
    REPO_ROOT / "scripts/release/runtime_vnext_cuda_correctness_build.py"
)
RUNTIME_VNEXT_PLAN_REFERENCE = (
    REPO_ROOT / "scripts/release/runtime_vnext_plan_reference.py"
)
RUNTIME_VNEXT_CUDA_CANDLE_BOUNDARY = (
    REPO_ROOT / "scripts/release/runtime_vnext_cuda_candle_boundary.py"
)
JSONL_PRODUCT_SESSION = REPO_ROOT / "scripts/release/jsonl_product_session.py"
RUNTIME_VNEXT_BASELINE_SCENARIOS = REPO_ROOT / "scripts/release/runtime_vnext_baseline_scenarios.py"
RUNTIME_VNEXT_BLOCKED_LANE = REPO_ROOT / "scripts/release/runtime_vnext_blocked_lane.py"
RUNTIME_VNEXT_RESOURCE_SAMPLER = REPO_ROOT / "scripts/release/runtime_vnext_resource_sampler.py"
RUNTIME_VNEXT_PERFORMANCE_COLLECTOR = REPO_ROOT / "scripts/release/runtime_vnext_performance_collector.py"
RUNTIME_VNEXT_G00A_CHECKPOINT = REPO_ROOT / "scripts/release/runtime_vnext_g00a_checkpoint.py"
RUNTIME_VNEXT_HISTORICAL_CORPUS = REPO_ROOT / "scripts/release/runtime_vnext_historical_corpus.py"
RUNTIME_VNEXT_HISTORICAL_REPLAY = REPO_ROOT / "scripts/release/runtime_vnext_historical_replay.py"
RUNTIME_VNEXT_G01A_CHECKPOINT = REPO_ROOT / "scripts/release/runtime_vnext_g01a_checkpoint.py"
RUNTIME_VNEXT_NUMERICAL_TOLERANCES = (
    REPO_ROOT / "scripts/release/runtime_vnext_numerical_tolerances.py"
)
RUNTIME_VNEXT_CHECKPOINT_ARTIFACT = (
    REPO_ROOT / "scripts/release/runtime_vnext_checkpoint_artifact.py"
)
RUNTIME_VNEXT_CUDA_DETERMINISM = (
    REPO_ROOT / "scripts/release/runtime_vnext_cuda_determinism.py"
)
RUNTIME_VNEXT_CUDA_DETERMINISM_COLLECT = (
    REPO_ROOT / "scripts/release/runtime_vnext_cuda_determinism_collect.py"
)
QWEN35_GGUF_LINEAR_ATTENTION_REFERENCE = (
    REPO_ROOT / "scripts/release/qwen35_gguf_linear_attention_reference.py"
)
QWEN35_GGUF_FULL_ATTENTION_REFERENCE = (
    REPO_ROOT / "scripts/release/qwen35_gguf_full_attention_reference.py"
)
QWEN35_GGUF_MODEL_REFERENCE = (
    REPO_ROOT / "scripts/release/qwen35_gguf_model_reference.py"
)
RUNTIME_VNEXT_QWEN35_LAYER_REFERENCE_GATE = (
    REPO_ROOT / "scripts/release/runtime_vnext_qwen35_layer_reference_gate.py"
)
RUNTIME_VNEXT_QWEN35_FULL_ATTENTION_GATE = (
    REPO_ROOT / "scripts/release/runtime_vnext_qwen35_full_attention_gate.py"
)
RUNTIME_VNEXT_QWEN35_MODEL_REFERENCE_GATE = (
    REPO_ROOT / "scripts/release/runtime_vnext_qwen35_model_reference_gate.py"
)
RUNTIME_VNEXT_G08A_METAL_OP_NUMERICS = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08a_metal_op_numerics.py"
)
RUNTIME_VNEXT_G08A_NUMERICS = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08a_numerics.py"
)
RUNTIME_VNEXT_G08A_TOKEN_PARITY_COLLECTOR = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08a_token_parity_collector.py"
)
RUNTIME_VNEXT_G08A_SOURCE_CONTRACT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08a_source_contract.py"
)
RUNTIME_VNEXT_G08A_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08a_checkpoint.py"
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
RUNTIME_VNEXT_G03_LIVE_CATALOG_COLLECTOR = (
    REPO_ROOT / "scripts/release/runtime_vnext_g03_live_catalog_collect.py"
)
RUNTIME_VNEXT_G03_LIVE_CATALOG_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g03_live_catalog_checkpoint.py"
)
RUNTIME_VNEXT_G08B_CUDA_MATRIX_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08b_cuda_matrix_checkpoint.py"
)
RUNTIME_VNEXT_G08B_CUDA_MATRIX_PREPARE = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08b_cuda_matrix_prepare.py"
)
RUNTIME_VNEXT_G08B_METAL_MATRIX_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08b_metal_matrix_checkpoint.py"
)
RUNTIME_VNEXT_G08B_METAL_MATRIX_PREPARE = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08b_metal_matrix_prepare.py"
)
RUNTIME_VNEXT_G08C_CUDA_MATRIX_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08c_cuda_matrix_checkpoint.py"
)
RUNTIME_VNEXT_G08C_CUDA_MATRIX_PREPARE = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08c_cuda_matrix_prepare.py"
)
RUNTIME_VNEXT_G08C_METAL_MATRIX_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08c_metal_matrix_checkpoint.py"
)
RUNTIME_VNEXT_G08C_METAL_MATRIX_PREPARE = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08c_metal_matrix_prepare.py"
)
RUNTIME_VNEXT_G08_PERFORMANCE_SMOKE = (
    REPO_ROOT / "scripts/release/runtime_vnext_g08_performance_smoke.py"
)
RUNTIME_VNEXT_G07B_NATIVE_CHAIN_VALIDATOR = (
    REPO_ROOT / "scripts/release/validate_runtime_vnext_g07b_native_chain.py"
)
RUNTIME_VNEXT_G07A_BUILD_ITERATION = (
    REPO_ROOT / "scripts/release/runtime_vnext_g07a_build_iteration.py"
)
RUNTIME_VNEXT_G07A_BUILD_ITERATION_VALIDATOR = (
    REPO_ROOT
    / "scripts/release/validate_runtime_vnext_g07a_build_iteration.py"
)
RUNTIME_VNEXT_G07A_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g07a_checkpoint.py"
)
RUNTIME_VNEXT_G07B_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g07b_checkpoint.py"
)
RUNTIME_VNEXT_G07_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_g07_checkpoint.py"
)
NATIVE_WORK_ATTRIBUTION_GATE = REPO_ROOT / "scripts/release/native_work_attribution_gate.py"
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
RUNTIME_VNEXT_S2_TOOL_SCHEMA_CHECKPOINT = (
    REPO_ROOT / "scripts/release/runtime_vnext_s2_tool_schema_checkpoint.py"
)
RUNTIME_VNEXT_S2_MULTITURN_CONCURRENCY_CHECKPOINT = (
    REPO_ROOT
    / "scripts/release/runtime_vnext_s2_multiturn_concurrency_checkpoint.py"
)
RUNTIME_VNEXT_S2_LATENCY_FAILURE_CHECKPOINT = (
    REPO_ROOT
    / "scripts/release/runtime_vnext_s2_latency_failure_checkpoint.py"
)
RUNTIME_VNEXT_S2_LATENCY_FAILURE_COLLECTOR = (
    REPO_ROOT
    / "scripts/release/runtime_vnext_s2_latency_failure_collector.py"
)
RUNTIME_VNEXT_S2_HISTORICAL_RESOURCE_SOURCE = (
    REPO_ROOT
    / "scripts/release/runtime_vnext_s2_historical_resource_source.py"
)
RUNTIME_VNEXT_S2_CUDA_PRODUCT_CONTRACT = (
    REPO_ROOT
    / "scripts/release/runtime_vnext_s2_cuda_product_contract.py"
)
RUNTIME_VNEXT_G02_CORE = REPO_ROOT / "scripts/release/runtime_vnext_g02_core.py"
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
RUNTIME_VNEXT_RELEASE_CONTROL_SELFTESTS = (
    (
        REPO_ROOT / "scripts/release/runtime_vnext_release_workflow_policy.py",
        "FERRUM RELEASE WORKFLOW POLICY SELFTEST PASS",
    ),
    (
        REPO_ROOT / "scripts/release/runtime_vnext_goal_gate.py",
        "FERRUM RUNTIME VNEXT GOAL GATE SELFTEST PASS",
    ),
    (
        REPO_ROOT / "scripts/release/runtime_vnext_sampled_final.py",
        "FERRUM RUNTIME VNEXT SAMPLED FINAL SELFTEST PASS",
    ),
    (
        REPO_ROOT / "scripts/release/runtime_vnext_g0_llama_sampled_execution.py",
        "FERRUM G0 LLAMA SAMPLED EXECUTION SELFTEST PASS",
    ),
    (
        REPO_ROOT / "scripts/release/runtime_vnext_crates_io_release.py",
        "FERRUM CRATES IO V0.8.0 SELFTEST PASS",
    ),
    (
        REPO_ROOT / "scripts/release/runtime_vnext_homebrew_release.py",
        "FERRUM HOMEBREW V0.8.0 SELFTEST PASS",
    ),
    (
        REPO_ROOT / "scripts/release/runtime_vnext_r2_ferrum_collector.py",
        "FERRUM RUNTIME VNEXT R2 FERRUM COLLECTOR SELFTEST PASS",
    ),
)
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
    data = json.loads((root / "summary.json").read_text())
    template = data["models"][0]
    template["default_startup"].update(
        {"max_sequences": 16, "min_required_max_sequences": 16, "max_allowed_max_sequences": 16}
    )
    template["serve_startup"].update(
        {"max_sequences": 16, "min_required_max_sequences": 16}
    )

    def model_fixture(key: str, concurrencies: tuple[int, ...]) -> dict[str, object]:
        model = json.loads(json.dumps(template))
        model["key"] = key
        original_cell = model["cells"][0]
        cells = []
        for concurrency in concurrencies:
            cell = json.loads(json.dumps(original_cell))
            cell.update(
                {
                    "concurrency": concurrency,
                    "prompts": concurrency,
                    "completed": concurrency,
                }
            )
            cell["quality"].update(
                {
                    "requests": concurrency,
                    "status_200": concurrency,
                    "marker_ok": concurrency,
                    "square_ok": concurrency,
                    "format_ok": concurrency,
                }
            )
            cells.append(cell)
        model["cells"] = cells
        return model

    data["models"] = [
        model_fixture("llama31_8b", (1, 8, 16)),
        model_fixture("qwen3_30b_a3b", (16,)),
    ]
    write_json(root / "summary.json", data)
    for key in ("llama31_8b", "qwen3_30b_a3b"):
        (root / f"{key}.server.stdout").write_text("server ready\n")
        (root / f"{key}.run.stdout").write_text("model answered normally\n")


def write_summary_gate_fixture(
    root: Path,
    directory: str,
    lane: str,
    child_gate: str,
    child_identity_field: str,
    child_identity: str,
    child_pass_prefix: str,
) -> None:
    artifact = root / directory
    child = {"status": "pass", child_identity_field: child_identity}
    if lane == "unit":
        child["source"] = {
            "git_sha": "1" * 40,
            "dirty_status": {"is_dirty": False, "status_short": []},
        }
    child_path = artifact / child_gate
    write_json(child_path, child)
    write_json(
        artifact / "gate.manifest.json",
        {
            "status": "pass",
            "lane": lane,
            "child_returncode": 0,
            "git_sha": "1" * 40,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "artifact_dir": str(artifact),
            "pass_line": f"FERRUM GATE {lane} PASS: {artifact}",
            "child_pass_line": child_pass_prefix + str(artifact),
            "child_artifacts": {
                "kind": "standard-g0-child",
                "child_manifest": {
                    "path": str(child_path),
                    "sha256": hashlib.sha256(child_path.read_bytes()).hexdigest(),
                    "size_bytes": child_path.stat().st_size,
                },
            },
        },
    )


def make_summary_artifact(root: Path) -> None:
    fixtures = (
        (
            "source-unit",
            "unit",
            "unit.gate.json",
            "lane",
            "unit",
            "G0 SOURCE unit PASS: ",
        ),
        (
            "source-metal",
            "metal",
            "metal.gate.json",
            "lane",
            "metal",
            "G0 SOURCE metal PASS: ",
        ),
        (
            "source-cuda-full",
            "cuda-full",
            "g0_cuda4090_full.gate.json",
            "lane",
            "g0_cuda4090_full",
            "G0 SOURCE g0_cuda4090_full PASS: ",
        ),
        (
            "source-cuda-llama-dense",
            "cuda-llama-dense",
            "g0_cuda4090_llama_dense.gate.json",
            "lane",
            "g0_cuda4090_llama_dense",
            "G0 SOURCE g0_cuda4090_llama_dense PASS: ",
        ),
        (
            "metal-tarball",
            "metal-tarball",
            "gate.json",
            "mode",
            "metal-tarball",
            "METAL TARBALL GATE PASS: ",
        ),
        (
            "cuda-tarball",
            "cuda-tarball",
            "gate.json",
            "mode",
            "cuda-tarball",
            "CUDA TARBALL GATE PASS: ",
        ),
        (
            "homebrew-metal",
            "homebrew-metal",
            "gate.json",
            "mode",
            "homebrew-metal",
            "HOMEBREW METAL GATE PASS: ",
        ),
        (
            "homebrew-cuda-fetch",
            "homebrew-cuda-fetch",
            "gate.json",
            "mode",
            "homebrew-cuda-fetch",
            "HOMEBREW CUDA FETCH GATE PASS: ",
        ),
    )
    for fixture in fixtures:
        write_summary_gate_fixture(root, *fixture)


def make_legacy_summary_artifact(root: Path) -> None:
    for relative in (
        "source-unit/unit.gate.json",
        "metal-tarball/gate.json",
        "cuda-tarball/gate.json",
        "homebrew-metal/gate.json",
        "homebrew-cuda-fetch/gate.json",
    ):
        write_json(root / relative, {"status": "pass"})
    import validate_release_completion_manifest as completion_validator

    completion_validator.make_selftest_manifest(
        root / "_completion-fixture.json",
        artifact_root=root,
    )


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
        validator_command = [
            sys.executable,
            str(METAL_VALIDATOR),
            str(root),
            "--require-release-matrix",
        ]
        ok = run(validator_command)
        require(ok.returncode == 0, ok.stderr or ok.stdout)
        require("METAL README GATE PASS" in ok.stdout, ok.stdout)

        data = json.loads((root / "summary.json").read_text())
        all_models = json.loads(json.dumps(data["models"]))
        data["models"] = data["models"][:1]
        write_json(root / "summary.json", data)
        missing_model = run(validator_command)
        require(missing_model.returncode != 0, "missing release model unexpectedly passed")
        require("must contain exactly" in missing_model.stderr, missing_model.stderr)

        data["models"] = json.loads(json.dumps(all_models))
        saved_cells = data["models"][0]["cells"]
        data["models"][0]["cells"] = []
        write_json(root / "summary.json", data)
        missing_cells = run(validator_command)
        require(missing_cells.returncode != 0, "missing performance cells unexpectedly passed")
        require("concurrency cells must be exactly" in missing_cells.stderr, missing_cells.stderr)

        data["models"][0]["cells"] = saved_cells
        data["models"][0]["default_startup"]["max_allowed_max_sequences"] = 15
        write_json(root / "summary.json", data)
        bad_default = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(bad_default.returncode != 0, "unsafe default max_sequences unexpectedly passed")
        require("default max_sequences 16 > allowed 15" in bad_default.stderr, bad_default.stderr)

        data["models"][0]["default_startup"]["max_allowed_max_sequences"] = 16
        write_json(root / "summary.json", data)
        data["models"][0]["chat"]["stateful_loop"]["repeated_prefixes"] = 1
        write_json(root / "summary.json", data)
        bad_loop = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(bad_loop.returncode != 0, "stateful loop regression unexpectedly passed")
        require("stateful_loop repeated_prefixes != 0" in bad_loop.stderr, bad_loop.stderr)

        data["models"][0]["chat"]["stateful_loop"]["repeated_prefixes"] = 0
        write_json(root / "summary.json", data)
        (root / "llama31_8b.run.stderr").write_text("thread panicked\n")
        bad = run([sys.executable, str(METAL_VALIDATOR), str(root)])
        require(bad.returncode != 0, "bad metal artifact unexpectedly passed")
        require("METAL README GATE FAIL" in bad.stderr, bad.stderr)


def test_summary_validator() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-summary-gate-") as tmp:
        root = Path(tmp)
        make_summary_artifact(root)
        command = [
            sys.executable,
            str(SUMMARY_VALIDATOR),
            str(root),
            "--profile",
            "v084",
        ]
        ok = run(command)
        require(ok.returncode == 0, ok.stderr or ok.stdout)
        require(ok.stdout == f"G0 RELEASE PASS: {root}\n", ok.stdout)
        require((root / "g0_release_summary.json").is_file(), "missing summary output")
        summary = json.loads((root / "g0_release_summary.json").read_text())
        require(
            set(summary["gates"])
            == {
                "source-unit/gate.manifest.json",
                "source-metal/gate.manifest.json",
                "source-cuda-full/gate.manifest.json",
                "source-cuda-llama-dense/gate.manifest.json",
                "metal-tarball/gate.manifest.json",
                "cuda-tarball/gate.manifest.json",
                "homebrew-metal/gate.manifest.json",
                "homebrew-cuda-fetch/gate.manifest.json",
            },
            summary,
        )
        require(
            summary["artifact_dir"] == str(root)
            and summary["release_candidate_sha"] == "1" * 40
            and summary["pass_line"] == f"G0 RELEASE PASS: {root}",
            summary,
        )
        require("release" not in summary and "release_candidate" not in summary, summary)

        legacy_root = root / "legacy-positive"
        make_legacy_summary_artifact(legacy_root)
        legacy_positive = run(
            [
                sys.executable,
                str(SUMMARY_VALIDATOR),
                str(legacy_root),
                "--profile",
                "legacy",
            ]
        )
        require(
            legacy_positive.returncode == 0
            and legacy_positive.stdout == f"G0 RELEASE PASS: {legacy_root}\n",
            legacy_positive.stderr or legacy_positive.stdout,
        )

        legacy_scope = run(
            [
                sys.executable,
                str(SUMMARY_VALIDATOR),
                str(root),
                "--profile",
                "legacy",
            ]
        )
        require(
            legacy_scope.returncode != 0
            and "vnext-g08-rc" in legacy_scope.stderr,
            "legacy profile did not retain its Runtime vNext inputs",
        )

        metal_manifest = root / "source-metal/gate.manifest.json"
        metal_doc = json.loads(metal_manifest.read_text())
        metal_doc["git_sha"] = "2" * 40
        write_json(metal_manifest, metal_doc)
        mixed_candidate = run(command)
        require(
            mixed_candidate.returncode != 0,
            "mixed release-candidate SHAs unexpectedly passed",
        )
        require(
            "do not bind one clean candidate git SHA" in mixed_candidate.stderr,
            mixed_candidate.stderr,
        )
        metal_doc["git_sha"] = "1" * 40
        write_json(metal_manifest, metal_doc)

        metal_doc["dirty_status"] = {
            "is_dirty": True,
            "status_short": [" M README.md"],
        }
        write_json(metal_manifest, metal_doc)
        dirty_candidate = run(command)
        require(
            dirty_candidate.returncode != 0,
            "dirty release-candidate gate unexpectedly passed",
        )
        require("candidate checkout is dirty" in dirty_candidate.stderr, dirty_candidate.stderr)
        metal_doc["dirty_status"] = {"is_dirty": False, "status_short": []}
        write_json(metal_manifest, metal_doc)

        unit_child_path = root / "source-unit/unit.gate.json"
        unit_child = json.loads(unit_child_path.read_text())
        unit_child["source"]["git_sha"] = "3" * 40
        write_json(unit_child_path, unit_child)
        unit_manifest_path = root / "source-unit/gate.manifest.json"
        unit_manifest = json.loads(unit_manifest_path.read_text())
        unit_binding = unit_manifest["child_artifacts"]["child_manifest"]
        unit_binding["sha256"] = hashlib.sha256(unit_child_path.read_bytes()).hexdigest()
        unit_binding["size_bytes"] = unit_child_path.stat().st_size
        write_json(unit_manifest_path, unit_manifest)
        stale_child = run(command)
        require(stale_child.returncode != 0, "stale child provenance unexpectedly passed")
        require("child gate candidate differs" in stale_child.stderr, stale_child.stderr)
        unit_child["source"]["git_sha"] = "1" * 40
        write_json(unit_child_path, unit_child)
        unit_binding["sha256"] = hashlib.sha256(unit_child_path.read_bytes()).hexdigest()
        unit_binding["size_bytes"] = unit_child_path.stat().st_size
        write_json(unit_manifest_path, unit_manifest)

        held_metal_manifest = metal_manifest.with_suffix(".json.missing")
        metal_manifest.rename(held_metal_manifest)
        missing_metal = run(command)
        require(
            missing_metal.returncode != 0,
            "release summary without Metal source evidence unexpectedly passed",
        )
        require("metal-source" in missing_metal.stderr, missing_metal.stderr)
        held_metal_manifest.rename(metal_manifest)

        cuda_child_path = root / "cuda-tarball/gate.json"
        cuda_child = json.loads(cuda_child_path.read_text())
        cuda_child["status"] = "fail"
        write_json(cuda_child_path, cuda_child)
        bad_child = run(command)
        require(bad_child.returncode != 0, "failed child gate unexpectedly passed")
        require("child gate not pass" in bad_child.stderr, bad_child.stderr)
        cuda_child["status"] = "pass"
        write_json(cuda_child_path, cuda_child)
        cuda_manifest_path = root / "cuda-tarball/gate.manifest.json"
        cuda_manifest = json.loads(cuda_manifest_path.read_text())
        cuda_binding = cuda_manifest["child_artifacts"]["child_manifest"]
        cuda_binding["sha256"] = hashlib.sha256(cuda_child_path.read_bytes()).hexdigest()
        cuda_binding["size_bytes"] = cuda_child_path.stat().st_size
        write_json(cuda_manifest_path, cuda_manifest)

        homebrew_manifest_path = root / "homebrew-metal/gate.manifest.json"
        homebrew_manifest = json.loads(homebrew_manifest_path.read_text())
        homebrew_manifest["lane"] = "cuda-tarball"
        write_json(homebrew_manifest_path, homebrew_manifest)
        wrong_lane = run(command)
        require(wrong_lane.returncode != 0, "wrong release lane unexpectedly passed")
        require("gate lane differs" in wrong_lane.stderr, wrong_lane.stderr)
        homebrew_manifest["lane"] = "homebrew-metal"
        write_json(homebrew_manifest_path, homebrew_manifest)

        metal_tarball_path = root / "metal-tarball/gate.manifest.json"
        metal_tarball = json.loads(metal_tarball_path.read_text())
        metal_tarball["pass_line"] = "WRONG PASS"
        write_json(metal_tarball_path, metal_tarball)
        wrong_outer_pass = run(command)
        require(
            wrong_outer_pass.returncode != 0,
            "wrong outer PASS line unexpectedly passed",
        )
        require("gate pass line differs" in wrong_outer_pass.stderr, wrong_outer_pass.stderr)
        metal_tarball["pass_line"] = (
            f"FERRUM GATE metal-tarball PASS: {root / 'metal-tarball'}"
        )
        write_json(metal_tarball_path, metal_tarball)

        cuda_full_path = root / "source-cuda-full/gate.manifest.json"
        cuda_full = json.loads(cuda_full_path.read_text())
        cuda_full["child_pass_line"] = "WRONG CHILD PASS"
        write_json(cuda_full_path, cuda_full)
        wrong_child_pass = run(command)
        require(
            wrong_child_pass.returncode != 0,
            "wrong delegated child PASS line unexpectedly passed",
        )
        require(
            "gate child pass line differs" in wrong_child_pass.stderr,
            wrong_child_pass.stderr,
        )
        cuda_full["child_pass_line"] = (
            f"G0 SOURCE g0_cuda4090_full PASS: {root / 'source-cuda-full'}"
        )
        write_json(cuda_full_path, cuda_full)

        dense_path = root / "source-cuda-llama-dense/gate.manifest.json"
        dense = json.loads(dense_path.read_text())
        copied_from = "/remote/evidence/ferrum-0.8.4/source-cuda-llama-dense"
        dense["artifact_dir"] = copied_from
        dense["pass_line"] = (
            f"FERRUM GATE cuda-llama-dense PASS: {copied_from}"
        )
        dense["child_pass_line"] = (
            f"G0 SOURCE g0_cuda4090_llama_dense PASS: {copied_from}"
        )
        dense["child_artifacts"]["child_manifest"]["path"] = (
            f"{copied_from}/g0_cuda4090_llama_dense.gate.json"
        )
        write_json(dense_path, dense)
        copied_evidence = run(command)
        require(
            copied_evidence.returncode == 0,
            copied_evidence.stderr or copied_evidence.stdout,
        )
        require(
            copied_evidence.stdout == f"G0 RELEASE PASS: {root}\n",
            copied_evidence.stdout,
        )

        metal_outer_path = root / "source-metal/gate.manifest.json"
        metal_outer = json.loads(metal_outer_path.read_text())
        recorded_metal_child = metal_outer["child_artifacts"]["child_manifest"]["path"]
        metal_outer["child_artifacts"]["child_manifest"]["path"] = (
            "/substituted/location/metal.gate.json"
        )
        write_json(metal_outer_path, metal_outer)
        wrong_child_path = run(command)
        require(wrong_child_path.returncode != 0, "wrong recorded child path unexpectedly passed")
        require(
            "child manifest recorded path differs" in wrong_child_path.stderr,
            wrong_child_path.stderr,
        )
        metal_outer["child_artifacts"]["child_manifest"]["path"] = recorded_metal_child
        write_json(metal_outer_path, metal_outer)

        substituted_path = root / "source-metal/metal.gate.json"
        substituted = json.loads(substituted_path.read_text())
        substituted["substituted"] = True
        write_json(substituted_path, substituted)
        substituted_outer_path = root / "source-metal/gate.manifest.json"
        substituted_outer = json.loads(substituted_outer_path.read_text())
        substituted_outer["child_artifacts"]["child_manifest"]["size_bytes"] = (
            substituted_path.stat().st_size
        )
        write_json(substituted_outer_path, substituted_outer)
        substituted_child = run(command)
        require(
            substituted_child.returncode != 0,
            "substituted child manifest unexpectedly passed",
        )
        require(
            "child manifest SHA256 differs" in substituted_child.stderr,
            substituted_child.stderr,
        )


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


def test_native_operator_source_bundle_selftest() -> None:
    ok = run([sys.executable, str(NATIVE_OPERATOR_SOURCE_BUNDLE), "self-test"])
    require(ok.returncode == 0, ok.stderr)
    require("FERRUM NATIVE SOURCE BUNDLE SELFTEST PASS" in ok.stdout, ok.stdout)


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


def test_runtime_vnext_cuda_correctness_build_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_CUDA_CORRECTNESS_BUILD), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM CUDA CORRECTNESS BUILD SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_plan_reference_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_PLAN_REFERENCE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM CUDA RELEASE PLAN REFERENCE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_cuda_candle_boundary_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_CUDA_CANDLE_BOUNDARY), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT CUDA CANDLE BOUNDARY SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_jsonl_product_session_selftest() -> None:
    ok = run([sys.executable, str(JSONL_PRODUCT_SESSION), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM JSONL PRODUCT SESSION SELFTEST PASS" in ok.stdout, ok.stdout)


def test_runtime_vnext_baseline_scenarios_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_BASELINE_SCENARIOS), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require("FERRUM RUNTIME VNEXT G00 SCENARIOS SELFTEST PASS" in ok.stdout, ok.stdout)


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


def test_runtime_vnext_cuda_determinism_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_CUDA_DETERMINISM), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT CUDA DETERMINISM SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_cuda_determinism_collect_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_CUDA_DETERMINISM_COLLECT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT CUDA DETERMINISM COLLECTOR SELF-TEST PASS" in ok.stdout,
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


def test_qwen35_gguf_full_attention_reference_selftest() -> None:
    ok = run(
        [sys.executable, str(QWEN35_GGUF_FULL_ATTENTION_REFERENCE), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "QWEN35 GGUF FULL ATTENTION REFERENCE SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_qwen35_gguf_model_reference_selftest() -> None:
    ok = run([sys.executable, str(QWEN35_GGUF_MODEL_REFERENCE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "QWEN35 GGUF MODEL REFERENCE SELF-TEST PASS" in ok.stdout,
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


def test_runtime_vnext_qwen35_full_attention_gate_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_QWEN35_FULL_ATTENTION_GATE), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT QWEN35 FULL ATTENTION NUMERICS SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_qwen35_model_reference_gate_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_QWEN35_MODEL_REFERENCE_GATE), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "RUNTIME VNEXT QWEN35 MODEL NUMERICS SELF-TEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08a_numerics_selftests() -> None:
    op = run(
        [sys.executable, str(RUNTIME_VNEXT_G08A_METAL_OP_NUMERICS), "self-test"]
    )
    require(op.returncode == 0, op.stderr or op.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08A METAL OP NUMERICS SELFTEST PASS" in op.stdout,
        op.stdout,
    )
    aggregate = run(
        [sys.executable, str(RUNTIME_VNEXT_G08A_NUMERICS), "--self-test"]
    )
    require(aggregate.returncode == 0, aggregate.stderr or aggregate.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08A NUMERICS SELFTEST PASS" in aggregate.stdout,
        aggregate.stdout,
    )
    collector = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08A_TOKEN_PARITY_COLLECTOR),
            "--self-test",
        ]
    )
    require(collector.returncode == 0, collector.stderr or collector.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08A TOKEN PARITY COLLECTOR SELFTEST PASS"
        in collector.stdout,
        collector.stdout,
    )


def test_runtime_vnext_g08a_source_contract_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_G08A_SOURCE_CONTRACT), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08A SOURCE OWNERSHIP SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08a_checkpoint_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_G08A_CHECKPOINT), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08A CHECKPOINT SELFTEST PASS" in ok.stdout,
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


def test_runtime_vnext_g03_live_catalog_collector_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G03_LIVE_CATALOG_COLLECTOR),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G03 LIVE CATALOG COLLECTOR SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g03_live_catalog_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G03_LIVE_CATALOG_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G03 LIVE CATALOG PASS:" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08b_cuda_matrix_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08B_CUDA_MATRIX_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08b_cuda_matrix_prepare_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08B_CUDA_MATRIX_PREPARE),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08B CUDA PREPARE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08b_metal_matrix_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08B_METAL_MATRIX_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08B METAL MODEL MATRIX SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08b_metal_matrix_prepare_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08B_METAL_MATRIX_PREPARE),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08B METAL PREPARE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08c_cuda_matrix_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08C_CUDA_MATRIX_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08C CUDA MODEL MATRIX SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08c_cuda_matrix_prepare_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08C_CUDA_MATRIX_PREPARE),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08C CUDA PREPARE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08c_metal_matrix_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08C_METAL_MATRIX_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08C METAL MODEL MATRIX SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08c_metal_matrix_prepare_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08C_METAL_MATRIX_PREPARE),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08C METAL PREPARE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g08_performance_smoke_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G08_PERFORMANCE_SMOKE),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G08 PERFORMANCE SMOKE SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g07b_native_chain_validator_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G07B_NATIVE_CHAIN_VALIDATOR),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G07B NATIVE CHAIN VALIDATOR SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g07a_build_iteration_selftests() -> None:
    collector = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G07A_BUILD_ITERATION),
            "--self-test",
        ]
    )
    require(collector.returncode == 0, collector.stderr or collector.stdout)
    require(
        "FERRUM RUNTIME VNEXT G07A BUILD ITERATION SELFTEST PASS"
        in collector.stdout,
        collector.stdout,
    )
    validator = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G07A_BUILD_ITERATION_VALIDATOR),
            "--self-test",
        ]
    )
    require(validator.returncode == 0, validator.stderr or validator.stdout)
    require(
        "FERRUM RUNTIME VNEXT G07A BUILD ITERATION VALIDATOR SELFTEST PASS"
        in validator.stdout,
        validator.stdout,
    )
    checkpoint = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G07A_CHECKPOINT),
            "--self-test",
        ]
    )
    require(
        checkpoint.returncode == 0,
        checkpoint.stderr or checkpoint.stdout,
    )
    require(
        "FERRUM RUNTIME VNEXT G07A CHECKPOINT SELFTEST PASS"
        in checkpoint.stdout,
        checkpoint.stdout,
    )


def test_runtime_vnext_g07b_checkpoint_selftest() -> None:
    checkpoint = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G07B_CHECKPOINT),
            "--self-test",
        ]
    )
    require(checkpoint.returncode == 0, checkpoint.stderr or checkpoint.stdout)
    require(
        "FERRUM RUNTIME VNEXT G07B CHECKPOINT SELFTEST PASS"
        in checkpoint.stdout,
        checkpoint.stdout,
    )


def test_runtime_vnext_g07_checkpoint_selftest() -> None:
    checkpoint = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_G07_CHECKPOINT),
            "--self-test",
        ]
    )
    require(checkpoint.returncode == 0, checkpoint.stderr or checkpoint.stdout)
    require(
        "FERRUM RUNTIME VNEXT G07 CHECKPOINT SELFTEST PASS"
        in checkpoint.stdout,
        checkpoint.stdout,
    )


def test_native_work_attribution_gate_selftest() -> None:
    ok = run([sys.executable, str(NATIVE_WORK_ATTRIBUTION_GATE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM NATIVE WORK ATTRIBUTION SELFTEST PASS" in ok.stdout,
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


def test_runtime_vnext_s2_tool_schema_checkpoint_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_S2_TOOL_SCHEMA_CHECKPOINT), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 TOOL SCHEMA PRIORITY SELFTEST PASS" in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_multiturn_concurrency_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_S2_MULTITURN_CONCURRENCY_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 MULTITURN CONCURRENCY SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_latency_failure_checkpoint_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_S2_LATENCY_FAILURE_CHECKPOINT),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_latency_failure_collector_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_S2_LATENCY_FAILURE_COLLECTOR),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE COLLECTOR SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_historical_resource_source_selftest() -> None:
    ok = run(
        [
            sys.executable,
            str(RUNTIME_VNEXT_S2_HISTORICAL_RESOURCE_SOURCE),
            "--self-test",
        ]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 HISTORICAL RESOURCE SOURCE SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_s2_cuda_product_contract_selftest() -> None:
    ok = run(
        [sys.executable, str(RUNTIME_VNEXT_S2_CUDA_PRODUCT_CONTRACT), "--self-test"]
    )
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT S2 CUDA PRODUCT CONTRACT SELFTEST PASS"
        in ok.stdout,
        ok.stdout,
    )


def test_runtime_vnext_g02_core_selftest() -> None:
    ok = run([sys.executable, str(RUNTIME_VNEXT_G02_CORE), "--self-test"])
    require(ok.returncode == 0, ok.stderr or ok.stdout)
    require(
        "FERRUM RUNTIME VNEXT G02 CORE L0 L1 SELFTEST PASS" in ok.stdout,
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


def test_runtime_vnext_release_control_selftests() -> None:
    for script, pass_line in RUNTIME_VNEXT_RELEASE_CONTROL_SELFTESTS:
        ok = run([sys.executable, str(script), "--self-test"])
        require(ok.returncode == 0, ok.stderr or ok.stdout)
        require(pass_line in ok.stdout, ok.stdout)


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
    test_native_operator_source_bundle_selftest()
    test_runtime_vnext_model_resolver_selftest()
    test_runtime_vnext_hardware_probe_selftest()
    test_runtime_vnext_build_timing_selftest()
    test_runtime_vnext_cuda_correctness_build_selftest()
    test_runtime_vnext_plan_reference_selftest()
    test_runtime_vnext_cuda_candle_boundary_selftest()
    test_jsonl_product_session_selftest()
    test_runtime_vnext_baseline_scenarios_selftest()
    test_runtime_vnext_blocked_lane_selftest()
    test_runtime_vnext_resource_sampler_selftest()
    test_runtime_vnext_performance_collector_selftest()
    test_runtime_vnext_g00a_checkpoint_selftest()
    test_runtime_vnext_historical_corpus_selftest()
    test_runtime_vnext_historical_replay_selftest()
    test_runtime_vnext_g01a_checkpoint_selftest()
    test_runtime_vnext_numerical_tolerances_selftest()
    test_runtime_vnext_checkpoint_artifact_selftest()
    test_runtime_vnext_cuda_determinism_selftest()
    test_runtime_vnext_cuda_determinism_collect_selftest()
    test_qwen35_gguf_linear_attention_reference_selftest()
    test_qwen35_gguf_full_attention_reference_selftest()
    test_qwen35_gguf_model_reference_selftest()
    test_runtime_vnext_qwen35_layer_reference_gate_selftest()
    test_runtime_vnext_qwen35_full_attention_gate_selftest()
    test_runtime_vnext_qwen35_model_reference_gate_selftest()
    test_runtime_vnext_g08a_numerics_selftests()
    test_runtime_vnext_g08a_source_contract_selftest()
    test_runtime_vnext_g08a_checkpoint_selftest()
    test_runtime_vnext_s1_cuda_checkpoint_selftest()
    test_runtime_vnext_s1_cuda_basic_collector_selftest()
    test_runtime_vnext_s1_cuda_capacity_selftest()
    test_runtime_vnext_s1_cuda_decode_capacity_selftest()
    test_runtime_vnext_g03_live_catalog_collector_selftest()
    test_runtime_vnext_g03_live_catalog_checkpoint_selftest()
    test_runtime_vnext_g08b_cuda_matrix_checkpoint_selftest()
    test_runtime_vnext_g08b_cuda_matrix_prepare_selftest()
    test_runtime_vnext_g08b_metal_matrix_checkpoint_selftest()
    test_runtime_vnext_g08b_metal_matrix_prepare_selftest()
    test_runtime_vnext_g08c_cuda_matrix_checkpoint_selftest()
    test_runtime_vnext_g08c_cuda_matrix_prepare_selftest()
    test_runtime_vnext_g08c_metal_matrix_checkpoint_selftest()
    test_runtime_vnext_g08c_metal_matrix_prepare_selftest()
    test_runtime_vnext_g08_performance_smoke_selftest()
    test_runtime_vnext_g07a_build_iteration_selftests()
    test_runtime_vnext_g07b_checkpoint_selftest()
    test_runtime_vnext_g07_checkpoint_selftest()
    test_runtime_vnext_g07b_native_chain_validator_selftest()
    test_native_work_attribution_gate_selftest()
    test_bounded_command_selftest()
    test_run_gate_selftest()
    test_run_scenarios_selftest()
    test_openai_tool_call_auto_choice_semantics()
    test_runtime_vnext_s2_response_format_checkpoint_selftest()
    test_runtime_vnext_s2_api_modality_checkpoint_selftest()
    test_runtime_vnext_s2_stream_disconnect_checkpoint_selftest()
    test_runtime_vnext_s2_tool_schema_checkpoint_selftest()
    test_runtime_vnext_s2_multiturn_concurrency_checkpoint_selftest()
    test_runtime_vnext_s2_latency_failure_checkpoint_selftest()
    test_runtime_vnext_s2_latency_failure_collector_selftest()
    test_runtime_vnext_s2_historical_resource_source_selftest()
    test_runtime_vnext_s2_cuda_product_contract_selftest()
    test_runtime_vnext_g02_core_selftest()
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
    test_runtime_vnext_release_control_selftests()
    test_m3_quality_gate_artifact_validators()
    print("G0 VALIDATOR SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
