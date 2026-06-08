#!/usr/bin/env python3
"""Run the Llama layer-split performance goal A/B gate on a 2x4090 host."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_PREFIX = "LAYER_SPLIT_PERF GOAL PASS"
SMOKE_PASS_PREFIX = "LAYER_SPLIT_PERF SMOKE PASS"
FULL_LANE = "layer-split-perf-full"
SMOKE_LANE = "layer-split-perf-smoke"
FULL_SOURCE_LANE = "g0_cuda2x4090_llama33_70b_4bit"
SMOKE_SOURCE_LANE = "g0_cuda2x4090_llama33_70b_4bit_smoke"
BASELINE_CONFIG = Path("scripts/release/configs/layer_split_perf_baseline_batch.json")
CANDIDATE_CONFIG = Path("scripts/release/configs/g0_cuda2x4090_llama33_70b_4bit.json")
SMOKE_CANDIDATE_CONFIG = Path(
    "scripts/release/configs/g0_cuda2x4090_llama33_70b_4bit_smoke.json"
)
SOURCE_GATE = Path("scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py")
FINAL_GATE = Path("scripts/release/layer_split_perf_goal_gate.py")
CUDA_BUILD_COMMAND = [
    "cargo",
    "build",
    "--release",
    "-p",
    "ferrum-cli",
    "--bin",
    "ferrum",
    "--features",
    "cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_status_short(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.splitlines()


def require_clean_worktree(repo: Path) -> None:
    dirty = git_status_short(repo)
    if dirty:
        raise RuntimeError(
            "refusing release-quality layer-split perf goal run from dirty worktree; "
            "final validator rejects dirty git metadata:\n" + "\n".join(dirty)
        )


def resolve_out_root(repo: Path, raw_out: Path) -> Path:
    if raw_out.is_absolute():
        out_root = raw_out
    else:
        out_root = repo / raw_out
    return out_root.resolve(strict=False)


def require_outside_repo(repo: Path, out_root: Path) -> None:
    repo_root_resolved = repo.resolve(strict=False)
    try:
        out_root.relative_to(repo_root_resolved)
    except ValueError:
        return
    raise RuntimeError(
        "refusing to write layer-split perf artifacts inside the git worktree; "
        "final validator requires clean git metadata. Use a path outside the repo, "
        "for example ../ferrum-infer-rs-records-20260608/layer-split-perf"
    )


def parse_nvidia_smi_gpu_query(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reader = csv.reader(text.splitlines())
    for row in reader:
        parts = [part.strip() for part in row]
        if len(parts) < 3:
            continue
        index, name, uuid = parts[:3]
        rows.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "uuid": uuid,
            }
        )
    return rows


def validate_gpu_preflight_rows(gpus: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(gpus) != 2:
        errors.append(f"expected exactly 2 GPUs, got {len(gpus)}")
    indices = [gpu.get("index") for gpu in gpus]
    if indices != [0, 1]:
        errors.append(f"expected GPU indices [0, 1], got {indices!r}")
    uuids = [str(gpu.get("uuid", "")).strip() for gpu in gpus]
    if len(set(uuids)) != len(uuids):
        errors.append("GPU UUIDs must be unique")
    for idx, gpu in enumerate(gpus):
        name = str(gpu.get("name", "")).lower()
        if "4090" not in name:
            errors.append(f"GPU {idx} is not an RTX 4090: {gpu.get('name')!r}")
    return errors


def query_gpu_preflight(repo: Path) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid",
        "--format=csv,noheader",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "schema_version": 1,
            "status": "fail",
            "cmd": cmd,
            "returncode": None,
            "error": str(exc),
            "gpus": [],
        }
    if proc.returncode != 0:
        return {
            "schema_version": 1,
            "status": "fail",
            "cmd": cmd,
            "returncode": proc.returncode,
            "error": proc.stderr.strip() or proc.stdout.strip() or "nvidia-smi failed",
            "gpus": [],
        }
    gpus = parse_nvidia_smi_gpu_query(proc.stdout)
    errors = validate_gpu_preflight_rows(gpus)
    status = "fail" if errors else "pass"
    return {
        "schema_version": 1,
        "status": status,
        "cmd": cmd,
        "returncode": proc.returncode,
        "gpus": gpus,
        "errors": errors,
    }


def require_2x4090_preflight(repo: Path, out_root: Path) -> dict[str, Any]:
    preflight = query_gpu_preflight(repo)
    write_json(out_root / "layer_split_perf_gpu_preflight.json", preflight)
    if preflight.get("status") != "pass":
        errors = preflight.get("errors")
        detail = (
            "; ".join(errors)
            if isinstance(errors, list) and errors
            else preflight.get("error")
        )
        raise RuntimeError(f"layer-split perf lane requires a 2x RTX 4090 host: {detail}")
    return preflight


def query_vllm_preflight() -> dict[str, Any]:
    path = shutil.which("vllm")
    if path is None:
        return {
            "schema_version": 1,
            "status": "fail",
            "cmd": ["vllm"],
            "error": "vllm executable not found on PATH",
            "path": None,
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "cmd": ["vllm"],
        "path": path,
    }


def require_vllm_preflight(out_root: Path) -> dict[str, Any]:
    preflight = query_vllm_preflight()
    write_json(out_root / "layer_split_perf_vllm_preflight.json", preflight)
    if preflight.get("status") != "pass":
        raise RuntimeError(preflight.get("error") or "vllm preflight failed")
    return preflight


def source_gate_command(
    repo: Path,
    ferrum_bin: Path,
    config: Path,
    out_dir: Path,
    source_lane: str,
) -> list[str]:
    config_path = config if config.is_absolute() else repo / config
    return [
        sys.executable,
        str(repo / SOURCE_GATE),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
        "--ferrum-bin",
        str(ferrum_bin),
        "--lane-name",
        source_lane,
    ]


def final_gate_command(
    repo: Path,
    out_dir: Path,
    baseline_artifact: Path,
    candidate_artifact: Path,
    optional_vllm_artifact: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo / FINAL_GATE),
        "--out",
        str(out_dir),
        "--baseline-artifact",
        str(baseline_artifact),
        "--candidate-artifact",
        str(candidate_artifact),
        "--correctness-artifact",
        str(candidate_artifact),
    ]
    if optional_vllm_artifact is not None:
        cmd.extend(["--optional-vllm-artifact", str(optional_vllm_artifact)])
    return cmd


def source_lane_for_goal_lane(lane_name: str) -> str:
    return SMOKE_SOURCE_LANE if lane_name == SMOKE_LANE else FULL_SOURCE_LANE


def candidate_config_path_for_lane(lane_name: str) -> Path:
    return SMOKE_CANDIDATE_CONFIG if lane_name == SMOKE_LANE else CANDIDATE_CONFIG


def baseline_config_for_run(repo: Path, out_root: Path, lane_name: str) -> Path:
    if lane_name != SMOKE_LANE:
        return repo / BASELINE_CONFIG
    baseline = load_json(repo / BASELINE_CONFIG)
    smoke = load_json(repo / SMOKE_CANDIDATE_CONFIG)
    if not isinstance(baseline, dict) or not isinstance(smoke, dict):
        raise RuntimeError("layer-split perf configs must be JSON objects")
    for key in [
        "concurrency_cells",
        "num_prompts",
        "warmup_requests",
        "n_repeats",
        "require_ci",
        "random_input_len",
        "random_output_len",
        "seed",
        "max_model_len",
        "kv_capacity",
        "max_num_seqs",
        "max_num_batched_tokens",
    ]:
        if key in smoke:
            baseline[key] = smoke[key]
    baseline["name"] = "layer-split-perf-baseline-batch-smoke"
    baseline["layer_split_pipeline_mode"] = "batch"
    baseline["run_vllm_baseline"] = False
    generated = out_root / "configs" / "baseline-batch-smoke.json"
    write_json(generated, baseline)
    return generated


def candidate_config_for_run(
    repo: Path,
    out_root: Path,
    run_same_pod_vllm_baseline: bool,
    lane_name: str,
) -> Path:
    base_config = candidate_config_path_for_lane(lane_name)
    if not run_same_pod_vllm_baseline:
        return repo / base_config
    config = load_json(repo / base_config)
    if not isinstance(config, dict):
        raise RuntimeError(f"candidate config must be a JSON object: {repo / base_config}")
    config["run_vllm_baseline"] = True
    config["same_pod_vllm_baseline"] = True
    generated = out_root / "configs" / "candidate-overlapped-with-vllm-baseline.json"
    write_json(generated, config)
    return generated


def command_plan(
    repo: Path,
    ferrum_bin: Path,
    out_root: Path,
    lane_name: str,
    baseline_config: Path,
    candidate_config: Path,
    optional_vllm_artifact: Path | None,
    run_same_pod_vllm_baseline: bool,
) -> dict[str, Any]:
    baseline = out_root / "baseline-batch"
    candidate = out_root / "candidate-overlapped"
    final = out_root / "final"
    source_lane = source_lane_for_goal_lane(lane_name)
    final_optional_vllm_artifact = optional_vllm_artifact
    if (
        lane_name == FULL_LANE
        and run_same_pod_vllm_baseline
        and final_optional_vllm_artifact is None
    ):
        final_optional_vllm_artifact = candidate
    if lane_name == SMOKE_LANE:
        expected_runtime_cost = "2x4090 host, 30-60 minutes, prefer about 1 USD/hour"
        stop_condition = (
            "any correctness failure, model load failure, CUDA OOM, or one candidate "
            "run still showing a flat throughput curve"
        )
        correctness_gate = "product run/serve smoke plus streaming usage"
        performance_command = (
            "ferrum bench-serve --fail-on-error --seed 9271 "
            "--concurrency-sweep 1,4 (diagnostic; no --require-ci)"
        )
    else:
        expected_runtime_cost = "2x4090 host, 1-3 hours, prefer about 1 USD/hour"
        if run_same_pod_vllm_baseline:
            expected_runtime_cost = (
                "2x4090 host, 2-4 hours including same-pod vLLM baseline, "
                "prefer about 1 USD/hour"
            )
        stop_condition = (
            "final PASS, any correctness failure, model load failure, CUDA OOM, "
            "or target miss with enough profiling evidence"
        )
        correctness_gate = "candidate g0_cuda2x4090_llama33_70b_4bit correctness matrix"
        performance_command = (
            "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
            "--n-repeats 3 --concurrency-sweep 1,4,8,16"
        )
    commands: dict[str, list[str]] = {
        "build_cuda_release": CUDA_BUILD_COMMAND,
        "baseline_batch": source_gate_command(
            repo, ferrum_bin, baseline_config, baseline, source_lane
        ),
        "candidate_overlapped": source_gate_command(
            repo, ferrum_bin, candidate_config, candidate, source_lane
        ),
    }
    if lane_name == FULL_LANE:
        commands["final_validator"] = final_gate_command(
            repo, final, baseline, candidate, final_optional_vllm_artifact
        )
    return {
        "schema_version": 1,
        "created_at": iso_now(),
        "lane": lane_name,
        "source_lane": source_lane,
        "expected_runtime_cost": expected_runtime_cost,
        "stop_condition": stop_condition,
        "correctness_gate": correctness_gate,
        "performance_command": performance_command,
        "hardware_preflight": "nvidia-smi must report exactly two RTX 4090 GPUs before build",
        "vllm_preflight": (
            "vllm executable must be on PATH before build"
            if run_same_pod_vllm_baseline
            else "not required unless --run-same-pod-vllm-baseline is set"
        ),
        "build_command": CUDA_BUILD_COMMAND,
        "baseline_config": str(baseline_config),
        "candidate_config": str(candidate_config),
        "run_same_pod_vllm_baseline": run_same_pod_vllm_baseline,
        "baseline_artifact": str(baseline),
        "candidate_artifact": str(candidate),
        "optional_vllm_artifact": str(final_optional_vllm_artifact)
        if final_optional_vllm_artifact is not None
        else None,
        "final_artifact": str(final) if lane_name == FULL_LANE else None,
        "smoke_summary": str(out_root / "layer_split_perf_smoke_summary.json")
        if lane_name == SMOKE_LANE
        else None,
        "commands": commands,
    }


def print_run_plan(plan: dict[str, Any]) -> None:
    print(f"Lane: {plan['lane']}")
    print(f"Expected runtime/cost: {plan['expected_runtime_cost']}")
    print(f"Stop condition: {plan['stop_condition']}")
    print(f"Correctness gate: {plan['correctness_gate']}")
    print(f"Performance command: {plan['performance_command']}")
    if plan["run_same_pod_vllm_baseline"]:
        print("Same-pod vLLM baseline: enabled; final target may use 80% of vLLM")
    else:
        print("Same-pod vLLM baseline: not collected by this orchestrator run")
    if plan["lane"] == FULL_LANE:
        print(f"Final artifact: {plan['final_artifact']}")
    else:
        print(f"Smoke summary: {plan['smoke_summary']}")
    sys.stdout.flush()


def write_failure_summary(
    out_root: Path,
    plan: dict[str, Any],
    failed_step: str,
    reason: str,
) -> None:
    write_json(
        out_root / "layer_split_perf_failure.json",
        {
            "schema_version": 1,
            "status": "fail",
            "created_at": iso_now(),
            "lane": plan["lane"],
            "failed_step": failed_step,
            "reason": reason,
            "stop_condition": plan["stop_condition"],
            "correctness_gate": plan["correctness_gate"],
            "performance_command": plan["performance_command"],
            "run_same_pod_vllm_baseline": plan["run_same_pod_vllm_baseline"],
            "baseline_artifact": plan["baseline_artifact"],
            "candidate_artifact": plan["candidate_artifact"],
            "optional_vllm_artifact": plan["optional_vllm_artifact"],
            "final_artifact": plan["final_artifact"],
            "smoke_summary": plan["smoke_summary"],
            "run_plan": str(out_root / "layer_split_perf_goal_run_plan.json"),
        },
    )


def write_smoke_summary(out_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    pass_line = f"{SMOKE_PASS_PREFIX}: {out_root}"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "diagnostic_only": True,
        "created_at": iso_now(),
        "lane": plan["lane"],
        "source_lane": plan["source_lane"],
        "baseline_artifact": plan["baseline_artifact"],
        "candidate_artifact": plan["candidate_artifact"],
        "stop_condition": plan["stop_condition"],
        "correctness_gate": plan["correctness_gate"],
        "performance_command": plan["performance_command"],
        "pass_line": pass_line,
        "final_goal_pass": None,
    }
    write_json(out_root / "layer_split_perf_smoke_summary.json", summary)
    return summary


def run_step(
    name: str,
    cmd: list[str],
    repo: Path,
    out_root: Path,
) -> subprocess.CompletedProcess[str]:
    started = iso_now()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    write_json(
        out_root / f"{name}.command.json",
        {
            "status": "pass" if proc.returncode == 0 else "fail",
            "started_at": started,
            "finished_at": iso_now(),
            "returncode": proc.returncode,
            "cmd": cmd,
            "env_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
        },
    )
    (out_root / f"{name}.stdout").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (out_root / f"{name}.stderr").write_text(proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed rc={proc.returncode}")
    return proc


def run_goal(args: argparse.Namespace) -> int:
    repo = repo_root()
    require_clean_worktree(repo)
    out_root = resolve_out_root(repo, args.out)
    require_outside_repo(repo, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ferrum_bin = args.ferrum_bin
    optional_vllm = args.optional_vllm_artifact
    if args.lane_name == SMOKE_LANE and (
        args.run_same_pod_vllm_baseline or optional_vllm is not None
    ):
        raise RuntimeError("same-pod vLLM baseline is only supported for layer-split-perf-full")
    if args.run_same_pod_vllm_baseline and optional_vllm is not None:
        raise RuntimeError(
            "--run-same-pod-vllm-baseline and --optional-vllm-artifact are mutually exclusive"
        )
    baseline_config = baseline_config_for_run(repo, out_root, args.lane_name)
    candidate_config = candidate_config_for_run(
        repo, out_root, args.run_same_pod_vllm_baseline, args.lane_name
    )
    plan = command_plan(
        repo,
        ferrum_bin,
        out_root,
        args.lane_name,
        baseline_config,
        candidate_config,
        optional_vllm,
        args.run_same_pod_vllm_baseline,
    )
    write_json(out_root / "layer_split_perf_goal_run_plan.json", plan)
    print_run_plan(plan)
    failed_step = "hardware_preflight"
    try:
        require_2x4090_preflight(repo, out_root)
        if args.run_same_pod_vllm_baseline:
            failed_step = "vllm_preflight"
            require_vllm_preflight(out_root)

        failed_step = "build_cuda_release"
        run_step("build_cuda_release", plan["commands"]["build_cuda_release"], repo, out_root)
        failed_step = "baseline_batch"
        run_step("baseline_batch", plan["commands"]["baseline_batch"], repo, out_root)
        failed_step = "candidate_overlapped"
        run_step("candidate_overlapped", plan["commands"]["candidate_overlapped"], repo, out_root)
        if args.lane_name == SMOKE_LANE:
            smoke = write_smoke_summary(out_root, plan)
            print(smoke["pass_line"])
            return 0
        failed_step = "final_validator"
        final = run_step("final_validator", plan["commands"]["final_validator"], repo, out_root)
        expected_pass_line = f"{PASS_PREFIX}: {plan['final_artifact']}"
        if expected_pass_line not in final.stdout.splitlines():
            raise RuntimeError(
                f"final validator output missing exact PASS line: {expected_pass_line}"
            )
    except RuntimeError as exc:
        write_failure_summary(out_root, plan, failed_step, str(exc))
        raise
    print(final.stdout.strip())
    return 0


def self_test() -> int:
    repo = Path("/repo")
    out = Path("/tmp/layer-split-perf")
    assert resolve_out_root(repo, Path("../records")) == Path("/records")
    require_outside_repo(repo, out)
    try:
        require_outside_repo(repo, Path("/repo/docs/release/g0/layer-split-perf"))
        raise AssertionError("repo-local output unexpectedly accepted")
    except RuntimeError as exc:
        assert "inside the git worktree" in str(exc)
    gpus = parse_nvidia_smi_gpu_query(
        "0, NVIDIA GeForce RTX 4090, GPU-selftest-0\n"
        "1, NVIDIA GeForce RTX 4090, GPU-selftest-1\n"
    )
    assert [gpu["index"] for gpu in gpus] == [0, 1]
    assert all("4090" in gpu["name"] for gpu in gpus)
    assert validate_gpu_preflight_rows(gpus) == []
    bad_gpus = parse_nvidia_smi_gpu_query("0, NVIDIA GeForce RTX 3090, GPU-selftest-0\n")
    assert len(bad_gpus) == 1
    assert any("not an RTX 4090" in error for error in validate_gpu_preflight_rows(bad_gpus))
    bad_indices = parse_nvidia_smi_gpu_query(
        "1, NVIDIA GeForce RTX 4090, GPU-selftest-1\n"
        "2, NVIDIA GeForce RTX 4090, GPU-selftest-2\n"
    )
    assert any("GPU indices" in error for error in validate_gpu_preflight_rows(bad_indices))
    duplicate_uuid = parse_nvidia_smi_gpu_query(
        "0, NVIDIA GeForce RTX 4090, GPU-same\n"
        "1, NVIDIA GeForce RTX 4090, GPU-same\n"
    )
    assert any("UUIDs" in error for error in validate_gpu_preflight_rows(duplicate_uuid))
    plan = command_plan(
        repo,
        Path("./target/release/ferrum"),
        out,
        FULL_LANE,
        repo / BASELINE_CONFIG,
        repo / CANDIDATE_CONFIG,
        None,
        False,
    )
    assert plan["baseline_artifact"] == str(out / "baseline-batch")
    assert plan["candidate_artifact"] == str(out / "candidate-overlapped")
    assert f"{PASS_PREFIX}: {plan['final_artifact']}" == (
        "LAYER_SPLIT_PERF GOAL PASS: /tmp/layer-split-perf/final"
    )
    baseline_cmd = plan["commands"]["baseline_batch"]
    candidate_cmd = plan["commands"]["candidate_overlapped"]
    final_cmd = plan["commands"]["final_validator"]
    assert plan["commands"]["build_cuda_release"] == CUDA_BUILD_COMMAND
    assert "two RTX 4090" in plan["hardware_preflight"]
    assert "not required" in plan["vllm_preflight"]
    assert plan["lane"] == FULL_LANE
    assert plan["source_lane"] == FULL_SOURCE_LANE
    assert any(str(BASELINE_CONFIG) in item for item in baseline_cmd)
    assert FULL_SOURCE_LANE in baseline_cmd
    assert any(str(CANDIDATE_CONFIG) in item for item in candidate_cmd)
    assert FULL_SOURCE_LANE in candidate_cmd
    assert str(out / "candidate-overlapped") in final_cmd
    assert "--correctness-artifact" in final_cmd
    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp)
        write_failure_summary(tmp_out, plan, "hardware_preflight", "missing nvidia-smi")
        failure = json.loads((tmp_out / "layer_split_perf_failure.json").read_text())
        assert failure["status"] == "fail"
        assert failure["failed_step"] == "hardware_preflight"
        assert failure["stop_condition"] == plan["stop_condition"]
        assert failure["performance_command"] == plan["performance_command"]
    with tempfile.TemporaryDirectory(prefix="ferrum-layer-split-vllm-plan-") as tmp:
        tmp_repo = Path(tmp) / "repo"
        tmp_repo.mkdir()
        config_dir = tmp_repo / CANDIDATE_CONFIG.parent
        config_dir.mkdir(parents=True)
        write_json(
            tmp_repo / CANDIDATE_CONFIG,
            {
                "name": "selftest",
                "model": "selftest/model",
                "run_vllm_baseline": False,
            },
        )
        tmp_out = Path(tmp) / "out"
        generated_config = candidate_config_for_run(tmp_repo, tmp_out, True, FULL_LANE)
        generated = load_json(generated_config)
        assert generated["run_vllm_baseline"] is True
        assert generated["same_pod_vllm_baseline"] is True
        vllm_plan = command_plan(
            tmp_repo,
            Path("./target/release/ferrum"),
            tmp_out,
            FULL_LANE,
            tmp_repo / BASELINE_CONFIG,
            generated_config,
            None,
            True,
        )
        assert vllm_plan["run_same_pod_vllm_baseline"] is True
        assert "2-4 hours" in vllm_plan["expected_runtime_cost"]
        assert "vllm executable" in vllm_plan["vllm_preflight"]
        assert vllm_plan["optional_vllm_artifact"] == str(tmp_out / "candidate-overlapped")
        assert str(generated_config) in vllm_plan["commands"]["candidate_overlapped"]
        assert str(tmp_out / "candidate-overlapped") in vllm_plan["commands"]["final_validator"]
    with tempfile.TemporaryDirectory(prefix="ferrum-layer-split-smoke-plan-") as tmp:
        tmp_repo = Path(tmp) / "repo"
        tmp_repo.mkdir()
        config_dir = tmp_repo / CANDIDATE_CONFIG.parent
        config_dir.mkdir(parents=True)
        write_json(
            tmp_repo / BASELINE_CONFIG,
            {
                "name": "baseline",
                "model": "selftest/model",
                "layer_split_pipeline_mode": "batch",
                "concurrency_cells": [1, 4, 8, 16],
                "num_prompts": 96,
                "warmup_requests": 10,
                "n_repeats": 3,
                "run_vllm_baseline": False,
            },
        )
        write_json(
            tmp_repo / SMOKE_CANDIDATE_CONFIG,
            {
                "name": "smoke",
                "model": "selftest/model",
                "layer_split_pipeline_mode": "overlapped",
                "concurrency_cells": [1, 4],
                "num_prompts": 24,
                "warmup_requests": 4,
                "n_repeats": 1,
                "require_ci": False,
                "run_vllm_baseline": False,
            },
        )
        tmp_out = Path(tmp) / "out"
        smoke_baseline = baseline_config_for_run(tmp_repo, tmp_out, SMOKE_LANE)
        generated_baseline = load_json(smoke_baseline)
        assert generated_baseline["layer_split_pipeline_mode"] == "batch"
        assert generated_baseline["concurrency_cells"] == [1, 4]
        assert generated_baseline["num_prompts"] == 24
        assert generated_baseline["n_repeats"] == 1
        smoke_candidate = candidate_config_for_run(tmp_repo, tmp_out, False, SMOKE_LANE)
        assert smoke_candidate == tmp_repo / SMOKE_CANDIDATE_CONFIG
        smoke_plan = command_plan(
            tmp_repo,
            Path("./target/release/ferrum"),
            tmp_out,
            SMOKE_LANE,
            smoke_baseline,
            smoke_candidate,
            None,
            False,
        )
        assert smoke_plan["lane"] == SMOKE_LANE
        assert smoke_plan["source_lane"] == SMOKE_SOURCE_LANE
        assert "30-60 minutes" in smoke_plan["expected_runtime_cost"]
        assert "final_validator" not in smoke_plan["commands"]
        assert smoke_plan["final_artifact"] is None
        assert smoke_plan["smoke_summary"] == str(tmp_out / "layer_split_perf_smoke_summary.json")
        assert SMOKE_SOURCE_LANE in smoke_plan["commands"]["candidate_overlapped"]
        smoke_summary = write_smoke_summary(tmp_out, smoke_plan)
        assert smoke_summary["diagnostic_only"] is True
        assert smoke_summary["pass_line"] == f"{SMOKE_PASS_PREFIX}: {tmp_out}"
    with tempfile.TemporaryDirectory(prefix="ferrum-layer-split-vllm-preflight-") as tmp:
        tmp_out = Path(tmp)
        empty_bin_dir = tmp_out / "empty-bin"
        empty_bin_dir.mkdir()
        old_path = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = str(empty_bin_dir)
            try:
                require_vllm_preflight(tmp_out)
                raise AssertionError("missing vllm preflight unexpectedly passed")
            except RuntimeError as exc:
                assert "vllm executable not found" in str(exc)
            missing = load_json(tmp_out / "layer_split_perf_vllm_preflight.json")
            assert missing["status"] == "fail"
        finally:
            os.environ["PATH"] = old_path
        bin_dir = tmp_out / "bin"
        bin_dir.mkdir()
        fake_vllm = bin_dir / "vllm"
        fake_vllm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_vllm.chmod(0o755)
        try:
            os.environ["PATH"] = str(bin_dir) + os.pathsep + old_path
            preflight = require_vllm_preflight(tmp_out)
            assert preflight["status"] == "pass"
            assert preflight["path"] == str(fake_vllm)
            assert (tmp_out / "layer_split_perf_vllm_preflight.json").is_file()
        finally:
            os.environ["PATH"] = old_path
    print("LAYER_SPLIT_PERF ORCHESTRATOR SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lane-name", default=FULL_LANE, choices=[FULL_LANE, SMOKE_LANE])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path, default=Path("./target/release/ferrum"))
    parser.add_argument("--optional-vllm-artifact", type=Path)
    parser.add_argument(
        "--run-same-pod-vllm-baseline",
        action="store_true",
        help=(
            "run vLLM baseline inside the candidate source gate and pass the "
            "candidate artifact to the final validator as same-pod vLLM evidence"
        ),
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.out is None:
        parser.error("--out is required")
    try:
        return run_goal(args)
    except RuntimeError as exc:
        print(f"{PASS_PREFIX.replace(' PASS', ' FAIL')}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
