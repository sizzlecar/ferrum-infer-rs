#!/usr/bin/env python3
"""Run the Llama layer-split performance goal A/B gate on a 2x4090 host."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_PREFIX = "LAYER_SPLIT_PERF GOAL PASS"
BASELINE_CONFIG = Path("scripts/release/configs/layer_split_perf_baseline_batch.json")
CANDIDATE_CONFIG = Path("scripts/release/configs/g0_cuda2x4090_llama33_70b_4bit.json")
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
        detail = "; ".join(errors) if isinstance(errors, list) and errors else preflight.get("error")
        raise RuntimeError(f"layer-split perf full lane requires a 2x RTX 4090 host: {detail}")
    return preflight


def source_gate_command(
    repo: Path,
    ferrum_bin: Path,
    config: Path,
    out_dir: Path,
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


def candidate_config_for_run(
    repo: Path,
    out_root: Path,
    run_same_pod_vllm_baseline: bool,
) -> Path:
    if not run_same_pod_vllm_baseline:
        return repo / CANDIDATE_CONFIG
    config = load_json(repo / CANDIDATE_CONFIG)
    if not isinstance(config, dict):
        raise RuntimeError(f"candidate config must be a JSON object: {repo / CANDIDATE_CONFIG}")
    config["run_vllm_baseline"] = True
    config["same_pod_vllm_baseline"] = True
    generated = out_root / "configs" / "candidate-overlapped-with-vllm-baseline.json"
    write_json(generated, config)
    return generated


def command_plan(
    repo: Path,
    ferrum_bin: Path,
    out_root: Path,
    candidate_config: Path,
    optional_vllm_artifact: Path | None,
    run_same_pod_vllm_baseline: bool,
) -> dict[str, Any]:
    baseline = out_root / "baseline-batch"
    candidate = out_root / "candidate-overlapped"
    final = out_root / "final"
    final_optional_vllm_artifact = optional_vllm_artifact
    if run_same_pod_vllm_baseline and final_optional_vllm_artifact is None:
        final_optional_vllm_artifact = candidate
    expected_runtime_cost = "2x4090 host, 1-3 hours, prefer about 1 USD/hour"
    if run_same_pod_vllm_baseline:
        expected_runtime_cost = (
            "2x4090 host, 2-4 hours including same-pod vLLM baseline, "
            "prefer about 1 USD/hour"
        )
    return {
        "schema_version": 1,
        "created_at": iso_now(),
        "lane": "layer-split-perf-full",
        "expected_runtime_cost": expected_runtime_cost,
        "stop_condition": (
            "final PASS, any correctness failure, model load failure, CUDA OOM, "
            "or target miss with enough profiling evidence"
        ),
        "correctness_gate": "candidate g0_cuda2x4090_llama33_70b_4bit correctness matrix",
        "performance_command": (
            "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
            "--n-repeats 3 --concurrency-sweep 1,4,8,16"
        ),
        "hardware_preflight": "nvidia-smi must report exactly two RTX 4090 GPUs before build",
        "build_command": CUDA_BUILD_COMMAND,
        "baseline_config": str(repo / BASELINE_CONFIG),
        "candidate_config": str(candidate_config),
        "run_same_pod_vllm_baseline": run_same_pod_vllm_baseline,
        "baseline_artifact": str(baseline),
        "candidate_artifact": str(candidate),
        "optional_vllm_artifact": str(final_optional_vllm_artifact)
        if final_optional_vllm_artifact is not None
        else None,
        "final_artifact": str(final),
        "commands": {
            "build_cuda_release": CUDA_BUILD_COMMAND,
            "baseline_batch": source_gate_command(repo, ferrum_bin, BASELINE_CONFIG, baseline),
            "candidate_overlapped": source_gate_command(
                repo, ferrum_bin, candidate_config, candidate
            ),
            "final_validator": final_gate_command(
                repo, final, baseline, candidate, final_optional_vllm_artifact
            ),
        },
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
    print(f"Final artifact: {plan['final_artifact']}")
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
            "run_plan": str(out_root / "layer_split_perf_goal_run_plan.json"),
        },
    )


def run_step(name: str, cmd: list[str], repo: Path, out_root: Path) -> subprocess.CompletedProcess[str]:
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
    if args.run_same_pod_vllm_baseline and optional_vllm is not None:
        raise RuntimeError(
            "--run-same-pod-vllm-baseline and --optional-vllm-artifact are mutually exclusive"
        )
    candidate_config = candidate_config_for_run(
        repo, out_root, args.run_same_pod_vllm_baseline
    )
    plan = command_plan(
        repo,
        ferrum_bin,
        out_root,
        candidate_config,
        optional_vllm,
        args.run_same_pod_vllm_baseline,
    )
    write_json(out_root / "layer_split_perf_goal_run_plan.json", plan)
    print_run_plan(plan)
    failed_step = "hardware_preflight"
    try:
        require_2x4090_preflight(repo, out_root)

        failed_step = "build_cuda_release"
        run_step("build_cuda_release", plan["commands"]["build_cuda_release"], repo, out_root)
        failed_step = "baseline_batch"
        run_step("baseline_batch", plan["commands"]["baseline_batch"], repo, out_root)
        failed_step = "candidate_overlapped"
        run_step("candidate_overlapped", plan["commands"]["candidate_overlapped"], repo, out_root)
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
    assert any(str(BASELINE_CONFIG) in item for item in baseline_cmd)
    assert any(str(CANDIDATE_CONFIG) in item for item in candidate_cmd)
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
        generated_config = candidate_config_for_run(tmp_repo, tmp_out, True)
        generated = load_json(generated_config)
        assert generated["run_vllm_baseline"] is True
        assert generated["same_pod_vllm_baseline"] is True
        vllm_plan = command_plan(
            tmp_repo,
            Path("./target/release/ferrum"),
            tmp_out,
            generated_config,
            None,
            True,
        )
        assert vllm_plan["run_same_pod_vllm_baseline"] is True
        assert "2-4 hours" in vllm_plan["expected_runtime_cost"]
        assert vllm_plan["optional_vllm_artifact"] == str(tmp_out / "candidate-overlapped")
        assert str(generated_config) in vllm_plan["commands"]["candidate_overlapped"]
        assert str(tmp_out / "candidate-overlapped") in vllm_plan["commands"]["final_validator"]
    print("LAYER_SPLIT_PERF ORCHESTRATOR SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
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
