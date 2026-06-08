#!/usr/bin/env python3
"""Run the Llama layer-split performance goal A/B gate on a 2x4090 host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def source_gate_command(
    repo: Path,
    ferrum_bin: Path,
    config: Path,
    out_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repo / SOURCE_GATE),
        "--config",
        str(repo / config),
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


def command_plan(
    repo: Path,
    ferrum_bin: Path,
    out_root: Path,
    optional_vllm_artifact: Path | None,
) -> dict[str, Any]:
    baseline = out_root / "baseline-batch"
    candidate = out_root / "candidate-overlapped"
    final = out_root / "final"
    return {
        "schema_version": 1,
        "created_at": iso_now(),
        "lane": "layer-split-perf-full",
        "expected_runtime_cost": "2x4090 host, 1-3 hours, prefer about 1 USD/hour",
        "stop_condition": (
            "final PASS, any correctness failure, model load failure, CUDA OOM, "
            "or target miss with enough profiling evidence"
        ),
        "correctness_gate": "candidate g0_cuda2x4090_llama33_70b_4bit correctness matrix",
        "performance_command": (
            "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
            "--n-repeats 3 --concurrency-sweep 1,4,8,16"
        ),
        "build_command": CUDA_BUILD_COMMAND,
        "baseline_artifact": str(baseline),
        "candidate_artifact": str(candidate),
        "final_artifact": str(final),
        "commands": {
            "build_cuda_release": CUDA_BUILD_COMMAND,
            "baseline_batch": source_gate_command(repo, ferrum_bin, BASELINE_CONFIG, baseline),
            "candidate_overlapped": source_gate_command(
                repo, ferrum_bin, CANDIDATE_CONFIG, candidate
            ),
            "final_validator": final_gate_command(
                repo, final, baseline, candidate, optional_vllm_artifact
            ),
        },
    }


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
    plan = command_plan(repo, ferrum_bin, out_root, optional_vllm)
    write_json(out_root / "layer_split_perf_goal_run_plan.json", plan)

    run_step("build_cuda_release", plan["commands"]["build_cuda_release"], repo, out_root)
    run_step("baseline_batch", plan["commands"]["baseline_batch"], repo, out_root)
    run_step("candidate_overlapped", plan["commands"]["candidate_overlapped"], repo, out_root)
    final = run_step("final_validator", plan["commands"]["final_validator"], repo, out_root)
    if PASS_PREFIX not in final.stdout:
        raise RuntimeError(f"final validator output missing {PASS_PREFIX!r}")
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
    plan = command_plan(repo, Path("./target/release/ferrum"), out, None)
    assert plan["baseline_artifact"] == str(out / "baseline-batch")
    assert plan["candidate_artifact"] == str(out / "candidate-overlapped")
    baseline_cmd = plan["commands"]["baseline_batch"]
    candidate_cmd = plan["commands"]["candidate_overlapped"]
    final_cmd = plan["commands"]["final_validator"]
    assert plan["commands"]["build_cuda_release"] == CUDA_BUILD_COMMAND
    assert any(str(BASELINE_CONFIG) in item for item in baseline_cmd)
    assert any(str(CANDIDATE_CONFIG) in item for item in candidate_cmd)
    assert str(out / "candidate-overlapped") in final_cmd
    assert "--correctness-artifact" in final_cmd
    print("LAYER_SPLIT_PERF ORCHESTRATOR SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path, default=Path("./target/release/ferrum"))
    parser.add_argument("--optional-vllm-artifact", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.out is None:
        parser.error("--out is required")
    return run_goal(args)


if __name__ == "__main__":
    raise SystemExit(main())
