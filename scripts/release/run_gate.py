#!/usr/bin/env python3
"""Unified release gate runner.

This is the product-facing release entrypoint. It delegates to the existing
source, binary, and summary validators, then writes one normalized
`gate.manifest.json` and prints the unified PASS line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_LANES = {
    "unit": "unit",
    "metal": "metal",
    "cuda-smoke": "cuda-smoke",
    "cuda-full": "cuda-full",
    "cuda-llama-dense": "cuda-llama-dense",
    "cuda-llama33-70b-4bit-2x4090": "cuda-llama33-70b-4bit-2x4090",
}
BINARY_LANES = {
    "metal-tarball",
    "cuda-tarball",
    "homebrew-metal",
    "homebrew-cuda-fetch",
}
LANES = (
    "unit",
    "metal",
    "cuda-smoke",
    "cuda-full",
    "cuda-llama-dense",
    "cuda-llama33-70b-4bit-2x4090",
    "metal-tarball",
    "cuda-tarball",
    "homebrew-metal",
    "homebrew-cuda-fetch",
    "release-summary",
    "release-complete",
)
ENV_ALLOW_PREFIXES = ("FERRUM_",)
ENV_ALLOW_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "LD_LIBRARY_PATH",
    "RUST_LOG",
)
SECRET_KEY_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")


@dataclass(frozen=True)
class LaneCommand:
    cmd: list[str]
    binary_path: Path | None = None
    model: str | None = None
    expected_child_pass_line: str | None = None


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()


def git_sha() -> str:
    return git_output(["rev-parse", "HEAD"])


def git_dirty_status() -> dict[str, Any]:
    text = git_output(["status", "--short"], default="")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "is_dirty": bool(lines),
        "status_short": lines,
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitized_env_summary() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if not (key in ENV_ALLOW_KEYS or any(key.startswith(prefix) for prefix in ENV_ALLOW_PREFIXES)):
            continue
        if any(fragment in key.upper() for fragment in SECRET_KEY_FRAGMENTS):
            out[key] = "<redacted>"
        elif len(value) > 512:
            out[key] = f"{value[:512]}...<truncated>"
        else:
            out[key] = value
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_lane_command(args: argparse.Namespace, out_dir: Path) -> LaneCommand:
    lane = args.lane
    if lane in SOURCE_LANES:
        return LaneCommand(
            cmd=[
                "scripts/release/g0_source_gate.sh",
                SOURCE_LANES[lane],
                str(out_dir),
            ],
            binary_path=Path("target/release/ferrum")
            if lane.startswith("cuda") or lane == "metal"
            else None,
            model=model_for_source_lane(lane),
            expected_child_pass_line=source_pass_line(lane, out_dir),
        )
    if lane in BINARY_LANES:
        if not args.version:
            raise GateError(f"{lane} requires --version")
        cmd = [
            sys.executable,
            "scripts/release/release_binary_gate.py",
            lane,
            "--version",
            args.version,
            "--out",
            str(out_dir),
        ]
        if args.asset_path is not None:
            cmd.extend(["--asset-path", str(args.asset_path)])
        if args.sha256 is not None:
            cmd.extend(["--sha256", args.sha256])
        if args.model is not None:
            cmd.extend(["--model", args.model])
        if args.model_name is not None:
            cmd.extend(["--model-name", args.model_name])
        if args.port is not None:
            cmd.extend(["--port", str(args.port)])
        return LaneCommand(
            cmd=cmd,
            model=args.model,
            expected_child_pass_line=binary_pass_line(lane, out_dir),
        )
    if lane == "release-summary":
        release_root = args.release_root or out_dir
        return LaneCommand(
            cmd=[sys.executable, "scripts/release/g0_release_summary.py", str(release_root)],
            expected_child_pass_line=f"G0 RELEASE PASS: {release_root}",
        )
    if lane == "release-complete":
        if args.completion_manifest is None:
            raise GateError("release-complete requires --completion-manifest")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/validate_release_completion_manifest.py",
                "--manifest",
                str(args.completion_manifest),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=f"FERRUM RELEASE COMPLETION PASS: {out_dir}",
        )
    raise GateError(f"unknown lane: {lane}")


def model_for_source_lane(lane: str) -> str | None:
    if lane in {"cuda-smoke", "cuda-full"}:
        return "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    if lane == "cuda-llama-dense":
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    if lane == "cuda-llama33-70b-4bit-2x4090":
        return "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4"
    return None


def source_pass_line(lane: str, out_dir: Path) -> str:
    delegated = {
        "unit": "unit",
        "metal": "metal",
        "cuda-smoke": "g0_cuda4090_smoke",
        "cuda-full": "g0_cuda4090_full",
        "cuda-llama-dense": "g0_cuda4090_llama_dense",
        "cuda-llama33-70b-4bit-2x4090": "g0_cuda2x4090_llama33_70b_4bit",
    }[lane]
    return f"G0 SOURCE {delegated} PASS: {out_dir}"


def binary_pass_line(lane: str, out_dir: Path) -> str:
    delegated = {
        "metal-tarball": "METAL TARBALL GATE",
        "cuda-tarball": "CUDA TARBALL GATE",
        "homebrew-metal": "HOMEBREW METAL GATE",
        "homebrew-cuda-fetch": "HOMEBREW CUDA FETCH GATE",
    }[lane]
    return f"{delegated} PASS: {out_dir}"


def verify_child_pass_line(lane_command: LaneCommand, stdout: str) -> None:
    expected = lane_command.expected_child_pass_line
    if expected is None:
        return
    if expected not in stdout.splitlines():
        raise GateError(f"delegated command did not print required PASS line: {expected}")


def run_child(cmd: list[str], out_dir: Path, timeout: int | None) -> subprocess.CompletedProcess[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    duration = time.monotonic() - started
    (out_dir / "run_gate.child.stdout").write_text(proc.stdout, errors="replace")
    (out_dir / "run_gate.child.stderr").write_text(proc.stderr, errors="replace")
    (out_dir / "run_gate.child.command.json").write_text(
        json.dumps({"cmd": cmd, "duration_sec": duration}, indent=2) + "\n"
    )
    return proc


def pass_line(lane: str, out_dir: Path) -> str:
    return f"FERRUM GATE {lane} PASS: {out_dir}"


def manifest(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    lane_command: LaneCommand | None,
    status: str,
    started_at: str,
    finished_at: str,
    duration_sec: float,
    child_returncode: int | None,
    child_pass_line: str | None,
    error: str | None,
) -> dict[str, Any]:
    binary_path = lane_command.binary_path if lane_command else None
    binary_sha = sha256(binary_path) if binary_path else None
    return {
        "schema_version": 1,
        "lane": args.lane,
        "status": status,
        "command_line": command_line(),
        "delegated_command_line": lane_command.cmd if lane_command else None,
        "child_returncode": child_returncode,
        "child_pass_line": child_pass_line,
        "git_sha": git_sha(),
        "dirty_status": git_dirty_status(),
        "artifact_dir": str(out_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "binary": {
            "path": str(binary_path) if binary_path else None,
            "sha256": binary_sha,
        },
        "model": lane_command.model if lane_command else args.model,
        "sanitized_env": sanitized_env_summary(),
        "pass_line": pass_line(args.lane, out_dir) if status == "pass" else None,
        "error": error,
    }


def list_lanes() -> None:
    for lane in LANES:
        print(lane)


def require_selftest(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_selftest_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def write_selftest_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_selftest_release_summary_artifact(root: Path) -> None:
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
        write_selftest_json(root / rel, {"status": "pass"})


def make_selftest_completion_manifest(path: Path) -> None:
    root = path.parent
    artifacts = {}
    for name in [
        "metal_source_gate_artifact",
        "cuda_full_source_gate_artifact",
        "cuda_dense_source_gate_artifact",
        "metal_tarball_gate_artifact",
        "cuda_tarball_gate_artifact",
        "homebrew_metal_gate_artifact",
        "homebrew_cuda_fetch_gate_artifact",
    ]:
        artifact = root / "artifacts" / name
        artifact.mkdir(parents=True)
        artifacts[name] = str(artifact)
    write_selftest_json(
        path,
        {
            "git_sha": "selftest",
            "dirty_status": {"is_dirty": False, "status_short": []},
            "tag": "v0.0.0-selftest",
            "github_release_url": "https://example.invalid/selftest",
            "release_assets": [
                {
                    "name": "ferrum-selftest.tar.gz",
                    "sha256": "0" * 64,
                }
            ],
            "cargo_workspace_crates": [
                {
                    "name": "ferrum-cli",
                    "version": "0.0.0-selftest",
                    "crates_io_visible": True,
                }
            ],
            **artifacts,
        },
    )


def self_test() -> int:
    this_script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="ferrum-run-gate-selftest-") as tmp:
        root = Path(tmp)

        listed = run_selftest_command([sys.executable, str(this_script), "--list-lanes"])
        require_selftest(listed.returncode == 0, listed.stderr or listed.stdout)
        require_selftest(listed.stdout.splitlines() == list(LANES), listed.stdout)

        dry_out = root / "unit-dry-run"
        dry = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "unit",
                "--out",
                str(dry_out),
                "--dry-run",
            ]
        )
        require_selftest(dry.returncode == 0, dry.stderr or dry.stdout)
        dry_manifest = json.loads((dry_out / "gate.manifest.json").read_text())
        require_selftest(dry_manifest["status"] == "dry-run", dry_manifest)
        require_selftest(dry_manifest["lane"] == "unit", dry_manifest)
        require_selftest(
            dry_manifest["delegated_command_line"][0] == "scripts/release/g0_source_gate.sh",
            dry_manifest,
        )
        require_selftest(
            dry_manifest["child_pass_line"] == source_pass_line("unit", dry_out),
            dry_manifest,
        )
        try:
            verify_child_pass_line(
                LaneCommand(["selftest"], expected_child_pass_line="SELFTEST PASS"),
                "no pass line here\n",
            )
            raise AssertionError("missing delegated PASS line unexpectedly passed")
        except GateError as exc:
            require_selftest("SELFTEST PASS" in str(exc), str(exc))

        release_root = root / "release-root"
        make_selftest_release_summary_artifact(release_root)
        summary_out = root / "release-summary"
        summary = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "release-summary",
                "--release-root",
                str(release_root),
                "--out",
                str(summary_out),
            ]
        )
        require_selftest(summary.returncode == 0, summary.stderr or summary.stdout)
        require_selftest(
            f"FERRUM GATE release-summary PASS: {summary_out}" in summary.stdout,
            summary.stdout,
        )
        summary_manifest = json.loads((summary_out / "gate.manifest.json").read_text())
        require_selftest(summary_manifest["status"] == "pass", summary_manifest)
        require_selftest(summary_manifest["pass_line"], summary_manifest)
        require_selftest(
            summary_manifest["child_pass_line"] == f"G0 RELEASE PASS: {release_root}",
            summary_manifest,
        )

        completion_manifest_path = root / "completion-manifest.json"
        make_selftest_completion_manifest(completion_manifest_path)
        completion_out = root / "release-complete"
        complete = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "release-complete",
                "--completion-manifest",
                str(completion_manifest_path),
                "--out",
                str(completion_out),
            ]
        )
        require_selftest(complete.returncode == 0, complete.stderr or complete.stdout)
        require_selftest(
            f"FERRUM GATE release-complete PASS: {completion_out}" in complete.stdout,
            complete.stdout,
        )
        complete_manifest = json.loads((completion_out / "gate.manifest.json").read_text())
        require_selftest(complete_manifest["status"] == "pass", complete_manifest)
        require_selftest(
            complete_manifest["child_pass_line"]
            == f"FERRUM RELEASE COMPLETION PASS: {completion_out}",
            complete_manifest,
        )
        require_selftest(
            (completion_out / "release_completion_gate.json").is_file(),
            "missing completion validator artifact",
        )
    print("FERRUM RUN GATE SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", nargs="?", choices=LANES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list-lanes", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--version")
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--model")
    parser.add_argument("--model-name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--completion-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.list_lanes:
        list_lanes()
        return 0
    if args.lane is None:
        parser.error("lane is required unless --list-lanes is set")
    if args.out is None:
        parser.error("--out is required")

    out_dir = args.out
    started_at = iso_now()
    start = time.monotonic()
    lane_command: LaneCommand | None = None
    child_returncode: int | None = None
    child_pass_line: str | None = None
    status = "fail"
    error: str | None = None
    try:
        lane_command = build_lane_command(args, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            write_json(
                out_dir / "gate.manifest.json",
                manifest(
                    args=args,
                    out_dir=out_dir,
                    lane_command=lane_command,
                    status="dry-run",
                    started_at=started_at,
                    finished_at=iso_now(),
                    duration_sec=time.monotonic() - start,
                    child_returncode=None,
                    child_pass_line=lane_command.expected_child_pass_line,
                    error=None,
                ),
            )
            print(" ".join(shlex.quote(part) for part in lane_command.cmd))
            return 0
        proc = run_child(lane_command.cmd, out_dir, args.timeout)
        child_returncode = proc.returncode
        if proc.returncode != 0:
            error = f"delegated command failed rc={proc.returncode}"
            status = "fail"
        else:
            verify_child_pass_line(lane_command, proc.stdout)
            child_pass_line = lane_command.expected_child_pass_line
            status = "pass"
    except (GateError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
        status = "fail"
    finished_at = iso_now()
    doc = manifest(
        args=args,
        out_dir=out_dir,
        lane_command=lane_command,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=time.monotonic() - start,
        child_returncode=child_returncode,
        child_pass_line=child_pass_line,
        error=error,
    )
    write_json(out_dir / "gate.manifest.json", doc)
    if status == "pass":
        print(doc["pass_line"])
        return 0
    print(f"FERRUM GATE {args.lane} FAIL: {out_dir}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
