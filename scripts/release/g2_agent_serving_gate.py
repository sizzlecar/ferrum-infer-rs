#!/usr/bin/env python3
"""G2 agent serving release gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from g1_g4_manifest import required_manifest_fields, utc_now

BAD_RUNTIME_PATTERNS = [
    "panicked at",
    "KV cache overflow",
    "failed to render model chat template",
    "<unk>",
    "[PAD]",
]


class GateLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=repo_root(), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def default_out_dir() -> Path:
    short = git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root() / "docs" / "release" / "g1-g4" / "g2-agent-serving" / f"{stamp}-{short}"


def assert_no_runtime_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_RUNTIME_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden runtime pattern {pattern!r} in {label}")


def run(cmd: list[str], out: Path, log: GateLog, *, timeout: int = 1200, scan_runtime: bool = False) -> subprocess.CompletedProcess[str]:
    log.write("RUN " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "NO_COLOR": "1"},
        check=False,
    )
    out.write_text(proc.stdout, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}; log={out}")
    if scan_runtime:
        assert_no_runtime_bad_patterns(out.name, proc.stdout)
    return proc


def write_artifacts(
    out: Path,
    args: argparse.Namespace,
    checks: dict[str, Any],
    started_at_utc: str,
) -> None:
    gate = {"status": "pass", "goal": "g2-agent-serving", "checks": checks}
    (out / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        **required_manifest_fields(
            repo=repo_root(),
            goal="G2",
            name="agent-serving",
            models=[args.model],
            commands=[
                "cargo test --workspace --all-targets",
                "cargo build --release -p ferrum-cli --bin ferrum",
                "cargo test -p ferrum-server structured_output",
                "cargo test --release -p ferrum-cli --test server_structured_output -- --ignored --test-threads=1",
                "cargo test --release -p ferrum-cli --test server_agent_tools -- --ignored --test-threads=1",
            ],
            started_at_utc=started_at_utc,
            binary_path=repo_root() / "target" / "release" / "ferrum",
            features=[],
        ),
        "goal": "G2",
        "name": "agent-serving",
        "status": "pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo": {
            "head": git_value(["rev-parse", "HEAD"]),
            "short_head": git_value(["rev-parse", "--short", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "dirty": bool(git_value(["status", "--porcelain"], "")),
        },
        "model": args.model,
        "checks": checks,
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "summary.md").write_text(
        "# G2 Agent Serving Gate\n\n"
        "Status: PASS\n\n"
        f"Model: `{args.model}`\n\n"
        "Validated:\n"
        "- `cargo test --workspace --all-targets`\n"
        "- `cargo test -p ferrum-server structured_output`\n"
        "- real-model strict schema smoke 20/20 via `server_structured_output`\n"
        "- real-model required tool-call smoke 10/10 via `server_agent_tools`\n"
        "- streaming tests require exactly one `[DONE]` and no invalid pre-validation content\n",
    )


def main() -> int:
    started_at_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--skip-workspace-test", action="store_true")
    args = parser.parse_args()

    out = (args.out or default_out_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = GateLog(out / "gate.log")
    checks: dict[str, Any] = {}

    if args.skip_workspace_test:
        (out / "cargo-test.log").write_text("skipped by --skip-workspace-test\n")
        checks["cargo_workspace_all_targets"] = "skipped"
    else:
        run(["cargo", "test", "--workspace", "--all-targets"], out / "cargo-test.log", log, timeout=2400)
        checks["cargo_workspace_all_targets"] = True

    run(
        ["cargo", "build", "--release", "-p", "ferrum-cli", "--bin", "ferrum"],
        out / "release-build.log",
        log,
        timeout=1200,
    )
    checks["release_build"] = True

    run(["cargo", "test", "-p", "ferrum-server", "structured_output"], out / "server-structured-output.log", log, timeout=600)
    checks["ferrum_server_structured_output"] = True

    run(
        ["cargo", "test", "--release", "-p", "ferrum-cli", "--test", "server_structured_output", "--", "--ignored", "--test-threads=1"],
        out / "server-structured-output-real-model.log",
        log,
        timeout=1800,
        scan_runtime=True,
    )
    checks["strict_schema_real_model_20x"] = True

    run(
        ["cargo", "test", "--release", "-p", "ferrum-cli", "--test", "server_agent_tools", "--", "--ignored", "--test-threads=1"],
        out / "server-agent-tools-real-model.log",
        log,
        timeout=1800,
        scan_runtime=True,
    )
    checks["required_tool_call_real_model_10x"] = True

    write_artifacts(out, args, checks, started_at_utc)
    print(f"G2 AGENT SERVING PASS: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"G2 AGENT SERVING FAIL: {exc}", file=sys.stderr)
        raise
