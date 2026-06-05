#!/usr/bin/env python3
"""Shared manifest helpers for G1-G4 local release gates."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_value(repo: Path, args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "sha256:missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def required_manifest_fields(
    *,
    repo: Path,
    goal: str,
    name: str,
    models: list[str],
    commands: list[str],
    started_at_utc: str,
    binary_path: Path,
    features: list[str] | None = None,
) -> dict[str, Any]:
    head = git_value(repo, ["rev-parse", "HEAD"])
    short = git_value(repo, ["rev-parse", "--short", "HEAD"])
    dirty = bool(git_value(repo, ["status", "--porcelain"], ""))
    branch = git_value(repo, ["branch", "--show-current"])
    finished_at_utc = utc_now()
    return {
        "goal": goal,
        "name": name,
        "status": "pass",
        "passed": True,
        "git_sha": head,
        "git_dirty": dirty,
        "binary_sha256": sha256_file(binary_path),
        "features": features or [],
        "models": models,
        "commands": commands,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "created_at": finished_at_utc,
        "repo": {
            "head": head,
            "short_head": short,
            "branch": branch,
            "dirty": dirty,
        },
    }
