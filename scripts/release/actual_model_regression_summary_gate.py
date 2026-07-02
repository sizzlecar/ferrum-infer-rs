#!/usr/bin/env python3
"""Validate the WP14 L2 actual model regression summary.

This gate records the real Metal/CUDA representative model artifacts that the
final hardening goal gate consumes. It is intentionally a summary validator, not
another product runner.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_LINE = "ACTUAL MODEL REGRESSION SUMMARY PASS"
SELFTEST_PASS_LINE = "ACTUAL MODEL REGRESSION SUMMARY SELFTEST PASS"
SCHEMA_VERSION = 1
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
REQUIRED_ENTRYPOINTS = {"run", "serve", "stream", "basic_concurrency"}


class ActualModelGateError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ActualModelGateError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ActualModelGateError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ActualModelGateError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def head_sha() -> str:
    value = git_value(["rev-parse", "HEAD"])
    if not GIT_SHA_RE.match(value):
        raise ActualModelGateError(f"current HEAD is not a git SHA: {value!r}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ActualModelGateError(message)


def require_string(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    require(isinstance(value, str) and value.strip(), f"{label}.{key} must be a non-empty string")
    return str(value)


def require_string_list(value: Any, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    require(all(isinstance(item, str) for item in value), f"{label} entries must be strings")
    return list(value)


def normalize_command(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        require(value.strip(), f"{label} must be non-empty")
        return [value]
    if isinstance(value, list):
        require(value, f"{label} must be non-empty")
        require(all(isinstance(item, str) and item.strip() for item in value), f"{label} entries must be non-empty strings")
        return list(value)
    raise ActualModelGateError(f"{label} must be a non-empty string or string array")


def require_git_sha(data: dict[str, Any], label: str, expected_sha: str) -> str:
    value = require_string(data, "git_sha", label)
    require(GIT_SHA_RE.match(value), f"{label}.git_sha must be a 40-character SHA")
    require(value.lower() == expected_sha.lower(), f"{label}.git_sha {value} is stale vs HEAD {expected_sha}")
    return value


def resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def pass_line_is_real(value: Any, label: str) -> str:
    require(isinstance(value, str) and " PASS:" in value, f"{label}.pass_line must be a gate PASS line")
    prefix = value.split(":", 1)[0].upper()
    require("SELFTEST" not in prefix and "SELF-TEST" not in prefix, f"{label}.pass_line must not be selftest evidence")
    return value


def validate_backend_resolution(artifact: dict[str, Any], label: str, backend: str) -> tuple[str, str]:
    requested = require_string(artifact, "requested_backend", label)
    effective = require_string(artifact, "effective_backend", label)
    require(effective == backend, f"{label}.effective_backend must be {backend}")
    require(
        requested == "auto" or requested == effective,
        f"{label}.requested_backend {requested!r} does not match effective_backend {effective!r}",
    )
    return requested, effective


def artifact_from_dir(path: Path) -> dict[str, Any]:
    manifest = read_json(path / "gate.manifest.json")
    summary_path = manifest.get("summary")
    if summary_path is None:
        outputs = manifest.get("outputs")
        if isinstance(outputs, dict):
            summary_path = outputs.get("summary")
    extra: dict[str, Any] = {}
    if isinstance(summary_path, str) and summary_path.strip():
        candidate = resolve_path(summary_path, base=path)
        if candidate.is_file():
            extra = read_json(candidate)
    return {
        **extra,
        "status": manifest.get("status", extra.get("status")),
        "git_sha": manifest.get("git_sha", extra.get("git_sha")),
        "artifact_dir": manifest.get("artifact_dir", str(path)),
        "pass_line": manifest.get("pass_line", extra.get("pass_line")),
    }


def load_artifact(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return artifact_from_dir(path)
    return read_json(path)


def resolve_artifact_dir(raw: str, *, artifact_source: Path, label: str) -> str:
    base = artifact_source if artifact_source.is_dir() else artifact_source.parent
    path = resolve_path(raw, base=base)
    require(path.is_dir(), f"{label}.artifact_dir must exist and be a directory: {path}")
    return str(path)


def validate_replay_index(value: Any, *, artifact_dir: str, label: str) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label}.replay_bundle_index must be a list when present")
    entries: list[dict[str, Any]] = []
    base = Path(artifact_dir)
    for index, entry in enumerate(value):
        entry_label = f"{label}.replay_bundle_index[{index}]"
        require(isinstance(entry, dict), f"{entry_label} must be an object")
        for key in ("request_id", "replay_command", "bundle_dir"):
            require(
                isinstance(entry.get(key), str) and entry[key].strip(),
                f"{entry_label}.{key} must be non-empty",
            )
        bundle_dir = resolve_path(entry["bundle_dir"], base=base)
        require(bundle_dir.is_dir(), f"{entry_label}.bundle_dir must exist: {bundle_dir}")
        entries.append({**entry, "bundle_dir": str(bundle_dir)})
    return entries


def validate_l2_artifact(
    path: Path,
    *,
    key: str,
    backend: str,
    expected_sha: str,
) -> dict[str, Any]:
    artifact = load_artifact(path)
    label = f"{key}"
    require(artifact.get("status") == "pass", f"{label}.status must be pass")
    require_git_sha(artifact, label, expected_sha)
    require(artifact.get("backend") == backend, f"{label}.backend must be {backend}")
    requested_backend, effective_backend = validate_backend_resolution(artifact, label, backend)
    entrypoints = set(artifact.get("entrypoints") or [])
    missing = sorted(REQUIRED_ENTRYPOINTS - entrypoints)
    require(not missing, f"{label}.entrypoints missing {missing}")
    artifact_dir = resolve_artifact_dir(
        require_string(artifact, "artifact_dir", label),
        artifact_source=path,
        label=label,
    )
    pass_line = pass_line_is_real(artifact.get("pass_line"), label)
    model_id = require_string(artifact, "model_id", label)
    architecture = require_string(artifact, "architecture", label)
    require(isinstance(artifact.get("git_dirty"), bool), f"{label}.git_dirty must be boolean")
    dirty_files = require_string_list(artifact.get("dirty_files", []), f"{label}.dirty_files")
    require(
        artifact["git_dirty"] is False,
        f"{label}.git_dirty must be false for release-regression-hardening L2 evidence",
    )
    require(not dirty_files, f"{label}.dirty_files must be empty for L2 evidence")
    command = normalize_command(artifact.get("command") or artifact.get("command_line"), f"{label}.command")
    profile_detail = artifact.get("profile_detail") or artifact.get("observability_profile_detail")
    require(isinstance(profile_detail, str) and profile_detail.strip(), f"{label}.profile_detail must be non-empty")
    replay_index = validate_replay_index(
        artifact.get("replay_bundle_index", []),
        artifact_dir=artifact_dir,
        label=label,
    )
    return {
        "status": "pass",
        "backend": backend,
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "git_sha": artifact["git_sha"],
        "git_dirty": artifact["git_dirty"],
        "dirty_files": dirty_files,
        "artifact_dir": artifact_dir,
        "pass_line": pass_line,
        "model_id": model_id,
        "architecture": architecture,
        "entrypoints": sorted(entrypoints),
        "command": command,
        "profile_detail": profile_detail.strip(),
        "replay_bundle_index": replay_index,
        "source": str(path),
    }


def validate_native_operator_selection(
    *,
    selected: bool | None,
    cuda_artifact: Path | None,
    non_selected_reason: str | None,
    expected_sha: str,
) -> dict[str, Any]:
    require(selected is not None, "--native-operator-selected or --native-operator-not-selected is required")
    if selected:
        require(cuda_artifact is not None, "native operator selected path requires --native-operator-cuda-artifact")
        artifact = load_artifact(cuda_artifact)
        require(artifact.get("status") == "pass", "native_operator_selection.cuda_artifact.status must be pass")
        require_git_sha(artifact, "native_operator_selection.cuda_artifact", expected_sha)
        require(artifact.get("backend") == "cuda", "native_operator_selection.cuda_artifact.backend must be cuda")
        return {
            "status": "pass",
            "selected": True,
            "cuda_artifact": str(cuda_artifact),
            "git_sha": artifact["git_sha"],
        }
    require(
        isinstance(non_selected_reason, str) and non_selected_reason.strip(),
        "native operator non-selected path requires --native-operator-non-selected-reason",
    )
    return {
        "status": "pass",
        "selected": False,
        "non_selected_reason": non_selected_reason.strip(),
        "git_sha": expected_sha,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    current_sha = head_sha()
    expected_sha = args.git_sha or current_sha
    require(GIT_SHA_RE.match(expected_sha), "--git-sha must be a 40-character SHA")
    require(
        expected_sha.lower() == current_sha.lower(),
        f"--git-sha {expected_sha} is stale vs current HEAD {current_sha}",
    )
    selected = None
    if args.native_operator_selected:
        selected = True
    if args.native_operator_not_selected:
        require(selected is None, "native operator selection flags are mutually exclusive")
        selected = False
    metal = validate_l2_artifact(
        args.metal_l2_artifact,
        key="metal_l2_artifact",
        backend="metal",
        expected_sha=expected_sha,
    )
    cuda = validate_l2_artifact(
        args.cuda_l2_artifact,
        key="cuda_l2_artifact",
        backend="cuda",
        expected_sha=expected_sha,
    )
    native_selection = validate_native_operator_selection(
        selected=selected,
        cuda_artifact=args.native_operator_cuda_artifact,
        non_selected_reason=args.native_operator_non_selected_reason,
        expected_sha=expected_sha,
    )
    dirty_files = git_value(["status", "--short"], default="").splitlines()
    pass_line = f"{PASS_LINE}: {out}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "actual_model_regression_summary",
        "git_sha": expected_sha,
        "pass_line": pass_line,
        "metal_l2_artifact": metal,
        "cuda_l2_artifact": cuda,
        "native_operator_selection": native_selection,
    }
    write_json(out / "actual_model_regression_summary.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "actual_model_regression_summary",
            "status": "pass",
            "repo_root": str(REPO_ROOT),
            "git_sha": current_sha,
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "outputs": {"summary": str(out / "actual_model_regression_summary.json")},
            "validation_summary": summary,
        },
    )
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    return summary


def make_l2_artifact(
    root: Path,
    *,
    backend: str,
    sha: str,
    suffix: str = "l2",
    entrypoints: list[str] | None = None,
) -> Path:
    artifact_dir = root / f"{backend}_{suffix}_artifact"
    request_dump_dir = artifact_dir / "request_dump"
    request_dump_dir.mkdir(parents=True, exist_ok=True)
    path = root / f"{backend}_{suffix}.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "status": "pass",
            "backend": backend,
            "requested_backend": backend,
            "effective_backend": backend,
            "git_sha": sha,
            "git_dirty": False,
            "dirty_files": [],
            "artifact_dir": str(artifact_dir),
            "pass_line": f"{backend.upper()} L2 ACTUAL MODEL PASS: fixtures/{backend}-l2",
            "model_id": f"fixture/{backend}-model",
            "architecture": "llama_dense" if backend == "metal" else "qwen3_moe",
            "entrypoints": entrypoints or sorted(REQUIRED_ENTRYPOINTS),
            "command": [
                "ferrum",
                "run",
                f"fixture/{backend}-model",
                "--profile-detail",
                "basic",
            ],
            "profile_detail": "basic",
            "replay_bundle_index": [
                {
                    "request_id": f"req-{backend}-fixture",
                    "entrypoint": "run",
                    "replay_command": f"ferrum run fixture/{backend}-model",
                    "bundle_dir": str(request_dump_dir),
                }
            ],
        },
    )
    return path


def make_native_artifact(root: Path, sha: str) -> Path:
    path = root / "native_cuda.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "status": "pass",
            "backend": "cuda",
            "git_sha": sha,
            "artifact_dir": "fixtures/native-cuda",
            "pass_line": "NATIVE OP ARTIFACT PASS: fixtures/native-cuda",
        },
    )
    return path


def run_selftest() -> dict[str, Any]:
    sha = head_sha()
    with tempfile.TemporaryDirectory(prefix="ferrum-actual-model-regression-") as tmp:
        root = Path(tmp)
        metal = make_l2_artifact(root, backend="metal", sha=sha)
        cuda = make_l2_artifact(root, backend="cuda", sha=sha)
        native = make_native_artifact(root, sha)
        out = root / "out-selected"
        run_gate(
            argparse.Namespace(
                out=out,
                git_sha=sha,
                metal_l2_artifact=metal,
                cuda_l2_artifact=cuda,
                native_operator_selected=True,
                native_operator_not_selected=False,
                native_operator_cuda_artifact=native,
                native_operator_non_selected_reason=None,
            )
        )
        run_gate(
            argparse.Namespace(
                out=root / "out-non-selected",
                git_sha=sha,
                metal_l2_artifact=metal,
                cuda_l2_artifact=cuda,
                native_operator_selected=False,
                native_operator_not_selected=True,
                native_operator_cuda_artifact=None,
                native_operator_non_selected_reason="FA2 disabled for this fixture",
            )
        )
        bad_cuda = make_l2_artifact(
            root,
            backend="cuda",
            sha=sha,
            suffix="missing_entrypoints",
            entrypoints=["run", "serve"],
        )
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-entrypoints",
                    git_sha=sha,
                    metal_l2_artifact=metal,
                    cuda_l2_artifact=bad_cuda,
                    native_operator_selected=True,
                    native_operator_not_selected=False,
                    native_operator_cuda_artifact=native,
                    native_operator_non_selected_reason=None,
                )
            )
            raise AssertionError("missing stream/basic_concurrency unexpectedly passed")
        except ActualModelGateError as exc:
            require("entrypoints" in str(exc), f"unexpected entrypoint failure: {exc}")
        missing_command = make_l2_artifact(root, backend="cuda", sha=sha, suffix="missing_command")
        missing_command_data = read_json(missing_command)
        missing_command_data.pop("command", None)
        write_json(missing_command, missing_command_data)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-missing-command",
                    git_sha=sha,
                    metal_l2_artifact=metal,
                    cuda_l2_artifact=missing_command,
                    native_operator_selected=False,
                    native_operator_not_selected=True,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason="fixture",
                )
            )
            raise AssertionError("missing L2 command unexpectedly passed")
        except ActualModelGateError as exc:
            require("command" in str(exc), f"unexpected missing command failure: {exc}")
        stale = make_l2_artifact(root, backend="metal", sha="2" * 40, suffix="stale")
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-stale",
                    git_sha=sha,
                    metal_l2_artifact=stale,
                    cuda_l2_artifact=cuda,
                    native_operator_selected=False,
                    native_operator_not_selected=True,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason="fixture",
                )
            )
            raise AssertionError("stale artifact unexpectedly passed")
        except ActualModelGateError as exc:
            require("stale" in str(exc), f"unexpected stale failure: {exc}")
        dirty = make_l2_artifact(root, backend="metal", sha=sha, suffix="dirty")
        dirty_data = read_json(dirty)
        dirty_data["git_dirty"] = True
        dirty_data["dirty_files"] = [" M crates/ferrum-cli/src/commands/run.rs"]
        write_json(dirty, dirty_data)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-dirty",
                    git_sha=sha,
                    metal_l2_artifact=dirty,
                    cuda_l2_artifact=cuda,
                    native_operator_selected=False,
                    native_operator_not_selected=True,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason="fixture",
                )
            )
            raise AssertionError("dirty artifact unexpectedly passed")
        except ActualModelGateError as exc:
            require("git_dirty" in str(exc), f"unexpected dirty failure: {exc}")
        fallback = make_l2_artifact(root, backend="metal", sha=sha, suffix="backend_fallback")
        fallback_data = read_json(fallback)
        fallback_data["requested_backend"] = "metal"
        fallback_data["effective_backend"] = "cpu"
        write_json(fallback, fallback_data)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-backend-fallback",
                    git_sha=sha,
                    metal_l2_artifact=fallback,
                    cuda_l2_artifact=cuda,
                    native_operator_selected=False,
                    native_operator_not_selected=True,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason="fixture",
                )
            )
            raise AssertionError("backend fallback artifact unexpectedly passed")
        except ActualModelGateError as exc:
            require("effective_backend" in str(exc), f"unexpected backend fallback failure: {exc}")
        missing_artifact_dir = make_l2_artifact(
            root,
            backend="metal",
            sha=sha,
            suffix="missing_artifact_dir",
        )
        missing_artifact_dir_data = read_json(missing_artifact_dir)
        missing_artifact_dir_data["artifact_dir"] = str(root / "does-not-exist")
        write_json(missing_artifact_dir, missing_artifact_dir_data)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-missing-artifact-dir",
                    git_sha=sha,
                    metal_l2_artifact=missing_artifact_dir,
                    cuda_l2_artifact=cuda,
                    native_operator_selected=False,
                    native_operator_not_selected=True,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason="fixture",
                )
            )
            raise AssertionError("missing artifact_dir unexpectedly passed")
        except ActualModelGateError as exc:
            require("artifact_dir" in str(exc), f"unexpected missing artifact_dir failure: {exc}")
        missing_replay_bundle = make_l2_artifact(
            root,
            backend="metal",
            sha=sha,
            suffix="missing_replay_bundle",
        )
        missing_replay_bundle_data = read_json(missing_replay_bundle)
        missing_replay_bundle_data["replay_bundle_index"][0]["bundle_dir"] = str(
            root / "missing-replay-bundle"
        )
        write_json(missing_replay_bundle, missing_replay_bundle_data)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-missing-replay-bundle",
                    git_sha=sha,
                    metal_l2_artifact=missing_replay_bundle,
                    cuda_l2_artifact=cuda,
                    native_operator_selected=False,
                    native_operator_not_selected=True,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason="fixture",
                )
            )
            raise AssertionError("missing replay bundle unexpectedly passed")
        except ActualModelGateError as exc:
            require("bundle_dir" in str(exc), f"unexpected missing replay bundle failure: {exc}")
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-native-selection",
                    git_sha=sha,
                    metal_l2_artifact=metal,
                    cuda_l2_artifact=cuda,
                    native_operator_selected=True,
                    native_operator_not_selected=False,
                    native_operator_cuda_artifact=None,
                    native_operator_non_selected_reason=None,
                )
            )
            raise AssertionError("selected native operator without artifact unexpectedly passed")
        except ActualModelGateError as exc:
            message = str(exc)
            require(
                "cuda_artifact" in message or "cuda-artifact" in message,
                f"unexpected native selection failure: {exc}",
            )
        return {"schema_version": SCHEMA_VERSION, "status": "pass"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--metal-l2-artifact", type=Path)
    parser.add_argument("--cuda-l2-artifact", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--native-operator-selected", action="store_true")
    group.add_argument("--native-operator-not-selected", action="store_true")
    parser.add_argument("--native-operator-cuda-artifact", type=Path)
    parser.add_argument("--native-operator-non-selected-reason")
    return parser.parse_args()


def require_normal_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in [
            ("--out", args.out),
            ("--metal-l2-artifact", args.metal_l2_artifact),
            ("--cuda-l2-artifact", args.cuda_l2_artifact),
        ]
        if value is None
    ]
    if missing:
        raise ActualModelGateError(f"missing required args: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        require_normal_args(args)
        summary = run_gate(args)
        print(summary["pass_line"])
        return 0
    except ActualModelGateError as exc:
        print(f"ACTUAL MODEL REGRESSION SUMMARY FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
