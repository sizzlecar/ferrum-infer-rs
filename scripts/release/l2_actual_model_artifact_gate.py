#!/usr/bin/env python3
"""Normalize real Metal/CUDA release artifacts into WP14 L2 model evidence."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from request_replay_bundle_gate import BundleError, make_bundle, validate_bundle_root


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_LINE_SUFFIX = "L2 ACTUAL MODEL PASS"
SELFTEST_PASS_LINE = "L2 ACTUAL MODEL ARTIFACT SELFTEST PASS"
SCHEMA_VERSION = 1
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
REQUIRED_ENTRYPOINTS = ["basic_concurrency", "run", "serve", "stream"]
ALLOWED_ARCHITECTURES = {
    "cuda": {"llama_dense", "qwen3_moe"},
    "metal": {"llama_dense", "qwen3", "qwen3_moe"},
}


class L2ArtifactError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise L2ArtifactError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise L2ArtifactError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise L2ArtifactError(f"{path}: expected JSON object")
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
        raise L2ArtifactError(f"current HEAD is not a git SHA: {value!r}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise L2ArtifactError(message)


def require_pass_line(value: Any, label: str) -> str:
    require(isinstance(value, str) and " PASS:" in value, f"{label}.pass_line must be a real PASS line")
    require("SELFTEST" not in value.upper(), f"{label}.pass_line must not be selftest evidence")
    return value


def require_source_sha(manifest: dict[str, Any], expected_sha: str, label: str) -> str:
    value = manifest.get("git_sha")
    require(isinstance(value, str) and GIT_SHA_RE.match(value), f"{label}.git_sha must be a 40-character SHA")
    require(value.lower() == expected_sha.lower(), f"{label}.git_sha {value} is stale vs HEAD {expected_sha}")
    return value


def source_dirty_files(manifest: dict[str, Any], *, source_artifact: Path) -> list[str]:
    raw: list[str] = []
    dirty_status = manifest.get("dirty_status")
    if isinstance(dirty_status, dict) and isinstance(dirty_status.get("status_short"), list):
        raw.extend(str(item) for item in dirty_status["status_short"])
    elif isinstance(manifest.get("dirty_files"), list):
        raw.extend(str(item) for item in manifest["dirty_files"])
    elif manifest.get("git_dirty") is True:
        raw.append("unknown")

    artifact_rel = source_artifact.resolve().relative_to(REPO_ROOT).as_posix() if source_artifact.resolve().is_relative_to(REPO_ROOT) else None
    filtered: list[str] = []
    for line in raw:
        path = line[3:] if len(line) > 3 and line[:2].strip() else line
        if artifact_rel and (path == artifact_rel or path.startswith(artifact_rel.rstrip("/") + "/")):
            continue
        filtered.append(line)
    return filtered


def load_source_manifest(source_artifact: Path, *, expected_sha: str) -> dict[str, Any]:
    manifest = read_json(source_artifact / "gate.manifest.json")
    require(manifest.get("status") == "pass", "source gate.manifest.status must be pass")
    require_pass_line(manifest.get("pass_line"), "source gate.manifest")
    require_source_sha(manifest, expected_sha, "source gate.manifest")
    dirty = source_dirty_files(manifest, source_artifact=source_artifact)
    require(not dirty, f"source artifact has dirty files outside artifact dir: {dirty}")
    return manifest


def infer_metal_architecture(model: dict[str, Any]) -> str:
    if model.get("moe") is True:
        return "qwen3_moe"
    text = f"{model.get('key', '')} {model.get('label', '')} {model.get('gguf', '')}".lower()
    if "qwen3" in text:
        return "qwen3"
    return "llama_dense"


def passed(value: Any) -> bool:
    return isinstance(value, dict) and value.get("passed") is True


def normalize_command_file(path: Path) -> str:
    if not path.exists():
        return str(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return str(path)
    if isinstance(data, list) and all(isinstance(part, str) for part in data):
        return " ".join(data)
    return str(path)


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "bundle"


def materialize_replay_bundle_index(replay_roots: list[Path], out: Path) -> list[dict[str, Any]]:
    require(replay_roots, "at least one --replay-bundle is required for L2 actual model evidence")
    replay_out = out / "replay_bundles"
    replay_out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for root_index, replay_root in enumerate(replay_roots):
        try:
            bundles = validate_bundle_root(replay_root)
        except BundleError as exc:
            raise L2ArtifactError(f"{replay_root}: invalid replay bundle: {exc}") from exc
        require(bundles, f"{replay_root}: replay bundle root must contain at least one bundle")
        for bundle_index, bundle in enumerate(bundles):
            request_id = str(bundle.get("request_id") or f"request-{root_index}-{bundle_index}")
            entrypoint = str(bundle.get("entrypoint") or "unknown")
            source_bundle = Path(str(bundle["bundle_dir"])).resolve()
            dest = replay_out / (
                f"{root_index:02d}_{bundle_index:02d}_"
                f"{safe_path_part(entrypoint)}_{safe_path_part(request_id)}"
            )
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source_bundle, dest)
            try:
                copied = validate_bundle_root(dest)
            except BundleError as exc:
                raise L2ArtifactError(f"{dest}: copied replay bundle is invalid: {exc}") from exc
            require(copied, f"{dest}: copied replay bundle root must contain at least one bundle")
            replay = read_json(dest / "replay.command.json")
            command = replay.get("command")
            require(
                isinstance(command, str) and command.strip(),
                f"{dest / 'replay.command.json'}.command must be non-empty",
            )
            copied_bundle = copied[0]
            entries.append(
                {
                    "request_id": str(copied_bundle.get("request_id") or request_id),
                    "entrypoint": str(copied_bundle.get("entrypoint") or entrypoint),
                    "replay_command": command,
                    "bundle_dir": str(dest),
                    "source_bundle_dir": str(source_bundle),
                }
            )
    return entries


def metal_l2(args: argparse.Namespace, source: Path, source_manifest: dict[str, Any], expected_sha: str) -> dict[str, Any]:
    summary_path = source / "metal-readme" / "summary.json"
    summary = read_json(summary_path)
    model_key = args.model_key
    require(isinstance(model_key, str) and model_key.strip(), "--model-key is required for metal source artifacts")
    models = summary.get("models")
    require(isinstance(models, list), f"{summary_path}.models must be a list")
    model = next((item for item in models if isinstance(item, dict) and item.get("key") == model_key), None)
    require(model is not None, f"metal summary does not contain model key {model_key!r}")

    require(passed(model.get("run")), f"metal {model_key}.run must pass")
    require(model.get("server_ready") is True, f"metal {model_key}.server_ready must be true")
    require(passed(model.get("serve_startup")), f"metal {model_key}.serve_startup must pass")
    chat = model.get("chat")
    require(isinstance(chat, dict), f"metal {model_key}.chat must be an object")
    for key in ("paris", "multiturn", "stream", "stateful_loop"):
        require(passed(chat.get(key)), f"metal {model_key}.chat.{key} must pass")
    concurrency_files = sorted((source / "metal-readme").glob(f"{model_key}.c*.json"))
    require(concurrency_files, f"metal {model_key} must include at least one concurrency bench JSON")

    architecture = args.architecture or infer_metal_architecture(model)
    require(architecture in ALLOWED_ARCHITECTURES["metal"], f"metal architecture {architecture!r} is not allowed")
    model_id = args.model_id or str(model.get("gguf") or model.get("label") or model_key)
    commands = [
        " ".join(source_manifest.get("command_line") or []),
        normalize_command_file(source / "metal-readme" / f"{model_key}.server_cmd.json"),
        f"ferrum run {model_id}",
        f"ferrum bench-serve {model_id}",
    ]
    return {
        "source_summary": str(summary_path),
        "model_key": model_key,
        "model_id": model_id,
        "architecture": architecture,
        "entrypoints": REQUIRED_ENTRYPOINTS,
        "command": [command for command in commands if command.strip()],
        "checks": {
            "run": "pass",
            "serve": "pass",
            "stream": "pass",
            "basic_concurrency": "pass",
            "concurrency_files": [str(path) for path in concurrency_files],
        },
    }


def cuda_l2(args: argparse.Namespace, source: Path, source_manifest: dict[str, Any], expected_sha: str) -> dict[str, Any]:
    gate = read_json(source / "gate.json")
    require(gate.get("status") == "pass", "cuda gate.json.status must be pass")
    checks = gate.get("checks")
    require(isinstance(checks, dict), "cuda gate.json.checks must be an object")
    require(passed(checks.get("run")), "cuda run check must pass")
    serve = checks.get("serve")
    require(isinstance(serve, dict), "cuda serve check must be an object")
    require(passed(serve.get("math")) and passed(serve.get("multi_turn")), "cuda serve math/multi_turn must pass")
    require(passed(serve.get("stream_usage")), "cuda stream_usage must pass")
    require((checks.get("concurrency_quality_regression") or {}).get("status") == "pass", "cuda concurrency quality must pass")
    require(passed(checks.get("bench_serve")), "cuda bench_serve must pass")
    for rel in [
        "run.command.json",
        "serve.command.json",
        "serve.stream.sse",
        "concurrency-quality-regression/concurrency_quality_regression.json",
        "bench-serve.command.json",
        "bench-serve.json",
    ]:
        require((source / rel).exists(), f"cuda source artifact missing {rel}")

    metadata = read_json(source / "metadata.json") if (source / "metadata.json").is_file() else {}
    architecture = args.architecture or metadata.get("model_architecture") or "llama_dense"
    require(architecture in ALLOWED_ARCHITECTURES["cuda"], f"cuda architecture {architecture!r} is not allowed")
    model_id = args.model_id or str(gate.get("model") or metadata.get("model") or source_manifest.get("model") or "cuda-model")
    commands = [
        " ".join(source_manifest.get("command_line") or []),
        normalize_command_file(source / "run.command.json"),
        normalize_command_file(source / "serve.command.json"),
        normalize_command_file(source / "bench-serve.command.json"),
    ]
    return {
        "source_summary": str(source / "gate.json"),
        "model_key": source_manifest.get("lane") or gate.get("lane") or "cuda",
        "model_id": model_id,
        "architecture": architecture,
        "entrypoints": REQUIRED_ENTRYPOINTS,
        "command": [command for command in commands if command.strip()],
        "checks": {
            "run": "pass",
            "serve": "pass",
            "stream": "pass",
            "basic_concurrency": "pass",
        },
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    source = args.source_artifact.resolve()
    require(source.is_dir(), f"--source-artifact must be a directory: {source}")
    expected_sha = args.git_sha or head_sha()
    require(GIT_SHA_RE.match(expected_sha), "--git-sha must be a 40-character SHA")
    require(expected_sha.lower() == head_sha().lower(), f"--git-sha {expected_sha} is stale vs current HEAD")
    source_manifest = load_source_manifest(source, expected_sha=expected_sha)
    backend = args.backend
    require(backend in {"cuda", "metal"}, "--backend must be cuda or metal")
    if backend == "metal":
        normalized = metal_l2(args, source, source_manifest, expected_sha)
    else:
        normalized = cuda_l2(args, source, source_manifest, expected_sha)
    replay_bundle_index = materialize_replay_bundle_index(args.replay_bundle, out)

    current_dirty = git_value(["status", "--short"], default="").splitlines()
    pass_line = f"{backend.upper()} {PASS_LINE_SUFFIX}: {out}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "l2_actual_model_artifact",
        "backend": backend,
        "requested_backend": backend,
        "effective_backend": backend,
        "git_sha": expected_sha,
        "git_dirty": False,
        "dirty_files": [],
        "artifact_dir": str(out),
        "pass_line": pass_line,
        "model_id": normalized["model_id"],
        "model_key": normalized["model_key"],
        "architecture": normalized["architecture"],
        "entrypoints": normalized["entrypoints"],
        "command": normalized["command"],
        "profile_detail": args.profile_detail,
        "replay_bundle_index": replay_bundle_index,
        "source_artifact": str(source),
        "source_pass_line": source_manifest.get("pass_line"),
        "source_summary": normalized["source_summary"],
        "checks": normalized["checks"],
    }
    write_json(out / "l2_actual_model_artifact.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "l2_actual_model_artifact",
            "status": "pass",
            "repo_root": str(REPO_ROOT),
            "git_sha": expected_sha,
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(current_dirty),
            "dirty_files": current_dirty,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "outputs": {"summary": str(out / "l2_actual_model_artifact.json")},
            "validation_summary": summary,
        },
    )
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    return summary


def make_source_manifest(root: Path, *, backend: str, sha: str) -> None:
    write_json(
        root / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "lane": backend if backend == "metal" else "cuda-llama-dense",
            "artifact_dir": str(root),
            "pass_line": f"FERRUM GATE {backend} PASS: {root}",
            "git_sha": sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "command_line": ["python3", "scripts/release/run_gate.py", backend, "--out", str(root)],
        },
    )


def make_metal_source(root: Path, sha: str) -> Path:
    make_source_manifest(root, backend="metal", sha=sha)
    readme = root / "metal-readme"
    readme.mkdir(parents=True)
    write_json(
        readme / "summary.json",
        {
            "models": [
                {
                    "key": "llama31_8b",
                    "label": "Llama-3.1-8B",
                    "gguf": "/models/llama31-8b.gguf",
                    "moe": False,
                    "run": {"passed": True},
                    "server_ready": True,
                    "serve_startup": {"passed": True},
                    "chat": {
                        "paris": {"passed": True},
                        "multiturn": {"passed": True},
                        "stream": {"passed": True},
                        "stateful_loop": {"passed": True},
                    },
                }
            ]
        },
    )
    write_json(readme / "llama31_8b.c1.json", {"concurrency": 1})
    write_json(readme / "llama31_8b.server_cmd.json", ["ferrum", "serve", "/models/llama31-8b.gguf"])
    return root


def make_cuda_source(root: Path, sha: str) -> Path:
    make_source_manifest(root, backend="cuda", sha=sha)
    write_json(
        root / "gate.json",
        {
            "status": "pass",
            "lane": "g0_cuda4090_llama_dense",
            "model": "fixture/llama-dense",
            "checks": {
                "run": {"passed": True},
                "serve": {
                    "math": {"passed": True},
                    "multi_turn": {"passed": True},
                    "stream_usage": {"passed": True},
                },
                "concurrency_quality_regression": {"status": "pass"},
                "bench_serve": {"passed": True},
            },
        },
    )
    write_json(root / "metadata.json", {"model": "fixture/llama-dense", "model_architecture": "llama_dense"})
    write_json(root / "run.command.json", ["ferrum", "run", "fixture/llama-dense"])
    write_json(root / "serve.command.json", ["ferrum", "serve", "--model", "fixture/llama-dense"])
    write_json(root / "bench-serve.command.json", ["ferrum", "bench-serve", "--model", "fixture/llama-dense"])
    write_json(root / "bench-serve.json", {"status": "pass"})
    (root / "serve.stream.sse").write_text("data: {}\n\ndata: [DONE]\n", encoding="utf-8")
    write_json(root / "concurrency-quality-regression/concurrency_quality_regression.json", {"status": "pass"})
    return root


def run_selftest() -> dict[str, Any]:
    sha = head_sha()
    with tempfile.TemporaryDirectory(prefix="ferrum-l2-actual-model-") as tmp:
        root = Path(tmp)
        metal_source = make_metal_source(root / "metal-source", sha)
        cuda_source = make_cuda_source(root / "cuda-source", sha)
        metal_replay = root / "metal-replay"
        cuda_replay = root / "cuda-replay"
        make_bundle(metal_replay, entrypoint="run")
        make_bundle(cuda_replay, entrypoint="serve", serve_replay=True)
        metal = run_gate(
            argparse.Namespace(
                out=root / "metal-out",
                source_artifact=metal_source,
                backend="metal",
                git_sha=sha,
                model_key="llama31_8b",
                model_id=None,
                architecture=None,
                profile_detail="release-gate",
                replay_bundle=[metal_replay],
            )
        )
        cuda = run_gate(
            argparse.Namespace(
                out=root / "cuda-out",
                source_artifact=cuda_source,
                backend="cuda",
                git_sha=sha,
                model_key=None,
                model_id=None,
                architecture=None,
                profile_detail="release-gate",
                replay_bundle=[cuda_replay],
            )
        )
        require(metal["backend"] == "metal", "metal normalization failed")
        require(cuda["backend"] == "cuda", "cuda normalization failed")
        summary_out = root / "actual-model-summary"
        summary_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/release/actual_model_regression_summary_gate.py"),
                "--out",
                str(summary_out),
                "--metal-l2-artifact",
                str(root / "metal-out"),
                "--cuda-l2-artifact",
                str(root / "cuda-out"),
                "--native-operator-not-selected",
                "--native-operator-non-selected-reason",
                "selftest fixture does not select FA2",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(summary_proc.returncode == 0, summary_proc.stderr or summary_proc.stdout)
        require(
            "ACTUAL MODEL REGRESSION SUMMARY PASS" in summary_proc.stdout,
            summary_proc.stdout,
        )

        stale_source = make_cuda_source(root / "stale-cuda-source", "1" * 40)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "stale-out",
                    source_artifact=stale_source,
                    backend="cuda",
                    git_sha=sha,
                    model_key=None,
                    model_id=None,
                    architecture=None,
                    profile_detail="release-gate",
                    replay_bundle=[cuda_replay],
                )
            )
            raise AssertionError("stale source artifact unexpectedly passed")
        except L2ArtifactError as exc:
            require("stale" in str(exc), f"unexpected stale failure: {exc}")

        broken_metal = make_metal_source(root / "broken-metal-source", sha)
        summary = read_json(broken_metal / "metal-readme/summary.json")
        summary["models"][0]["chat"]["stream"]["passed"] = False
        write_json(broken_metal / "metal-readme/summary.json", summary)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "broken-metal-out",
                    source_artifact=broken_metal,
                    backend="metal",
                    git_sha=sha,
                    model_key="llama31_8b",
                    model_id=None,
                    architecture=None,
                    profile_detail="release-gate",
                    replay_bundle=[metal_replay],
                )
            )
            raise AssertionError("broken metal stream unexpectedly passed")
        except L2ArtifactError as exc:
            require("stream" in str(exc), f"unexpected stream failure: {exc}")
    return {"status": "pass"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--backend", choices=["cuda", "metal"])
    parser.add_argument("--git-sha")
    parser.add_argument("--model-key")
    parser.add_argument("--model-id")
    parser.add_argument("--architecture")
    parser.add_argument("--profile-detail", default="release-gate")
    parser.add_argument("--replay-bundle", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        for name in ("out", "source_artifact", "backend"):
            if getattr(args, name) is None:
                raise L2ArtifactError(f"--{name.replace('_', '-')} is required unless --self-test is set")
        run_gate(args)
        print(f"{args.backend.upper()} {PASS_LINE_SUFFIX}: {args.out}")
        return 0
    except L2ArtifactError as exc:
        print(f"L2 ACTUAL MODEL ARTIFACT FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
