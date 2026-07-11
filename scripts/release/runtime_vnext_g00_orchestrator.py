#!/usr/bin/env python3
"""Prepare and resume the Runtime vNext G00 evidence root.

This program is intentionally orchestration-only. It never resolves a model over
the network, downloads weights, starts a product process, or allocates GPU work.
It turns already-collected immutable model and hardware facts into the native G00
``models.lock.json`` contract and writes a hash-bound phase plan for the six
primary product lanes.

The final source of truth remains ``runtime_vnext_baseline_gate.py``. A prepared
root is incomplete until that validator emits its exact PASS line.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import runtime_vnext_baseline_gate as baseline
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_baseline_gate as baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()
ORCHESTRATION_DIR = "orchestration"
ORCHESTRATION_MANIFEST = f"{ORCHESTRATION_DIR}/manifest.json"
PHASE_INPUTS_PATH = f"{ORCHESTRATION_DIR}/phase-inputs.json"
ROOT_MANIFEST = "manifest.json"
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "runtime_vnext_g00_orchestration"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 ORCHESTRATOR SELFTEST PASS"


class OrchestrationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OrchestrationError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"cannot read strict JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OrchestrationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, canonical_json_bytes(value))


def atomic_copy(source: Path, destination: Path) -> None:
    require(source.is_file() and not source.is_symlink(), f"input artifact must be a regular file: {source}")
    if source.resolve() == destination.resolve(strict=False):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def checked_artifact(root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, str) and bool(raw.strip()), f"{label} must be a non-empty relative path")
    path = Path(raw)
    require(not path.is_absolute(), f"{label} must be relative")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise OrchestrationError(f"{label} escapes its artifact directory: {raw}") from exc
    return resolved


def external_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return root
    raise OrchestrationError(f"G00 output must be outside the source worktree: {root}")


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def collector_identity() -> dict[str, Any]:
    status = run_git("status", "--short").splitlines()
    return {
        "git_sha": run_git("rev-parse", "HEAD"),
        "tree_sha": run_git("rev-parse", "HEAD^{tree}"),
        "dirty_status": {"is_dirty": bool(status), "status_short": status},
    }


def file_ref(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"artifact is not a regular file: {path}")
    rendered = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
    return {"path": rendered, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def normalized_file_locks(raw: Any, label: str) -> list[dict[str, Any]]:
    require(isinstance(raw, list) and raw, f"{label} must be a non-empty list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        require(isinstance(item, dict), f"{label}[{index}] must be an object")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        require(isinstance(path, str) and path and path not in seen, f"{label}[{index}].path is invalid or duplicate")
        require(isinstance(digest, str) and baseline.SHA256_RE.fullmatch(digest) is not None, f"{label}[{index}].sha256 is invalid")
        require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}[{index}].size_bytes is invalid")
        seen.add(path)
        rows.append({"path": path, "sha256": digest, "size_bytes": size})
    return sorted(rows, key=lambda row: row["path"])


def ordered_file_locks(raw: Any, label: str) -> list[dict[str, Any]]:
    """Keep resolver order while reducing rows to the lock's public identity."""
    normalized = {row["path"]: row for row in normalized_file_locks(raw, label)}
    return [copy.deepcopy(normalized[str(row["path"])]) for row in raw]


def input_probe_bundle(probe_path: Path) -> dict[str, Any]:
    probe_path = probe_path.expanduser().resolve()
    probe = read_json(probe_path)
    commands = probe.get("commands")
    require(isinstance(commands, list) and commands, f"{probe_path}.commands must be non-empty")
    artifacts = [file_ref(probe_path)]
    for index, command in enumerate(commands):
        require(isinstance(command, dict), f"{probe_path}.commands[{index}] must be an object")
        for stream in ("stdout", "stderr"):
            source = checked_artifact(probe_path.parent, command.get(stream), f"probe.commands[{index}].{stream}")
            expected = command.get(f"{stream}_sha256")
            require(source.is_file() and not source.is_symlink(), f"probe command artifact is missing: {source}")
            actual = sha256_file(source)
            require(actual == expected, f"probe command artifact SHA mismatch: {source}")
            artifacts.append(file_ref(source, relative_to=probe_path.parent))
    return {"path": str(probe_path), "probe": probe, "artifacts": sorted(artifacts, key=lambda row: row["path"])}


def ingest_probe(root: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    source_probe = Path(bundle["path"])
    probe = copy.deepcopy(bundle["probe"])
    hardware_id = probe.get("hardware_id")
    require(isinstance(hardware_id, str) and hardware_id, f"{source_probe}.hardware_id is invalid")
    target_dir = root / "hardware" / hardware_id
    for command in probe["commands"]:
        for stream in ("stdout", "stderr"):
            rel = command[stream]
            source = checked_artifact(source_probe.parent, rel, f"probe.{stream}")
            destination = checked_artifact(target_dir, rel, f"probe.{stream}")
            atomic_copy(source, destination)
    target_probe = target_dir / "probe.json"
    atomic_write(target_probe, source_probe.read_bytes())
    normalized = probe.get("normalized")
    require(isinstance(normalized, dict) and normalized.get("schema_version") == 1, f"{source_probe}.normalized is invalid")
    fingerprint = probe.get("fingerprint")
    require(fingerprint == canonical_json_sha256(normalized), f"{source_probe}.fingerprint is not derived from normalized facts")
    row = {
        "id": hardware_id,
        **{key: copy.deepcopy(value) for key, value in normalized.items() if key != "schema_version"},
        "fingerprint_material": copy.deepcopy(normalized),
        "fingerprint": fingerprint,
        "probe": {
            "path": target_probe.relative_to(root).as_posix(),
            "sha256": sha256_file(target_probe),
        },
    }
    return row


def resolution_lane_index(resolution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(resolution.get("schema_version") == 1, "model-resolution.schema_version must be 1")
    require(resolution.get("artifact_type") == "runtime_vnext_model_resolution", "model-resolution artifact_type mismatch")
    rows = resolution.get("lanes")
    require(isinstance(rows, list), "model-resolution.lanes must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"model-resolution.lanes[{index}] must be an object")
        lane_id = row.get("catalog_lane_id")
        require(isinstance(lane_id, str) and lane_id and lane_id not in indexed, f"model-resolution lane id is invalid or duplicate: {lane_id}")
        indexed[lane_id] = row
    return indexed


def license_lock(
    source: dict[str, Any],
    *,
    model_key: str,
    backend: str,
    allow_synthetic: bool,
) -> dict[str, str]:
    license_info = source.get("license")
    require(isinstance(license_info, dict), "resolved weight source lacks license evidence")
    spdx = license_info.get("hugging_face_id")
    require(isinstance(spdx, str) and spdx.strip(), "resolved weight source has no Hugging Face license id")
    repo = source.get("repo")
    revision = source.get("revision")
    require(isinstance(repo, str) and repo and isinstance(revision, str) and revision, "resolved weight source identity is incomplete")
    source_url = (
        f"https://example.invalid/{model_key}/{backend}/license"
        if allow_synthetic
        else f"https://huggingface.co/{repo}/tree/{revision}"
    )
    return {
        "spdx": spdx.strip(),
        "source": source_url,
    }


def source_lock(source: dict[str, Any], label: str) -> dict[str, Any]:
    repo = source.get("repo")
    revision = source.get("revision")
    require(isinstance(repo, str) and repo and isinstance(revision, str) and revision, f"{label} identity is incomplete")
    return {"repo": repo, "revision": revision, "files": ordered_file_locks(source.get("files"), f"{label}.files")}


def materialize_models(
    resolution: dict[str, Any],
    catalog: dict[str, Any],
    preset_catalog: dict[str, Any],
    hardware_by_policy: dict[str, dict[str, Any]],
    *,
    allow_synthetic: bool,
) -> list[dict[str, Any]]:
    resolved = resolution_lane_index(resolution)
    catalog_rows = catalog.get("models")
    require(isinstance(catalog_rows, list) and len(catalog_rows) == 12, "model catalog must contain 12 lanes")
    catalog_by_key_backend: dict[tuple[str, str], dict[str, Any]] = {}
    for row in catalog_rows:
        require(isinstance(row, dict), "model catalog lane must be an object")
        model_key = baseline.CATALOG_MODEL_KEYS.get(row.get("model_id"))
        backend = row.get("backend")
        require(model_key is not None and backend in {"cuda", "metal"}, "model catalog lane identity is invalid")
        catalog_by_key_backend[(model_key, str(backend))] = row
    expected_ids = {str(row["id"]) for row in catalog_rows}
    require(set(resolved) == expected_ids, "model-resolution does not exactly cover the 12 catalog lanes")

    models: list[dict[str, Any]] = []
    matrices = (("primary", baseline.PRIMARY_MODELS), ("supplemental", baseline.SUPPLEMENTAL_MODELS))
    for role, mapping in matrices:
        for model_key, official_model_id in mapping.items():
            lanes: dict[str, dict[str, Any]] = {}
            for backend in ("cuda", "metal"):
                catalog_lane = catalog_by_key_backend[(model_key, backend)]
                lane_id = str(catalog_lane["id"])
                source_lane = resolved[lane_id]
                require(source_lane.get("backend") == backend, f"{lane_id} backend differs from catalog")
                require(source_lane.get("model_id") == catalog_lane.get("model_id"), f"{lane_id} model id differs from catalog")
                weight = source_lane.get("weight_source")
                semantic = source_lane.get("semantic_source")
                require(isinstance(weight, dict) and isinstance(semantic, dict), f"{lane_id} source locks are incomplete")
                policy = catalog_lane.get("hardware_policy")
                hardware = hardware_by_policy.get(str(policy))
                require(hardware is not None, f"{lane_id} has no probe for hardware policy {policy}")
                lane: dict[str, Any] = {
                    "catalog_lane_id": lane_id,
                    "repo": weight.get("repo"),
                    "revision": weight.get("revision"),
                    "format": catalog_lane.get("format"),
                    "hardware_policy": policy,
                    "hardware_id": hardware["id"],
                    "files": ordered_file_locks(weight.get("files"), f"{lane_id}.weight_source.files"),
                    "semantic_source": source_lock(semantic, f"{lane_id}.semantic_source"),
                    "license": license_lock(
                        weight,
                        model_key=model_key,
                        backend=backend,
                        allow_synthetic=allow_synthetic,
                    ),
                    "generation_config": copy.deepcopy(source_lane.get("generation_config")),
                    "chat_template": copy.deepcopy(source_lane.get("chat_template")),
                }
                tokenizer = source_lane.get("tokenizer_source")
                if tokenizer is not None:
                    require(isinstance(tokenizer, dict), f"{lane_id}.tokenizer_source must be an object")
                    lane["tokenizer_source"] = source_lock(tokenizer, f"{lane_id}.tokenizer_source")
                official = source_lane.get("official_upstream")
                if official is not None:
                    require(isinstance(official, dict), f"{lane_id}.official_upstream must be an object")
                    lane["official_upstream"] = copy.deepcopy(official)
                lanes[backend] = lane
            model: dict[str, Any] = {
                "key": model_key,
                "official_model_id": official_model_id,
                "role": role,
                "lanes": lanes,
            }
            if role == "primary":
                policies = preset_catalog.get("models")
                require(isinstance(policies, dict) and isinstance(policies.get(model_key), dict), f"preset policy missing {model_key}")
                policy = policies[model_key]
                model["generation_presets"] = copy.deepcopy(policy.get("presets"))
                evidence = policy.get("evidence")
                require(isinstance(evidence, dict) and evidence, f"preset evidence missing {model_key}")
                evidence_rows = []
                for path, digest in evidence.items():
                    sizes: set[int] = set()
                    for backend in ("cuda", "metal"):
                        by_path = {row["path"]: row for row in lanes[backend]["semantic_source"]["files"]}
                        require(path in by_path and by_path[path]["sha256"] == digest, f"{model_key}/{backend} preset evidence differs at {path}")
                        sizes.add(int(by_path[path]["size_bytes"]))
                    require(len(sizes) == 1, f"{model_key} preset evidence size differs across backends at {path}")
                    evidence_rows.append({"path": path, "sha256": digest, "size_bytes": sizes.pop()})
                model["generation_preset_evidence"] = evidence_rows
            models.append(model)
    return models


def expectations_binding(root: Path, path: Path, *, allow_synthetic: bool) -> dict[str, str]:
    path = path.expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"expectations catalog is missing: {path}")
    if allow_synthetic:
        try:
            rendered = path.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise OrchestrationError("synthetic expectations must live inside the artifact root") from exc
    else:
        require(path == baseline.CORRECTNESS_EXPECTATIONS_PATH.resolve(), "real G00 must bind the checked-in correctness expectations catalog")
        rendered = path.relative_to(REPO_ROOT).as_posix()
    return {"path": rendered, "sha256": sha256_file(path)}


def build_models_lock(
    root: Path,
    resolution: dict[str, Any],
    hardware: list[dict[str, Any]],
    expectation_ref: dict[str, str],
    *,
    allow_synthetic: bool,
) -> dict[str, Any]:
    catalog = read_json(baseline.MODELS_CATALOG_PATH)
    preset_catalog = read_json(baseline.PRESETS_CATALOG_PATH)
    hardware_by_policy: dict[str, dict[str, Any]] = {}
    for item in hardware:
        policy = item.get("policy_id")
        require(isinstance(policy, str) and policy not in hardware_by_policy, f"duplicate or invalid hardware policy: {policy}")
        hardware_by_policy[policy] = item
    lock = {
        "schema_version": 1,
        "source_git_sha": baseline.FROZEN_LEGACY_SHA,
        "source_tree_sha": baseline.frozen_tree_sha(),
        "dirty_status": {"is_dirty": False, "status_short": []},
        "catalog_id": catalog.get("catalog_id"),
        "catalog_sha256": sha256_file(baseline.MODELS_CATALOG_PATH),
        "preset_catalog_id": preset_catalog.get("catalog_id"),
        "preset_catalog_sha256": sha256_file(baseline.PRESETS_CATALOG_PATH),
        "expectations_catalog": expectation_ref,
        "model_resolution": {"path": "model-resolution.json", "sha256": sha256_file(root / "model-resolution.json")},
        "hardware": sorted(hardware, key=lambda row: str(row["backend"])),
        "models": materialize_models(
            resolution,
            catalog,
            preset_catalog,
            hardware_by_policy,
            allow_synthetic=allow_synthetic,
        ),
    }
    return lock


def lane_expectation_statuses(expectations: dict[str, Any], lane: str) -> set[str]:
    lanes = expectations.get("lanes")
    require(isinstance(lanes, dict) and isinstance(lanes.get(lane), dict), f"expectations catalog lacks lane {lane}")
    rules = lanes[lane].get("rules")
    require(isinstance(rules, list) and rules, f"expectations lane {lane} has no rules")
    statuses = {row.get("expected_status") for row in rules if isinstance(row, dict)}
    require(statuses and all(isinstance(item, str) for item in statuses), f"expectations lane {lane} has invalid statuses")
    return {str(item) for item in statuses}


def artifact_state(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    if not path.is_file():
        return {"state": "missing", "path": rel}
    require(not path.is_symlink(), f"phase artifact must not be a symlink: {path}")
    return {"state": "artifact-present-awaiting-validation", **file_ref(path, relative_to=root)}


def build_phase_inputs(
    root: Path,
    lock: dict[str, Any],
    expectations: dict[str, Any],
    input_fingerprint: str,
) -> dict[str, Any]:
    models = {row["key"]: row for row in lock["models"]}
    global_outputs = {
        "legacy-binaries": "legacy-binaries.json",
        "build-timings-cuda": "build-timings/summary.json",
        "coupling-inventory": "coupling-inventory.json",
        "historical-bug-corpus": "historical-bug-corpus.json",
    }
    global_phases: list[dict[str, Any]] = [
        {"id": "model-resolution", "state": "complete", "dependencies": [], "output": file_ref(root / "model-resolution.json", relative_to=root)},
        {"id": "hardware-probes", "state": "complete", "dependencies": [], "outputs": [row["probe"] for row in lock["hardware"]]},
        {"id": "models-lock", "state": "complete", "dependencies": ["model-resolution", "hardware-probes"], "output": file_ref(root / "models.lock.json", relative_to=root)},
    ]
    for phase_id, rel in global_outputs.items():
        state = artifact_state(root, rel)
        global_phases.append({"id": phase_id, "dependencies": ["models-lock"], **state})

    lanes: list[dict[str, Any]] = []
    for model_key in sorted(baseline.PRIMARY_MODELS):
        model = models[model_key]
        for backend in ("cuda", "metal"):
            lane_id = f"{model_key}/{backend}"
            statuses = lane_expectation_statuses(expectations, lane_id)
            discovery_required = "discovery-required" in statuses
            blocked_only = statuses == {"blocked"}
            discovery_rel = f"discovery/{model_key}/{backend}/scenario-report.json"
            correctness_rel = f"correctness/{model_key}/{backend}/lane.json"
            external_rel = f"external-baselines/{model_key}/{backend}/summary.json"
            performance_rel = f"performance/{model_key}/{backend}/summary.json"
            discovery_artifact = artifact_state(root, discovery_rel)
            correctness_artifact = artifact_state(root, correctness_rel)
            external_artifact = artifact_state(root, external_rel)
            performance_artifact = artifact_state(root, performance_rel)
            if discovery_required:
                discovery_state = "observation-present" if discovery_artifact["state"] != "missing" else "required"
                amendment_state = "required-after-observation" if discovery_state == "observation-present" else "waiting-for-discovery"
                formal_state = "blocked-by-expectation-amendment"
                if correctness_artifact["state"] != "missing":
                    formal_state = "stale-artifact-present-before-amendment"
            else:
                discovery_state = "not-required"
                amendment_state = "not-required"
                formal_state = correctness_artifact["state"]
            if blocked_only and correctness_artifact["state"] == "missing":
                formal_state = "truthful-blocked-artifact-required"
            lane_lock = model["lanes"][backend]
            lane_binding = {
                "model_lock_sha256": sha256_file(root / "models.lock.json"),
                "catalog_lane_id": lane_lock["catalog_lane_id"],
                "model_revision": lane_lock["revision"],
                "model_files_sha256": canonical_json_sha256(lane_lock["files"]),
                "semantic_source_sha256": canonical_json_sha256(lane_lock["semantic_source"]),
                "hardware_id": lane_lock["hardware_id"],
                "hardware_fingerprint": next(row["fingerprint"] for row in lock["hardware"] if row["id"] == lane_lock["hardware_id"]),
                "expectations_lane_sha256": canonical_json_sha256(expectations["lanes"][lane_id]),
            }
            lanes.append(
                {
                    "lane_id": lane_id,
                    "model_key": model_key,
                    "backend": backend,
                    "formal_expected_result": "blocked" if blocked_only else "pass",
                    "expectation_statuses": sorted(statuses),
                    "binding": lane_binding,
                    "phases": {
                        "discovery": {
                            "required": discovery_required,
                            "state": discovery_state,
                            "dependencies": ["global.models-lock", "global.legacy-binaries"],
                            "output": discovery_artifact,
                            "mode": "discover",
                        },
                        "expectation-amendment": {
                            "required": discovery_required,
                            "state": amendment_state,
                            "dependencies": [f"{lane_id}.discovery"] if discovery_required else [],
                            "acceptance": "checked-in catalog contains no discovery-required rule for this lane",
                        },
                        "formal-correctness": {
                            "state": formal_state,
                            "dependencies": ["global.models-lock", "global.legacy-binaries"]
                            + ([f"{lane_id}.expectation-amendment"] if discovery_required else []),
                            "output": correctness_artifact,
                            "entrypoints": ["ferrum run", "ferrum serve"],
                        },
                        "external-performance": {
                            "state": external_artifact["state"] if formal_state == "artifact-present-awaiting-validation" else "waiting-for-formal-correctness",
                            "dependencies": [f"{lane_id}.formal-correctness"],
                            "output": external_artifact,
                            "comparison": "standalone" if blocked_only else "ABBA-BAAB-shared-host",
                        },
                        "legacy-performance": {
                            "state": performance_artifact["state"] if external_artifact["state"] != "missing" and formal_state == "artifact-present-awaiting-validation" else "waiting-for-correctness-and-external",
                            "dependencies": [f"{lane_id}.formal-correctness", f"{lane_id}.external-performance"],
                            "output": performance_artifact,
                            "comparable": not blocked_only,
                        },
                    },
                }
            )
    require(len(lanes) == 6, "phase plan must contain exactly six primary product lanes")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g00_phase_inputs",
        "input_fingerprint": input_fingerprint,
        "source_git_sha": baseline.FROZEN_LEGACY_SHA,
        "source_tree_sha": baseline.frozen_tree_sha(),
        "global_phases": global_phases,
        "lanes": lanes,
        "final_validator": {
            "dependencies": [phase["id"] for phase in global_phases[3:]]
            + [f"{lane['lane_id']}.formal-correctness" for lane in lanes]
            + [f"{lane['lane_id']}.external-performance" for lane in lanes]
            + [f"{lane['lane_id']}.legacy-performance" for lane in lanes],
            "argv": [
                "python3",
                "scripts/release/runtime_vnext_baseline_gate.py",
                "--out",
                str(root),
                "--require-full-self-test",
            ],
            "required_pass_line": f"{baseline.PASS_PREFIX}: {root}",
        },
    }


def missing_phases(phase_inputs: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for phase in phase_inputs["global_phases"]:
        if phase["state"] not in {"complete", "artifact-present-awaiting-validation"}:
            missing.append(f"global.{phase['id']}")
    for lane in phase_inputs["lanes"]:
        lane_id = lane["lane_id"]
        for phase_name, phase in lane["phases"].items():
            if phase["state"] not in {"not-required", "artifact-present-awaiting-validation"}:
                missing.append(f"{lane_id}.{phase_name}")
    if not missing:
        missing.append("global.final-validator")
    return sorted(missing)


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    excluded = {ROOT_MANIFEST, ORCHESTRATION_MANIFEST, PHASE_INPUTS_PATH}
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise OrchestrationError(f"artifact root contains a forbidden symlink: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded or rel.startswith(f"{ORCHESTRATION_DIR}/."):
            continue
        rows.append(file_ref(path, relative_to=root))
    return rows


def verify_prior_freeze(root: Path, prior: dict[str, Any], input_fingerprint: str) -> None:
    require(prior.get("artifact_type") == ARTIFACT_TYPE, "existing orchestration manifest has the wrong artifact type")
    require(prior.get("input_fingerprint") == input_fingerprint, "orchestration inputs changed; use a new G00 output root after an expectation/model/hardware amendment")
    rows = prior.get("frozen_artifacts")
    require(isinstance(rows, list), "existing orchestration manifest lacks frozen_artifacts")
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"frozen_artifacts[{index}] must be an object")
        path = checked_artifact(root, row.get("path"), f"frozen_artifacts[{index}].path")
        require(path.is_file() and not path.is_symlink(), f"previously frozen artifact disappeared: {path}")
        require(sha256_file(path) == row.get("sha256") and path.stat().st_size == row.get("size_bytes"), f"previously frozen artifact changed: {path}")


def input_material(
    resolution_path: Path,
    probe_bundles: list[dict[str, Any]],
    expectation_path: Path,
    collector: dict[str, Any],
    *,
    allow_synthetic: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "internal-selftest" if allow_synthetic else "real-evidence",
        "legacy_source": {"git_sha": baseline.FROZEN_LEGACY_SHA, "tree_sha": baseline.frozen_tree_sha()},
        "collector_source": collector,
        "orchestrator": file_ref(ORCHESTRATOR_PATH, relative_to=REPO_ROOT),
        "validator": file_ref(Path(baseline.__file__).resolve(), relative_to=REPO_ROOT),
        "model_resolver": file_ref(baseline.MODEL_RESOLVER_PATH, relative_to=REPO_ROOT),
        "hardware_probe_collector": file_ref(baseline.HARDWARE_PROBE_PATH, relative_to=REPO_ROOT),
        "model_catalog": file_ref(baseline.MODELS_CATALOG_PATH, relative_to=REPO_ROOT),
        "preset_catalog": file_ref(baseline.PRESETS_CATALOG_PATH, relative_to=REPO_ROOT),
        "expectations_catalog": file_ref(expectation_path),
        "model_resolution": file_ref(resolution_path),
        "hardware_probe_bundles": [{"path": row["path"], "artifacts": row["artifacts"]} for row in probe_bundles],
    }


def locate_probe(root: Path, backend: str) -> Path:
    candidates = []
    for path in sorted((root / "hardware").glob("*/probe.json")):
        try:
            probe = read_json(path)
        except OrchestrationError:
            continue
        normalized = probe.get("normalized")
        if isinstance(normalized, dict) and normalized.get("backend") == backend:
            candidates.append(path)
    require(
        len(candidates) == 1,
        f"when --{backend}-probe is omitted, exactly one {backend} probe must exist under {root / 'hardware'}",
    )
    return candidates[0]


def prepare_root(
    out: Path,
    *,
    model_resolution: Path | None,
    cuda_probe: Path | None,
    metal_probe: Path | None,
    expectations_path: Path | None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    root = external_root(out)
    final_manifest_path = root / ROOT_MANIFEST
    if final_manifest_path.is_file():
        existing_final = read_json(final_manifest_path)
        if existing_final.get("status") == "pass" and isinstance(existing_final.get("pass_line"), str):
            raise OrchestrationError("G00 root already has a validator PASS manifest and is immutable")
    root.mkdir(parents=True, exist_ok=True)
    resolution_path = (model_resolution or (root / "model-resolution.json")).expanduser().resolve()
    cuda_probe_path = (cuda_probe or locate_probe(root, "cuda")).expanduser().resolve()
    metal_probe_path = (metal_probe or locate_probe(root, "metal")).expanduser().resolve()
    expectations_path = (expectations_path or baseline.CORRECTNESS_EXPECTATIONS_PATH).expanduser().resolve()
    require(resolution_path.is_file(), f"model-resolution input is missing: {resolution_path}")
    resolution = read_json(resolution_path)
    resolution_lane_index(resolution)
    probe_bundles = [input_probe_bundle(cuda_probe_path), input_probe_bundle(metal_probe_path)]
    backends = {
        bundle["probe"].get("normalized", {}).get("backend")
        for bundle in probe_bundles
        if isinstance(bundle["probe"].get("normalized"), dict)
    }
    require(backends == {"cuda", "metal"}, "hardware inputs must contain one CUDA and one Metal probe")
    collector = collector_identity()
    if not allow_synthetic:
        require(
            collector["dirty_status"] == {"is_dirty": False, "status_short": []},
            "real G00 orchestration requires a clean committed collector worktree",
        )
    material = input_material(
        resolution_path,
        probe_bundles,
        expectations_path,
        collector,
        allow_synthetic=allow_synthetic,
    )
    input_fingerprint = canonical_json_sha256(material)
    prior_path = root / ORCHESTRATION_MANIFEST
    prior = read_json(prior_path) if prior_path.is_file() else None
    if prior is not None:
        verify_prior_freeze(root, prior, input_fingerprint)

    target_resolution = root / "model-resolution.json"
    atomic_copy(resolution_path, target_resolution)
    ingested_hardware = [ingest_probe(root, bundle) for bundle in probe_bundles]
    expectation_ref = expectations_binding(root, expectations_path, allow_synthetic=allow_synthetic)
    lock = build_models_lock(
        root,
        resolution,
        ingested_hardware,
        expectation_ref,
        allow_synthetic=allow_synthetic,
    )
    lock_path = root / "models.lock.json"
    lock_payload = canonical_json_bytes(lock)
    if prior is not None and lock_path.is_file():
        require(lock_path.read_bytes() == lock_payload, "deterministic models.lock reconstruction differs on resume")
    atomic_write(lock_path, lock_payload)
    try:
        baseline.validate_models_lock(root, allow_synthetic=allow_synthetic)
    except baseline.BaselineError as exc:
        raise OrchestrationError(f"materialized models.lock is rejected by the G00 validator: {exc}") from exc

    expectations = read_json(expectations_path)
    phase_inputs = build_phase_inputs(root, lock, expectations, input_fingerprint)
    atomic_write_json(root / PHASE_INPUTS_PATH, phase_inputs)
    missing = missing_phases(phase_inputs)
    frozen = artifact_inventory(root)
    generation = int(prior.get("generation", 0)) + 1 if prior is not None else 1
    prepared_at = now_iso()
    orchestration_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "ready-for-final-validator" if missing == ["global.final-validator"] else "collecting",
        "generation": generation,
        "first_prepared_at": prior.get("first_prepared_at", prepared_at) if prior is not None else prepared_at,
        "updated_at": prepared_at,
        "artifact_dir": str(root),
        "input_fingerprint": input_fingerprint,
        "input_material": material,
        "models_lock": file_ref(lock_path, relative_to=root),
        "phase_inputs": file_ref(root / PHASE_INPUTS_PATH, relative_to=root),
        "frozen_artifacts": frozen,
        "frozen_artifact_count": len(frozen),
        "missing_phases": missing,
        "missing_phase_count": len(missing),
        "next_phase": missing[0],
        "orchestrator_started_paid_gpu": False,
        "orchestrator_started_downloads": False,
        "final_validator_pass_line": None,
    }
    atomic_write_json(prior_path, orchestration_manifest)
    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g00_prepared_root",
        "status": orchestration_manifest["status"],
        "source_git_sha": baseline.FROZEN_LEGACY_SHA,
        "source_tree_sha": baseline.frozen_tree_sha(),
        "artifact_dir": str(root),
        "prepared_at": prepared_at,
        "orchestration_manifest": file_ref(prior_path, relative_to=root),
        "models_lock_sha256": sha256_file(lock_path),
        "missing_phases": missing,
        "pass_line": None,
    }
    atomic_write_json(final_manifest_path, root_manifest)
    return orchestration_manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-runtime-vnext-g00-orchestrator-") as temporary:
        temp = Path(temporary)
        seed = temp / "seed"
        baseline.make_synthetic_root(seed)
        seed_lock = read_json(seed / "models.lock.json")
        expectation_rel = seed_lock["expectations_catalog"]["path"]
        expectation_source = seed / expectation_rel
        probe_paths = {
            row["backend"]: seed / row["probe"]["path"]
            for row in seed_lock["hardware"]
        }

        partial = temp / "partial"
        partial.mkdir()
        partial_expectations = partial / "legacy-correctness-expectations.json"
        atomic_copy(expectation_source, partial_expectations)
        first = prepare_root(
            partial,
            model_resolution=seed / "model-resolution.json",
            cuda_probe=probe_paths["cuda"],
            metal_probe=probe_paths["metal"],
            expectations_path=partial_expectations,
            allow_synthetic=True,
        )
        require(first["missing_phase_count"] > 20, "partial self-test root did not enumerate missing phases")
        require("global.legacy-binaries" in first["missing_phases"], "partial self-test root missed the binary phase")
        require(len(read_json(partial / PHASE_INPUTS_PATH)["lanes"]) == 6, "self-test phase plan is not six lanes")
        second = prepare_root(
            partial,
            model_resolution=seed / "model-resolution.json",
            cuda_probe=probe_paths["cuda"],
            metal_probe=probe_paths["metal"],
            expectations_path=partial_expectations,
            allow_synthetic=True,
        )
        require(second["generation"] == 2, "resume did not advance orchestration generation")
        require(second["input_fingerprint"] == first["input_fingerprint"], "resume changed a stable input fingerprint")
        real_expectations = read_json(baseline.CORRECTNESS_EXPECTATIONS_PATH)
        real_plan = build_phase_inputs(
            partial,
            read_json(partial / "models.lock.json"),
            real_expectations,
            second["input_fingerprint"],
        )
        discovery_lanes = {
            lane["lane_id"]
            for lane in real_plan["lanes"]
            if lane["phases"]["discovery"]["required"]
        }
        require(
            discovery_lanes
            == {
                "m1-qwen35-4b/cuda",
                "m2-qwen35-35b-a3b/cuda",
                "m3-qwen3-30b-a3b/cuda",
                "m3-qwen3-30b-a3b/metal",
            },
            "checked-in discovery/formal dependency matrix drifted",
        )
        lock_path = partial / "models.lock.json"
        lock_bytes = lock_path.read_bytes()
        atomic_write(lock_path, lock_bytes + b" ")
        try:
            prepare_root(
                partial,
                model_resolution=seed / "model-resolution.json",
                cuda_probe=probe_paths["cuda"],
                metal_probe=probe_paths["metal"],
                expectations_path=partial_expectations,
                allow_synthetic=True,
            )
        except OrchestrationError as exc:
            require("previously frozen artifact changed" in str(exc), "resume drift rejection reported the wrong failure")
        else:
            raise OrchestrationError("resume accepted a changed frozen models.lock")
        atomic_write(lock_path, lock_bytes)

        (seed / "models.lock.json").unlink()
        complete = prepare_root(
            seed,
            model_resolution=seed / "model-resolution.json",
            cuda_probe=probe_paths["cuda"],
            metal_probe=probe_paths["metal"],
            expectations_path=expectation_source,
            allow_synthetic=True,
        )
        require(complete["models_lock"]["sha256"] == sha256_file(seed / "models.lock.json"), "self-test lock receipt mismatch")
        accepted = baseline.validate_root(seed, allow_synthetic=True)
        require(accepted.get("status") == "pass", "existing G00 validator rejected the orchestrated synthetic root")
        require(accepted.get("pass_line") == f"{baseline.PASS_PREFIX}: {seed}", "existing validator PASS line mismatch")
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="G00 artifact root outside the repository")
    parser.add_argument("--model-resolution", type=Path, help="already-collected model-resolution.json")
    parser.add_argument("--cuda-probe", type=Path, help="already-collected CUDA hardware probe.json")
    parser.add_argument("--metal-probe", type=Path, help="already-collected Metal hardware probe.json")
    parser.add_argument("--expectations", type=Path, default=baseline.CORRECTNESS_EXPECTATIONS_PATH)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and args.out is None:
        parser.error("--out is required")
    if args.self_test and any((args.out, args.model_resolution, args.cuda_probe, args.metal_probe)):
        parser.error("--self-test does not accept collection inputs")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        manifest = prepare_root(
            args.out,
            model_resolution=args.model_resolution,
            cuda_probe=args.cuda_probe,
            metal_probe=args.metal_probe,
            expectations_path=args.expectations,
        )
    except (OSError, ValueError, OrchestrationError, baseline.BaselineError) as exc:
        print(f"FERRUM RUNTIME VNEXT G00 ORCHESTRATOR FAIL: {exc}", file=sys.stderr)
        return 1
    if manifest["status"] == "ready-for-final-validator":
        print(f"FERRUM RUNTIME VNEXT G00 ORCHESTRATOR READY FOR VALIDATOR: {manifest['artifact_dir']}")
    else:
        print(
            "FERRUM RUNTIME VNEXT G00 ORCHESTRATOR INCOMPLETE: "
            f"{manifest['artifact_dir']}: {manifest['missing_phase_count']} missing phases; "
            f"next={manifest['next_phase']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
