#!/usr/bin/env python3
"""Finalize one R2 Ferrum lane after the legacy CUDA terminal-exit audit race.

The recovery never rewrites raw evidence and never launches a model.  It keeps
the original rejected audit, writes a derived accepted-prefix sidecar plus an
idle-GPU postflight, and lets the final R2 validator reclassify only the exact
process-exit edge.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import runtime_vnext_r2_ferrum_collector as collector
    import runtime_vnext_r2_performance_build_profile as aggregate
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_r2_ferrum_collector as collector
    from scripts.release import runtime_vnext_r2_performance_build_profile as aggregate


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_RELATIVE_PATH = SCRIPT_PATH.relative_to(collector.REPO_ROOT).as_posix()
PASS_PREFIX = "FERRUM RUNTIME VNEXT R2 FERRUM TERMINAL RECOVERY PASS"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise collector.R2CollectorError(message)


def write_text_once(path: Path, payload: str) -> None:
    if path.exists():
        require(path.read_text(encoding="utf-8") == payload, f"existing derived file differs: {path}")
        return
    collector.atomic_write_text(path, payload)


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        require(collector.read_json(path) == value, f"existing derived JSON differs: {path}")
        return
    collector.atomic_write_json(path, value)


def collect_idle_postflight(
    bridge: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
    binary = Path(str(bridge.get("real_nvidia_smi_path", ""))).resolve()
    require(
        binary.is_file()
        and collector.file_sha256(binary) == bridge.get("real_nvidia_smi_sha256"),
        "CUDA idle postflight nvidia-smi identity differs",
    )
    compute_argv = [str(binary), *collector.CUDA_COMPUTE_QUERY]
    compute = subprocess.run(
        compute_argv, capture_output=True, text=True, check=False, timeout=30.0
    )
    require(compute.returncode == 0, "CUDA idle postflight compute query failed")
    compute_apps = collector.parse_cuda_compute_rows(
        compute.stdout, "CUDA idle postflight compute query"
    )
    require(not compute_apps, "CUDA idle postflight found a live compute application")
    gpu_argv = [str(binary), *collector.CUDA_GPU_UUID_QUERY]
    gpu = subprocess.run(
        gpu_argv, capture_output=True, text=True, check=False, timeout=30.0
    )
    gpu_uuids = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    require(
        gpu.returncode == 0 and gpu_uuids == preflight.get("gpu_uuids"),
        "CUDA idle postflight GPU identity differs",
    )
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_r2_cuda_idle_postflight",
        "captured_at": collector.now_iso(),
        "nvidia_smi_path": str(binary),
        "nvidia_smi_sha256": collector.file_sha256(binary),
        "compute_argv": compute_argv,
        "returncode": compute.returncode,
        "compute_stdout": compute.stdout,
        "compute_stderr": compute.stderr,
        "compute_apps": compute_apps,
        "gpu_argv": gpu_argv,
        "gpu_returncode": gpu.returncode,
        "gpu_stdout": gpu.stdout,
        "gpu_stderr": gpu.stderr,
        "gpu_uuids": gpu_uuids,
    }


def finalize(root: Path, config_path: Path, sample_ordinal: int) -> Path:
    root = root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    require(root.is_dir(), f"artifact root is missing: {root}")
    config, context = collector.normalize_config(collector.read_json(config_path))
    require(config["backend"] == "cuda", "terminal recovery is CUDA-only")
    lane = collector.lane_dir(root, config)
    plan_path = lane / "plan.json"
    plan = collector.read_json(plan_path)
    fingerprint = plan.get("config_fingerprint")
    require(
        isinstance(fingerprint, str) and fingerprint,
        "collection plan fingerprint is missing",
    )
    require(
        plan.get("collector", {}).get("sha256")
        == collector.file_sha256(collector.COLLECTOR_PATH),
        "checked-in collector differs from the frozen collection plan",
    )

    inputs = collector.stage_inputs(root, lane, config, context)
    server_path = lane / "server-session.json"
    server = collector.read_json(server_path)
    collector.validate_server_bundle(root, server, fingerprint, config)
    run_paths = [
        lane / "run-samples" / f"sample-{ordinal}.json"
        for ordinal in range(1, collector.RUN_SAMPLE_COUNT + 1)
    ]
    require(all(path.is_file() for path in run_paths), "three run sample bundles are required")
    runs = [collector.read_json(path) for path in run_paths]
    for ordinal, bundle in enumerate(runs, start=1):
        if ordinal != sample_ordinal:
            collector.validate_run_bundle(root, bundle, fingerprint, ordinal)

    recovered_sample = runs[sample_ordinal - 1]["sample"]
    bridge = recovered_sample["resources"]["cuda_pid_namespace_bridge"]
    original_audit_path = collector.validate_artifact_ref(
        root, bridge["audit"], "terminal recovery original audit"
    )
    audit_rows = [
        json.loads(line)
        for line in original_audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(audit_rows) >= 2, "terminal recovery audit is too short")

    derived = lane / "derived" / f"terminal-exit-run-{sample_ordinal}"
    derived.mkdir(parents=True, exist_ok=True)
    prefix_path = derived / "accepted-prefix-audit.jsonl"
    prefix_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in audit_rows[:-1]
    )
    write_text_once(prefix_path, prefix_payload)
    prefix_ref = collector.artifact_ref(
        root, prefix_path, kind="cuda-pid-namespace-accepted-prefix"
    )

    preflight_path = collector.validate_artifact_ref(
        root, bridge["preflight"], "terminal recovery CUDA preflight"
    )
    postflight_path = derived / "cuda-idle-postflight.json"
    if postflight_path.exists():
        postflight = collector.read_json(postflight_path)
    else:
        postflight = collect_idle_postflight(
            bridge, collector.read_json(preflight_path)
        )
        write_json_once(postflight_path, postflight)
    postflight_ref = collector.artifact_ref(
        root, postflight_path, kind="cuda-idle-postflight"
    )

    receipt_path = derived / "recovery.json"
    receipt = {
        "schema_version": 1,
        "contract": aggregate.TERMINAL_CUDA_EXIT_RECOVERY_CONTRACT,
        "artifact_type": "runtime_vnext_r2_cuda_terminal_exit_recovery",
        "status": "pass",
        "created_at": collector.now_iso(),
        "sample_ordinal": sample_ordinal,
        "run_sample": collector.artifact_ref(
            root, run_paths[sample_ordinal - 1], kind="run-sample-bundle"
        ),
        "original_audit": copy.deepcopy(bridge["audit"]),
        "accepted_prefix_audit": prefix_ref,
        "cuda_idle_postflight": postflight_ref,
        "terminal_rejection_sha256": aggregate.canonical_json_sha256(
            audit_rows[-1]
        ),
        "raw_evidence_mutated": False,
        "recovery_source": {
            "path": SCRIPT_RELATIVE_PATH,
            "sha256": collector.file_sha256(SCRIPT_PATH),
        },
        "classifier_source": {
            "path": aggregate.SCRIPT_PATH.relative_to(collector.REPO_ROOT).as_posix(),
            "sha256": collector.file_sha256(aggregate.SCRIPT_PATH),
        },
        "frozen_collector_source": copy.deepcopy(plan["collector"]),
    }
    write_json_once(receipt_path, receipt)
    receipt_ref = collector.artifact_ref(
        root, receipt_path, kind="cuda-terminal-exit-recovery"
    )
    aggregate.validate_terminal_cuda_exit_recovery(
        root,
        runs[sample_ordinal - 1],
        fingerprint=fingerprint,
        sample_ordinal=sample_ordinal,
        recovery=receipt,
    )

    collector.collector_support.append_jsonl(
        lane / "command-log.jsonl",
        {
            "event": "terminal-exit-recovery-complete",
            "sample_ordinal": sample_ordinal,
            "created_at": receipt["created_at"],
            "receipt": receipt_ref["path"],
            "receipt_sha256": receipt_ref["sha256"],
            "raw_evidence_mutated": False,
        },
    )
    index_material = [
        *runs,
        {
            "terminal_exit_recovery": receipt_ref,
            "accepted_prefix_audit": prefix_ref,
            "cuda_idle_postflight": postflight_ref,
        },
    ]
    index_path = collector.write_artifact_index(
        root,
        lane,
        plan_path,
        inputs,
        server,
        server_path,
        index_material,
        run_paths,
    )
    summary = collector.run_summary(runs, server["run_serve_parity_report"])
    manifest_path = lane / "manifest.json"
    manifest = {
        "schema_version": collector.SCHEMA_VERSION,
        "contract": collector.CONTRACT,
        "artifact_type": "runtime_vnext_r2_ferrum_lane_manifest",
        "status": "pass",
        "formal_r2_aggregate_status": "not-evaluated",
        "model_key": config["model_key"],
        "backend": config["backend"],
        "hardware": copy.deepcopy(config["hardware"]),
        "config_fingerprint": fingerprint,
        "profile_detail": "off",
        "source_git_sha": config["candidate"]["source_git_sha"],
        "source_tree_sha": config["candidate"]["source_tree_sha"],
        "dirty_status": copy.deepcopy(config["candidate"]["dirty_status"]),
        "candidate_binary_sha256": inputs["binary"]["sha256"],
        "model_revision": context["lane"]["revision"],
        "model_files": copy.deepcopy(context["model_files"]),
        "plan": collector.artifact_ref(root, plan_path, kind="collection-plan"),
        "inputs": {
            key: copy.deepcopy(value)
            for key, value in inputs.items()
            if isinstance(value, dict)
        },
        "server_session": collector.artifact_ref(
            root, server_path, kind="server-session-bundle"
        ),
        "formal_http_cell_count": len(collector.expected_cells(config["backend"])),
        "formal_http_cells": [
            collector.cell_id(cell)
            for cell in collector.expected_cells(config["backend"])
        ],
        "run_serve_parity_probe": copy.deepcopy(
            server["run_serve_parity_report"]
        ),
        "run_samples": [
            collector.artifact_ref(root, path, kind="run-sample-bundle")
            for path in run_paths
        ],
        "run_performance": summary,
        "terminal_exit_recovery": receipt_ref,
        "raw_artifact_index": collector.artifact_ref(
            root, index_path, kind="raw-artifact-index"
        ),
        "pass_line": (
            f"{collector.PASS_PREFIX}: {config['model_key']}/{config['backend']}: "
            f"{manifest_path}"
        ),
    }
    require(not manifest_path.exists(), f"refusing to overwrite manifest: {manifest_path}")
    collector.atomic_write_json(manifest_path, manifest)
    collector.validate_final_manifest(root, manifest, fingerprint)
    aggregate.default_collector_verifier(root, manifest, config, server, runs)
    print(f"{PASS_PREFIX}: {manifest_path}")
    print(manifest["pass_line"])
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-ordinal", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        finalize(args.artifact_root, args.config, args.sample_ordinal)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"runtime vNext R2 Ferrum terminal recovery failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
