#!/usr/bin/env python3
"""Collect and validate bounded CUDA decode-capacity pressure evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import runtime_vnext_s1_cuda_capacity as common


PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS"
COLLECT_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY COLLECTED"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY FAIL"
CALIBRATION_TOKEN_BUDGET = 1
TARGET_TOKEN_BUDGET = 1024
MAX_NUM_SEQS = 3
MAX_MODEL_LEN = 512
PREFILL_FIRST_UNTIL_ACTIVE = 3
STREAM_MAX_TOKENS = 128
MAX_DECODE_CAPACITY_EVENTS = 2048
ALLOWED_EXECUTION_STAGES = {
    "sequence_extension",
    "step_admission",
    "submission_wave",
}
SERVER_POLICY = {
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": MAX_NUM_SEQS,
    "prefill_first_until_active": PREFILL_FIRST_UNTIL_ACTIVE,
    "calibration_max_num_batched_tokens": CALIBRATION_TOKEN_BUDGET,
    "target_max_num_batched_tokens": TARGET_TOKEN_BUDGET,
    "stream_max_tokens": STREAM_MAX_TOKENS,
}
STOP_POLICY = {
    "no_progress_timeout_seconds": common.MAX_PRESSURE_NO_PROGRESS_SECONDS,
    "joint_stream_timeout_seconds": common.MAX_PRESSURE_JOINT_STREAM_SECONDS,
    "max_trace_bytes": common.MAX_PRESSURE_TRACE_BYTES,
    "max_decode_capacity_events": MAX_DECODE_CAPACITY_EVENTS,
}


class DecodeCapacityGateError(common.CapacityGateError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeCapacityGateError(message)


def source_key(source: Any) -> str:
    return json.dumps(source, sort_keys=True, separators=(",", ":"))


def wait_source_epochs(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, list) and value, f"{label}: current wait sources are missing")
    epochs: dict[str, int] = {}
    for entry in value:
        require(isinstance(entry, dict), f"{label}: current wait source is invalid")
        source = entry.get("source")
        key = source_key(source)
        require(key not in epochs, f"{label}: current wait source is duplicated")
        epoch = entry.get("epoch")
        require(isinstance(epoch, int) and epoch > 0, f"{label}: current wait epoch is invalid")
        epochs[key] = epoch
    return epochs


def compare_wait_sources(
    wait_condition: dict[str, Any],
    current_wait_sources: Any,
    *,
    label: str,
    expect_changed: bool,
) -> None:
    observed = {
        source_key(entry["source"]): entry["epoch"]
        for entry in wait_condition["observed"]
    }
    current = wait_source_epochs(current_wait_sources, label)
    require(current.keys() == observed.keys(), f"{label}: exact wait source set changed")
    require(
        all(current[key] >= observed[key] for key in observed),
        f"{label}: exact wait source generation regressed",
    )
    changed = any(current[key] > observed[key] for key in observed)
    require(changed == expect_changed, f"{label}: exact wait source change evidence is inconsistent")


def event_request_ids(row: dict[str, Any], label: str) -> list[str]:
    attributes = row.get("attributes")
    require(isinstance(attributes, dict), f"{label}: trace attributes are missing")
    request_ids = attributes.get("request_ids")
    require(isinstance(request_ids, list) and request_ids, f"{label}: request IDs are missing")
    require(
        all(isinstance(request_id, str) and request_id for request_id in request_ids),
        f"{label}: request identity is invalid",
    )
    require(len(set(request_ids)) == len(request_ids), f"{label}: request IDs are duplicated")
    return request_ids


def validate_decode_deferral(row: dict[str, Any], label: str) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    shape = row.get("shape")
    require(isinstance(shape, dict), f"{label}: shape is missing")
    decision = shape.get("decision")
    require(decision in {"split_cohort", "wait_for_release"}, f"{label}: decision is invalid")
    width = shape.get("attempted_decode_width")
    require(isinstance(width, int) and width > 0, f"{label}: attempted width is invalid")
    stage = shape.get("execution_stage")
    require(stage in ALLOWED_EXECUTION_STAGES, f"{label}: execution stage is invalid")
    require(shape.get("decode_submit_observed") is False, f"{label}: decode submit preceded defer")
    request_ids = event_request_ids(row, label)
    require(len(request_ids) == width, f"{label}: attempted width/request count mismatch")
    if decision == "split_cohort":
        require(width >= 2, f"{label}: split cohort is not wide")
    else:
        require(width == 1, f"{label}: a non-exact cohort was parked")

    evidence = row.get("attributes", {}).get("capacity_evidence")
    require(isinstance(evidence, dict), f"{label}: capacity evidence is missing")
    observed = common.validate_admission_epochs(evidence.get("observed"), label)
    wait_condition = common.validate_capacity_wait_condition(
        evidence.get("wait_condition"),
        coordinator_id=observed["coordinator_id"],
        label=label,
    )
    scheduler_snapshot = row.get("attributes", {}).get("scheduler_snapshot")
    require(isinstance(scheduler_snapshot, dict), f"{label}: scheduler snapshot is missing")
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "decision": decision,
        "width": width,
        "stage": stage,
        "request_ids": request_ids,
        "observed": observed,
        "wait_condition": wait_condition,
    }


def validate_decode_queue_transition(
    row: dict[str, Any], label: str, *, resumed: bool
) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    request_id = row.get("request_id")
    require(isinstance(request_id, str) and request_id, f"{label}: request identity is missing")
    shape = row.get("shape")
    attributes = row.get("attributes")
    require(isinstance(shape, dict) and isinstance(attributes, dict), f"{label}: trace payload is missing")
    require(shape.get("decode_submit_observed") is False, f"{label}: retry happened after submit")
    require(shape.get("probe_performed") is False, f"{label}: scheduler performed an admission probe")
    evidence = attributes.get("deferral_evidence")
    require(isinstance(evidence, dict), f"{label}: deferral evidence is missing")
    require(evidence.get("action") == "wait_for_release", f"{label}: deferred action is invalid")
    observed = common.validate_admission_epochs(evidence.get("observed"), label)
    current = common.validate_admission_epochs(evidence.get("current"), label)
    require(
        current["coordinator_id"] == observed["coordinator_id"],
        f"{label}: admission coordinator changed",
    )
    require(
        current["release_epoch"] >= observed["release_epoch"]
        and current["capacity_epoch"] >= observed["capacity_epoch"],
        f"{label}: global audit epoch regressed",
    )
    wait_condition = common.validate_capacity_wait_condition(
        evidence.get("wait_condition"),
        coordinator_id=observed["coordinator_id"],
        label=label,
    )
    if resumed:
        exact_changed = shape.get("exact_source_changed")
        policy_changed = shape.get("policy_epoch_changed")
        require(isinstance(exact_changed, bool), f"{label}: exact-source flag is missing")
        require(isinstance(policy_changed, bool), f"{label}: policy flag is missing")
        require(exact_changed or policy_changed, f"{label}: resume has no wake reason")
        expected_decision = "exact_source_changed" if exact_changed else "policy_epoch_changed"
        require(shape.get("decision") == expected_decision, f"{label}: resume reason is inconsistent")
        compare_wait_sources(
            wait_condition,
            evidence.get("current_wait_sources"),
            label=label,
            expect_changed=exact_changed,
        )
    else:
        require(shape.get("decision") == "skipped_unchanged", f"{label}: skip decision is invalid")
        compare_wait_sources(
            wait_condition,
            evidence.get("current_wait_sources"),
            label=label,
            expect_changed=False,
        )
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "request_id": request_id,
        "exact_source_changed": shape.get("exact_source_changed", False),
    }


def validate_decode_trace(
    rows: list[dict[str, Any]], *, started_wall_ns: int, finished_wall_ns: int
) -> dict[str, Any]:
    require(started_wall_ns > 0 and finished_wall_ns >= started_wall_ns, "invalid trace window")
    window = [
        row
        for row in rows
        if isinstance(row.get("ts_unix_nanos"), int)
        and started_wall_ns <= row["ts_unix_nanos"] <= finished_wall_ns
    ]
    deferral_rows = [row for row in window if row.get("phase") == "vnext.decode_capacity_deferred"]
    require(deferral_rows, "target produced no typed decode-capacity deferral")
    require(
        len(deferral_rows) <= MAX_DECODE_CAPACITY_EVENTS,
        "decode-capacity deferrals exceeded the bounded event ceiling",
    )
    deferrals = [
        validate_decode_deferral(row, f"decode deferral {index}")
        for index, row in enumerate(deferral_rows)
    ]
    splits = [event for event in deferrals if event["decision"] == "split_cohort"]
    parks = [event for event in deferrals if event["decision"] == "wait_for_release"]
    require(splits, "target never adaptively split a capacity-blocked decode cohort")

    skip_rows = [row for row in window if row.get("phase") == "vnext.decode_capacity_skipped_unchanged"]
    resume_rows = [row for row in window if row.get("phase") == "vnext.decode_capacity_resumed"]
    require(
        len(skip_rows) + len(resume_rows) <= MAX_DECODE_CAPACITY_EVENTS,
        "decode-capacity wake observations exceeded the bounded event ceiling",
    )
    skips = [
        validate_decode_queue_transition(row, f"decode skip {index}", resumed=False)
        for index, row in enumerate(skip_rows)
    ]
    resumes = [
        validate_decode_queue_transition(row, f"decode resume {index}", resumed=True)
        for index, row in enumerate(resume_rows)
    ]
    for park in parks:
        request_id = park["request_ids"][0]
        matching = [
            resume
            for resume in resumes
            if resume["ts_unix_nanos"] > park["ts_unix_nanos"]
            and common.request_identity_matches(resume["request_id"], request_id)
            and resume["exact_source_changed"]
        ]
        require(matching, f"parked decode {request_id} did not resume after its exact source changed")

    deferred_request_ids = sorted(
        {request_id for event in deferrals for request_id in event["request_ids"]}
    )
    completed_rows = [
        row for row in rows if str(row.get("phase", "")).endswith("request_completed")
    ]
    for request_id in deferred_request_ids:
        require(
            any(common.request_identity_matches(row.get("request_id"), request_id) for row in completed_rows),
            f"decode-capacity request {request_id} has no completion event",
        )
    return {
        "deferral_events": len(deferrals),
        "split_events": len(splits),
        "park_events": len(parks),
        "skip_events": len(skips),
        "resume_events": len(resumes),
        "stages": sorted({event["stage"] for event in deferrals}),
        "max_attempted_decode_width": max(event["width"] for event in deferrals),
        "deferred_request_ids": deferred_request_ids,
    }


def start_stream_group(
    server: common.ServerSession,
    *,
    out_dir: Path,
    prefix: str,
    timeout: float,
) -> dict[str, common.StreamTask]:
    tasks = {
        slot: common.StreamTask(
            port=server.port,
            model=server.model_id,
            role=f"{prefix}-{slot}",
            workload_slot=slot,
            max_tokens=STREAM_MAX_TOKENS,
            out_dir=out_dir,
            timeout=timeout,
        )
        for slot in ("A", "B", "C")
    }
    for task in tasks.values():
        task.start()
    return tasks


def wait_stream_group(
    tasks: dict[str, common.StreamTask],
    *,
    trace_path: Path,
    trace_baseline_bytes: int,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(set(tasks) == {"A", "B", "C"}, "stream group must contain A/B/C")
    started = time.monotonic()
    deadline = started + timeout
    guard = common.PressureStopGuard(
        initial_progress={role: task.live_content_chunks() for role, task in tasks.items()},
        started_monotonic=started,
        no_progress_timeout=STOP_POLICY["no_progress_timeout_seconds"],
        max_unchanged_skips=MAX_DECODE_CAPACITY_EVENTS,
        max_trace_bytes=STOP_POLICY["max_trace_bytes"],
    )
    while True:
        now = time.monotonic()
        active_roles = {role for role, task in tasks.items() if task.is_alive()}
        for role, task in tasks.items():
            if role not in active_roles and isinstance(task.result, dict) and task.result.get("error"):
                raise DecodeCapacityGateError(f"{role} stream failed: {task.result['error']}")
        trace_bytes = trace_path.stat().st_size if trace_path.is_file() else 0
        require(trace_bytes >= trace_baseline_bytes, "scheduler trace was truncated")
        guard.observe(
            progress={role: task.live_content_chunks() for role, task in tasks.items()},
            unchanged_skips=0,
            trace_bytes=trace_bytes - trace_baseline_bytes,
            now_monotonic=now,
            active_roles=active_roles,
        )
        if not active_roles:
            break
        require(now < deadline, "stream group exceeded the bounded joint timeout")
        time.sleep(min(0.05, max(0.0, deadline - now)))

    results = {role: task.join(0) for role, task in tasks.items()}
    for role, result in results.items():
        common.validate_stream(result, role)
    require(
        max(result["started_wall_ns"] for result in results.values())
        < min(result["finished_wall_ns"] for result in results.values()),
        "three streams were not concurrently live",
    )
    return results, {
        "duration_seconds": time.monotonic() - started,
        "content_chunks_by_role": {
            role: task.live_content_chunks() for role, task in tasks.items()
        },
        "max_stall_seconds_by_role": guard.max_stall_seconds_by_role,
        "trace_baseline_bytes": trace_baseline_bytes,
        "trace_bytes": (
            trace_path.stat().st_size - trace_baseline_bytes if trace_path.is_file() else 0
        ),
    }


def server_session(
    *,
    repo: Path,
    binary: Path,
    model: Path,
    port: int,
    out_dir: Path,
    runtime_budget: int | None,
    max_num_batched_tokens: int,
    startup_timeout: float,
) -> common.ServerSession:
    return common.ServerSession(
        repo=repo,
        binary=binary,
        model=model,
        port=port,
        out_dir=out_dir,
        runtime_budget=runtime_budget,
        startup_timeout=startup_timeout,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=max_num_batched_tokens,
        prefill_first_until_active=PREFILL_FIRST_UNTIL_ACTIVE,
    )


def collect(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    binary = args.binary.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    require(not out.exists(), f"collection output already exists: {out}")
    require(binary.is_file(), f"missing binary: {binary}")
    require(model.is_dir(), f"missing model directory: {model}")
    require(args.port < 65535, "base port leaves no target-server port")
    require(
        0 < args.request_timeout <= STOP_POLICY["joint_stream_timeout_seconds"],
        "request timeout exceeds the bounded stop policy",
    )
    out.mkdir(parents=True)
    provenance = {
        "schema_version": 1,
        "command_line": sys.argv,
        "git_sha": common.command_output(["git", "rev-parse", "HEAD"], repo),
        "dirty_status": common.command_output(["git", "status", "--short"], repo),
        "binary_path": str(binary),
        "binary_sha256": common.sha256(binary),
        "model_path": str(model),
        "nvidia_smi": common.command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            repo,
        ).splitlines(),
        "sanitized_env": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "LD_LIBRARY_PATH", "RUST_LOG")
            if key in os.environ
        },
        "started_wall_ns": time.time_ns(),
    }
    common.write_json(out / "provenance.json", provenance)
    require(common.GIT_SHA_RE.fullmatch(provenance["git_sha"]) is not None, "invalid git SHA")
    require(not provenance["dirty_status"], "CUDA evidence requires a clean checkout")
    sessions: list[common.ServerSession] = []
    tasks: dict[str, common.StreamTask] = {}
    collection: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_decode_capacity_collection",
        "status": "reject",
        "server_policy": SERVER_POLICY,
        "stop_policy": STOP_POLICY,
        "error": None,
    }
    try:
        run_summary = common.collect_run_smoke(
            repo=repo,
            binary=binary,
            model=model,
            out_dir=out / "run",
            timeout=args.request_timeout,
        )
        calibration = server_session(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port,
            out_dir=out / "calibration",
            runtime_budget=None,
            max_num_batched_tokens=CALIBRATION_TOKEN_BUDGET,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(calibration)
        calibration_trace_baseline = (
            calibration.trace_path.stat().st_size if calibration.trace_path.is_file() else 0
        )
        tasks = start_stream_group(
            calibration,
            out_dir=out / "calibration" / "clients",
            prefix="calibration",
            timeout=args.request_timeout,
        )
        calibration_clients, calibration_monitor = wait_stream_group(
            tasks,
            trace_path=calibration.trace_path,
            trace_baseline_bytes=calibration_trace_baseline,
            timeout=args.request_timeout,
        )
        tasks = {}
        calibration_health = calibration.health("health.final.json")
        calibration_executor = common.find_executor_snapshot(calibration_health)
        require(calibration_executor is not None, "calibration health has no vNext executor")
        calibration_pool = common.quiescent_pool_snapshot(
            calibration_executor, "decode calibration"
        )
        exact_budget = calibration_pool["budget_claimed_bytes"]
        calibration.stop()

        target = server_session(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port + 1,
            out_dir=out / "target",
            runtime_budget=exact_budget,
            max_num_batched_tokens=TARGET_TOKEN_BUDGET,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(target)
        target_trace_baseline = target.trace_path.stat().st_size if target.trace_path.is_file() else 0
        tasks = start_stream_group(
            target,
            out_dir=out / "target" / "clients",
            prefix="target",
            timeout=args.request_timeout,
        )
        target_clients, target_monitor = wait_stream_group(
            tasks,
            trace_path=target.trace_path,
            trace_baseline_bytes=target_trace_baseline,
            timeout=args.request_timeout,
        )
        tasks = {}
        target_started = min(result["started_wall_ns"] for result in target_clients.values())
        target_finished = max(result["finished_wall_ns"] for result in target_clients.values())
        decode_summary = validate_decode_trace(
            common.read_trace(target.trace_path),
            started_wall_ns=target_started,
            finished_wall_ns=target_finished,
        )
        target_health = target.health("health.final.json")
        target_executor = common.find_executor_snapshot(target_health)
        require(target_executor is not None, "target health has no vNext executor")
        target_pool = common.quiescent_pool_snapshot(target_executor, "decode target")
        common.require_replayed_pool_snapshot(calibration_pool, target_pool, exact_budget)
        target.stop()

        collection.update(
            {
                "status": "collected",
                "source_git_sha": provenance["git_sha"],
                "binary_sha256": provenance["binary_sha256"],
                "model_path": str(model),
                "run": run_summary,
                "calibration": {
                    "clients": calibration_clients,
                    "monitor": calibration_monitor,
                    "pool_snapshot": calibration_pool,
                    "health_final": "calibration/health.final.json",
                    "trace": "calibration/scheduler-trace.jsonl",
                },
                "exact_budget_bytes": exact_budget,
                "target": {
                    "clients": target_clients,
                    "monitor": target_monitor,
                    "pool_snapshot": target_pool,
                    "decode_summary": decode_summary,
                    "health_final": "target/health.final.json",
                    "trace": "target/scheduler-trace.jsonl",
                },
                "finished_wall_ns": time.time_ns(),
                "error": None,
            }
        )
        common.write_json(out / "collection.json", collection)
        print(f"{COLLECT_PREFIX}: {out}")
        return 0
    except Exception as error:
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception:
                pass
        unsettled = common.settle_stream_tasks(tasks, timeout=10.0)
        collection["failure_cleanup"] = {
            "unsettled_client_roles": unsettled,
            "finished_wall_ns": time.time_ns(),
        }
        collection["error"] = str(error)
        collection["finished_wall_ns"] = time.time_ns()
        common.write_json(out / "collection.json", collection)
        print(f"{FAIL_PREFIX}: {out}: {error}", file=sys.stderr)
        return 1
    finally:
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception as error:
                print(f"decode capacity session cleanup failed: {error}", file=sys.stderr)


def validate_stream_group(
    root: Path,
    phase: str,
    results: dict[str, Any],
) -> tuple[int, int, dict[str, float]]:
    require(isinstance(results, dict) and set(results) == {"A", "B", "C"}, f"{phase}: invalid client set")
    for role, result in results.items():
        require(isinstance(result, dict), f"{phase}-{role}: result is invalid")
        common.validate_stream(result, f"{phase}-{role}")
    started = min(result["started_wall_ns"] for result in results.values())
    finished = max(result["finished_wall_ns"] for result in results.values())
    require(
        max(result["started_wall_ns"] for result in results.values())
        < min(result["finished_wall_ns"] for result in results.values()),
        f"{phase}: streams did not overlap",
    )
    silences: dict[str, float] = {}
    for role, result in results.items():
        events = root / phase / "clients" / f"{phase}-{role}.events.jsonl"
        silences[role] = common.max_stream_silence_seconds(
            result,
            common.read_stream_content_times(events),
            monitored_from_wall_ns=started,
        )
        require(
            silences[role] < STOP_POLICY["no_progress_timeout_seconds"],
            f"{phase}-{role}: token progress stalled for {silences[role]:.3f}s",
        )
    return started, finished, silences


def validate(root: Path, out: Path) -> int:
    root = root.resolve()
    out = out.resolve()
    require(root.is_dir(), f"missing collection directory: {root}")
    out.mkdir(parents=True, exist_ok=True)
    collection = common.read_json(root / "collection.json")
    provenance = common.read_json(root / "provenance.json")
    require(collection.get("status") == "collected", f"collection is unusable: {collection.get('error')}")
    require(collection.get("server_policy") == SERVER_POLICY, "collection used a non-canonical server policy")
    require(collection.get("stop_policy") == STOP_POLICY, "collection used a non-canonical stop policy")
    source_git_sha = collection.get("source_git_sha")
    require(common.GIT_SHA_RE.fullmatch(str(source_git_sha)) is not None, "invalid source git SHA")
    require(source_git_sha == provenance.get("git_sha"), "collection/provenance SHA mismatch")
    require(not provenance.get("dirty_status"), "CUDA evidence used a dirty checkout")
    binary_sha256 = collection.get("binary_sha256")
    require(common.SHA256_RE.fullmatch(str(binary_sha256)) is not None, "invalid binary SHA256")
    require(binary_sha256 == provenance.get("binary_sha256"), "binary SHA mismatch")
    gpu_rows = provenance.get("nvidia_smi")
    require(
        isinstance(gpu_rows, list) and len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0],
        "artifact is not from exactly one RTX 4090",
    )
    require("Qwen3.5-4B" in str(collection.get("model_path")), "artifact model is not Qwen3.5-4B")

    calibration = collection.get("calibration")
    target = collection.get("target")
    require(isinstance(calibration, dict) and isinstance(target, dict), "scenario summaries are missing")
    calibration_health = common.read_json(root / str(calibration.get("health_final")))
    target_health = common.read_json(root / str(target.get("health_final")))
    calibration_executor = common.find_executor_snapshot(calibration_health)
    target_executor = common.find_executor_snapshot(target_health)
    require(calibration_executor is not None and target_executor is not None, "raw executor snapshots are missing")
    for identity in ("model_id", "family_fingerprint", "program_fingerprint", "plan_hash"):
        require(
            calibration_executor.get(identity) == target_executor.get(identity),
            f"calibration/target changed {identity}",
        )
    calibration_pool = common.quiescent_pool_snapshot(calibration_executor, "raw decode calibration")
    target_pool = common.quiescent_pool_snapshot(target_executor, "raw decode target")
    exact_budget = collection.get("exact_budget_bytes")
    require(
        isinstance(exact_budget, int) and exact_budget == calibration_pool["budget_claimed_bytes"],
        "exact budget does not match calibrated installed backing",
    )
    require(calibration.get("pool_snapshot") == calibration_pool, "calibration summary differs from raw health")
    require(target.get("pool_snapshot") == target_pool, "target summary differs from raw health")
    common.require_replayed_pool_snapshot(calibration_pool, target_pool, exact_budget)
    policy = target_executor.get("runtime_memory_policy")
    require(isinstance(policy, dict), "target runtime memory policy is missing")
    require(
        policy.get("capacity_bytes", 0) - policy.get("reserve_bytes", 0) == exact_budget,
        "target runtime did not use the calibrated exact budget",
    )

    calibration_started, calibration_finished, calibration_silence = validate_stream_group(
        root, "calibration", calibration.get("clients")
    )
    target_started, target_finished, target_silence = validate_stream_group(
        root, "target", target.get("clients")
    )
    for slot in ("A", "B", "C"):
        common.validate_replayed_workload(
            slot,
            {
                "calibration": calibration["clients"][slot],
                "target": target["clients"][slot],
            },
        )
    calibration_rows = common.read_trace(root / str(calibration.get("trace")))
    require(
        not any(
            row.get("phase") == "vnext.decode_capacity_deferred"
            and isinstance(row.get("ts_unix_nanos"), int)
            and calibration_started <= row["ts_unix_nanos"] <= calibration_finished
            for row in calibration_rows
        ),
        "single-token calibration unexpectedly hit decode capacity pressure",
    )
    target_trace_path = root / str(target.get("trace"))
    target_trace_bytes = target_trace_path.stat().st_size
    require(target_trace_bytes <= STOP_POLICY["max_trace_bytes"], "target trace exceeds its byte ceiling")
    decode_summary = validate_decode_trace(
        common.read_trace(target_trace_path),
        started_wall_ns=target_started,
        finished_wall_ns=target_finished,
    )
    require(target.get("decode_summary") == decode_summary, "decode summary differs from raw trace")

    counters = target_executor.get("counters")
    require(isinstance(counters, dict), "target executor counters are missing")
    counter_by_stage = {
        "sequence_extension": "extension_deferrals",
        "step_admission": "step_deferrals",
        "submission_wave": "wave_deferrals",
    }
    for stage in decode_summary["stages"]:
        counter = counter_by_stage[stage]
        require(counters.get(counter, 0) > 0, f"target counter {counter} did not record its trace stage")
    require(target_executor.get("active_sequences") == 0, "target still has active sequences")
    require(target_executor.get("pending_sequences") == 0, "target still has pending sequences")
    require(target_executor.get("pending_prefill_maintenance") == 0, "target still has prefill maintenance")
    require(target_executor.get("executing_prefills") == 0, "target still has executing prefills")
    require(target_executor.get("staged_prefill_requests") == 0, "target still has staged prefill requests")
    require(target_executor.get("staged_prefill_sequences") == 0, "target still has staged prefill sequences")
    require(target_health.get("engine", {}).get("active_requests") == 0, "target engine still has active requests")
    require(target_health.get("engine", {}).get("queued_requests") == 0, "target engine still has queued requests")

    run_result = common.read_json(root / "run" / "result.json")
    require(run_result.get("returncode") == 0, "ferrum run smoke failed")
    run_rows = [
        json.loads(line)
        for line in (root / "run" / "stdout.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(any("paris" in str(row.get("content", "")).lower() for row in run_rows), "run smoke is not Paris")
    run_phases = {
        row.get("phase") for row in common.read_trace(root / "run" / "scheduler-trace.jsonl")
    }
    require("vnext.operation_submitted" in run_phases, "run smoke has no vNext submission")
    require("vnext.request_completed" in run_phases, "run smoke has no vNext completion")

    for path in [*root.rglob("*.log"), *root.rglob("*.sse")]:
        text = path.read_text(errors="replace").lower()
        require("\ufffd" not in text, f"Unicode replacement character in {path}")
        for pattern in common.FORBIDDEN_PATTERNS:
            require(pattern not in text, f"forbidden pattern {pattern!r} in {path}")

    pass_line = f"{PASS_PREFIX}: {out}"
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_decode_capacity_validation",
        "status": "pass",
        "source_git_sha": source_git_sha,
        "binary_sha256": binary_sha256,
        "model_path": collection["model_path"],
        "source_artifact": str(root),
        "source_collection_sha256": common.sha256(root / "collection.json"),
        "exact_budget_bytes": exact_budget,
        "server_policy": SERVER_POLICY,
        "stop_policy": STOP_POLICY,
        "decode_summary": decode_summary,
        "target_trace_bytes": target_trace_bytes,
        "calibration_window_ns": [calibration_started, calibration_finished],
        "target_window_ns": [target_started, target_finished],
        "max_silence_seconds": {
            "calibration": calibration_silence,
            "target": target_silence,
        },
        "does_not_prove": ["S1", "G04", "performance", "release"],
        "pass_line": pass_line,
    }
    common.write_json(out / "validation.json", manifest)
    common.write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def self_test() -> int:
    wait_condition = {
        "coordinator_id": 7,
        "observed": [{"source": {"domain": 5}, "epoch": 3}],
    }
    capacity_evidence = {
        "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
        "wait_condition": wait_condition,
    }

    def deferral(ts: int, decision: str, request_ids: list[str]) -> dict[str, Any]:
        return {
            "ts_unix_nanos": ts,
            "phase": "vnext.decode_capacity_deferred",
            "status": "ok",
            "error": None,
            "shape": {
                "decision": decision,
                "attempted_decode_width": len(request_ids),
                "execution_stage": "step_admission",
                "decode_submit_observed": False,
            },
            "attributes": {
                "request_ids": request_ids,
                "capacity_evidence": capacity_evidence,
                "scheduler_snapshot": {},
            },
        }

    rows = [
        deferral(100, "split_cohort", ["A", "B", "C"]),
        deferral(110, "wait_for_release", ["B"]),
        {
            "ts_unix_nanos": 120,
            "phase": "vnext.decode_capacity_skipped_unchanged",
            "status": "ok",
            "error": None,
            "request_id": "B",
            "shape": {
                "decision": "skipped_unchanged",
                "decode_submit_observed": False,
                "probe_performed": False,
            },
            "attributes": {
                "deferral_evidence": {
                    "action": "wait_for_release",
                    "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
                    "current": {"coordinator_id": 7, "release_epoch": 12, "capacity_epoch": 14},
                    "wait_condition": wait_condition,
                    "current_wait_sources": [{"source": {"domain": 5}, "epoch": 3}],
                }
            },
        },
        {
            "ts_unix_nanos": 130,
            "phase": "vnext.decode_capacity_resumed",
            "status": "ok",
            "error": None,
            "request_id": "request.product.B",
            "shape": {
                "decision": "exact_source_changed",
                "decode_submit_observed": False,
                "probe_performed": False,
                "exact_source_changed": True,
                "policy_epoch_changed": False,
            },
            "attributes": {
                "deferral_evidence": {
                    "action": "wait_for_release",
                    "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
                    "current": {"coordinator_id": 7, "release_epoch": 13, "capacity_epoch": 14},
                    "wait_condition": wait_condition,
                    "current_wait_sources": [{"source": {"domain": 5}, "epoch": 4}],
                }
            },
        },
        *[
            {
                "ts_unix_nanos": 140 + index,
                "phase": "vnext.request_completed",
                "request_id": request_id,
            }
            for index, request_id in enumerate(("A", "B", "C"))
        ],
    ]
    summary = validate_decode_trace(rows, started_wall_ns=90, finished_wall_ns=150)
    require(summary["split_events"] == 1, "self-test lost split evidence")
    require(summary["park_events"] == 1, "self-test lost park evidence")
    require(summary["resume_events"] == 1, "self-test lost resume evidence")

    unchanged_resume = json.loads(json.dumps(rows))
    unchanged_resume[3]["attributes"]["deferral_evidence"]["current_wait_sources"][0]["epoch"] = 3
    try:
        validate_decode_trace(unchanged_resume, started_wall_ns=90, finished_wall_ns=150)
        raise AssertionError("unchanged exact source unexpectedly resumed")
    except common.CapacityGateError:
        pass
    try:
        validate_decode_trace(rows[1:], started_wall_ns=90, finished_wall_ns=150)
        raise AssertionError("trace without adaptive split unexpectedly passed")
    except common.CapacityGateError:
        pass
    print("FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--repo", type=Path, default=Path.cwd())
    collect_parser.add_argument("--binary", type=Path, required=True)
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--port", type=int, default=18130)
    collect_parser.add_argument("--startup-timeout", type=float, default=600)
    collect_parser.add_argument("--request-timeout", type=float, default=300)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    validate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.command == "collect":
            return collect(args)
        if args.command == "validate":
            return validate(args.artifact, args.out)
        parser.error("a command is required")
    except common.CapacityGateError as error:
        target = getattr(args, "out", Path("."))
        print(f"{FAIL_PREFIX}: {target}: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
