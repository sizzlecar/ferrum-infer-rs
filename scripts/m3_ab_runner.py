#!/usr/bin/env python3
"""Reusable M3 server A/B runner.

The runner keeps the old shell-wrapper ergonomics while centralizing the
fragile parts: preflight capture, server lifecycle, correctness gates,
bench-serve invocation, manifest writing, cleanup, and summary extraction.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3_validate_runner_artifact import (
    ValidationError as ArtifactValidationError,
    VALIDATION_TOUCHED_AREAS,
    required_gates_for_touched_areas,
    validate_profile_event,
)


RUNTIME_PRESET_ENV: dict[str, dict[str, str]] = {
    "m3_qwen3_30b_a3b_int4": {
        "HF_HOME": "/workspace/hf-cache",
    }
}

ARTIFACT_VERDICTS = {"pass", "fail", "diagnostic-only"}
VALIDATION_CHANGE_TYPES = {
    "default_path",
    "opt_in_experiment",
    "diagnostic",
    "api_only",
    "build_loop",
}
DEFAULT_PATH_REQUIRED_CONCURRENCY_CELLS = [1, 4, 16, 32]
LOG_SNIPPET_PROFILE_KEYS = {
    "snippet_regex",
    "required_patterns",
    "required_any_patterns",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def run_capture(args: list[str], *, timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, str(exc)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def parse_process_scan(ps: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in ps.splitlines():
        line = line.strip()
        if not line or line.startswith("PID "):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, etime, cmd = parts
        try:
            pid_int = int(pid)
            ppid_int = int(ppid)
        except ValueError:
            continue
        entries.append(
            {
                "pid": pid_int,
                "ppid": ppid_int,
                "stat": stat,
                "etime": etime,
                "cmd": cmd,
            }
        )
    return entries


@dataclass
class RunPaths:
    case_dir: Path
    server_log: Path
    bench_log: Path
    bench_json: Path
    manifest: Path
    health_json: Path
    effective_config_json: Path
    decision_trace_jsonl: Path
    profile_jsonl: Path


class Runner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo = Path.cwd()
        self.out_root = Path(config["out_root"])
        self.model_dir = Path(config["model_dir"])
        self.bin = Path(config.get("bin", "./target/release/ferrum"))
        self.hf_model = config.get("hf_model", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")
        self.server_pid: int | None = None
        self.server_proc: subprocess.Popen[str] | None = None
        self.run_manifest: dict[str, Any] = {}

    def validate(self) -> None:
        required = ["out_root", "model_dir", "cases"]
        for key in required:
            if key not in self.config:
                raise SystemExit(f"missing required config field: {key}")
        if not isinstance(self.config.get("cases"), list) or not self.config["cases"]:
            raise SystemExit("config.cases must be a non-empty list")
        if not self.model_dir.is_dir():
            raise SystemExit(f"MODEL_DIR does not exist: {self.model_dir}")
        preset = self.config.get("preset")
        if preset and preset not in RUNTIME_PRESET_ENV:
            raise SystemExit(f"unknown runtime preset: {preset}")
        profile = self.config.get("profile", {})
        if profile and not isinstance(profile, dict):
            raise SystemExit("profile must be an object when set")
        log_profile_keys = sorted(key for key in LOG_SNIPPET_PROFILE_KEYS if key in profile)
        if log_profile_keys:
            raise SystemExit(
                "text-log profile matching is not supported in m3_ab_runner configs; "
                f"remove {log_profile_keys} and use profile.structured=true with "
                "required_events/required_any_events"
            )
        profile_env_cases = profile.get("profile_env_cases", [])
        if profile_env_cases and not isinstance(profile_env_cases, list):
            raise SystemExit("profile.profile_env_cases must be a list when set")
        self.validate_verdict(str(self.config.get("artifact_verdict", "pass")), "config")
        case_names = set()
        for case in self.config["cases"]:
            if "name" not in case:
                raise SystemExit("every case requires name")
            case_names.add(str(case["name"]))
            self.validate_verdict(
                str(case.get("artifact_verdict", self.config.get("artifact_verdict", "pass"))),
                case["name"],
            )
        missing_profile_cases = [
            str(name) for name in profile_env_cases if str(name) not in case_names
        ]
        if missing_profile_cases:
            raise SystemExit(
                "profile.profile_env_cases references unknown cases: "
                + ", ".join(missing_profile_cases)
            )
        self.validate_validation_config()

    def validate_verdict(self, verdict: str, where: str) -> None:
        if verdict not in ARTIFACT_VERDICTS:
            raise SystemExit(
                f"{where}: artifact_verdict must be one of {sorted(ARTIFACT_VERDICTS)}"
            )

    def validate_validation_config(self) -> None:
        validation = self.config.get("validation", {})
        if validation is None:
            validation = {}
        if not isinstance(validation, dict):
            raise SystemExit("validation must be an object when set")
        change_type = str(validation.get("change_type", "opt_in_experiment"))
        if change_type not in VALIDATION_CHANGE_TYPES:
            raise SystemExit(
                f"validation.change_type must be one of {sorted(VALIDATION_CHANGE_TYPES)}"
            )
        for key in ("touched_areas", "required_correctness_gates"):
            value = validation.get(key, [])
            if value is not None and not isinstance(value, list):
                raise SystemExit(f"validation.{key} must be a list when set")
        touched = [str(item) for item in self.validation_touched_areas()]
        invalid_touched = sorted(area for area in touched if area not in VALIDATION_TOUCHED_AREAS)
        if invalid_touched:
            raise SystemExit(
                "validation.touched_areas invalid: "
                + ", ".join(invalid_touched)
                + f"; allowed={sorted(VALIDATION_TOUCHED_AREAS)}"
            )
        required_gates = set(self.validation_required_correctness_gates())
        missing_area_gates = sorted(required_gates_for_touched_areas(touched) - required_gates)
        if missing_area_gates:
            raise SystemExit(
                "validation.required_correctness_gates missing gates required by "
                f"touched_areas: {missing_area_gates}"
            )
        for key in ("local_gates", "skipped_gates"):
            value = validation.get(key, [])
            if value is not None and not isinstance(value, list):
                raise SystemExit(f"validation.{key} must be a list when set")
            for item in value or []:
                if not isinstance(item, dict):
                    raise SystemExit(f"validation.{key} entries must be objects")
                if not str(item.get("name", "")).strip():
                    raise SystemExit(f"validation.{key} entries require non-empty name")
        if (
            "performance_regression_required" in validation
            and not isinstance(validation["performance_regression_required"], bool)
        ):
            raise SystemExit("validation.performance_regression_required must be boolean")
        impact = validation.get("benchmark_impact")
        if impact is not None:
            if not isinstance(impact, dict):
                raise SystemExit("validation.benchmark_impact must be an object when set")
            if not isinstance(impact.get("m3_benchmark_exercised"), bool):
                raise SystemExit(
                    "validation.benchmark_impact.m3_benchmark_exercised must be boolean"
                )
            for key in ("reason", "evidence"):
                if not str(impact.get(key, "")).strip():
                    raise SystemExit(f"validation.benchmark_impact.{key} must be non-empty")
        cells = validation.get("required_concurrency_cells")
        if cells is not None:
            if not isinstance(cells, list):
                raise SystemExit("validation.required_concurrency_cells must be a list")
            for cell in cells:
                if not isinstance(cell, int) or isinstance(cell, bool) or cell <= 0:
                    raise SystemExit(
                        "validation.required_concurrency_cells entries must be positive integers"
                    )

    def verdict_for_case(self, case: dict[str, Any]) -> str:
        return str(case.get("artifact_verdict", self.config.get("artifact_verdict", "pass")))

    def publishability_for_verdict(
        self,
        verdict: str,
        case: dict[str, Any] | None = None,
        *,
        failure_reason: str | None = None,
    ) -> tuple[bool, str | None]:
        case = case or {}
        not_publishable = bool(
            case.get("not_publishable", self.config.get("not_publishable", False))
        ) or verdict in {"fail", "diagnostic-only"}
        reason = (
            failure_reason
            or case.get("not_publishable_reason")
            or self.config.get("not_publishable_reason")
        )
        if not_publishable and not reason:
            reason = f"{verdict} artifact"
        return not_publishable, reason

    def validation_config(self) -> dict[str, Any]:
        validation = self.config.get("validation", {})
        return validation if isinstance(validation, dict) else {}

    def validation_change_type(self) -> str:
        validation = self.validation_config()
        configured = validation.get("change_type")
        if configured:
            return str(configured)
        if str(self.config.get("artifact_verdict", "pass")) == "diagnostic-only":
            return "diagnostic"
        return "opt_in_experiment"

    def validation_touched_areas(self) -> list[str]:
        validation = self.validation_config()
        touched = validation.get("touched_areas") or validation.get("touched") or []
        result = [str(item) for item in touched if str(item).strip()]
        return result or ["benchmark_harness"]

    def validation_required_correctness_gates(self) -> list[str]:
        validation = self.validation_config()
        configured = validation.get("required_correctness_gates")
        if configured:
            return [str(item) for item in configured]
        gates = self.config.get("gates", {})
        required: list[str] = []
        if gates.get("paris", True):
            required.append("paris")
        if gates.get("multi_turn", False):
            required.append("multi_turn_paris")
        required.append("bench_completion")
        return required

    def validation_performance_required(self) -> bool:
        validation = self.validation_config()
        if self.validation_change_type() == "default_path":
            return True
        if "performance_regression_required" in validation:
            return bool(validation["performance_regression_required"])
        return bool(self.config.get("baseline_case"))

    def validation_required_concurrency_cells(self) -> list[int]:
        validation = self.validation_config()
        configured = validation.get("required_concurrency_cells")
        if configured is not None:
            return sorted({int(cell) for cell in configured})
        if self.validation_change_type() == "default_path" and self.validation_performance_required():
            return list(DEFAULT_PATH_REQUIRED_CONCURRENCY_CELLS)
        return []

    def normalize_validation_gates(self, key: str) -> list[dict[str, Any]]:
        gates = self.validation_config().get(key, []) or []
        normalized: list[dict[str, Any]] = []
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            normalized.append(
                {
                    "name": str(gate.get("name", "")),
                    "ok": gate.get("ok"),
                    "evidence": gate.get("evidence"),
                    "required": bool(gate.get("required", True)),
                    "reason": gate.get("reason"),
                }
            )
        return normalized

    def validation_checklist(
        self,
        case: dict[str, Any] | None = None,
        *,
        correctness_gates: list[dict[str, Any]] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        correctness_gates = correctness_gates or []
        metrics = metrics or {}
        observed = [
            {
                "name": str(gate.get("name", "")),
                "ok": bool(gate.get("ok", False)),
                "evidence": gate.get("path") or gate.get("content"),
            }
            for gate in correctness_gates
        ]
        bench_required = "bench_completion" in self.validation_required_correctness_gates()
        completed = metrics.get("completed")
        errored = metrics.get("errored")
        bench_ok = None
        if completed is not None or errored is not None:
            try:
                bench_ok = int(completed or 0) > 0 and int(errored or 0) == 0
            except (TypeError, ValueError):
                bench_ok = False
        checklist = {
            "schema_version": 1,
            "change_type": self.validation_change_type(),
            "touched_areas": self.validation_touched_areas(),
            "required_correctness_gates": self.validation_required_correctness_gates(),
            "observed_correctness_gates": observed,
            "bench_completion": {
                "required": bench_required,
                "ok": bench_ok,
                "completed": completed,
                "errored": errored,
            },
            "local_gates": self.normalize_validation_gates("local_gates"),
            "skipped_gates": self.normalize_validation_gates("skipped_gates"),
            "performance_regression_required": self.validation_performance_required(),
            "baseline_case": self.config.get("baseline_case"),
            "case": None if case is None else str(case.get("name")),
        }
        benchmark_impact = self.validation_config().get("benchmark_impact")
        if isinstance(benchmark_impact, dict):
            checklist["benchmark_impact"] = {
                "m3_benchmark_exercised": bool(
                    benchmark_impact.get("m3_benchmark_exercised")
                ),
                "reason": str(benchmark_impact.get("reason", "")),
                "evidence": str(benchmark_impact.get("evidence", "")),
            }
        return checklist

    def maybe_build(self) -> None:
        if not self.config.get("build", False):
            return
        features = self.config.get("features", "")
        cmd = ["cargo", "build", "--release", "-p", "ferrum-cli"]
        if features:
            cmd += ["--features", features]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    def preflight(self) -> dict[str, Any]:
        git_head_rc, git_head = run_capture(["git", "rev-parse", "HEAD"])
        git_status_rc, git_status = run_capture(["git", "status", "--short"])
        smi_rc, smi = run_capture(["nvidia-smi"], timeout=10)
        ps_rc, ps = run_capture(["ps", "-eo", "pid,ppid,stat,etime,cmd"], timeout=10)
        return {
            "date": now_iso(),
            "host": os.uname().nodename,
            "repo": str(self.repo),
            "git_head": git_head.strip() if git_head_rc == 0 else None,
            "git_status_short": git_status.splitlines() if git_status_rc == 0 else [],
            "nvidia_smi": smi,
            "gpu_process_state": ps,
            "binary_sha256": sha256_file(self.bin),
            "features": self.config.get("features", ""),
            "preset": self.config.get("preset"),
            "model_dir": str(self.model_dir),
            "hf_model": self.hf_model,
            "bin": str(self.bin),
        }

    def make_run_manifest(self, preflight: dict[str, Any]) -> None:
        verdict = str(self.config.get("artifact_verdict", "pass"))
        not_publishable, reason = self.publishability_for_verdict(verdict)
        self.run_manifest = {
            "runner": "scripts/m3_ab_runner.py",
            "schema_version": 1,
            "name": self.config.get("name", "m3-ab-run"),
            "created_at": now_iso(),
            "artifact_verdict": verdict,
            "not_publishable": not_publishable,
            "not_publishable_reason": reason,
            "validation_checklist": self.validation_checklist(),
            "preflight": preflight,
            "runtime_preset": self.config.get("preset"),
            "cases": [],
            "summary_json": str(self.out_root / "summary.json"),
        }
        write_json(self.out_root / "manifest.json", self.run_manifest)

    def ensure_binary(self) -> None:
        if not self.bin.is_file() or not os.access(self.bin, os.X_OK):
            raise SystemExit(f"ferrum binary not executable: {self.bin}")

    def paths_for_case(self, case: dict[str, Any]) -> RunPaths:
        concurrency = self.config.get("concurrency", 32)
        repeats = self.config.get("repeats", 1)
        case_dir = self.out_root / f"{case['name']}_c{concurrency}_n{repeats}"
        return RunPaths(
            case_dir=case_dir,
            server_log=case_dir / "server.log",
            bench_log=case_dir / "bench.log",
            bench_json=case_dir / "bench.json",
            manifest=case_dir / "manifest.json",
            health_json=case_dir / "health.json",
            effective_config_json=case_dir / "effective_config.json",
            decision_trace_jsonl=case_dir / "decision_trace.jsonl",
            profile_jsonl=case_dir / "profile.jsonl",
        )

    def preset_env(self) -> dict[str, str]:
        preset = self.config.get("preset")
        if not preset:
            return {}
        return dict(RUNTIME_PRESET_ENV[preset])

    def model_path_env(self) -> dict[str, str]:
        return {"FERRUM_MODEL_PATH": str(self.model_dir)}

    def merged_env(self, case: dict[str, Any]) -> dict[str, str]:
        env = dict(os.environ)
        for source in (
            self.model_path_env(),
            self.preset_env(),
            self.config.get("base_env", {}),
            case.get("env", {}),
        ):
            for key, value in source.items():
                env[str(key)] = str(value)
        return env

    def case_env_only(self, case: dict[str, Any]) -> dict[str, str]:
        merged = self.model_path_env()
        merged.update(self.preset_env())
        merged.update(self.config.get("base_env", {}))
        merged.update(case.get("env", {}))
        return {str(k): str(v) for k, v in merged.items()}

    def runtime_effects(self, key: str) -> list[str]:
        effects: list[str] = []
        if any(part in key for part in ("PROF", "TRACE", "DUMP", "LOG_CONFIG", "DIAG")):
            effects.append("diagnostics")
        if any(part in key for part in ("KV", "MAX_BLOCKS", "PAGED_MAX_SEQS")):
            effects.append("memory")
        if any(part in key for part in ("PREFIX_CACHE", "MODEL_PATH", "SPEC_", "REF_", "DTYPE")):
            effects.append("correctness")
        if any(
            part in key
            for part in (
                "MOE",
                "ATTN",
                "GRAPH",
                "GREEDY",
                "SCHED",
                "BACKEND",
                "CUDA",
                "FA",
                "FLASH",
                "TRITON",
                "MARLIN",
                "VLLM",
            )
        ):
            effects.append("performance")
        if not effects:
            effects.append("performance")
        return sorted(set(effects))

    def runtime_effect(self, key: str) -> str:
        return self.runtime_effects(key)[0]

    def runtime_config_snapshot(self, case: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, str] = {}
        sources: dict[str, str] = {}
        for source, source_name in (
            (self.model_path_env(), "runner_model_dir"),
            (self.config.get("base_env", {}), "script_case"),
            (case.get("env", {}), "script_case"),
        ):
            for key, value in source.items():
                if not str(key).startswith("FERRUM_"):
                    continue
                values[str(key)] = str(value)
                sources[str(key)] = source_name

        entries = [
            {
                "key": key,
                "effective_value": values[key],
                "source": sources[key],
                "effect": self.runtime_effect(key),
                "affects": self.runtime_effects(key),
            }
            for key in sorted(values)
        ]
        return {
            "schema_version": 1,
            "preset": self.config.get("preset"),
            "env_hash": sha256_text(values),
            "entries": entries,
        }

    def server_runtime_config_snapshot(self, path: Path) -> dict[str, Any]:
        data = load_json(path)
        entries = []
        for entry in data.get("entries", []):
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", ""))
            affects = entry.get("affects")
            if not isinstance(affects, list) or not affects:
                affects = self.runtime_effects(key)
            entries.append(
                {
                    "key": key,
                    "effective_value": str(entry.get("effective_value", "")),
                    "source": str(entry.get("source", "default")),
                    "effect": str(affects[0]),
                    "affects": [str(effect) for effect in affects],
                }
            )
        entries.sort(key=lambda item: item["key"])
        return {
            "schema_version": data.get("schema_version", 1),
            "preset": data.get("preset"),
            "env_hash": data.get("env_hash"),
            "entries": entries,
        }

    def runtime_config_diff(
        self, baseline: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        def by_key(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
            entries = snapshot.get("entries", [])
            if not isinstance(entries, list):
                return {}
            return {
                str(entry.get("key")): entry
                for entry in entries
                if isinstance(entry, dict) and entry.get("key") is not None
            }

        base_entries = by_key(baseline)
        candidate_entries = by_key(candidate)
        base_keys = set(base_entries)
        candidate_keys = set(candidate_entries)
        changed = []
        for key in sorted(base_keys & candidate_keys):
            if canonical_json(base_entries[key]) != canonical_json(candidate_entries[key]):
                changed.append(
                    {
                        "key": key,
                        "baseline": base_entries[key],
                        "candidate": candidate_entries[key],
                    }
                )
        return {
            "baseline_env_hash": baseline.get("env_hash"),
            "candidate_env_hash": candidate.get("env_hash"),
            "added": [candidate_entries[key] for key in sorted(candidate_keys - base_keys)],
            "removed": [base_entries[key] for key in sorted(base_keys - candidate_keys)],
            "changed": changed,
        }

    def performance_gate_thresholds(self) -> dict[str, Any]:
        configured = self.config.get("performance_gates", {})
        if configured is False:
            return {"enabled": False, "reason": "disabled by config"}
        if configured is True:
            configured = {}
        if configured is None:
            configured = {}
        if not isinstance(configured, dict):
            configured = {}
        concurrency = int(self.config.get("concurrency", 0) or 0)
        throughput_floor = float(configured.get("throughput_min_delta_pct", -3.0))
        if concurrency >= 32:
            throughput_floor = float(
                configured.get("c32_throughput_min_delta_pct", throughput_floor)
            )
        return {
            "enabled": bool(configured.get("enabled", True)),
            "throughput_min_delta_pct": throughput_floor,
            "ttft_max_regression_pct": float(
                configured.get("ttft_max_regression_pct", 10.0)
            ),
            "tpot_max_regression_pct": float(
                configured.get("tpot_max_regression_pct", 5.0)
            ),
            "itl_p95_max_regression_pct": float(
                configured.get("itl_p95_max_regression_pct", 10.0)
            ),
        }

    def metric_delta_pct(self, baseline: Any, candidate: Any) -> float | None:
        if baseline is None or candidate is None:
            return None
        try:
            baseline_f = float(baseline)
            candidate_f = float(candidate)
        except (TypeError, ValueError):
            return None
        if baseline_f <= 0:
            return None
        return (candidate_f / baseline_f - 1.0) * 100.0

    def performance_metric_gate(
        self,
        metric: str,
        baseline_row: dict[str, Any],
        candidate_row: dict[str, Any],
        *,
        threshold_type: str,
        threshold_value: float,
    ) -> dict[str, Any]:
        baseline_value = baseline_row.get(metric)
        candidate_value = candidate_row.get(metric)
        delta_pct = self.metric_delta_pct(baseline_value, candidate_value)
        if delta_pct is None:
            ok = False
            reason = "missing_or_invalid_metric"
        elif threshold_type == "min_delta_pct":
            ok = delta_pct >= threshold_value
            reason = "ok" if ok else "below_min_delta_pct"
        elif threshold_type == "max_regression_pct":
            ok = delta_pct <= threshold_value
            reason = "ok" if ok else "above_max_regression_pct"
        else:
            ok = False
            reason = "unknown_threshold_type"
        return {
            "metric": metric,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta_pct": delta_pct,
            "threshold": {"type": threshold_type, "value": threshold_value},
            "ok": ok,
            "reason": reason,
        }

    def performance_regression_gates(
        self, rows: list[dict[str, Any]], baseline_name: str | None
    ) -> dict[str, Any]:
        thresholds = self.performance_gate_thresholds()
        required_cells = self.validation_required_concurrency_cells()
        observed_cells = sorted(
            {
                int(row["concurrency"])
                for row in rows
                if isinstance(row.get("concurrency"), int)
                and not isinstance(row.get("concurrency"), bool)
            }
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "enabled": bool(thresholds.get("enabled", False)) and bool(baseline_name),
            "baseline_case": baseline_name,
            "thresholds": thresholds,
            "cases": {},
            "required_concurrency_cells": required_cells,
            "observed_concurrency_cells": observed_cells,
            "concurrency_cells_ok": all(cell in observed_cells for cell in required_cells),
        }
        if not result["enabled"]:
            result["reason"] = thresholds.get("reason") or "no baseline_case configured"
            return result
        baseline = next((row for row in rows if row.get("name") == baseline_name), None)
        if baseline is None:
            result["enabled"] = False
            result["reason"] = f"baseline case not found: {baseline_name}"
            return result

        for row in rows:
            case_name = row.get("name")
            if not case_name or case_name == baseline_name:
                continue
            metric_gates = [
                self.performance_metric_gate(
                    "throughput_mean",
                    baseline,
                    row,
                    threshold_type="min_delta_pct",
                    threshold_value=float(thresholds["throughput_min_delta_pct"]),
                ),
                self.performance_metric_gate(
                    "ttft_p50",
                    baseline,
                    row,
                    threshold_type="max_regression_pct",
                    threshold_value=float(thresholds["ttft_max_regression_pct"]),
                ),
                self.performance_metric_gate(
                    "tpot_p50",
                    baseline,
                    row,
                    threshold_type="max_regression_pct",
                    threshold_value=float(thresholds["tpot_max_regression_pct"]),
                ),
                self.performance_metric_gate(
                    "itl_p95",
                    baseline,
                    row,
                    threshold_type="max_regression_pct",
                    threshold_value=float(thresholds["itl_p95_max_regression_pct"]),
                ),
            ]
            case_result = {
                "baseline_case": baseline_name,
                "ok": all(gate["ok"] for gate in metric_gates),
                "metrics": metric_gates,
            }
            result["cases"][str(case_name)] = case_result
            row["performance_gate_ok"] = case_result["ok"]
        return result

    def structured_profile_enabled(self) -> bool:
        profile = self.config.get("profile", {})
        return bool(profile.get("structured"))

    def profile_env_enabled(self, case: dict[str, Any]) -> bool:
        profile = self.config.get("profile", {})
        if self.structured_profile_enabled():
            return True
        profile_env_cases = {str(name) for name in profile.get("profile_env_cases", [])}
        return bool(case.get("profile_env")) or str(case.get("name")) in profile_env_cases

    def profile_server_args(
        self,
        paths: RunPaths,
        case: dict[str, Any],
    ) -> list[str]:
        if not self.profile_env_enabled(case):
            return []
        git_head = self.run_manifest.get("preflight", {}).get("git_head")
        args = [
            "--profile-jsonl",
            str(paths.profile_jsonl),
            "--profile-model",
            self.hf_model,
            "--profile-concurrency",
            str(self.config.get("concurrency", 32)),
        ]
        if git_head:
            args.extend(["--profile-commit-sha", str(git_head)])
        return args

    def start_server(
        self,
        case: dict[str, Any],
        paths: RunPaths,
        env: dict[str, str],
        profile_args: list[str],
    ) -> None:
        port = int(case.get("port", self.config.get("port_base", 18480)))
        paths.case_dir.mkdir(parents=True, exist_ok=True)
        log_handle = paths.server_log.open("w")
        cmd = [
            str(self.bin),
            "serve",
            str(self.model_dir),
            "--port",
            str(port),
            "--effective-config-json",
            str(paths.effective_config_json),
            "--decision-trace-jsonl",
            str(paths.decision_trace_jsonl),
        ]
        if self.config.get("preset"):
            cmd.extend(["--runtime-preset", str(self.config["preset"])])
        cmd.extend(profile_args)
        self.server_proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.server_pid = self.server_proc.pid

    def runner_process_leaks(
        self,
        ps: str,
        *,
        paths: RunPaths,
        port: int,
        server_pid: int | None,
    ) -> list[dict[str, Any]]:
        leaks: list[dict[str, Any]] = []
        current_pid = os.getpid()
        bin_text = str(self.bin)
        bench_json_text = str(paths.bench_json)
        port_text = str(port)
        for proc in parse_process_scan(ps):
            pid = proc["pid"]
            if pid == current_pid:
                continue
            cmd = str(proc["cmd"])
            reason = None
            if server_pid is not None and (pid == server_pid or proc["ppid"] == server_pid):
                reason = "server-pid-descendant"
            elif (
                "bench-serve" in cmd
                and (bench_json_text in cmd or f"127.0.0.1:{port_text}" in cmd)
            ):
                reason = "bench-serve-case"
            elif (
                " serve " in f" {cmd} "
                and port_text in cmd
                and (bin_text in cmd or "target/release/ferrum" in cmd)
            ):
                reason = "server-port"
            if reason is not None:
                leaks.append({**proc, "reason": reason})
        return leaks

    def global_process_hygiene_findings(
        self,
        ps: str,
        *,
        runner_leaks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        runner_leak_pids = {int(proc["pid"]) for proc in runner_leaks}
        findings: list[dict[str, Any]] = []
        current_pid = os.getpid()
        for proc in parse_process_scan(ps):
            pid = proc["pid"]
            if pid == current_pid or pid in runner_leak_pids:
                continue
            cmd = str(proc["cmd"])
            reason = None
            if re.search(r"(^|[/\s])bench-serve(\s|$)", cmd) or re.search(
                r"(^|[/\s])ferrum\s+bench-serve(\s|$)", cmd
            ):
                reason = "bench-serve-global"
            elif "target/release/ferrum" in cmd or re.search(r"(^|[/\s])ferrum\s+serve\b", cmd):
                reason = "ferrum-server-global"
            elif re.search(r"(^|[/\s])cargo(\s|$)", cmd):
                reason = "cargo-global"
            elif re.search(r"(^|[/\s])nvcc(\s|$)", cmd):
                reason = "nvcc-global"
            elif re.search(r"(^|[/\s])vllm(\s|$)", cmd) or "python -m vllm" in cmd:
                reason = "vllm-global"
            if reason is not None:
                findings.append({**proc, "reason": reason})
        return findings

    def cleanup(self, paths: RunPaths, port: int) -> dict[str, Any]:
        server_pid = self.server_pid
        status: dict[str, Any] = {
            "server_pid": server_pid,
            "sent_int": False,
            "sent_kill": False,
            "returncode": None,
            "post_process_scan": "",
            "process_leak_ok": True,
            "process_leaks": [],
            "global_process_hygiene_ok": True,
            "global_process_findings": [],
        }
        proc = self.server_proc
        if proc is not None and proc.poll() is None:
            status["sent_int"] = True
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            for _ in range(60):
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            if proc.poll() is None:
                status["sent_kill"] = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
        if proc is not None:
            status["returncode"] = proc.poll()
        rc, ps = run_capture(["ps", "-eo", "pid,ppid,stat,etime,cmd"], timeout=10)
        status["post_process_scan"] = ps if rc == 0 else ""
        if rc == 0:
            leaks = self.runner_process_leaks(
                ps,
                paths=paths,
                port=port,
                server_pid=server_pid,
            )
            status["process_leaks"] = leaks
            status["process_leak_ok"] = not leaks
            global_findings = self.global_process_hygiene_findings(
                ps,
                runner_leaks=leaks,
            )
            status["global_process_findings"] = global_findings
            status["global_process_hygiene_ok"] = not global_findings
        self.server_proc = None
        self.server_pid = None
        return status

    def wait_health(self, port: int, paths: RunPaths) -> dict[str, Any]:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + float(self.config.get("health_timeout_s", 360))
        last_error = ""
        while time.time() < deadline:
            if self.server_proc is not None and self.server_proc.poll() is not None:
                raise RuntimeError(
                    f"server exited before health on port {port}; see {paths.server_log}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = response.read().decode("utf-8")
                data = json.loads(body)
                write_json(paths.health_json, data)
                return {
                    "ok": True,
                    "url": url,
                    "path": str(paths.health_json),
                    "runtime_config": data.get("config"),
                    "auto_config": data.get("auto_config"),
                }
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                time.sleep(2)
        raise RuntimeError(f"server health timeout on port {port}: {last_error}")

    def post_chat(self, port: int, payload: dict[str, Any], out_path: Path) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
        out_path.write_text(body)
        return json.loads(body)

    def run_gates(self, case: dict[str, Any], paths: RunPaths, port: int) -> list[dict[str, Any]]:
        gates: list[dict[str, Any]] = []
        gate_config = self.config.get("gates", {})
        if gate_config.get("paris", True):
            payload = {
                "model": self.hf_model,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "max_tokens": 64,
                "temperature": 0.0,
            }
            data = self.post_chat(port, payload, paths.case_dir / "paris.json")
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            ok = "Paris" in content
            gates.append({"name": "paris", "ok": ok, "content": content})
            print(f"PARIS_CONTENT= {content}", flush=True)
            if not ok:
                raise RuntimeError("Paris gate failed")

        if gate_config.get("multi_turn", False):
            payload = {
                "model": self.hf_model,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                    {"role": "user", "content": "Reply with only that city name."},
                ],
                "max_tokens": 32,
                "temperature": 0.0,
            }
            data = self.post_chat(port, payload, paths.case_dir / "multiturn.json")
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            ok = "Paris" in content
            gates.append({"name": "multi_turn_paris", "ok": ok, "content": content})
            print(f"MULTITURN_CONTENT= {content}", flush=True)
            if not ok:
                raise RuntimeError("multi-turn Paris gate failed")

        return gates

    def run_bench(self, paths: RunPaths, port: int) -> None:
        cmd = [
            str(self.bin),
            "bench-serve",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--model",
            self.hf_model,
            "--tokenizer",
            str(self.model_dir),
            "--dataset",
            self.config.get("dataset", "random"),
            "--random-input-len",
            str(self.config.get("random_input_len", 256)),
            "--random-output-len",
            str(self.config.get("random_output_len", 128)),
            "--num-prompts",
            str(self.config.get("num_prompts", 128)),
            "--warmup-requests",
            str(self.config.get("warmup_requests", 10)),
            "--n-repeats",
            str(self.config.get("repeats", 1)),
            "--concurrency",
            str(self.config.get("concurrency", 32)),
            "--output",
            "json",
            "--out",
            str(paths.bench_json),
        ]
        with paths.bench_log.open("w") as log:
            subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT, check=True)

    def collect_profile(self, paths: RunPaths) -> dict[str, Any]:
        profile = self.config.get("profile", {})
        if profile.get("structured"):
            return self.collect_structured_profile(paths)

        return {
            "enabled": False,
            "profile_jsonl": profile.get("profile_jsonl"),
        }

    def collect_structured_profile(self, paths: RunPaths) -> dict[str, Any]:
        profile = self.config.get("profile", {})
        required = profile.get("required_events", [])
        required_any = profile.get("required_any_events", [])
        events: list[dict[str, Any]] = []
        errors: list[str] = []

        if paths.profile_jsonl.exists():
            with paths.profile_jsonl.open() as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ArtifactValidationError("profile event must be an object")
                        validate_profile_event(f"{paths.profile_jsonl}:{line_no}", event)
                        events.append(event)
                    except (json.JSONDecodeError, ArtifactValidationError) as exc:
                        errors.append(f"{paths.profile_jsonl}:{line_no}: {exc}")
        else:
            errors.append(f"profile_jsonl missing: {paths.profile_jsonl}")

        event_names = [str(event.get("event")) for event in events]
        missing = [name for name in required if name not in event_names]
        missing_any = [
            group for group in required_any if not any(name in event_names for name in group)
        ]
        return {
            "enabled": True,
            "mode": "structured_jsonl",
            "profile_jsonl": str(paths.profile_jsonl),
            "source": "server_profile_sink",
            "event_count": len(events),
            "events": sorted(set(event_names)),
            "required_events": required,
            "required_any_events": required_any,
            "missing_events": missing,
            "missing_any_events": missing_any,
            "errors": errors,
            "ok": not errors and not missing and not missing_any,
        }

    def extract_metrics(self, bench_path: Path) -> dict[str, Any]:
        if not bench_path.exists():
            return {}
        data = load_json(bench_path)
        throughput = data.get("output_throughput_tps") or {}
        tpot = data.get("tpot_ms") or {}
        itl = data.get("itl_ms") or {}
        ttft = data.get("ttft_ms") or {}
        completed = data.get("completed", data.get("completed_requests"))
        errored = data.get("errored", data.get("errored_requests", data.get("errors")))
        if completed is None and isinstance(data.get("completed_per_run"), list):
            completed = sum(data["completed_per_run"])
        if errored is None and isinstance(data.get("errored_per_run"), list):
            errored = sum(data["errored_per_run"])
        return {
            "throughput_mean": throughput.get("mean", data.get("output_throughput")),
            "throughput_stddev": throughput.get("stddev", throughput.get("std")),
            "throughput_ci95_hw": throughput.get("ci95_hw"),
            "ttft_p50": (ttft.get("p50") or {}).get("mean", data.get("median_ttft_ms")),
            "tpot_p50": (tpot.get("p50") or {}).get("mean", data.get("median_tpot_ms")),
            "itl_p95": (itl.get("p95") or {}).get("mean"),
            "completed": completed,
            "errored": errored,
        }

    def count_decision_trace(self, path: Path) -> int:
        if not path.exists():
            raise RuntimeError(f"server did not write decision trace: {path}")
        count = 0
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_no}: invalid decision JSON: {exc}") from exc
                if not isinstance(event, dict) or not event.get("selection"):
                    raise RuntimeError(f"{path}:{line_no}: invalid decision event")
                count += 1
        if count == 0:
            raise RuntimeError(f"server wrote empty decision trace: {path}")
        return count

    def run_case(self, case: dict[str, Any], index: int) -> dict[str, Any]:
        port = int(case.get("port", int(self.config.get("port_base", 18480)) + index))
        paths = self.paths_for_case(case)
        case_env = self.case_env_only(case)
        case_env_hash = sha256_text(case_env)
        runtime_config_snapshot = self.runtime_config_snapshot(case)
        env = self.merged_env(case)
        profile_env_enabled = self.profile_env_enabled(case)
        profile_args = self.profile_server_args(paths, case)
        artifact_verdict = self.verdict_for_case(case)
        not_publishable, not_publishable_reason = self.publishability_for_verdict(
            artifact_verdict, case
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "name": case["name"],
            "started_at": now_iso(),
            "artifact_verdict": artifact_verdict,
            "not_publishable": not_publishable,
            "not_publishable_reason": not_publishable_reason,
            "port": port,
            "git_head": self.run_manifest.get("preflight", {}).get("git_head"),
            "git_status_short": self.run_manifest.get("preflight", {}).get("git_status_short", []),
            "binary_sha256": self.run_manifest.get("preflight", {}).get("binary_sha256"),
            "features": self.config.get("features", ""),
            "runtime_preset": self.config.get("preset"),
            "env_hash": case_env_hash,
            "runtime_config_snapshot": runtime_config_snapshot,
            "preset_env": self.preset_env(),
            "base_env": {str(k): str(v) for k, v in self.config.get("base_env", {}).items()},
            "effective_env": case_env,
            "case_env": {str(k): str(v) for k, v in case.get("env", {}).items()},
            "model_dir": str(self.model_dir),
            "server_log": str(paths.server_log),
            "bench_json": str(paths.bench_json),
            "bench_log": str(paths.bench_log),
            "effective_config_json": str(paths.effective_config_json),
            "decision_trace_jsonl": str(paths.decision_trace_jsonl),
            "auto_config_decision_count": 0,
            "profile_jsonl": str(paths.profile_jsonl),
            "profile_env_enabled": profile_env_enabled,
            "correctness_gates": [],
            "validation_checklist": self.validation_checklist(case),
            "cleanup_status": {},
            "status": "running",
        }
        write_json(paths.manifest, manifest)

        try:
            print(f"=== {case['name']} ===", flush=True)
            self.start_server(case, paths, env, profile_args)
            manifest["health"] = self.wait_health(port, paths)
            if not paths.effective_config_json.exists():
                raise RuntimeError(
                    f"server did not write effective config: {paths.effective_config_json}"
                )
            manifest["auto_config_decision_count"] = self.count_decision_trace(
                paths.decision_trace_jsonl
            )
            runtime_config_snapshot = self.server_runtime_config_snapshot(
                paths.effective_config_json
            )
            manifest["runtime_config_snapshot"] = runtime_config_snapshot
            manifest["env_hash"] = runtime_config_snapshot.get("env_hash", case_env_hash)
            write_json(paths.manifest, manifest)
            manifest["correctness_gates"] = self.run_gates(case, paths, port)
            self.run_bench(paths, port)
            manifest["metrics"] = self.extract_metrics(paths.bench_json)
            manifest["validation_checklist"] = self.validation_checklist(
                case,
                correctness_gates=manifest["correctness_gates"],
                metrics=manifest["metrics"],
            )
            manifest["profile"] = self.collect_profile(paths)
            if manifest["profile"].get("ok") is False:
                missing = [
                    *manifest["profile"].get("missing_patterns", []),
                    *manifest["profile"].get("missing_events", []),
                    *[
                        "|".join(group)
                        for group in manifest["profile"].get("missing_any_patterns", [])
                    ],
                    *[
                        "|".join(group)
                        for group in manifest["profile"].get("missing_any_events", [])
                    ],
                    *manifest["profile"].get("errors", []),
                ]
                raise RuntimeError(f"profile validation failed: {', '.join(missing)}")
            manifest["status"] = "pass"
            return manifest
        except BaseException as exc:
            manifest["status"] = "fail"
            manifest["artifact_verdict"] = "fail"
            manifest["not_publishable"] = True
            manifest["not_publishable_reason"] = str(exc)
            manifest["error"] = str(exc)
            manifest["validation_checklist"] = self.validation_checklist(
                case,
                correctness_gates=manifest.get("correctness_gates", []),
                metrics=manifest.get("metrics", {}),
            )
            raise
        finally:
            active_exc = sys.exc_info()[1]
            manifest["cleanup_status"] = self.cleanup(paths, port)
            if manifest["cleanup_status"].get("process_leaks"):
                leak_count = len(manifest["cleanup_status"]["process_leaks"])
                leak_reason = f"runner-owned process leak after cleanup: {leak_count}"
                manifest["status"] = "fail"
                manifest["artifact_verdict"] = "fail"
                manifest["not_publishable"] = True
                manifest["not_publishable_reason"] = leak_reason
                manifest["error"] = leak_reason
            elif manifest["cleanup_status"].get("global_process_findings"):
                finding_count = len(manifest["cleanup_status"]["global_process_findings"])
                hygiene_reason = f"global process hygiene failed after cleanup: {finding_count}"
                manifest["status"] = "fail"
                manifest["artifact_verdict"] = "fail"
                manifest["not_publishable"] = True
                manifest["not_publishable_reason"] = hygiene_reason
                manifest["error"] = hygiene_reason
            manifest["finished_at"] = now_iso()
            manifest["validation_checklist"] = self.validation_checklist(
                case,
                correctness_gates=manifest.get("correctness_gates", []),
                metrics=manifest.get("metrics", {}),
            )
            write_json(paths.manifest, manifest)
            if active_exc is None and (
                manifest["cleanup_status"].get("process_leaks")
                or manifest["cleanup_status"].get("global_process_findings")
            ):
                raise RuntimeError(manifest["not_publishable_reason"])

    def summarize(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for case in cases:
            metrics = case.get("metrics", {})
            rows.append(
                {
                    "name": case.get("name"),
                    "status": case.get("status"),
                    "artifact_verdict": case.get("artifact_verdict"),
                    "not_publishable": case.get("not_publishable"),
                    "concurrency": int(self.config.get("concurrency", 32)),
                    "env_hash": case.get("env_hash"),
                    "runtime_config_entry_count": len(
                        case.get("runtime_config_snapshot", {}).get("entries", [])
                    ),
                    **metrics,
                }
            )

        summary: dict[str, Any] = {"rows": rows}
        baseline_name = self.config.get("baseline_case")
        if baseline_name:
            baseline = next((r for r in rows if r["name"] == baseline_name), None)
            baseline_case = next((c for c in cases if c.get("name") == baseline_name), None)
            if baseline and baseline.get("throughput_mean"):
                deltas = {}
                base = float(baseline["throughput_mean"])
                for row in rows:
                    value = row.get("throughput_mean")
                    if value is not None and row["name"] != baseline_name:
                        deltas[row["name"]] = (float(value) / base - 1.0) * 100.0
                summary["throughput_delta_pct_vs_baseline"] = deltas
            if baseline_case:
                diffs = {}
                baseline_snapshot = baseline_case.get("runtime_config_snapshot", {})
                for case in cases:
                    case_name = case.get("name")
                    if case_name and case_name != baseline_name:
                        diffs[str(case_name)] = self.runtime_config_diff(
                            baseline_snapshot,
                            case.get("runtime_config_snapshot", {}),
                        )
                summary["runtime_config_diff_vs_baseline"] = diffs

        summary["performance_regression_gates"] = self.performance_regression_gates(
            rows, str(baseline_name) if baseline_name else None
        )

        write_json(self.out_root / "summary.json", summary)
        for row in rows:
            print(
                "SUMMARY {name}: throughput={throughput_mean} stddev={throughput_stddev} "
                "ci95={throughput_ci95_hw} ttft_p50={ttft_p50} tpot_p50={tpot_p50} "
                "itl_p95={itl_p95} completed={completed} errored={errored} "
                "status={status} verdict={artifact_verdict} "
                "not_publishable={not_publishable}".format(
                    **row
                ),
                flush=True,
            )
        for name, delta in summary.get("throughput_delta_pct_vs_baseline", {}).items():
            print(f"DELTA {name} vs {baseline_name}: {delta:.2f}%", flush=True)
        for name, gate in summary["performance_regression_gates"].get("cases", {}).items():
            print(
                f"PERF_GATE {name} vs {gate['baseline_case']}: ok={gate['ok']}",
                flush=True,
            )
        return summary

    def run(self) -> None:
        self.validate()
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.maybe_build()
        self.ensure_binary()
        self.make_run_manifest(self.preflight())

        completed_cases: list[dict[str, Any]] = []
        try:
            for index, case in enumerate(self.config["cases"]):
                completed_cases.append(self.run_case(case, index))
        finally:
            summary = self.summarize(completed_cases)
            self.run_manifest["cases"] = [
                {
                    "name": case.get("name"),
                    "manifest": str(self.paths_for_case(case).manifest),
                }
                for case in self.config["cases"]
            ]
            self.run_manifest["validation_checklist"] = self.validation_checklist()
            self.run_manifest["validation_checklist"]["performance_regression_gates"] = summary.get(
                "performance_regression_gates"
            )
            self.run_manifest["finished_at"] = now_iso()
            write_json(self.out_root / "manifest.json", self.run_manifest)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = {
            "name": "self-test",
            "out_root": str(root / "out"),
            "model_dir": str(root / "model"),
            "bin": sys.executable,
            "base_env": {"FERRUM_MOE_GRAPH": "1", "HF_HOME": "/tmp/hf"},
            "cases": [{"name": "a", "env": {"FERRUM_FA_LAYOUT_VARLEN": "1"}}],
        }
        (root / "model").mkdir()
        runner = Runner(cfg)
        runner.validate()
        env = runner.case_env_only(cfg["cases"][0])
        assert env == {
            "FERRUM_FA_LAYOUT_VARLEN": "1",
            "FERRUM_MODEL_PATH": str(root / "model"),
            "FERRUM_MOE_GRAPH": "1",
            "HF_HOME": "/tmp/hf",
        }
        snapshot = runner.runtime_config_snapshot(cfg["cases"][0])
        assert [entry["key"] for entry in snapshot["entries"]] == [
            "FERRUM_FA_LAYOUT_VARLEN",
            "FERRUM_MODEL_PATH",
            "FERRUM_MOE_GRAPH",
        ]
        assert snapshot["entries"][1]["source"] == "runner_model_dir"
        assert snapshot["entries"][1]["affects"] == ["correctness"]
        assert snapshot["entries"][2]["source"] == "script_case"
        assert snapshot["entries"][2]["affects"] == ["performance"]
        assert sha256_text(env).startswith("sha256:")
        default_snapshot = runner.runtime_config_snapshot({"name": "default", "env": {}})
        diff = runner.runtime_config_diff(default_snapshot, snapshot)
        assert diff["candidate_env_hash"] == snapshot["env_hash"]
        assert [entry["key"] for entry in diff["added"]] == ["FERRUM_FA_LAYOUT_VARLEN"]
        assert diff["changed"] == []
        assert runner.runtime_effect("FERRUM_FA_LAYOUT_VARLEN") == "performance"

        api_cfg = {
            **cfg,
            "out_root": str(root / "api-out"),
            "validation": {
                "change_type": "api_only",
                "touched_areas": ["openai_server_api"],
                "required_correctness_gates": ["api_contract_tests", "bench_completion"],
                "performance_regression_required": False,
                "benchmark_impact": {
                    "m3_benchmark_exercised": False,
                    "reason": "OpenAI route code is outside the M3 bench path",
                    "evidence": "self-test fixture",
                },
            },
        }
        api_runner = Runner(api_cfg)
        api_runner.validate()
        api_checklist = api_runner.validation_checklist()
        assert api_checklist["benchmark_impact"] == api_cfg["validation"]["benchmark_impact"]

        bad_api_cfg = {
            **api_cfg,
            "out_root": str(root / "bad-api-out"),
            "validation": {
                **api_cfg["validation"],
                "benchmark_impact": {
                    "m3_benchmark_exercised": False,
                    "reason": "",
                    "evidence": "self-test fixture",
                },
            },
        }
        try:
            Runner(bad_api_cfg).validate()
            raise AssertionError("empty benchmark impact reason should fail validation")
        except SystemExit as exc:
            assert "validation.benchmark_impact.reason" in str(exc)

        preset_cfg = {
            "name": "preset-self-test",
            "out_root": str(root / "preset-out"),
            "model_dir": str(root / "model"),
            "bin": sys.executable,
            "preset": "m3_qwen3_30b_a3b_int4",
            "cases": [{"name": "preset", "env": {"FERRUM_FA_LAYOUT_VARLEN": "1"}}],
        }
        preset_runner = Runner(preset_cfg)
        preset_runner.validate()
        preset_env = preset_runner.case_env_only(preset_cfg["cases"][0])
        assert preset_env["FERRUM_MODEL_PATH"] == str(root / "model")
        assert "FERRUM_MOE_GRAPH" not in preset_env
        assert preset_env["HF_HOME"] == "/workspace/hf-cache"
        preset_snapshot = preset_runner.runtime_config_snapshot(preset_cfg["cases"][0])
        preset_entry = {entry["key"]: entry for entry in preset_snapshot["entries"]}
        assert preset_entry["FERRUM_MODEL_PATH"]["source"] == "runner_model_dir"
        assert "FERRUM_MOE_GRAPH" not in preset_entry
        assert "FERRUM_VLLM_MOE" not in preset_entry
        assert preset_entry["FERRUM_FA_LAYOUT_VARLEN"]["source"] == "script_case"

        runner.config["baseline_case"] = "default"
        summary_metrics = {
            "throughput_stddev": None,
            "throughput_ci95_hw": None,
            "ttft_p50": 1.0,
            "tpot_p50": 1.0,
            "itl_p95": 1.0,
            "completed": 1,
            "errored": 0,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            summary = runner.summarize(
                [
                    {
                        "name": "default",
                        "status": "pass",
                        "artifact_verdict": "pass",
                        "not_publishable": False,
                        "env_hash": default_snapshot["env_hash"],
                        "runtime_config_snapshot": default_snapshot,
                        "metrics": {"throughput_mean": 1.0, **summary_metrics},
                    },
                    {
                        "name": "a",
                        "status": "pass",
                        "artifact_verdict": "pass",
                        "not_publishable": False,
                        "env_hash": snapshot["env_hash"],
                        "runtime_config_snapshot": snapshot,
                        "metrics": {"throughput_mean": 2.0, **summary_metrics},
                    },
                ]
            )
        assert summary["throughput_delta_pct_vs_baseline"]["a"] == 100.0
        assert (
            summary["runtime_config_diff_vs_baseline"]["a"]["added"][0]["key"]
            == "FERRUM_FA_LAYOUT_VARLEN"
        )
        perf_gate = summary["performance_regression_gates"]["cases"]["a"]
        assert perf_gate["ok"]
        assert perf_gate["metrics"][0]["metric"] == "throughput_mean"
        assert summary["performance_regression_gates"]["required_concurrency_cells"] == []
        assert summary["performance_regression_gates"]["observed_concurrency_cells"] == [32]
        assert summary["performance_regression_gates"]["concurrency_cells_ok"]

        runner.config["validation"] = {
            **runner.validation_config(),
            "change_type": "default_path",
        }
        default_path_gates = runner.performance_regression_gates(
            summary["rows"], baseline_name="default"
        )
        assert default_path_gates["required_concurrency_cells"] == [1, 4, 16, 32]
        assert default_path_gates["observed_concurrency_cells"] == [32]
        assert not default_path_gates["concurrency_cells_ok"]
        bench = root / "bench.json"
        write_json(
            bench,
            {
                "output_throughput_tps": {"mean": 1.0, "stddev": 0.1, "ci95_hw": 0.2},
                "ttft_ms": {"p50": {"mean": 3.0}},
                "tpot_ms": {"p50": {"mean": 4.0}},
                "itl_ms": {"p95": {"mean": 5.0}},
                "completed_per_run": [6],
                "errored_per_run": [0],
            },
        )
        metrics = runner.extract_metrics(bench)
        assert metrics["throughput_mean"] == 1.0
        assert metrics["completed"] == 6
        assert metrics["errored"] == 0
        paths = RunPaths(
            case_dir=root / "case",
            server_log=root / "case" / "server.log",
            bench_log=root / "case" / "bench.log",
            bench_json=root / "case" / "bench.json",
            manifest=root / "case" / "manifest.json",
            health_json=root / "case" / "health.json",
            effective_config_json=root / "case" / "effective_config.json",
            decision_trace_jsonl=root / "case" / "decision_trace.jsonl",
            profile_jsonl=root / "case" / "profile.jsonl",
        )
        decision_selections = [
            "attention_prefill_mixed_backend",
            "attention_decode_backend",
            "moe_implementation",
            "moe_graph_policy",
            "kv_block_count",
            "max_sequences",
            "max_batched_tokens",
            "prefix_cache_policy",
            "scheduler_admission_policy",
            "sampling_readback_path",
        ]
        decisions = [
            {
                "schema_version": 1,
                "selection": selection,
                "selected": "fixture",
                "source": "default",
                "source_key": None,
                "candidates": ["fixture"],
                "rejected": [],
                "affects": ["performance"],
            }
            for selection in decision_selections
        ]
        auto_config_inputs = {
            "model_capabilities": {
                "architecture": "qwen3_moe",
                "quantization": "gptq_int4",
                "moe": {
                    "num_experts": 128,
                    "experts_per_token": 8,
                    "moe_intermediate_size": 768,
                },
                "max_context_len": 40960,
                "num_hidden_layers": 48,
                "head_dim": 128,
                "kv_heads": 4,
                "estimated_weight_bytes": 19327352832,
                "supported_dtypes": ["fp16"],
                "graph_safe_moe": True,
            },
            "hardware_capabilities": {
                "backend": "cuda",
                "cuda_runtime": "12.8",
                "compute_capability": "8.9",
                "vram_bytes": 25753026560,
                "sm_count": 128,
                "supported_dtypes": ["fp16", "fp32"],
                "supported_kv_dtypes": ["fp16", "int8"],
                "graph_support": True,
                "compiled_features": {
                    "cuda": True,
                    "vllm_paged_attn": True,
                    "vllm_moe_marlin": True,
                    "cuda_graph": True,
                    "greedy_argmax": True,
                    "fa2_source": False,
                    "fa2_direct_ffi": False,
                },
            },
            "workload_profile": {
                "preset": "m3_qwen3_30b_a3b_int4",
                "serving_mode": "bench_serve",
                "target_concurrency": 32,
                "prompt_length_class": "random_256",
                "output_length_class": "random_128",
                "priority": "throughput",
            },
        }
        write_json(
            paths.effective_config_json,
            {
                "schema_version": 1,
                "preset": snapshot["preset"],
                "env_hash": snapshot["env_hash"],
                "entries": [
                    {
                        "key": entry["key"],
                        "effective_value": entry["effective_value"],
                        "source": entry["source"],
                        "affects": entry["affects"],
                    }
                    for entry in snapshot["entries"]
                ],
                **auto_config_inputs,
                "decisions": decisions,
            },
        )
        paths.decision_trace_jsonl.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in decisions)
        )
        server_snapshot = runner.server_runtime_config_snapshot(paths.effective_config_json)
        assert server_snapshot["env_hash"] == snapshot["env_hash"]
        assert server_snapshot["entries"][0]["effect"] == "performance"
        effective_fixture = load_json(paths.effective_config_json)
        assert effective_fixture["hardware_capabilities"]["backend"] == "cuda"
        assert effective_fixture["workload_profile"]["target_concurrency"] == 32
        assert set(decision_selections) <= {decision["selection"] for decision in decisions}
        assert paths.effective_config_json.exists()
        assert paths.decision_trace_jsonl.exists()
        ps = "\n".join(
            [
                "PID PPID STAT ELAPSED CMD",
                f"123 1 S 00:01 {sys.executable} serve /model --port 18840",
                f"124 123 S 00:01 child-worker",
                f"125 1 S 00:01 {sys.executable} bench-serve --base-url http://127.0.0.1:18840 --out {paths.bench_json}",
                "126 1 S 00:01 target/release/ferrum serve /model --port 19999",
                "127 1 S 00:01 cargo build --release -p ferrum-cli",
            ]
        )
        leaks = runner.runner_process_leaks(ps, paths=paths, port=18840, server_pid=123)
        assert [leak["reason"] for leak in leaks] == [
            "server-pid-descendant",
            "server-pid-descendant",
            "bench-serve-case",
        ]
        global_findings = runner.global_process_hygiene_findings(ps, runner_leaks=leaks)
        assert [finding["reason"] for finding in global_findings] == [
            "ferrum-server-global",
            "cargo-global",
        ]
        write_json(
            paths.manifest,
            {
                "git_head": "abc",
                "env_hash": "sha256:test",
                "effective_env": {"FERRUM_MOE_GRAPH": "1"},
                "runtime_config_snapshot": snapshot,
            },
        )
        profile_event = {
            "event": "unified_prof",
            "commit_sha": "abc",
            "env_hash": "sha256:test",
            "model": runner.hf_model,
            "concurrency": 32,
            "shape": {},
            "stage_us": {"total": 1.0, "model": 2.0},
            "graph_enabled": True,
            "runtime_flags": {},
        }
        paths.profile_jsonl.write_text(canonical_json(profile_event) + "\n")
        runner.config["profile"] = {
            "structured": True,
            "required_events": ["unified_prof"],
        }
        profile_args = runner.profile_server_args(paths, cfg["cases"][0])
        assert profile_args[:4] == [
            "--profile-jsonl",
            str(paths.profile_jsonl),
            "--profile-model",
            runner.hf_model,
        ]
        assert "--profile-env-hash" not in profile_args
        assert "--profile-runtime-flags-json" not in profile_args
        structured = runner.collect_profile(paths)
        assert structured["ok"]
        assert structured["mode"] == "structured_jsonl"
        assert structured["source"] == "server_profile_sink"
        assert structured["event_count"] == 1
        bad_profile_cfg = dict(cfg)
        bad_profile_cfg["profile"] = {
            "snippet_regex": "unified-prof",
            "required_patterns": ["unified-prof"],
        }
        bad_runner = Runner(bad_profile_cfg)
        try:
            bad_runner.validate()
            raise AssertionError("log-snippet profile config should fail validation")
        except SystemExit as exc:
            assert "text-log profile matching" in str(exc)
        runner.config["profile"] = {"profile_env_cases": ["a"]}
        case_profile_args = runner.profile_server_args(paths, cfg["cases"][0])
        assert "--profile-jsonl" in case_profile_args
        other_profile_args = runner.profile_server_args(paths, {"name": "other"})
        assert other_profile_args == []
    print("m3_ab_runner self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON runner config")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.config:
        parser.error("--config is required unless --self-test is set")

    config = load_json(args.config)
    runner = Runner(config)
    runner.validate()
    if args.validate_only:
        print("config ok")
        return
    runner.run()


if __name__ == "__main__":
    main()
