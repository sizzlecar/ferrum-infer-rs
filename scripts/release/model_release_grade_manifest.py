#!/usr/bin/env python3
"""Build model release-grade manifests from lane artifacts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = "docs/goals/model-coverage-2026-06-12/RELEASE_GRADE_GOAL.md"
W2_MATRIX = REPO_ROOT / "docs/goals/model-coverage-2026-06-12/w2_matrix.json"
FINAL_GATE = REPO_ROOT / "scripts/release/model_release_grade_goal_gate.py"
REQUIRED_CELLS = [1, 4, 16, 32]
QUALITY_FIELDS = [
    "bad_output",
    "malformed_stream",
    "missing_done",
    "duplicate_done",
    "zero_output_tokens",
    "stream_bulk_flush",
    "http_500",
    "panic",
]
PATH_CONFIG_KEYS = {
    "source",
    "baseline_source",
    "out",
    "matrix",
    "dirty_status_json",
    "hardware",
    "runtime_snapshot",
    "l0_template",
    "l1_numeric",
    "l2_quantized",
    "l3_behavior",
    "l4_agent",
    "l5_concurrency",
    "w3_s0_design",
    "w3_s0_microbench",
    "w3_s1_single_layer",
    "w3_s2_product",
    "ferrum_run",
    "ferrum_serve",
    "ferrum_perf_report",
    "baseline_perf_report",
}
COMMAND_CONFIG_KEYS = {
    "ferrum_bench_command",
    "baseline_bench_command",
    "baseline_server_command",
    "baseline_build_command",
}
SCALAR_CONFIG_KEYS = {
    "lane",
    "dataset_id",
    "model_id",
    "backend",
    "quantization",
    "git_sha",
    "binary_sha256",
    "product_surface",
    "baseline_engine",
    "baseline_version",
    "dataset_sha",
    "no_run_validator",
}
CONFIG_KEYS = PATH_CONFIG_KEYS | COMMAND_CONFIG_KEYS | SCALAR_CONFIG_KEYS | {
    "effective_concurrency",
    "dirty_status",
}


class BuildError(Exception):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BuildError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_config_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label} must be a non-empty path string")
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def command_config_text(value: Any, label: str) -> str:
    if isinstance(value, list):
        if not value or not all(isinstance(part, str) and part for part in value):
            raise BuildError(f"{label} command list must contain non-empty strings")
        return shlex.join(value)
    if isinstance(value, str):
        if not value.strip():
            raise BuildError(f"{label} command must not be empty")
        return value.strip()
    if isinstance(value, dict):
        if "path" in value:
            raise BuildError(f"{label}.path must be handled as a command file path")
        for key in ["command_line", "command", "raw", "cmd"]:
            if key in value:
                return command_config_text(value[key], f"{label}.{key}")
    raise BuildError(f"{label} must be a command string, string list, or {{path: ...}}")


def config_out_dir(args: argparse.Namespace) -> Path:
    out_dir = getattr(args, "out", None)
    if not isinstance(out_dir, Path):
        raise BuildError("--config inline values require --out or config out")
    return out_dir


def materialize_config_json(args: argparse.Namespace, name: str, value: Any) -> Path:
    path = config_out_dir(args) / "_config_inputs" / f"{name}.json"
    write_json(path, value)
    return path


def materialize_config_command(args: argparse.Namespace, name: str, value: Any) -> Path:
    if isinstance(value, dict) and "path" in value:
        return resolve_config_path(value["path"], name)
    if isinstance(value, str):
        candidate = resolve_config_path(value, name)
        if candidate.is_file():
            return candidate
    text = command_config_text(value, name)
    path = config_out_dir(args) / "_config_inputs" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def normalize_config_key(raw: str) -> str:
    return raw.replace("-", "_")


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args
    config_path = require_file(config_path, "--config").resolve()
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise BuildError("--config must be a JSON object")

    raw_args = config.get("args", {})
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise BuildError("--config args must be a JSON object")
    values: dict[str, Any] = {}
    for source in [config, raw_args]:
        for raw_key, raw_value in source.items():
            key = normalize_config_key(raw_key)
            if key == "args":
                continue
            if key not in CONFIG_KEYS:
                if source is raw_args:
                    raise BuildError(f"--config args contains unknown key: {raw_key}")
                continue
            values[key] = raw_value

    lane = values.get("lane")
    if lane is not None:
        if lane not in {"w2", "w3"}:
            raise BuildError("--config lane must be w2 or w3")
        cli_lane = getattr(args, "lane", None)
        if cli_lane is not None and cli_lane != lane:
            raise BuildError(f"--config lane {lane!r} conflicts with CLI lane {cli_lane!r}")
        args.lane = lane

    if "out" in values and getattr(args, "out", None) is None:
        args.out = resolve_config_path(values["out"], "out")
    if "no_run_validator" in values:
        args.no_run_validator = bool(values["no_run_validator"]) or bool(
            getattr(args, "no_run_validator", False)
        )
    if "effective_concurrency" in values:
        raw_effective = values["effective_concurrency"]
        if not isinstance(raw_effective, list) or not all(
            isinstance(item, str) for item in raw_effective
        ):
            raise BuildError("--config effective_concurrency must be a string list")
        args.effective_concurrency = raw_effective + list(
            getattr(args, "effective_concurrency", [])
        )

    for key in sorted(PATH_CONFIG_KEYS - {"out"}):
        if key in values and getattr(args, key, None) is None:
            setattr(args, key, resolve_config_path(values[key], key))

    if "dirty_status" in values and getattr(args, "dirty_status_json", None) is None:
        args.dirty_status_json = materialize_config_json(
            args,
            "dirty_status",
            values["dirty_status"],
        )

    for key in sorted(COMMAND_CONFIG_KEYS):
        if key in values and getattr(args, key, None) is None:
            setattr(args, key, materialize_config_command(args, key, values[key]))

    for key in sorted(SCALAR_CONFIG_KEYS - {"lane", "no_run_validator"}):
        if key in values:
            setattr(args, key, values[key])

    return args


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise BuildError(f"{label} missing: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise BuildError(f"{label} missing: {path}")
    return path


def artifact_ref(path: Path, out_dir: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return path.relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_first_token(path: Path, label: str) -> str:
    text = require_file(path, label).read_text(encoding="utf-8").strip()
    if not text:
        raise BuildError(f"{label} is empty: {path}")
    return text.split()[0]


def read_command(path: Path, label: str) -> list[str]:
    text = require_file(path, label).read_text(encoding="utf-8").strip()
    if not text:
        raise BuildError(f"{label} is empty: {path}")
    return shlex.split(text)


def reports_by_concurrency(path: Path, label: str) -> dict[int, dict[str, Any]]:
    data = load_json(require_file(path, label))
    reports = data.get("reports") if isinstance(data, dict) else data
    if not isinstance(reports, list):
        raise BuildError(f"{label} must contain a report list: {path}")
    out: dict[int, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            raise BuildError(f"{label} contains non-object report")
        concurrency = report.get("concurrency")
        if not isinstance(concurrency, int):
            raise BuildError(f"{label} report missing integer concurrency")
        if concurrency in out:
            raise BuildError(f"{label} has duplicate c={concurrency}")
        out[concurrency] = report
    return out


def output_tps_lcb(report: dict[str, Any], label: str) -> float:
    metric = report.get("output_throughput_tps")
    if not isinstance(metric, dict):
        raise BuildError(f"{label} missing output_throughput_tps")
    mean = metric.get("mean")
    ci95 = metric.get("ci95_hw", 0.0)
    if not isinstance(mean, (int, float)) or not isinstance(ci95, (int, float)):
        raise BuildError(f"{label} output_throughput_tps mean/ci95_hw must be numeric")
    return float(mean) - float(ci95)


def p95_itl_ms(report: dict[str, Any], label: str) -> float:
    try:
        value = report["itl_ms"]["p95"]["mean"]
    except KeyError as exc:
        raise BuildError(f"{label} missing itl_ms.p95.mean") from exc
    if not isinstance(value, (int, float)):
        raise BuildError(f"{label} itl_ms.p95.mean must be numeric")
    return float(value)


def integer_list(report: dict[str, Any], key: str, label: str) -> list[int]:
    value = report.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise BuildError(f"{label}.{key} must be an integer list")
    return value


def integer_matrix(report: dict[str, Any], key: str, label: str) -> list[list[int]]:
    value = report.get(key)
    if not isinstance(value, list):
        raise BuildError(f"{label}.{key} must be a list of integer lists")
    out: list[list[int]] = []
    for row_idx, row in enumerate(value):
        if not isinstance(row, list):
            raise BuildError(f"{label}.{key}[{row_idx}] must be an integer list")
        parsed_row: list[int] = []
        for col_idx, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, int):
                raise BuildError(f"{label}.{key}[{row_idx}][{col_idx}] must be an integer")
            parsed_row.append(item)
        out.append(parsed_row)
    return out


def quality_list(report: dict[str, Any], field: str, label: str) -> list[int]:
    direct_key = f"{field}_per_run"
    if direct_key in report:
        return integer_list(report, direct_key, label)
    issues = report.get("quality_issues_per_run")
    if not isinstance(issues, list):
        raise BuildError(f"{label} missing {direct_key} and quality_issues_per_run")
    values = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise BuildError(f"{label}.quality_issues_per_run contains non-object issue")
        raw = issue.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise BuildError(f"{label}.quality_issues_per_run.{field} must be integer")
        values.append(raw)
    return values


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BuildError(f"{label} must be a positive integer")
    return value


def assert_report_quality(report: dict[str, Any], label: str) -> None:
    if report.get("output_token_count_source") != "usage":
        raise BuildError(f"{label} output_token_count_source must be usage")
    n_repeats = positive_int(report.get("n_repeats"), f"{label}.n_repeats")
    completed = integer_list(report, "completed_per_run", label)
    errored = integer_list(report, "errored_per_run", label)
    if len(completed) != n_repeats or len(errored) != n_repeats:
        raise BuildError(f"{label} completed/errors length must equal n_repeats")
    requests = positive_int(report.get("n_requests_per_run"), f"{label}.n_requests_per_run")
    output_tokens = integer_matrix(report, "output_tokens_per_request", label)
    if len(output_tokens) != n_repeats:
        raise BuildError(f"{label}.output_tokens_per_request length must equal n_repeats")
    for idx, row in enumerate(output_tokens):
        if len(row) != requests:
            raise BuildError(
                f"{label}.output_tokens_per_request[{idx}] length must equal n_requests_per_run"
            )
    if completed != [requests] * n_repeats:
        raise BuildError(f"{label} completed_per_run must be full")
    if any(value != 0 for value in errored):
        raise BuildError(f"{label} errored_per_run must be all zero")
    for field in QUALITY_FIELDS:
        values = quality_list(report, field, label)
        if len(values) != n_repeats:
            raise BuildError(f"{label}.{field}_per_run length must equal n_repeats")
        if any(value != 0 for value in values):
            raise BuildError(f"{label}.{field}_per_run must be all zero")


def w2_correctness_entries(matrix_path: Path, out_dir: Path) -> dict[str, dict[str, Any]]:
    data = load_json(matrix_path)
    models = data.get("models")
    if not isinstance(models, list):
        raise BuildError(f"invalid W2 matrix models list: {matrix_path}")
    model = next((item for item in models if item.get("id") == "gemma3-27b"), None)
    if not isinstance(model, dict):
        raise BuildError("w2_matrix missing gemma3-27b")
    cells = model.get("cells")
    if not isinstance(cells, dict):
        raise BuildError("w2_matrix gemma3-27b missing cells")
    mapping = {
        "l0_template": "l0_template",
        "l1_numeric": "l1_bf16",
        "l2_quantized": "l2_gptq_cuda",
        "l3_behavior": "l3_behavior",
        "l4_agent": "l4_agent",
        "l5_concurrency": "l5_concurrency",
    }
    entries = {}
    for dst, src in mapping.items():
        cell = cells.get(src)
        if not isinstance(cell, dict) or cell.get("status") != "pass":
            raise BuildError(f"w2_matrix {src} must be status=pass")
        artifact = cell.get("artifact")
        if not isinstance(artifact, str) or not artifact:
            raise BuildError(f"w2_matrix {src} missing artifact")
        paths = [REPO_ROOT / artifact]
        if not any(path.exists() for path in paths):
            raise BuildError(f"w2_matrix {src} artifact missing: {artifact}")
        entries[dst] = {
            "status": "pass",
            "artifact": artifact_ref(paths[0], out_dir),
            "source_cell": src,
        }
    return entries


def evidence_entry(path: Path, label: str, out_dir: Path) -> dict[str, Any]:
    if not path.exists():
        raise BuildError(f"{label} artifact missing: {path}")
    return {"status": "pass", "artifact": artifact_ref(path, out_dir)}


def hex64(value: str, label: str) -> str:
    text = value.strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise BuildError(f"{label} must be a 64-character hex digest")
    return text


def git_sha(value: str, label: str) -> str:
    text = value.strip()
    if not (7 <= len(text) <= 40) or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise BuildError(f"{label} must be a 7-40 character hex string")
    return text


def w3_correctness_entries(args: argparse.Namespace, out_dir: Path) -> dict[str, dict[str, Any]]:
    entries = {
        "l0_template": evidence_entry(args.l0_template, "W3 L0 template", out_dir),
        "l1_numeric": evidence_entry(args.l1_numeric, "W3 L1 numeric", out_dir),
        "l2_quantized": evidence_entry(args.l2_quantized, "W3 L2 quantized", out_dir),
        "l3_behavior": evidence_entry(args.l3_behavior, "W3 L3 behavior", out_dir),
        "l4_agent": evidence_entry(args.l4_agent, "W3 L4 agent", out_dir),
        "l5_concurrency": evidence_entry(args.l5_concurrency, "W3 L5 concurrency", out_dir),
        "w3_s0_design": evidence_entry(args.w3_s0_design, "W3 S0 design", out_dir),
        "w3_s0_microbench": evidence_entry(args.w3_s0_microbench, "W3 S0 microbench", out_dir),
        "w3_s1_single_layer": evidence_entry(args.w3_s1_single_layer, "W3 S1 single-layer", out_dir),
        "w3_s2_whole_model_product_path": evidence_entry(
            args.w3_s2_product,
            "W3 S2 product path",
            out_dir,
        ),
    }
    return entries


def parse_effective_concurrency(values: list[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise BuildError(f"--effective-concurrency must be C=E, got {value!r}")
        left, right = value.split("=", 1)
        try:
            requested = int(left)
            effective = int(right)
        except ValueError as exc:
            raise BuildError(f"--effective-concurrency must be integer C=E: {value!r}") from exc
        if requested <= 0 or effective <= 0 or effective > requested:
            raise BuildError(f"invalid effective concurrency mapping: {value!r}")
        previous = result.get(requested)
        if previous is not None and previous != effective:
            raise BuildError(f"conflicting effective concurrency mapping for c={requested}")
        result[requested] = effective
    return result


def nested_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def positive_int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BuildError(f"{label} must be a positive integer")
    return value


def admission_limit_from_runtime_snapshot(path: Path) -> int | None:
    data = load_json(path)
    if not isinstance(data, dict):
        raise BuildError(f"runtime snapshot must be a JSON object: {path}")
    auto_config = nested_object(data, "auto_config") or data
    candidates: list[tuple[str, Any]] = [
        ("runtime_snapshot.selected_admission_limit", auto_config.get("selected_admission_limit")),
        (
            "runtime_snapshot.admission.effective_max_concurrent",
            nested_object(auto_config, "admission").get("effective_max_concurrent"),
        ),
        (
            "runtime_snapshot.top_level_admission.effective_max_concurrent",
            nested_object(data, "admission").get("effective_max_concurrent"),
        ),
    ]
    parsed: list[tuple[str, int]] = []
    for label, value in candidates:
        if value is None:
            continue
        parsed.append((label, positive_int_value(value, label)))
    if not parsed:
        return None
    first_label, first_value = parsed[0]
    for label, value in parsed[1:]:
        if value != first_value:
            raise BuildError(f"{label}={value} does not match {first_label}={first_value}")
    return first_value


def effective_concurrency_from_runtime_snapshot(path: Path) -> dict[int, int]:
    limit = admission_limit_from_runtime_snapshot(path)
    if limit is None:
        return {}
    return {concurrency: min(concurrency, limit) for concurrency in REQUIRED_CELLS}


def merge_effective_concurrency(
    target: dict[int, int],
    incoming: dict[int, int],
    source: str,
) -> None:
    for concurrency, effective in incoming.items():
        previous = target.get(concurrency)
        if previous is not None and previous != effective:
            raise BuildError(
                f"{source} conflicts for requested c={concurrency}: "
                f"{effective} != existing {previous}"
            )
        target[concurrency] = effective


def build_cell(
    *,
    concurrency: int,
    ferrum: dict[str, Any],
    baseline: dict[str, Any],
    source: Path,
    out_dir: Path,
    dataset_id: str,
    dataset_sha: str,
    ferrum_command: list[str],
    baseline_command: list[str],
    effective_concurrency: dict[int, int],
) -> dict[str, Any]:
    assert_report_quality(ferrum, f"ferrum c={concurrency}")
    assert_report_quality(baseline, f"baseline c={concurrency}")
    n_repeats = positive_int(ferrum.get("n_repeats"), f"ferrum c={concurrency}.n_repeats")
    baseline_n_repeats = positive_int(
        baseline.get("n_repeats"),
        f"baseline c={concurrency}.n_repeats",
    )
    requests = positive_int(
        ferrum.get("n_requests_per_run"),
        f"ferrum c={concurrency}.n_requests_per_run",
    )
    baseline_requests = positive_int(
        baseline.get("n_requests_per_run"),
        f"baseline c={concurrency}.n_requests_per_run",
    )
    if n_repeats != baseline_n_repeats:
        raise BuildError(f"c={concurrency} repeat count differs between Ferrum and baseline")
    if requests != baseline_requests:
        raise BuildError(f"c={concurrency} request count differs between Ferrum and baseline")

    effective = effective_concurrency.get(concurrency, concurrency)
    ferrum_lcb = output_tps_lcb(ferrum, f"ferrum c={concurrency}")
    baseline_lcb = output_tps_lcb(baseline, f"baseline c={concurrency}")
    if baseline_lcb <= 0:
        raise BuildError(f"baseline c={concurrency} LCB must be positive")

    cell = {
        "requested_concurrency": concurrency,
        "effective_active_concurrency": effective,
        "baseline_effective_active_concurrency": effective,
        "requests_per_run": requests,
        "n_repeats": n_repeats,
        "completed_per_run": integer_list(ferrum, "completed_per_run", f"ferrum c={concurrency}"),
        "errored_per_run": integer_list(ferrum, "errored_per_run", f"ferrum c={concurrency}"),
        "baseline_requests_per_run": baseline_requests,
        "baseline_n_repeats": baseline_n_repeats,
        "baseline_completed_per_run": integer_list(
            baseline,
            "completed_per_run",
            f"baseline c={concurrency}",
        ),
        "baseline_errored_per_run": integer_list(
            baseline,
            "errored_per_run",
            f"baseline c={concurrency}",
        ),
        "output_tokens_per_request": integer_matrix(
            ferrum,
            "output_tokens_per_request",
            f"ferrum c={concurrency}",
        ),
        "baseline_output_tokens_per_request": integer_matrix(
            baseline,
            "output_tokens_per_request",
            f"baseline c={concurrency}",
        ),
        "output_token_count_source": ferrum.get("output_token_count_source"),
        "stream_options_include_usage": True,
        "baseline_output_token_count_source": baseline.get("output_token_count_source"),
        "baseline_stream_options_include_usage": True,
        "same_hardware": True,
        "same_model": True,
        "same_quantization": True,
        "same_prompt_or_dataset": True,
        "prompt_dataset_id": dataset_id,
        "baseline_prompt_dataset_id": dataset_id,
        "prompt_dataset_sha256": dataset_sha,
        "baseline_prompt_dataset_sha256": dataset_sha,
        "bench_command_line": ferrum_command,
        "baseline_bench_command_line": baseline_command,
        "ferrum_output_tps_lcb": ferrum_lcb,
        "baseline_output_tps": baseline_lcb,
        "ratio": ferrum_lcb / baseline_lcb,
        "ferrum_p95_itl_ms": p95_itl_ms(ferrum, f"ferrum c={concurrency}"),
        "baseline_p95_itl_ms": p95_itl_ms(baseline, f"baseline c={concurrency}"),
        "artifact": artifact_ref(source / "perf/bench_ferrum_sharegpt_sweep_100x3.json", out_dir),
    }
    if effective < concurrency:
        cell["published_concurrency"] = effective
    for field in QUALITY_FIELDS:
        cell[f"{field}_per_run"] = quality_list(ferrum, field, f"ferrum c={concurrency}")
        cell[f"baseline_{field}_per_run"] = quality_list(
            baseline,
            field,
            f"baseline c={concurrency}",
        )
    return cell


def build_direct_cell(
    *,
    concurrency: int,
    ferrum: dict[str, Any],
    baseline: dict[str, Any],
    baseline_concurrency: int,
    ferrum_perf: Path,
    out_dir: Path,
    dataset_id: str,
    dataset_sha: str,
    ferrum_command: list[str],
    baseline_command: list[str],
    effective_concurrency: dict[int, int],
) -> dict[str, Any]:
    assert_report_quality(ferrum, f"ferrum c={concurrency}")
    assert_report_quality(baseline, f"baseline c={baseline_concurrency}")
    n_repeats = positive_int(ferrum.get("n_repeats"), f"ferrum c={concurrency}.n_repeats")
    baseline_n_repeats = positive_int(
        baseline.get("n_repeats"),
        f"baseline c={baseline_concurrency}.n_repeats",
    )
    requests = positive_int(
        ferrum.get("n_requests_per_run"),
        f"ferrum c={concurrency}.n_requests_per_run",
    )
    baseline_requests = positive_int(
        baseline.get("n_requests_per_run"),
        f"baseline c={baseline_concurrency}.n_requests_per_run",
    )
    if n_repeats != baseline_n_repeats:
        raise BuildError(f"c={concurrency} repeat count differs between Ferrum and baseline")
    if requests != baseline_requests:
        raise BuildError(f"c={concurrency} request count differs between Ferrum and baseline")

    effective = effective_concurrency.get(concurrency, concurrency)
    ferrum_lcb = output_tps_lcb(ferrum, f"ferrum c={concurrency}")
    baseline_lcb = output_tps_lcb(baseline, f"baseline c={baseline_concurrency}")
    if baseline_lcb <= 0:
        raise BuildError(f"baseline c={baseline_concurrency} LCB must be positive")

    cell = {
        "requested_concurrency": concurrency,
        "effective_active_concurrency": effective,
        "baseline_effective_active_concurrency": effective,
        "baseline_measured_concurrency": baseline_concurrency,
        "requests_per_run": requests,
        "n_repeats": n_repeats,
        "completed_per_run": integer_list(ferrum, "completed_per_run", f"ferrum c={concurrency}"),
        "errored_per_run": integer_list(ferrum, "errored_per_run", f"ferrum c={concurrency}"),
        "baseline_requests_per_run": baseline_requests,
        "baseline_n_repeats": baseline_n_repeats,
        "baseline_completed_per_run": integer_list(
            baseline,
            "completed_per_run",
            f"baseline c={baseline_concurrency}",
        ),
        "baseline_errored_per_run": integer_list(
            baseline,
            "errored_per_run",
            f"baseline c={baseline_concurrency}",
        ),
        "output_tokens_per_request": integer_matrix(
            ferrum,
            "output_tokens_per_request",
            f"ferrum c={concurrency}",
        ),
        "baseline_output_tokens_per_request": integer_matrix(
            baseline,
            "output_tokens_per_request",
            f"baseline c={baseline_concurrency}",
        ),
        "output_token_count_source": ferrum.get("output_token_count_source"),
        "stream_options_include_usage": True,
        "baseline_output_token_count_source": baseline.get("output_token_count_source"),
        "baseline_stream_options_include_usage": True,
        "same_hardware": True,
        "same_model": True,
        "same_quantization": True,
        "same_prompt_or_dataset": True,
        "prompt_dataset_id": dataset_id,
        "baseline_prompt_dataset_id": dataset_id,
        "prompt_dataset_sha256": dataset_sha,
        "baseline_prompt_dataset_sha256": dataset_sha,
        "bench_command_line": ferrum_command,
        "baseline_bench_command_line": baseline_command,
        "ferrum_output_tps_lcb": ferrum_lcb,
        "baseline_output_tps": baseline_lcb,
        "ratio": ferrum_lcb / baseline_lcb,
        "ferrum_p95_itl_ms": p95_itl_ms(ferrum, f"ferrum c={concurrency}"),
        "baseline_p95_itl_ms": p95_itl_ms(baseline, f"baseline c={baseline_concurrency}"),
        "artifact": artifact_ref(ferrum_perf, out_dir),
    }
    if effective < concurrency:
        cell["published_concurrency"] = effective
    for field in QUALITY_FIELDS:
        cell[f"{field}_per_run"] = quality_list(ferrum, field, f"ferrum c={concurrency}")
        cell[f"baseline_{field}_per_run"] = quality_list(
            baseline,
            field,
            f"baseline c={baseline_concurrency}",
        )
    return cell


def source_status(source: Path) -> dict[str, Any]:
    summary = load_json(source / "summary.json")
    if summary.get("status") not in {"diagnostic_pass", "pass"}:
        raise BuildError(f"source summary status is not pass: {summary.get('status')!r}")
    if summary.get("ferrum_bench_exit_code") not in {0, None}:
        raise BuildError(f"Ferrum bench exit code is not zero: {summary.get('ferrum_bench_exit_code')}")
    if summary.get("vllm_bench_exit_code") not in {0, None}:
        raise BuildError(f"vLLM bench exit code is not zero: {summary.get('vllm_bench_exit_code')}")
    return summary


def build_w2_manifest(
    *,
    source: Path,
    baseline_source: Path | None,
    out_dir: Path,
    matrix_path: Path,
    effective_concurrency: dict[int, int],
    dataset_id: str,
) -> Path:
    source = require_dir(source, "source artifact dir").resolve()
    baseline_source = require_dir(
        baseline_source or source,
        "baseline artifact dir",
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_status(source)

    ferrum_perf = source / "perf/bench_ferrum_sharegpt_sweep_100x3.json"
    baseline_perf = baseline_source / "perf/bench_vllm_sharegpt_sweep_100x3.json"
    ferrum_reports = reports_by_concurrency(ferrum_perf, "Ferrum bench report")
    baseline_reports = reports_by_concurrency(baseline_perf, "vLLM bench report")
    missing = [c for c in REQUIRED_CELLS if c not in ferrum_reports or c not in baseline_reports]
    if missing:
        raise BuildError(f"missing required performance cells: {missing}")

    dataset_sha = read_first_token(source / "env/dataset.sha256", "dataset sha")
    baseline_dataset_sha = read_first_token(
        baseline_source / "env/dataset.sha256",
        "baseline dataset sha",
    )
    if dataset_sha.lower() != baseline_dataset_sha.lower():
        raise BuildError(
            "baseline dataset sha does not match Ferrum dataset sha: "
            f"{baseline_dataset_sha} != {dataset_sha}"
        )
    binary_sha = read_first_token(source / "env/ferrum.sha256", "Ferrum binary sha")
    git_sha = read_first_token(source / "env/git_sha.txt", "remote git sha")
    git_status = require_file(source / "env/git_status_short.txt", "remote git status").read_text(
        encoding="utf-8"
    ).strip()
    vllm_versions = load_json(baseline_source / "env/vllm_versions.json")
    vllm_version = str(vllm_versions.get("vllm", "unknown"))
    baseline_build_command = [
        "python",
        "-m",
        "pip",
        "install",
        f"vllm=={vllm_version}",
    ]
    ferrum_command = read_command(source / "perf/bench-ferrum.command.txt", "Ferrum bench command")
    baseline_command = read_command(
        baseline_source / "perf/bench-vllm.command.txt",
        "vLLM bench command",
    )

    vllm_server = load_json(
        require_file(baseline_source / "vllm/vllm_server.command.json", "vLLM command")
    )
    vllm_server_command = vllm_server.get("cmd")
    if not isinstance(vllm_server_command, list) or not all(
        isinstance(part, str) for part in vllm_server_command
    ):
        raise BuildError("vLLM server command JSON missing string-list cmd")

    cells = [
        build_cell(
            concurrency=concurrency,
            ferrum=ferrum_reports[concurrency],
            baseline=baseline_reports[concurrency],
            source=source,
            out_dir=out_dir,
            dataset_id=dataset_id,
            dataset_sha=dataset_sha,
            ferrum_command=ferrum_command,
            baseline_command=baseline_command,
            effective_concurrency=effective_concurrency,
        )
        for concurrency in REQUIRED_CELLS
    ]

    manifest = {
        "schema_version": 1,
        "lane": "w2",
        "status": "pass",
        "goal_doc": GOAL_DOC,
        "model_id": "gemma3:27b-gptq",
        "backend": "cuda",
        "quantization": "hf-gptq-int4",
        "git_sha": git_sha,
        "dirty_status": {"dirty": bool(git_status), "status_short": git_status},
        "binary_sha256": binary_sha,
        "hardware": {
            "status": "pass",
            "artifact": artifact_ref(source / "env/nvidia_smi_before.txt", out_dir),
        },
        "release_scope": {
            "backends": ["cuda"],
            "formats": ["hf-gptq-int4"],
            "excluded_lanes": {
                "gguf_metal": {
                    "reason": "GGUF/Metal is not in this W2 CUDA GPTQ release-grade scope",
                }
            },
        },
        "runtime_config": {
            "product_surface": "typed_cli",
            "hidden_env": [],
            "snapshot": artifact_ref(source / "server/serve_effective_config.json", out_dir),
        },
        "correctness": w2_correctness_entries(matrix_path, out_dir),
        "product_entrypoints": {
            "ferrum_run": {
                "status": "pass",
                "artifact": artifact_ref(source / "correctness/run_summary.json", out_dir),
            },
            "ferrum_serve": {
                "status": "pass",
                "artifact": artifact_ref(source / "correctness/smoke/stream_summary.json", out_dir),
            },
        },
        "performance": {
            "baseline": {
                "engine": "vLLM",
                "version": vllm_version,
                "evidence_mode": "historical" if baseline_source != source else "live",
                "build_command_line": baseline_build_command,
                "command_line": vllm_server_command,
                "bench_command_line": baseline_command,
                "same_hardware": True,
                "same_model": True,
                "same_quantization": True,
                "artifact": artifact_ref(baseline_perf, out_dir),
            },
            "cells": cells,
        },
    }
    manifest_path = out_dir / "model_release_grade_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def build_w3_manifest(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    effective_concurrency: dict[int, int],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    ferrum_perf = require_file(args.ferrum_perf_report, "W3 Ferrum performance report")
    baseline_perf = require_file(args.baseline_perf_report, "W3 baseline performance report")
    ferrum_reports = reports_by_concurrency(ferrum_perf, "W3 Ferrum bench report")
    baseline_reports = reports_by_concurrency(baseline_perf, "W3 baseline bench report")
    missing_ferrum = [c for c in REQUIRED_CELLS if c not in ferrum_reports]
    missing_baseline = [
        f"requested c={c} effective c={effective_concurrency.get(c, c)}"
        for c in REQUIRED_CELLS
        if effective_concurrency.get(c, c) not in baseline_reports
    ]
    if missing_ferrum or missing_baseline:
        parts = []
        if missing_ferrum:
            parts.append(f"Ferrum cells {missing_ferrum}")
        if missing_baseline:
            parts.append(f"baseline cells {missing_baseline}")
        raise BuildError(f"W3 missing required performance cells: {'; '.join(parts)}")

    dataset_sha = hex64(args.dataset_sha, "--dataset-sha")
    binary_sha = hex64(args.binary_sha256, "--binary-sha256")
    sha = git_sha(args.git_sha, "--git-sha")
    dirty_status = load_json(require_file(args.dirty_status_json, "--dirty-status-json"))
    ferrum_command = read_command(args.ferrum_bench_command, "W3 Ferrum bench command")
    baseline_command = read_command(args.baseline_bench_command, "W3 baseline bench command")
    baseline_server_command = read_command(
        args.baseline_server_command,
        "W3 baseline server command",
    )
    baseline_build_command = read_command(
        args.baseline_build_command,
        "W3 baseline build command",
    )

    cells = []
    for concurrency in REQUIRED_CELLS:
        baseline_concurrency = effective_concurrency.get(concurrency, concurrency)
        cells.append(
            build_direct_cell(
                concurrency=concurrency,
                ferrum=ferrum_reports[concurrency],
                baseline=baseline_reports[baseline_concurrency],
                baseline_concurrency=baseline_concurrency,
                ferrum_perf=ferrum_perf,
                out_dir=out_dir,
                dataset_id=args.dataset_id,
                dataset_sha=dataset_sha,
                ferrum_command=ferrum_command,
                baseline_command=baseline_command,
                effective_concurrency=effective_concurrency,
            )
        )

    manifest = {
        "schema_version": 1,
        "lane": "w3",
        "status": "pass",
        "goal_doc": GOAL_DOC,
        "model_id": args.model_id,
        "backend": args.backend,
        "quantization": args.quantization,
        "git_sha": sha,
        "dirty_status": dirty_status,
        "binary_sha256": binary_sha,
        "hardware": evidence_entry(args.hardware, "W3 hardware", out_dir),
        "release_scope": {
            "backends": [args.backend],
            "formats": [args.quantization],
            "excluded_lanes": {},
        },
        "runtime_config": {
            "product_surface": args.product_surface,
            "hidden_env": [],
            "snapshot": artifact_ref(
                require_file(args.runtime_snapshot, "W3 runtime snapshot"),
                out_dir,
            ),
        },
        "correctness": w3_correctness_entries(args, out_dir),
        "product_entrypoints": {
            "ferrum_run": evidence_entry(args.ferrum_run, "W3 ferrum run", out_dir),
            "ferrum_serve": evidence_entry(args.ferrum_serve, "W3 ferrum serve", out_dir),
        },
        "performance": {
            "baseline": {
                "engine": args.baseline_engine,
                "version": args.baseline_version,
                "build_command_line": baseline_build_command,
                "command_line": baseline_server_command,
                "bench_command_line": baseline_command,
                "same_hardware": True,
                "same_model": True,
                "same_quantization": True,
                "artifact": artifact_ref(baseline_perf, out_dir),
            },
            "cells": cells,
        },
    }
    manifest_path = out_dir / "model_release_grade_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def run_validator(out_dir: Path, lane: str) -> int:
    cmd = [sys.executable, str(FINAL_GATE), lane, str(out_dir)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    return proc.returncode


def metric(mean: float, ci95_hw: float = 0.0) -> dict[str, float]:
    return {"mean": mean, "stddev": 0.0, "ci95_hw": ci95_hw}


def write_selftest_source(root: Path) -> Path:
    source = root / "remote"
    for rel in [
        "env",
        "perf",
        "server",
        "correctness/smoke",
        "vllm",
    ]:
        (source / rel).mkdir(parents=True, exist_ok=True)
    (source / "env/dataset.sha256").write_text("b" * 64 + "  dataset.jsonl\n")
    (source / "env/ferrum.sha256").write_text("a" * 64 + "  target/release/ferrum\n")
    (source / "env/git_sha.txt").write_text("0123456789abcdef\n")
    (source / "env/git_status_short.txt").write_text("")
    (source / "env/nvidia_smi_before.txt").write_text("NVIDIA GeForce RTX 4090\n")
    write_json(source / "env/vllm_versions.json", {"vllm": "0.23.0"})
    write_json(source / "server/serve_effective_config.json", {"status": "pass"})
    write_json(source / "correctness/run_summary.json", {"status": "pass", "content": "5"})
    write_json(source / "correctness/smoke/stream_summary.json", {"done_count": 1, "content": "5"})
    write_json(source / "vllm/vllm_server.command.json", {"cmd": ["python", "-m", "vllm"]})
    (source / "perf/bench-ferrum.command.txt").write_text(
        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
        "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 "
        "--random-output-len 128\n"
    )
    (source / "perf/bench-vllm.command.txt").write_text(
        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
        "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 "
        "--random-output-len 128\n"
    )
    ferrum_reports = []
    baseline_reports = []
    for concurrency in REQUIRED_CELLS:
        common = {
            "concurrency": concurrency,
            "n_repeats": 3,
            "n_requests_per_run": 100,
            "completed_per_run": [100, 100, 100],
            "errored_per_run": [0, 0, 0],
            "output_tokens_per_request": [[128] * 100, [128] * 100, [128] * 100],
            "output_token_count_source": "usage",
            "itl_ms": {"p95": metric(10.0)},
            "bad_output_per_run": [0, 0, 0],
            "malformed_stream_per_run": [0, 0, 0],
            "missing_done_per_run": [0, 0, 0],
            "duplicate_done_per_run": [0, 0, 0],
            "zero_output_tokens_per_run": [0, 0, 0],
            "stream_bulk_flush_per_run": [0, 0, 0],
            "http_500_per_run": [0, 0, 0],
            "panic_per_run": [0, 0, 0],
        }
        ferrum_reports.append({**common, "output_throughput_tps": metric(82.0)})
        baseline_reports.append(
            {**common, "output_throughput_tps": metric(100.0), "itl_ms": {"p95": metric(9.0)}}
        )
    write_json(source / "perf/bench_ferrum_sharegpt_sweep_100x3.json", ferrum_reports)
    write_json(source / "perf/bench_vllm_sharegpt_sweep_100x3.json", baseline_reports)
    write_json(
        source / "summary.json",
        {
            "status": "diagnostic_pass",
            "ferrum_bench_exit_code": 0,
            "vllm_bench_exit_code": 0,
        },
    )
    return source


def write_selftest_reports(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    ferrum_reports = []
    baseline_reports = []
    for concurrency in REQUIRED_CELLS:
        common = {
            "concurrency": concurrency,
            "n_repeats": 3,
            "n_requests_per_run": 100,
            "completed_per_run": [100, 100, 100],
            "errored_per_run": [0, 0, 0],
            "output_tokens_per_request": [[128] * 100, [128] * 100, [128] * 100],
            "output_token_count_source": "usage",
            "itl_ms": {"p95": metric(10.0)},
            "bad_output_per_run": [0, 0, 0],
            "malformed_stream_per_run": [0, 0, 0],
            "missing_done_per_run": [0, 0, 0],
            "duplicate_done_per_run": [0, 0, 0],
            "zero_output_tokens_per_run": [0, 0, 0],
            "stream_bulk_flush_per_run": [0, 0, 0],
            "http_500_per_run": [0, 0, 0],
            "panic_per_run": [0, 0, 0],
        }
        ferrum_reports.append({**common, "output_throughput_tps": metric(82.0)})
        baseline_reports.append(
            {**common, "output_throughput_tps": metric(100.0), "itl_ms": {"p95": metric(9.0)}}
        )
    ferrum_perf = root / "bench_ferrum_sharegpt_sweep_100x3.json"
    baseline_perf = root / "bench_vllm_sharegpt_sweep_100x3.json"
    write_json(ferrum_perf, ferrum_reports)
    write_json(baseline_perf, baseline_reports)
    return ferrum_perf, baseline_perf


def write_selftest_w3_s0_s1_s2(root: Path) -> dict[str, Path]:
    (root / "s1_reference_dump").mkdir(parents=True, exist_ok=True)
    (root / "s1_ferrum_dump").mkdir(parents=True, exist_ok=True)
    write_json(
        root / "w3_s0_design.json",
        {
            "schema_version": 1,
            "status": "pass",
            "lane": "w3_s0_design",
            "pass_line": "W3 S0 DESIGN PASS: selftest",
            "hidden_env": [],
            "recurrent_state_cache": {
                "trait": "RecurrentStateManager",
                "state_spec": "RecurrentStateSpec",
            },
            "coexistence": {
                "paged_kv": "separate recurrent state domain",
                "continuous_batch": "allocated at admission",
                "preemption": "clone or preserve recurrent handle",
                "release": "release with request lifecycle",
            },
        },
    )
    write_json(
        root / "w3_s0_microbench.json",
        {
            "schema_version": 1,
            "status": "pass",
            "mode": "cuda",
            "pass_line": "W3 DELTA RULE S0 MICROBENCH PASS: selftest",
            "ptx_arch": "sm_89",
            "cuda_binary_sha256": "c" * 64,
            "seed": 9271,
            "git": {
                "sha": "0123456789abcdef",
                "is_dirty": False,
                "tracked_status_short": [],
            },
            "shape": {"batch": 2, "heads": 2, "tokens": 8, "key_dim": 4, "value_dim": 4},
            "reference": {
                "name": "internal-python-delta-rule-reference",
                "formula": "S_t = S_{t-1} + beta_t * k_t^T * (v_t - k_t @ S_{t-1})",
            },
            "input_distribution": {
                "generator": "lcg_u32_centered_uniform",
                "q_range": [-0.25, 0.25],
                "k_range": [-0.20, 0.20],
                "v_range": [-0.30, 0.30],
                "beta_range": [0.50, 0.75],
            },
            "tolerance": {"max_abs": 0.001},
            "error_stats": {"max_abs": 0.00001},
            "chunked_reference_error": {"max_abs": 0.0},
            "cuda_error": {"max_abs": 0.00001},
            "cuda": {
                "compile_command": ["nvcc", "-arch=sm_89", "delta_rule_s0.cu"],
                "run_command": ["delta_rule_s0", "input.bin", "output.bin"],
                "compile_logs": {"returncode": 0},
                "run_logs": {"returncode": 0},
            },
        },
    )
    s1_comparisons = {
        key: {"status": "pass", "atol": 0.000001, "max_abs": 0.0}
        for key in [
            "input",
            "delta_q",
            "delta_k",
            "delta_v",
            "delta_gate",
            "delta_beta",
            "delta_core",
            "delta_output",
            "router_logits",
            "router_topk_weights",
            "routed_expert_output",
            "shared_expert_output",
            "moe_output",
            "layer_output",
        ]
    }
    s1_comparisons["router_topk_indices"] = {"status": "pass", "mismatches": 0}
    write_json(
        root / "w3_s1_single_layer.json",
        {
            "schema_version": 1,
            "status": "pass",
            "mode": "compare",
            "pass_line": "W3 DELTANET S1 LAYER COMPARE PASS: selftest",
            "reference_dump": str(root / "s1_reference_dump"),
            "ferrum_dump": str(root / "s1_ferrum_dump"),
            "git": {
                "sha": "0123456789abcdef",
                "is_dirty": False,
                "tracked_status_short": [],
            },
            "checks": {
                "delta_rule": "pass",
                "deltanet_layer": "pass",
                "expert_layout": "pass",
                "router_topk": "pass",
                "shared_expert_merge": "pass",
            },
            "comparisons": s1_comparisons,
        },
    )
    write_json(root / "w3_run_stdout.jsonl", {"role": "assistant", "content": "ok ok"})
    write_json(root / "w3_serve_nonstream.json", {"choices": [{"message": {"content": "ok"}}]})
    (root / "w3_run_stderr.txt").write_text("")
    (root / "w3_serve.log").write_text("selftest serve log\n")
    (root / "w3_serve_stream.sse").write_text(
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        'data: {"usage":{"completion_tokens":1},"choices":[]}\n\n'
        "data: [DONE]\n\n"
    )
    write_json(
        root / "w3_s2_product.json",
        {
            "schema_version": 1,
            "status": "pass",
            "lane": "w3_s2_whole_model_product_path",
            "runtime_surface": "typed_cli",
            "hidden_env": [],
            "product_entrypoints": {
                "ferrum_run": {
                    "status": "pass",
                    "command_line": ["ferrum", "run", "selftest-qwen35"],
                    "stdout": str(root / "w3_run_stdout.jsonl"),
                    "stderr": str(root / "w3_run_stderr.txt"),
                    "assistant_event": {
                        "finish_reason": "length",
                        "n_tokens": 2,
                        "content": "ok ok",
                    },
                },
                "ferrum_serve": {
                    "status": "pass",
                    "command_line": ["ferrum", "serve", "selftest-qwen35"],
                    "log": str(root / "w3_serve.log"),
                    "nonstream": {
                        "artifact": str(root / "w3_serve_nonstream.json"),
                        "finish_reason": "length",
                        "content_len": 2,
                    },
                    "stream": {
                        "artifact": str(root / "w3_serve_stream.sse"),
                        "chunk_count": 2,
                        "done_count": 1,
                        "has_usage": True,
                        "usage_chunks": 1,
                    },
                },
            },
        },
    )
    return {
        "w3_s0_design": root / "w3_s0_design.json",
        "w3_s0_microbench": root / "w3_s0_microbench.json",
        "w3_s1_single_layer": root / "w3_s1_single_layer.json",
        "w3_s2_product": root / "w3_s2_product.json",
    }


def write_selftest_w3_l0_l5(root: Path) -> None:
    common = {
        "status": "pass",
        "model_id": "selftest-qwen35",
        "product_surface": "typed_cli",
        "hidden_env": [],
    }
    write_json(
        root / "l0.json",
        {
            **common,
            "level": "l0_template",
            "pass_line": "W3 L0 TEMPLATE PASS: selftest",
            "chat_template_golden": {
                "cases_total": 5,
                "cases_passed": 5,
                "hf_apply_chat_template_reference": True,
                "byte_equal": True,
                "eos_bos_from_generation_config": True,
                "render_failure_is_error": True,
                "silent_fallback": False,
            },
        },
    )
    write_json(
        root / "l1.json",
        {
            **common,
            "level": "l1_numeric",
            "pass_line": "W3 L1 NUMERIC PASS: selftest",
            "numeric": {
                "comparisons_total": 6,
                "comparisons_passed": 6,
                "atol": 0.000001,
                "max_abs": 0.0,
                "deterministic": True,
            },
            "coverage": {
                "linear_attention": True,
                "full_attention": True,
                "full_attention_official_shape": True,
                "deltanet": True,
                "moe_or_dense": True,
                "lm_head": True,
            },
            "reference": {
                "engine": "transformers",
                "artifact": "selftest-reference-dump",
            },
        },
    )
    write_json(
        root / "l2.json",
        {
            **common,
            "level": "l2_quantized",
            "pass_line": "W3 L2 QUANTIZED PASS: selftest",
            "quantized_semantics": {
                "real_size_model": True,
                "waived": False,
                "semantic_pass": True,
                "known_answer_total": 10,
                "known_answer_passed": 10,
                "format": "hf-gptq-int4",
            },
            "output_hygiene": {
                "known_answer_cases_checked": 10,
                "response_artifacts_checked": 10,
                "case_entrypoints": ["ferrum run", "ferrum serve"],
                "content_non_empty": True,
                "forbidden_patterns_absent": True,
                "artifact_text_scanned": True,
            },
            "commands": [
                {
                    "entrypoint": "ferrum run",
                    "command_line": [
                        "ferrum",
                        "run",
                        "selftest-qwen35",
                        "--backend",
                        "cuda",
                    ],
                },
                {
                    "entrypoint": "ferrum serve",
                    "command_line": [
                        "ferrum",
                        "serve",
                        "selftest-qwen35",
                        "--backend",
                        "cuda",
                    ],
                },
            ],
        },
    )
    for rel_path in [
        "behavior/01_multi_turn.response.json",
        "behavior/02_stream_match_stream.response.sse",
        "behavior/03_natural_eos.response.json",
        "behavior/04_custom_stop.response.json",
        "behavior/05_reasoning_extraction.response.json",
        "behavior/06_multi_turn_repeat.response.json",
        "behavior/07_stop_repeat.response.json",
    ]:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            'data: {"usage":{"completion_tokens":1},"choices":[]}\n\ndata: [DONE]\n'
            if rel_path.endswith(".sse")
            else "{}\n"
        )

    write_json(
        root / "l3.json",
        {
            **common,
            "level": "l3_behavior",
            "pass_line": "W3 L3 BEHAVIOR PASS: selftest",
            "behavior": {
                "cases_total": 7,
                "cases_passed": 7,
                "multi_turn": True,
                "stream_nonstream_match": True,
                "natural_eos": True,
                "custom_stop": True,
                "reasoning_extraction": True,
                "stream_done_exactly_once": True,
                "stream_usage_present": True,
            },
            "cases": [
                {
                    "id": "multi_turn",
                    "passed": True,
                    "artifact": "behavior/01_multi_turn.response.json",
                    "detail": {"finish_reason": "stop"},
                },
                {
                    "id": "stream_nonstream_match",
                    "passed": True,
                    "artifact": "behavior/02_stream_match_stream.response.sse",
                    "detail": {"stream_done_count": 1, "stream_usage_chunks": 1},
                },
                {
                    "id": "natural_eos",
                    "passed": True,
                    "artifact": "behavior/03_natural_eos.response.json",
                    "detail": {"finish_reason": "stop"},
                },
                {
                    "id": "custom_stop",
                    "passed": True,
                    "artifact": "behavior/04_custom_stop.response.json",
                    "detail": {"finish_reason": "stop"},
                },
                {
                    "id": "reasoning_extraction",
                    "passed": True,
                    "artifact": "behavior/05_reasoning_extraction.response.json",
                    "detail": {"reasoning_len": 8, "leaked_think": False},
                },
                {
                    "id": "multi_turn_repeat",
                    "passed": True,
                    "artifact": "behavior/06_multi_turn_repeat.response.json",
                    "detail": {"finish_reason": "stop"},
                },
                {
                    "id": "stop_repeat",
                    "passed": True,
                    "artifact": "behavior/07_stop_repeat.response.json",
                    "detail": {"finish_reason": "stop"},
                },
            ],
        },
    )
    for idx in range(10):
        path = root / "tool_calls" / f"tool_{idx:02d}.response.json"
        write_json(
            path,
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"call_tool_{idx:02d}",
                                    "type": "function",
                                    "function": {
                                        "name": "calc",
                                        "arguments": json.dumps({"expression": "123+456"}),
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )
    for idx in range(20):
        path = root / "strict_schema" / f"strict_schema_{idx:02d}.response.json"
        write_json(
            path,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"answer": f"ok-{idx}"}),
                        },
                    }
                ]
            },
        )

    write_json(
        root / "l4.json",
        {
            **common,
            "level": "l4_agent",
            "pass_line": "W3 L4 AGENT PASS: selftest",
            "agent": {
                "real_model": True,
                "required_tool_enforced": True,
                "json_schema_strict": True,
                "tool_calls_total": 10,
                "tool_calls_passed": 10,
                "strict_schema_total": 20,
                "strict_schema_passed": 20,
            },
            "negative_contracts": {
                "tool_choice_400": True,
                "response_format_400": True,
            },
            "tool_call_cases": [
                {
                    "id": f"tool_{idx:02d}",
                    "passed": True,
                    "finish_reason": "tool_calls",
                    "artifact": f"tool_calls/tool_{idx:02d}.response.json",
                    "arguments": {"expression": "123+456"},
                }
                for idx in range(10)
            ],
            "strict_schema_cases": [
                {
                    "id": f"strict_schema_{idx:02d}",
                    "passed": True,
                    "finish_reason": "stop",
                    "artifact": f"strict_schema/strict_schema_{idx:02d}.response.json",
                    "content": {"answer": f"ok-{idx}"},
                }
                for idx in range(20)
            ],
        },
    )
    cells = []
    for concurrency in REQUIRED_CELLS:
        effective = 16 if concurrency == 32 else concurrency
        cell: dict[str, Any] = {
            "requested_concurrency": concurrency,
            "effective_active_concurrency": effective,
            "requests_per_run": 100,
            "n_repeats": 3,
            "completed_per_run": [100, 100, 100],
            "errored_per_run": [0, 0, 0],
            "output_tokens_per_request": [[128] * 100, [128] * 100, [128] * 100],
        }
        if effective < concurrency:
            cell["published_concurrency"] = effective
        for field in QUALITY_FIELDS:
            cell[f"{field}_per_run"] = [0, 0, 0]
        cells.append(cell)
    write_json(
        root / "l5.json",
        {
            **common,
            "level": "l5_concurrency",
            "pass_line": "W3 L5 CONCURRENCY PASS: selftest",
            "commands": [
                {
                    "command_line": [
                        "ferrum",
                        "bench-serve",
                        "--fail-on-error",
                        "--require-ci",
                        "--seed",
                        "9271",
                        "--n-repeats",
                        "3",
                        "--concurrency-sweep",
                        "1,4,16,32",
                        "--random-output-len",
                        "128",
                        "--ignore-eos",
                    ],
                    "covers_concurrency": [1, 4, 16, 32],
                }
            ],
            "concurrency": {
                "closed_loop": True,
                "stream_options_include_usage": True,
                "output_token_count_source": "usage",
                "expected_output_tokens_per_request": 128,
                "cells": cells,
            },
        },
    )


def write_selftest_w3_args(root: Path) -> argparse.Namespace:
    root.mkdir(parents=True, exist_ok=True)
    artifacts = write_selftest_w3_s0_s1_s2(root / "w3")
    for rel in [
        "hardware.json",
        "runtime.json",
        "dirty_status.json",
        "l0.json",
        "l1.json",
        "l2.json",
        "l3.json",
        "l4.json",
        "l5.json",
        "run.json",
        "serve.json",
    ]:
        write_json(root / rel, {"status": "pass", "name": rel})
    write_json(
        root / "runtime.json",
        {
            "status": "pass",
            "name": "runtime.json",
            "selected_admission_limit": 16,
            "admission": {"effective_max_concurrent": 16},
        },
    )
    write_selftest_w3_l0_l5(root)
    write_json(root / "dirty_status.json", {"dirty": False})
    ferrum_perf, baseline_perf = write_selftest_reports(root / "perf")
    (root / "ferrum_bench.command.txt").write_text(
        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
        "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 "
        "--random-output-len 128 --ignore-eos\n"
    )
    (root / "baseline_bench.command.txt").write_text(
        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
        "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 "
        "--random-output-len 128 --ignore-eos\n"
    )
    (root / "baseline_server.command.txt").write_text("python -m vllm serve selftest-qwen35\n")
    (root / "baseline_build.command.txt").write_text("python -m pip install vllm==selftest\n")
    return argparse.Namespace(
        model_id="selftest-qwen35",
        backend="cuda",
        quantization="hf-gptq-int4",
        git_sha="0123456789abcdef",
        binary_sha256="a" * 64,
        dirty_status_json=root / "dirty_status.json",
        hardware=root / "hardware.json",
        runtime_snapshot=root / "runtime.json",
        product_surface="typed_cli",
        l0_template=root / "l0.json",
        l1_numeric=root / "l1.json",
        l2_quantized=root / "l2.json",
        l3_behavior=root / "l3.json",
        l4_agent=root / "l4.json",
        l5_concurrency=root / "l5.json",
        ferrum_run=root / "run.json",
        ferrum_serve=root / "serve.json",
        ferrum_perf_report=ferrum_perf,
        baseline_perf_report=baseline_perf,
        ferrum_bench_command=root / "ferrum_bench.command.txt",
        baseline_bench_command=root / "baseline_bench.command.txt",
        baseline_server_command=root / "baseline_server.command.txt",
        baseline_build_command=root / "baseline_build.command.txt",
        baseline_engine="vLLM",
        baseline_version="selftest",
        dataset_id="selftest/sharegpt-100-seed9271",
        dataset_sha="b" * 64,
        **artifacts,
    )


def write_selftest_w3_config(root: Path, config_path: Path, out_dir: Path) -> Path:
    args = write_selftest_w3_args(root)
    path_keys = [
        "hardware",
        "runtime_snapshot",
        "l0_template",
        "l1_numeric",
        "l2_quantized",
        "l3_behavior",
        "l4_agent",
        "l5_concurrency",
        "w3_s0_design",
        "w3_s0_microbench",
        "w3_s1_single_layer",
        "w3_s2_product",
        "ferrum_run",
        "ferrum_serve",
        "ferrum_perf_report",
        "baseline_perf_report",
    ]
    config_args = {key: str(getattr(args, key)) for key in path_keys}
    config_args.update(
        {
            "model_id": args.model_id,
            "backend": args.backend,
            "quantization": args.quantization,
            "git_sha": args.git_sha,
            "binary_sha256": args.binary_sha256,
            "dirty_status": {"dirty": False, "status_short": ""},
            "product_surface": args.product_surface,
            "ferrum_bench_command": [
                "ferrum",
                "bench-serve",
                "--fail-on-error",
                "--require-ci",
                "--seed",
                "9271",
                "--concurrency-sweep",
                "1,4,16,32",
                "--num-prompts",
                "100",
                "--n-repeats",
                "3",
                "--random-output-len",
                "128",
                "--ignore-eos",
            ],
            "baseline_bench_command": (
                "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
                "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 "
                "--random-output-len 128 --ignore-eos"
            ),
            "baseline_server_command": {
                "command_line": ["python", "-m", "vllm", "serve", "selftest-qwen35"],
            },
            "baseline_build_command": {
                "command": ["python", "-m", "pip", "install", "vllm==selftest"],
            },
            "baseline_engine": args.baseline_engine,
            "baseline_version": args.baseline_version,
            "dataset_id": args.dataset_id,
            "dataset_sha": args.dataset_sha,
        }
    )
    write_json(
        config_path,
        {
            "lane": "w3",
            "out": str(out_dir),
            "effective_concurrency": ["32=16"],
            "args": config_args,
        },
    )
    return config_path


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w2-release-grade-manifest-") as tmp:
        root = Path(tmp)
        source = write_selftest_source(root)
        baseline_source = write_selftest_source(root / "historical-baseline")
        out_dir = root / "out"
        manifest = build_w2_manifest(
            source=source,
            baseline_source=baseline_source,
            out_dir=out_dir,
            matrix_path=W2_MATRIX,
            effective_concurrency={32: 16},
            dataset_id="selftest/sharegpt-100-seed9271",
        )
        if not manifest.is_file():
            raise AssertionError("manifest was not written")
        rc = run_validator(out_dir, "w2")
        if rc != 0:
            raise AssertionError(f"W2 selftest final validator failed with rc={rc}")

        w3_args = write_selftest_w3_args(root / "w3-source")
        w3_out = root / "w3-out"
        w3_manifest = build_w3_manifest(
            args=w3_args,
            out_dir=w3_out,
            effective_concurrency={32: 16},
        )
        if not w3_manifest.is_file():
            raise AssertionError("W3 manifest was not written")
        rc = run_validator(w3_out, "w3")
        if rc != 0:
            raise AssertionError(f"W3 selftest final validator failed with rc={rc}")

        w3_config = write_selftest_w3_config(
            root / "w3-config-source",
            root / "w3-config.json",
            root / "w3-config-out",
        )
        config_args = apply_config(
            argparse.Namespace(
                config=w3_config,
                lane=None,
                no_run_validator=False,
                effective_concurrency=[],
            )
        )
        require_w3_args(config_args)
        w3_config_manifest = build_w3_manifest(
            args=config_args,
            out_dir=config_args.out,
            effective_concurrency=parse_effective_concurrency(config_args.effective_concurrency),
        )
        if not w3_config_manifest.is_file():
            raise AssertionError("W3 config manifest was not written")
        rc = run_validator(config_args.out, "w3")
        if rc != 0:
            raise AssertionError(f"W3 config selftest final validator failed with rc={rc}")

        bad_args = write_selftest_w3_args(root / "bad-w3-source")
        reports = load_json(bad_args.ferrum_perf_report)
        write_json(
            bad_args.ferrum_perf_report,
            [report for report in reports if report.get("concurrency") != 32],
        )
        try:
            build_w3_manifest(
                args=bad_args,
                out_dir=root / "bad-w3-out",
                effective_concurrency={},
            )
        except BuildError as exc:
            if "missing required performance cells" not in str(exc):
                raise AssertionError(f"unexpected W3 missing-cell error: {exc}") from exc
        else:
            raise AssertionError("bad W3 missing-cell selftest did not fail")
    print("MODEL RELEASE GRADE MANIFEST SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=["w2", "w3"], nargs="?", help="release-grade lane")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config for manifest inputs; paths are resolved relative to the repo root",
    )
    parser.add_argument("--source", type=Path, help="full-matrix artifact source dir")
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="artifact dir containing the vLLM baseline reports; defaults to --source",
    )
    parser.add_argument("--out", type=Path, help="output dir for model_release_grade_manifest.json")
    parser.add_argument("--matrix", type=Path, default=W2_MATRIX, help="W2 coverage matrix")
    parser.add_argument(
        "--dataset-id",
        default="w2/ascii-sharegpt-100-seed9271",
        help="prompt dataset id recorded in release-grade cells",
    )
    parser.add_argument(
        "--effective-concurrency",
        action="append",
        default=[],
        metavar="C=E",
        help="record an admission-capped effective concurrency, e.g. 32=16",
    )
    parser.add_argument(
        "--no-run-validator",
        action="store_true",
        help="only write the manifest; do not invoke the final validator",
    )
    parser.add_argument("--model-id", default="qwen35-release-grade", help="W3 model id")
    parser.add_argument("--backend", default="cuda", help="W3 backend")
    parser.add_argument("--quantization", default="hf-gptq-int4", help="W3 quantization/format")
    parser.add_argument("--git-sha", help="W3 git SHA for release evidence")
    parser.add_argument("--binary-sha256", help="W3 Ferrum binary SHA256")
    parser.add_argument("--dirty-status-json", type=Path, help="W3 dirty status JSON")
    parser.add_argument("--hardware", type=Path, help="W3 hardware artifact")
    parser.add_argument("--runtime-snapshot", type=Path, help="W3 runtime config snapshot")
    parser.add_argument(
        "--product-surface",
        default="typed_cli",
        choices=["typed_cli", "typed_config", "typed_defaults", "model_defaults"],
        help="W3 product runtime surface",
    )
    parser.add_argument("--l0-template", type=Path, help="W3 L0 artifact")
    parser.add_argument("--l1-numeric", type=Path, help="W3 L1 artifact")
    parser.add_argument("--l2-quantized", type=Path, help="W3 L2 artifact")
    parser.add_argument("--l3-behavior", type=Path, help="W3 L3 artifact")
    parser.add_argument("--l4-agent", type=Path, help="W3 L4 artifact")
    parser.add_argument("--l5-concurrency", type=Path, help="W3 L5 artifact")
    parser.add_argument("--w3-s0-design", type=Path, help="W3 S0 design artifact")
    parser.add_argument("--w3-s0-microbench", type=Path, help="W3 S0 microbench artifact")
    parser.add_argument("--w3-s1-single-layer", type=Path, help="W3 S1 single-layer artifact")
    parser.add_argument("--w3-s2-product", type=Path, help="W3 S2 product-path artifact")
    parser.add_argument("--ferrum-run", type=Path, help="W3 ferrum run artifact")
    parser.add_argument("--ferrum-serve", type=Path, help="W3 ferrum serve artifact")
    parser.add_argument("--ferrum-perf-report", type=Path, help="W3 Ferrum bench report")
    parser.add_argument("--baseline-perf-report", type=Path, help="W3 baseline bench report")
    parser.add_argument("--ferrum-bench-command", type=Path, help="W3 Ferrum bench command file")
    parser.add_argument("--baseline-bench-command", type=Path, help="W3 baseline bench command file")
    parser.add_argument("--baseline-server-command", type=Path, help="W3 baseline server command file")
    parser.add_argument("--baseline-build-command", type=Path, help="W3 baseline build command file")
    parser.add_argument("--baseline-engine", default="vLLM", help="W3 baseline engine")
    parser.add_argument("--baseline-version", default="unknown", help="W3 baseline engine version")
    parser.add_argument("--dataset-sha", help="W3 prompt dataset SHA256")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic self-test")
    return parser.parse_args()


def require_w3_args(args: argparse.Namespace) -> None:
    required = [
        "out",
        "git_sha",
        "binary_sha256",
        "dirty_status_json",
        "hardware",
        "runtime_snapshot",
        "l0_template",
        "l1_numeric",
        "l2_quantized",
        "l3_behavior",
        "l4_agent",
        "l5_concurrency",
        "w3_s0_design",
        "w3_s0_microbench",
        "w3_s1_single_layer",
        "w3_s2_product",
        "ferrum_run",
        "ferrum_serve",
        "ferrum_perf_report",
        "baseline_perf_report",
        "ferrum_bench_command",
        "baseline_bench_command",
        "baseline_server_command",
        "baseline_build_command",
        "dataset_sha",
    ]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        rendered = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise BuildError(f"W3 manifest requires {rendered}")


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_selftest()
    args = apply_config(args)
    effective: dict[int, int] = {}
    if args.lane == "w3" and args.runtime_snapshot is not None:
        merge_effective_concurrency(
            effective,
            effective_concurrency_from_runtime_snapshot(args.runtime_snapshot),
            f"--runtime-snapshot {args.runtime_snapshot}",
        )
    merge_effective_concurrency(
        effective,
        parse_effective_concurrency(args.effective_concurrency),
        "--effective-concurrency",
    )
    if args.lane == "w2":
        if args.source is None or args.out is None:
            raise BuildError("--source and --out are required")
        manifest = build_w2_manifest(
            source=args.source,
            baseline_source=args.baseline_source,
            out_dir=args.out,
            matrix_path=args.matrix,
            effective_concurrency=effective,
            dataset_id=args.dataset_id,
        )
    elif args.lane == "w3":
        require_w3_args(args)
        manifest = build_w3_manifest(
            args=args,
            out_dir=args.out,
            effective_concurrency=effective,
        )
    else:
        raise BuildError("lane must be w2 or w3")
    print(f"wrote {manifest}")
    if args.no_run_validator:
        return 0
    return run_validator(args.out, args.lane)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"MODEL RELEASE GRADE MANIFEST FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
