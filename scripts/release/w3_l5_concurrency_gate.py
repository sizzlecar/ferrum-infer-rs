#!/usr/bin/env python3
"""Package bench-serve reports into the W3 L5 concurrency artifact."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_NAME = "w3_l5_concurrency.json"
PASS_LINE_PREFIX = "W3 L5 CONCURRENCY PASS"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
DEFAULT_EXPECTED_OUTPUT_LEN = 128
REQUIRED_CONCURRENCY = {1, 4, 16, 32}
ZERO_FIELDS = [
    "bad_output_per_run",
    "malformed_stream_per_run",
    "missing_done_per_run",
    "duplicate_done_per_run",
    "zero_output_tokens_per_run",
    "stream_bulk_flush_per_run",
    "http_500_per_run",
    "panic_per_run",
]


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reports(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GateError(f"missing bench report file: {path}") from exc
    stripped = text.strip()
    if not stripped:
        raise GateError(f"empty bench report file: {path}")
    if stripped[0] in "[{":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        if data is not None:
            raise GateError(f"{path}: JSON report must be an object or object list")
    reports: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise GateError(f"{path}:{lineno}: JSONL row must be an object")
        reports.append(item)
    if not reports:
        raise GateError(f"{path}: no JSONL reports found")
    return reports


def int_list(report: dict[str, Any], field: str, n_repeats: int, label: str) -> list[int]:
    values = report.get(field)
    if not isinstance(values, list):
        raise GateError(f"{label}.{field} must be a list")
    out: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise GateError(f"{label}.{field} contains non-integer value {value!r}")
        out.append(value)
    if len(out) != n_repeats:
        raise GateError(f"{label}.{field} length {len(out)} != n_repeats {n_repeats}")
    return out


def int_matrix(
    report: dict[str, Any],
    field: str,
    n_repeats: int,
    row_len: int,
    label: str,
) -> list[list[int]]:
    values = report.get(field)
    if not isinstance(values, list):
        raise GateError(f"{label}.{field} must be a list of integer lists")
    out: list[list[int]] = []
    for row_idx, row in enumerate(values):
        if not isinstance(row, list):
            raise GateError(f"{label}.{field}[{row_idx}] must be an integer list")
        parsed_row: list[int] = []
        for col_idx, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise GateError(f"{label}.{field}[{row_idx}][{col_idx}] must be an integer")
            parsed_row.append(value)
        if len(parsed_row) != row_len:
            raise GateError(f"{label}.{field}[{row_idx}] length {len(parsed_row)} != {row_len}")
        out.append(parsed_row)
    if len(out) != n_repeats:
        raise GateError(f"{label}.{field} length {len(out)} != n_repeats {n_repeats}")
    return out


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateError(f"{label} must be a positive integer")
    return value


def command_parts(raw: Any, label: str) -> list[str]:
    if isinstance(raw, str):
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            raise GateError(f"{label} is not a valid shell command: {exc}") from exc
    elif isinstance(raw, list) and all(isinstance(part, str) for part in raw):
        parts = raw
    else:
        raise GateError(f"{label} must be a command string or string list")
    if not parts:
        raise GateError(f"{label} must not be empty")
    return parts


def has_flag(parts: list[str], flag: str) -> bool:
    return flag in parts or any(part.startswith(f"{flag}=") for part in parts)


def flag_values(parts: list[str], flag: str) -> list[str]:
    values: list[str] = []
    prefix = f"{flag}="
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if part.startswith(prefix):
            values.append(part[len(prefix) :])
        elif part == flag and idx + 1 < len(parts):
            values.append(parts[idx + 1])
            idx += 1
        idx += 1
    return values


def single_flag_value(parts: list[str], flag: str, label: str) -> str | None:
    values = flag_values(parts, flag)
    if len(values) > 1:
        raise GateError(f"{label} must not repeat {flag}")
    return values[0] if values else None


def parse_positive_int_set(raw: str, label: str) -> set[int]:
    out: set[int] = set()
    for item in raw.split(","):
        text = item.strip()
        if not text:
            raise GateError(f"{label} contains an empty concurrency cell")
        try:
            parsed = int(text)
        except ValueError as exc:
            raise GateError(f"{label} contains non-integer concurrency cell {text!r}") from exc
        if parsed <= 0:
            raise GateError(f"{label} contains non-positive concurrency cell {parsed}")
        out.add(parsed)
    return out


def parse_effective_concurrency_overrides(raw_values: list[str] | None) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for raw in raw_values or []:
        if not isinstance(raw, str) or not raw.strip():
            raise GateError("--effective-concurrency must be requested=effective")
        text = raw.strip()
        separator = "=" if "=" in text else ":"
        if separator not in text:
            raise GateError("--effective-concurrency must be requested=effective")
        requested_text, effective_text = [part.strip() for part in text.split(separator, 1)]
        try:
            requested = int(requested_text)
            effective = int(effective_text)
        except ValueError as exc:
            raise GateError(f"--effective-concurrency contains non-integer value: {raw!r}") from exc
        if requested <= 0 or effective <= 0:
            raise GateError(f"--effective-concurrency values must be positive: {raw!r}")
        if effective > requested:
            raise GateError(
                f"--effective-concurrency effective value {effective} exceeds requested {requested}"
            )
        previous = overrides.get(requested)
        if previous is not None and previous != effective:
            raise GateError(
                f"conflicting --effective-concurrency overrides for requested c={requested}"
            )
        overrides[requested] = effective
    return overrides


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError(f"{label} must be a JSON object: {path}")
    return data


def nested_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def effective_limit_from_config(data: dict[str, Any], label: str) -> int:
    auto_config = nested_object(data, "auto_config") or data
    candidates: list[tuple[str, Any]] = [
        (f"{label}.selected_admission_limit", auto_config.get("selected_admission_limit")),
        (
            f"{label}.admission.effective_max_concurrent",
            nested_object(auto_config, "admission").get("effective_max_concurrent"),
        ),
        (
            f"{label}.top_level_admission.effective_max_concurrent",
            nested_object(data, "admission").get("effective_max_concurrent"),
        ),
    ]
    parsed: list[tuple[str, int]] = []
    for source, value in candidates:
        if value is None:
            continue
        parsed.append((source, positive_int(value, source)))
    if not parsed:
        raise GateError(
            f"{label} must contain selected_admission_limit or admission.effective_max_concurrent"
        )
    first_source, first_value = parsed[0]
    for source, value in parsed[1:]:
        if value != first_value:
            raise GateError(
                f"{source}={value} does not match {first_source}={first_value}"
            )
    return first_value


def effective_concurrency_from_config(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    data = load_json_object(path, "--effective-config")
    limit = effective_limit_from_config(data, f"--effective-config {path}")
    mapping = {
        requested: min(requested, limit)
        for requested in sorted(REQUIRED_CONCURRENCY)
    }
    return mapping, {
        "source": "effective_config",
        "path": str(path),
        "selected_admission_limit": limit,
        "effective_concurrency": [
            {
                "requested_concurrency": requested,
                "effective_active_concurrency": effective,
            }
            for requested, effective in sorted(mapping.items())
        ],
    }


def merge_effective_concurrency(
    merged: dict[int, int],
    incoming: dict[int, int],
    source: str,
) -> None:
    for requested, effective in incoming.items():
        previous = merged.get(requested)
        if previous is not None and previous != effective:
            raise GateError(
                f"{source} conflicts for requested c={requested}: "
                f"{effective} != existing {previous}"
            )
        merged[requested] = effective


def command_concurrency_cells(parts: list[str], label: str) -> set[int]:
    cells: set[int] = set()
    for raw in flag_values(parts, "--concurrency-sweep"):
        cells.update(parse_positive_int_set(raw, f"{label} --concurrency-sweep"))
    for flag in ["--concurrency", "--max-concurrency"]:
        for raw in flag_values(parts, flag):
            try:
                parsed = int(raw)
            except ValueError as exc:
                raise GateError(f"{label} {flag} must be integer, got {raw!r}") from exc
            if parsed <= 0:
                raise GateError(f"{label} {flag} must be positive, got {parsed}")
            cells.add(parsed)
    if not cells:
        raise GateError(f"{label} must include --concurrency-sweep or --concurrency")
    return cells


def validate_bench_command(
    raw: Any,
    label: str,
    *,
    expected_output_len: int,
) -> tuple[list[str], set[int]]:
    parts = command_parts(raw, label)
    if not any(part == "bench-serve" or part.endswith("/bench-serve") for part in parts):
        raise GateError(f"{label} must invoke ferrum bench-serve")
    if has_flag(parts, "--request-rate"):
        raise GateError(f"{label} must use closed-loop concurrency, not --request-rate")
    for flag in ["--fail-on-error", "--require-ci"]:
        if not has_flag(parts, flag):
            raise GateError(f"{label} missing {flag}")
    if single_flag_value(parts, "--seed", label) != "9271":
        raise GateError(f"{label} must include --seed 9271")
    if single_flag_value(parts, "--n-repeats", label) != "3":
        raise GateError(f"{label} must include --n-repeats 3")
    if not has_flag(parts, "--ignore-eos"):
        raise GateError(f"{label} missing --ignore-eos")
    if single_flag_value(parts, "--random-output-len", label) != str(expected_output_len):
        raise GateError(f"{label} must include --random-output-len {expected_output_len}")
    for part in parts:
        if re.match(r"^FERRUM_[A-Z0-9_]+=", part):
            raise GateError(f"{label} uses hidden env override: {part.split('=', 1)[0]}")
    return parts, command_concurrency_cells(parts, label)


def validate_bench_commands(
    commands: list[Any],
    expected_cells: set[int],
    *,
    expected_output_len: int,
) -> list[dict[str, Any]]:
    if not commands:
        raise GateError("at least one bench-serve command must be supplied with --command")
    normalized: list[dict[str, Any]] = []
    covered: set[int] = set()
    for idx, raw in enumerate(commands):
        label = f"commands[{idx}]"
        parts, cells = validate_bench_command(
            raw,
            label,
            expected_output_len=expected_output_len,
        )
        covered.update(cells)
        entry: dict[str, Any] = {
            "command_line": parts,
            "covers_concurrency": sorted(cells),
        }
        if isinstance(raw, str):
            entry["raw"] = raw
        normalized.append(entry)
    missing = sorted(expected_cells - covered)
    if missing:
        raise GateError(f"bench commands missing required concurrency cells: {missing}")
    return normalized


def report_concurrency(report: dict[str, Any], label: str) -> int:
    if report.get("scenario") != "closed_loop":
        raise GateError(f"{label}.scenario must be closed_loop")
    return positive_int(report.get("concurrency"), f"{label}.concurrency")


def cell_from_report(
    report: dict[str, Any],
    source: str,
    label: str,
    *,
    expected_output_len: int,
) -> dict[str, Any]:
    concurrency = report_concurrency(report, label)
    n_repeats = positive_int(report.get("n_repeats"), f"{label}.n_repeats")
    requests = positive_int(report.get("n_requests_per_run"), f"{label}.n_requests_per_run")
    if n_repeats < 3:
        raise GateError(f"{label}.n_repeats must be >= 3")
    if report.get("request_rate") is not None:
        raise GateError(f"{label} must be closed-loop, not request_rate")
    if report.get("output_token_count_source") != "usage":
        raise GateError(f"{label}.output_token_count_source must be usage")

    completed = int_list(report, "completed_per_run", n_repeats, label)
    errored = int_list(report, "errored_per_run", n_repeats, label)
    if completed != [requests] * n_repeats:
        raise GateError(f"{label}.completed_per_run must be full for every repeat")
    if any(value != 0 for value in errored):
        raise GateError(f"{label}.errored_per_run must be all zero")
    output_tokens = int_matrix(
        report,
        "output_tokens_per_request",
        n_repeats,
        requests,
        label,
    )
    for row_idx, row in enumerate(output_tokens):
        if any(value != expected_output_len for value in row):
            raise GateError(
                f"{label}.output_tokens_per_request[{row_idx}] must equal "
                f"--random-output-len {expected_output_len}"
            )

    raw_effective = report.get("effective_active_concurrency", concurrency)
    effective = positive_int(raw_effective, f"{label}.effective_active_concurrency")
    if effective > concurrency:
        raise GateError(
            f"{label}.effective_active_concurrency {effective} exceeds requested c={concurrency}"
        )

    cell = {
        "requested_concurrency": concurrency,
        "effective_active_concurrency": effective,
        "n_repeats": n_repeats,
        "requests_per_run": requests,
        "completed_per_run": completed,
        "errored_per_run": errored,
        "output_tokens_per_request": output_tokens,
        "output_token_count_source": "usage",
        "source_report": source,
    }
    for field in ZERO_FIELDS:
        values = int_list(report, field, n_repeats, label)
        if any(value != 0 for value in values):
            raise GateError(f"{label}.{field} must be all zero")
        cell[field] = values
    return cell


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    expected_output_len = positive_int(
        getattr(args, "expected_output_len", DEFAULT_EXPECTED_OUTPUT_LEN),
        "expected_output_len",
    )
    effective_sources: list[dict[str, Any]] = []
    effective_overrides: dict[int, int] = {}
    for path in getattr(args, "effective_config", None) or []:
        mapping, source = effective_concurrency_from_config(path)
        merge_effective_concurrency(
            effective_overrides,
            mapping,
            f"--effective-config {path}",
        )
        effective_sources.append(source)
    manual_overrides = parse_effective_concurrency_overrides(
        getattr(args, "effective_concurrency", None)
    )
    merge_effective_concurrency(
        effective_overrides,
        manual_overrides,
        "--effective-concurrency",
    )
    reports: list[tuple[dict[str, Any], str]] = []
    for path in args.report:
        for report in load_reports(path):
            reports.append((report, str(path)))
    if not reports:
        raise GateError("no reports supplied")

    cells: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx, (report, source) in enumerate(reports):
        cell = cell_from_report(
            report,
            source,
            f"reports[{idx}]",
            expected_output_len=expected_output_len,
        )
        concurrency = cell["requested_concurrency"]
        if concurrency in seen:
            raise GateError(f"duplicate concurrency cell c={concurrency}")
        seen.add(concurrency)
        if concurrency in effective_overrides:
            reported_effective = cell["effective_active_concurrency"]
            if reported_effective != concurrency and reported_effective != effective_overrides[concurrency]:
                raise GateError(
                    f"reports[{idx}].effective_active_concurrency {reported_effective} "
                    f"conflicts with configured effective concurrency {effective_overrides[concurrency]}"
                )
            cell["effective_active_concurrency"] = effective_overrides[concurrency]
        if cell["effective_active_concurrency"] < concurrency:
            cell["published_concurrency"] = cell["effective_active_concurrency"]
        if concurrency in REQUIRED_CONCURRENCY:
            cells.append(cell)
    missing = sorted(REQUIRED_CONCURRENCY - seen)
    if missing:
        raise GateError(f"missing required concurrency cells: {missing}")
    unknown_overrides = sorted(set(effective_overrides) - seen)
    if unknown_overrides:
        raise GateError(f"--effective-concurrency references missing cells: {unknown_overrides}")
    cells.sort(key=lambda item: item["requested_concurrency"])
    commands = validate_bench_commands(
        args.command or [],
        REQUIRED_CONCURRENCY,
        expected_output_len=expected_output_len,
    )

    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l5_concurrency",
        "model_id": args.model_id,
        "product_surface": "typed_cli",
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"{PASS_LINE_PREFIX}: {args.out}",
        "commands": commands,
        "concurrency": {
            "closed_loop": True,
            "stream_options_include_usage": True,
            "output_token_count_source": "usage",
            "expected_output_tokens_per_request": expected_output_len,
            "effective_concurrency_overrides": [
                {
                    "requested_concurrency": requested,
                    "effective_active_concurrency": effective,
                }
                for requested, effective in sorted(effective_overrides.items())
                if effective != requested
            ],
            "effective_concurrency_sources": effective_sources,
            "cells": cells,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / ARTIFACT_NAME, artifact)
    return artifact


def fake_report(concurrency: int) -> dict[str, Any]:
    zeros = [0, 0, 0]
    report = {
        "model": DEFAULT_MODEL_ID,
        "backend": "cuda",
        "scenario": "closed_loop",
        "concurrency": concurrency,
        "request_rate": None,
        "n_prompt": 256,
        "n_gen": 128,
        "output_token_count_source": "usage",
        "n_repeats": 3,
        "n_requests_per_run": 8,
        "warmup_requests": 0,
        "completed_per_run": [8, 8, 8],
        "errored_per_run": zeros,
        "output_tokens_per_request": [[DEFAULT_EXPECTED_OUTPUT_LEN] * 8 for _ in range(3)],
    }
    for field in ZERO_FIELDS:
        report[field] = zeros
    return report


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-l5-") as tmp:
        root = Path(tmp)
        report_path = root / "bench.jsonl"
        report_path.write_text(
            "\n".join(json.dumps(fake_report(c), sort_keys=True) for c in [1, 4, 16, 32])
            + "\n",
            encoding="utf-8",
        )
        effective_config_path = root / "effective_config.json"
        write_json(
            effective_config_path,
            {
                "selected_admission_limit": 8,
                "admission": {"effective_max_concurrent": 8},
            },
        )
        args = argparse.Namespace(
            report=[report_path],
            out=root / "out",
            model_id=DEFAULT_MODEL_ID,
            expected_output_len=DEFAULT_EXPECTED_OUTPUT_LEN,
            effective_config=[effective_config_path],
            effective_concurrency=[],
            command=[
                "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
                "--n-repeats 3 --concurrency-sweep 1,4,16,32 "
                "--random-output-len 128 --ignore-eos"
            ],
        )
        artifact = build_artifact(args)
        if len(artifact["concurrency"]["cells"]) != 4:
            raise AssertionError("selftest did not preserve four cells")
        cells_by_c = {
            cell["requested_concurrency"]: cell
            for cell in artifact["concurrency"]["cells"]
        }
        if cells_by_c[16]["effective_active_concurrency"] != 8:
            raise AssertionError("selftest did not preserve c16 effective concurrency")
        if cells_by_c[32]["effective_active_concurrency"] != 8:
            raise AssertionError("selftest did not preserve c32 effective concurrency")
        if cells_by_c[32].get("published_concurrency") != 8:
            raise AssertionError("selftest did not publish capped c32 concurrency")
        if not artifact["concurrency"].get("effective_concurrency_sources"):
            raise AssertionError("selftest did not record effective config source")
        if artifact["commands"][0]["covers_concurrency"] != [1, 4, 16, 32]:
            raise AssertionError("selftest did not preserve command concurrency coverage")

        from model_release_grade_goal_gate import validate_w3_l0_l5_artifact  # type: ignore

        problems: list[str] = []
        validate_w3_l0_l5_artifact(
            "l5_concurrency",
            {"artifact": str(args.out / ARTIFACT_NAME)},
            args.out,
            problems,
        )
        if problems:
            raise AssertionError(f"L5 selftest artifact failed validator: {problems}")

        bad_path = root / "bad.jsonl"
        bad = fake_report(1)
        bad["errored_per_run"] = [0, 1, 0]
        bad_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        try:
            build_artifact(
                argparse.Namespace(
                    report=[bad_path],
                    out=root / "bad_out",
                    model_id=DEFAULT_MODEL_ID,
                    expected_output_len=DEFAULT_EXPECTED_OUTPUT_LEN,
                    effective_config=[],
                    effective_concurrency=[],
                    command=[],
                )
            )
        except GateError as exc:
            if "errored_per_run" not in str(exc) and "missing required" not in str(exc):
                raise AssertionError(f"unexpected bad-report error: {exc}") from exc
        else:
            raise AssertionError("bad L5 report unexpectedly passed")

        bad_command_path = root / "bad_command.jsonl"
        bad_command_path.write_text(
            "\n".join(json.dumps(fake_report(c), sort_keys=True) for c in [1, 4, 16, 32])
            + "\n",
            encoding="utf-8",
        )
        try:
            build_artifact(
                argparse.Namespace(
                    report=[bad_command_path],
                    out=root / "bad_command_out",
                    model_id=DEFAULT_MODEL_ID,
                    expected_output_len=DEFAULT_EXPECTED_OUTPUT_LEN,
                    effective_config=[],
                    effective_concurrency=[],
                    command=[
                        "ferrum bench-serve --fail-on-error --seed 9271 "
                        "--n-repeats 3 --concurrency-sweep 1,4,16 "
                        "--random-output-len 128 --ignore-eos"
                    ],
                )
            )
        except GateError as exc:
            if "--require-ci" not in str(exc):
                raise AssertionError(f"unexpected bad-command error: {exc}") from exc
        else:
            raise AssertionError("bad L5 command unexpectedly passed")

        bad_output_path = root / "bad_output.jsonl"
        bad_output = fake_report(1)
        bad_output["output_tokens_per_request"][0][0] = DEFAULT_EXPECTED_OUTPUT_LEN - 1
        bad_output_path.write_text(json.dumps(bad_output) + "\n", encoding="utf-8")
        try:
            build_artifact(
                argparse.Namespace(
                    report=[bad_output_path],
                    out=root / "bad_output_out",
                    model_id=DEFAULT_MODEL_ID,
                    expected_output_len=DEFAULT_EXPECTED_OUTPUT_LEN,
                    effective_config=[],
                    effective_concurrency=[],
                    command=[
                        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
                        "--n-repeats 3 --concurrency-sweep 1,4,16,32 "
                        "--random-output-len 128 --ignore-eos"
                    ],
                )
            )
        except GateError as exc:
            if "output_tokens_per_request" not in str(exc):
                raise AssertionError(f"unexpected bad-output error: {exc}") from exc
        else:
            raise AssertionError("bad L5 output-token report unexpectedly passed")

        missing_ignore_eos_path = root / "missing_ignore_eos.jsonl"
        missing_ignore_eos_path.write_text(
            "\n".join(json.dumps(fake_report(c), sort_keys=True) for c in [1, 4, 16, 32])
            + "\n",
            encoding="utf-8",
        )
        try:
            build_artifact(
                argparse.Namespace(
                    report=[missing_ignore_eos_path],
                    out=root / "missing_ignore_eos_out",
                    model_id=DEFAULT_MODEL_ID,
                    expected_output_len=DEFAULT_EXPECTED_OUTPUT_LEN,
                    effective_config=[],
                    effective_concurrency=[],
                    command=[
                        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
                        "--n-repeats 3 --concurrency-sweep 1,4,16,32 "
                        "--random-output-len 128"
                    ],
                )
            )
        except GateError as exc:
            if "--ignore-eos" not in str(exc):
                raise AssertionError(f"unexpected missing-ignore-eos error: {exc}") from exc
        else:
            raise AssertionError("missing-ignore-eos L5 command unexpectedly passed")

        try:
            build_artifact(
                argparse.Namespace(
                    report=[report_path],
                    out=root / "bad_effective_out",
                    model_id=DEFAULT_MODEL_ID,
                    expected_output_len=DEFAULT_EXPECTED_OUTPUT_LEN,
                    effective_config=[],
                    effective_concurrency=["16=17"],
                    command=[
                        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
                        "--n-repeats 3 --concurrency-sweep 1,4,16,32 "
                        "--random-output-len 128 --ignore-eos"
                    ],
                )
            )
        except GateError as exc:
            if "exceeds requested" not in str(exc):
                raise AssertionError(f"unexpected bad-effective error: {exc}") from exc
        else:
            raise AssertionError("bad effective concurrency unexpectedly passed")
    print("W3 L5 CONCURRENCY SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--expected-output-len", type=int, default=DEFAULT_EXPECTED_OUTPUT_LEN)
    parser.add_argument("--command", action="append", help="bench command line used")
    parser.add_argument(
        "--effective-config",
        type=Path,
        action="append",
        default=[],
        help="Effective config JSON used to derive admission-capped concurrency",
    )
    parser.add_argument(
        "--effective-concurrency",
        action="append",
        default=[],
        help="Record a requested-to-effective active concurrency cap, e.g. 32=8",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        if not args.report:
            raise GateError("missing required arg: --report")
        if args.out is None:
            raise GateError("missing required arg: --out")
        artifact = build_artifact(args)
    except (GateError, json.JSONDecodeError) as exc:
        print(f"W3 L5 CONCURRENCY FAIL: {exc}", file=sys.stderr)
        return 1
    print(artifact["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
