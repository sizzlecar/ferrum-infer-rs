#!/usr/bin/env python3
"""Package bench-serve reports into the W3 L5 concurrency artifact."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_NAME = "w3_l5_concurrency.json"
PASS_LINE_PREFIX = "W3 L5 CONCURRENCY PASS"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
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


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateError(f"{label} must be a positive integer")
    return value


def report_concurrency(report: dict[str, Any], label: str) -> int:
    if report.get("scenario") != "closed_loop":
        raise GateError(f"{label}.scenario must be closed_loop")
    return positive_int(report.get("concurrency"), f"{label}.concurrency")


def cell_from_report(report: dict[str, Any], source: str, label: str) -> dict[str, Any]:
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

    cell = {
        "requested_concurrency": concurrency,
        "effective_active_concurrency": report.get("effective_active_concurrency", concurrency),
        "n_repeats": n_repeats,
        "requests_per_run": requests,
        "completed_per_run": completed,
        "errored_per_run": errored,
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
    reports: list[tuple[dict[str, Any], str]] = []
    for path in args.report:
        for report in load_reports(path):
            reports.append((report, str(path)))
    if not reports:
        raise GateError("no reports supplied")

    cells: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx, (report, source) in enumerate(reports):
        cell = cell_from_report(report, source, f"reports[{idx}]")
        concurrency = cell["requested_concurrency"]
        if concurrency in seen:
            raise GateError(f"duplicate concurrency cell c={concurrency}")
        seen.add(concurrency)
        if concurrency in REQUIRED_CONCURRENCY:
            cells.append(cell)
    missing = sorted(REQUIRED_CONCURRENCY - seen)
    if missing:
        raise GateError(f"missing required concurrency cells: {missing}")
    cells.sort(key=lambda item: item["requested_concurrency"])

    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l5_concurrency",
        "model_id": args.model_id,
        "product_surface": "typed_cli",
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"{PASS_LINE_PREFIX}: {args.out}",
        "commands": args.command or [],
        "concurrency": {
            "closed_loop": True,
            "stream_options_include_usage": True,
            "output_token_count_source": "usage",
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
        args = argparse.Namespace(
            report=[report_path],
            out=root / "out",
            model_id=DEFAULT_MODEL_ID,
            command=[
                "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
                "--n-repeats 3 --concurrency-sweep 1,4,16,32"
            ],
        )
        artifact = build_artifact(args)
        if len(artifact["concurrency"]["cells"]) != 4:
            raise AssertionError("selftest did not preserve four cells")

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
                    command=[],
                )
            )
        except GateError as exc:
            if "errored_per_run" not in str(exc) and "missing required" not in str(exc):
                raise AssertionError(f"unexpected bad-report error: {exc}") from exc
        else:
            raise AssertionError("bad L5 report unexpectedly passed")
    print("W3 L5 CONCURRENCY SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--command", action="append", help="bench command line used")
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
