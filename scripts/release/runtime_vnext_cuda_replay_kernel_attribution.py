#!/usr/bin/env python3
"""Join Ferrum reusable-execution spans to Nsight kernel activity by fingerprint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PASS_PREFIX = "CUDA REPLAY KERNEL ATTRIBUTION PASS:"
REJECT_PREFIX = "CUDA REPLAY KERNEL ATTRIBUTION REJECT:"
RANGE_PREFIX = "ferrum.cuda.replay/"
FINGERPRINT_RE = re.compile(r"ferrum\.cuda\.replay/([0-9a-f]{64})")


class AttributionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionError(message)


def read_replay_spans(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    elapsed_by_fingerprint: dict[str, int] = defaultdict(int)
    spans_by_fingerprint: dict[str, int] = defaultdict(int)
    physical_span_count = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AttributionError(
                    f"{path}:{line_number}: malformed JSON: {exc}"
                ) from exc
            if event.get("phase") != "vnext.device_execution_span":
                continue
            physical_span_count += 1
            shape = event.get("shape")
            attributes = event.get("attributes")
            require(
                isinstance(shape, dict) and isinstance(attributes, dict),
                f"{path}:{line_number}: physical span lacks shape or attributes",
            )
            require(
                attributes.get("device_timing_span_kind") == "reusable_executable",
                f"{path}:{line_number}: physical span is not reusable execution",
            )
            require(
                attributes.get("device_timing_status") == "measured",
                f"{path}:{line_number}: reusable execution span is not measured",
            )
            fingerprint = attributes.get("reusable_executable_fingerprint")
            require(
                isinstance(fingerprint, str)
                and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None,
                f"{path}:{line_number}: reusable execution span lacks a valid fingerprint",
            )
            elapsed_ns = shape.get("device_elapsed_ns")
            require(
                isinstance(elapsed_ns, int)
                and not isinstance(elapsed_ns, bool)
                and elapsed_ns > 0,
                f"{path}:{line_number}: reusable execution span lacks positive device_elapsed_ns",
            )
            elapsed_by_fingerprint[fingerprint] += elapsed_ns
            spans_by_fingerprint[fingerprint] += 1

    require(physical_span_count > 0, f"{path}: no reusable execution spans")
    return dict(elapsed_by_fingerprint), dict(spans_by_fingerprint)


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def find_csv_table(text: str) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        try:
            header = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = [normalized_header(value) for value in header]
        if "name" not in normalized or not any(
            value.startswith("total time") for value in normalized
        ):
            continue
        rows: list[list[str]] = []
        for candidate in lines[index + 1 :]:
            try:
                row = next(csv.reader([candidate]))
            except csv.Error:
                continue
            if len(row) == len(header):
                rows.append(row)
        return header, rows
    raise AttributionError("Nsight CSV lacks a Name/Total Time table")


def parse_nsight_kernel_summary(
    path: Path,
) -> tuple[dict[str, int], dict[tuple[str, str], dict[str, int]]]:
    header, rows = find_csv_table(path.read_text(encoding="utf-8", errors="replace"))
    normalized = [normalized_header(value) for value in header]
    name_index = normalized.index("name")
    total_indices = [
        index for index, value in enumerate(normalized) if value.startswith("total time")
    ]
    require(len(total_indices) == 1, "Nsight CSV has ambiguous Total Time columns")
    total_index = total_indices[0]
    require(
        "ns" in normalized[total_index].split(),
        "Nsight Total Time must be emitted in nanoseconds",
    )
    instances_index = normalized.index("instances") if "instances" in normalized else None

    elapsed_by_fingerprint: dict[str, int] = defaultdict(int)
    kernels: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"elapsed_ns": 0, "instances": 0}
    )
    for row in rows:
        name = row[name_index]
        match = FINGERPRINT_RE.search(name)
        if match is None:
            continue
        fingerprint = match.group(1)
        try:
            elapsed_ns = int(row[total_index])
            instances = int(row[instances_index]) if instances_index is not None else 0
        except ValueError as exc:
            raise AttributionError(f"invalid numeric Nsight row: {row}") from exc
        require(elapsed_ns > 0, f"Nsight row has non-positive duration: {row}")
        suffix = name[match.end() :].lstrip(" :/|-")
        kernel_name = suffix or "<unnamed>"
        elapsed_by_fingerprint[fingerprint] += elapsed_ns
        kernels[(fingerprint, kernel_name)]["elapsed_ns"] += elapsed_ns
        kernels[(fingerprint, kernel_name)]["instances"] += instances

    require(
        elapsed_by_fingerprint,
        f"{path}: no Nsight kernel rows carry the {RANGE_PREFIX} fingerprint",
    )
    return dict(elapsed_by_fingerprint), dict(kernels)


def parse_nsight_projection_trace(
    path: Path,
) -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    required_headers = {
        "name",
        "projected duration ns",
        "lvl",
        "numchild",
    }
    header: list[str] | None = None
    rows: list[list[str]] = []
    for index, line in enumerate(lines):
        try:
            candidate = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = [normalized_header(value) for value in candidate]
        if not required_headers.issubset(normalized):
            continue
        header = normalized
        for row_line in lines[index + 1 :]:
            try:
                row = next(csv.reader([row_line]))
            except csv.Error:
                continue
            if len(row) == len(candidate):
                rows.append(row)
        break
    require(header is not None, "Nsight projection CSV lacks the required trace table")

    name_index = header.index("name")
    duration_index = header.index("projected duration ns")
    level_index = header.index("lvl")
    children_index = header.index("numchild")
    projected_by_fingerprint: dict[str, int] = defaultdict(int)
    counts_by_fingerprint: dict[str, int] = defaultdict(int)
    invalid_ranges: list[dict[str, Any]] = []
    for row in rows:
        name = row[name_index]
        match = FINGERPRINT_RE.search(name)
        if match is None:
            continue
        fingerprint = match.group(1)
        try:
            projected_ns = int(row[duration_index])
            level = int(row[level_index])
            child_count = int(row[children_index])
        except ValueError as exc:
            raise AttributionError(f"invalid numeric Nsight projection row: {row}") from exc
        require(
            projected_ns > 0,
            f"Nsight projection row has non-positive duration: {row}",
        )
        projected_by_fingerprint[fingerprint] += projected_ns
        counts_by_fingerprint[fingerprint] += 1
        if level != 0 or child_count != 0:
            invalid_ranges.append(
                {
                    "fingerprint": fingerprint,
                    "level": level,
                    "child_count": child_count,
                    "projected_ns": projected_ns,
                }
            )

    require(
        projected_by_fingerprint,
        f"{path}: no projected NVTX ranges carry the {RANGE_PREFIX} fingerprint",
    )
    return (
        dict(projected_by_fingerprint),
        dict(counts_by_fingerprint),
        invalid_ranges,
    )


def analyze(
    trace_path: Path,
    nsys_kernel_path: Path,
    nsys_projection_path: Path,
    minimum_coverage: float,
    maximum_coverage: float,
) -> dict[str, Any]:
    require(
        0.0 < minimum_coverage <= 1.0,
        "minimum coverage must be within (0, 1]",
    )
    require(maximum_coverage >= 1.0, "maximum coverage must be at least 1")
    internal_elapsed, span_counts = read_replay_spans(trace_path)
    nsys_kernel_elapsed, kernels = parse_nsight_kernel_summary(nsys_kernel_path)
    nsys_projected_elapsed, projection_counts, invalid_ranges = (
        parse_nsight_projection_trace(nsys_projection_path)
    )

    internal_fingerprints = set(internal_elapsed)
    kernel_fingerprints = set(nsys_kernel_elapsed)
    projected_fingerprints = set(nsys_projected_elapsed)
    missing_kernel = sorted(internal_fingerprints - kernel_fingerprints)
    foreign_kernel = sorted(kernel_fingerprints - internal_fingerprints)
    missing_projection = sorted(internal_fingerprints - projected_fingerprints)
    foreign_projection = sorted(projected_fingerprints - internal_fingerprints)
    internal_total = sum(internal_elapsed.values())
    projected_total = sum(
        nsys_projected_elapsed.get(key, 0) for key in internal_fingerprints
    )
    kernel_work_total = sum(
        nsys_kernel_elapsed.get(key, 0) for key in internal_fingerprints
    )
    projection_coverage = projected_total / internal_total

    per_fingerprint = []
    count_mismatches = []
    for fingerprint in sorted(internal_fingerprints):
        expected_ns = internal_elapsed[fingerprint]
        projected_ns = nsys_projected_elapsed.get(fingerprint, 0)
        expected_count = span_counts[fingerprint]
        projected_count = projection_counts.get(fingerprint, 0)
        if expected_count != projected_count:
            count_mismatches.append(
                {
                    "fingerprint": fingerprint,
                    "ferrum_span_count": expected_count,
                    "nsight_range_count": projected_count,
                }
            )
        per_fingerprint.append(
            {
                "fingerprint": fingerprint,
                "ferrum_span_count": expected_count,
                "nsight_range_count": projected_count,
                "ferrum_replay_elapsed_ns": expected_ns,
                "nsight_projected_elapsed_ns": projected_ns,
                "projection_coverage": projected_ns / expected_ns,
                "nsight_kernel_work_ns": nsys_kernel_elapsed.get(fingerprint, 0),
            }
        )

    top_kernels = []
    for (fingerprint, kernel_name), values in kernels.items():
        if fingerprint not in internal_fingerprints:
            continue
        elapsed_ns = values["elapsed_ns"]
        top_kernels.append(
            {
                "fingerprint": fingerprint,
                "kernel_name": kernel_name,
                "elapsed_ns": elapsed_ns,
                "instances": values["instances"],
                "kernel_work_to_ferrum_replay_ratio": elapsed_ns / internal_total,
            }
        )
    top_kernels.sort(key=lambda row: (-row["elapsed_ns"], row["kernel_name"]))

    errors = []
    if missing_kernel:
        errors.append(
            f"{len(missing_kernel)} Ferrum replay fingerprints are absent from Nsight kernels"
        )
    if foreign_kernel:
        errors.append(
            f"{len(foreign_kernel)} Nsight kernel fingerprints are absent from Ferrum trace"
        )
    if missing_projection:
        errors.append(
            f"{len(missing_projection)} Ferrum replay fingerprints are absent from Nsight projections"
        )
    if foreign_projection:
        errors.append(
            f"{len(foreign_projection)} Nsight projection fingerprints are absent from Ferrum trace"
        )
    if count_mismatches:
        errors.append(
            f"{len(count_mismatches)} replay fingerprints have span/range count mismatches"
        )
    if invalid_ranges:
        errors.append(
            f"{len(invalid_ranges)} projected replay ranges are nested or own child ranges"
        )
    if projection_coverage < minimum_coverage:
        errors.append(
            f"projected wall-time coverage {projection_coverage:.6f} is below minimum "
            f"{minimum_coverage:.6f}"
        )
    if projection_coverage > maximum_coverage:
        errors.append(
            f"projected wall-time coverage {projection_coverage:.6f} exceeds maximum "
            f"{maximum_coverage:.6f}"
        )
    if not top_kernels:
        errors.append("no correlated kernel activity remains after the fingerprint join")

    return {
        "schema_version": 2,
        "trace_jsonl": str(trace_path),
        "nsys_kernel_csv": str(nsys_kernel_path),
        "nsys_projection_csv": str(nsys_projection_path),
        "range_prefix": RANGE_PREFIX,
        "minimum_coverage": minimum_coverage,
        "maximum_coverage": maximum_coverage,
        "replay_fingerprint_count": len(internal_fingerprints),
        "replay_span_count": sum(span_counts.values()),
        "ferrum_replay_elapsed_ns": internal_total,
        "nsight_projected_replay_elapsed_ns": projected_total,
        "projection_coverage": projection_coverage,
        "nsight_kernel_work_ns": kernel_work_total,
        "kernel_work_to_projection_ratio": (
            kernel_work_total / projected_total if projected_total else None
        ),
        "missing_kernel_fingerprints": missing_kernel,
        "foreign_kernel_fingerprints": foreign_kernel,
        "missing_projection_fingerprints": missing_projection,
        "foreign_projection_fingerprints": foreign_projection,
        "range_count_mismatches": count_mismatches,
        "invalid_projected_range_count": len(invalid_ranges),
        "invalid_projected_ranges": invalid_ranges[:50],
        "per_fingerprint": per_fingerprint,
        "top_kernels": top_kernels[:50],
        "dominant_kernel": top_kernels[0] if top_kernels else None,
        "errors": errors,
        "pass": not errors,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_self_test() -> None:
    fingerprint = "a" * 64
    with tempfile.TemporaryDirectory(prefix="ferrum-replay-kernel-attribution-") as tmp:
        root = Path(tmp)
        trace = root / "trace.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "phase": "vnext.device_execution_span",
                    "shape": {"device_elapsed_ns": 1_000},
                    "attributes": {
                        "device_timing_span_kind": "reusable_executable",
                        "device_timing_status": "measured",
                        "reusable_executable_fingerprint": fingerprint,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        nsys = root / "kernels.csv"
        nsys.write_text(
            "Profiler preamble\n"
            "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
            f"70.0,700,10,70.0,{RANGE_PREFIX}{fingerprint}:marlin\n"
            f"27.0,270,10,27.0,{RANGE_PREFIX}{fingerprint}:attention\n",
            encoding="utf-8",
        )
        projection = root / "projection.csv"
        projection.write_text(
            "Name,Projected Start (ns),Projected Duration (ns),Orig Start (ns),"
            "Orig Duration (ns),Style,PID,TID,NumGPUOps,Lvl,NumChild,RangeId,"
            "ParentId,RangeStack\n"
            f"{RANGE_PREFIX}{fingerprint},10,970,1,10,PushPop,1,1,2,0,0,1,,1\n",
            encoding="utf-8",
        )
        summary = analyze(trace, nsys, projection, 0.95, 1.10)
        require(summary["pass"], f"self-test valid fixture rejected: {summary}")
        require(
            summary["projection_coverage"] == 0.97,
            "self-test projection coverage mismatch",
        )
        require(
            summary["kernel_work_to_projection_ratio"] == 1.0,
            "self-test kernel work ratio mismatch",
        )
        require(
            summary["dominant_kernel"]["kernel_name"] == "marlin",
            "self-test dominant kernel mismatch",
        )

        nsys.write_text(
            "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
            "100.0,1000,1,1000.0,untagged_kernel\n",
            encoding="utf-8",
        )
        try:
            analyze(trace, nsys, projection, 0.95, 1.10)
        except AttributionError as exc:
            require("no Nsight kernel rows" in str(exc), f"unexpected rejection: {exc}")
        else:
            raise AttributionError("self-test accepted an uncorrelated Nsight fixture")

        nsys.write_text(
            "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
            f"100.0,970,1,970.0,{RANGE_PREFIX}{fingerprint}:marlin\n",
            encoding="utf-8",
        )
        projection.write_text(
            "Name,Projected Start (ns),Projected Duration (ns),Orig Start (ns),"
            "Orig Duration (ns),Style,PID,TID,NumGPUOps,Lvl,NumChild,RangeId,"
            "ParentId,RangeStack\n"
            f"{RANGE_PREFIX}{fingerprint},10,970,1,10,PushPop,1,1,2,1,1,1,,1\n",
            encoding="utf-8",
        )
        summary = analyze(trace, nsys, projection, 0.95, 1.10)
        require(not summary["pass"], "self-test accepted nested replay ranges")
        require(
            "nested or own child ranges" in summary["errors"][0],
            f"unexpected nested-range rejection: {summary}",
        )
    print("CUDA REPLAY KERNEL ATTRIBUTION SELFTEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--nsys-kernel-csv", type=Path)
    parser.add_argument("--nsys-projection-csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--maximum-coverage", type=float, default=1.10)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        if (
            args.trace_jsonl
            or args.nsys_kernel_csv
            or args.nsys_projection_csv
            or args.out
        ):
            parser.error("--self-test cannot be combined with artifact inputs")
        run_self_test()
        return 0
    if (
        not args.trace_jsonl
        or not args.nsys_kernel_csv
        or not args.nsys_projection_csv
        or not args.out
    ):
        parser.error(
            "--trace-jsonl, --nsys-kernel-csv, --nsys-projection-csv, and --out "
            "are required"
        )

    try:
        summary = analyze(
            args.trace_jsonl,
            args.nsys_kernel_csv,
            args.nsys_projection_csv,
            args.minimum_coverage,
            args.maximum_coverage,
        )
    except AttributionError as exc:
        summary = {
            "schema_version": 2,
            "trace_jsonl": str(args.trace_jsonl),
            "nsys_kernel_csv": str(args.nsys_kernel_csv),
            "nsys_projection_csv": str(args.nsys_projection_csv),
            "errors": [str(exc)],
            "pass": False,
        }
    write_json(args.out, summary)
    prefix = PASS_PREFIX if summary["pass"] else REJECT_PREFIX
    print(f"{prefix} {args.out}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
