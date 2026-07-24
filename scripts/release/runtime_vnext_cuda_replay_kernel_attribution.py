#!/usr/bin/env python3
"""Join Ferrum reusable-execution spans to Nsight kernel activity by fingerprint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import tempfile
from bisect import bisect_left
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


def read_replay_program_layouts(path: Path) -> dict[str, dict[str, Any]]:
    native_by_occurrence: dict[tuple[str, int], dict[int, dict[str, Any]]] = (
        defaultdict(dict)
    )
    replay_spans: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AttributionError(
                    f"{path}:{line_number}: malformed JSON: {exc}"
                ) from exc
            phase = event.get("phase")
            if phase not in {
                "vnext.device_native_work",
                "vnext.device_execution_span",
            }:
                continue
            attributes = event.get("attributes")
            shape = event.get("shape")
            require(
                isinstance(attributes, dict) and isinstance(shape, dict),
                f"{path}:{line_number}: replay ownership row lacks shape or attributes",
            )
            if attributes.get("execution_path") != "replayed":
                continue
            submission = attributes.get("physical_submission_fingerprint")
            timestamp = event.get("ts_unix_nanos")
            require(
                isinstance(submission, str)
                and re.fullmatch(r"[0-9a-f]{64}", submission) is not None
                and isinstance(timestamp, int)
                and not isinstance(timestamp, bool),
                f"{path}:{line_number}: replay ownership row lacks occurrence identity",
            )
            occurrence = (submission, timestamp)
            if phase == "vnext.device_native_work":
                command_index = shape.get("command_index")
                require(
                    isinstance(command_index, int)
                    and not isinstance(command_index, bool)
                    and command_index >= 0,
                    f"{path}:{line_number}: replay command lacks command_index",
                )
                require(
                    command_index not in native_by_occurrence[occurrence],
                    f"{path}:{line_number}: duplicate replay command {command_index}",
                )
                native_by_occurrence[occurrence][command_index] = {
                    "command_index": command_index,
                    "native_op_id": attributes.get("native_op_id"),
                    "node_id": attributes.get("node_id"),
                    "operation_id": attributes.get("operation_id"),
                    "provider_id": attributes.get("provider_id"),
                    "graph_node_count": shape.get("reusable_graph_node_count"),
                }
                continue

            require(
                attributes.get("device_timing_span_kind") == "reusable_executable",
                f"{path}:{line_number}: replay ownership span is not reusable execution",
            )
            fingerprint = attributes.get("reusable_executable_fingerprint")
            start = shape.get("start_command_index")
            end = shape.get("end_command_index")
            require(
                isinstance(fingerprint, str)
                and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None
                and isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end,
                f"{path}:{line_number}: replay ownership span has invalid identity or range",
            )
            replay_spans.append(
                {
                    "occurrence": occurrence,
                    "fingerprint": fingerprint,
                    "start_command_index": start,
                    "end_command_index": end,
                }
            )

    require(replay_spans, f"{path}: no replay spans for graph ownership")
    layouts: dict[str, dict[str, Any]] = {}
    for span in replay_spans:
        occurrence = span["occurrence"]
        start = span["start_command_index"]
        end = span["end_command_index"]
        command_rows = native_by_occurrence.get(occurrence, {})
        require(
            all(index in command_rows for index in range(start, end)),
            f"{path}: replay span [{start}, {end}) lacks native command rows",
        )
        graph_start = 0
        commands = []
        for ordinal, command_index in enumerate(range(start, end)):
            row = command_rows[command_index]
            graph_node_count = row["graph_node_count"]
            require(
                isinstance(graph_node_count, int)
                and not isinstance(graph_node_count, bool)
                and graph_node_count >= 0,
                f"{path}: replay command {command_index} lacks actual graph-node count",
            )
            native_op_id = row["native_op_id"]
            require(
                isinstance(native_op_id, str) and native_op_id,
                f"{path}: replay command {command_index} lacks native_op_id",
            )
            commands.append(
                {
                    "ordinal": ordinal,
                    "command_index": command_index,
                    "native_op_id": native_op_id,
                    "node_id": row["node_id"],
                    "operation_id": row["operation_id"],
                    "provider_id": row["provider_id"],
                    "graph_node_start": graph_start,
                    "graph_node_end": graph_start + graph_node_count,
                    "graph_node_count": graph_node_count,
                }
            )
            graph_start += graph_node_count
        require(
            graph_start > 0,
            f"{path}: replay fingerprint {span['fingerprint']} has no graph nodes",
        )
        normalized = [
            {
                key: value
                for key, value in command.items()
                if key != "command_index"
            }
            for command in commands
        ]
        fingerprint = span["fingerprint"]
        current = layouts.get(fingerprint)
        if current is None:
            layouts[fingerprint] = {
                "fingerprint": fingerprint,
                "span_count": 1,
                "graph_node_count": graph_start,
                "commands": commands,
                "_normalized": normalized,
            }
            continue
        require(
            current["_normalized"] == normalized,
            f"{path}: replay fingerprint {fingerprint} has inconsistent command layout",
        )
        current["span_count"] += 1

    for layout in layouts.values():
        layout.pop("_normalized")
    return layouts


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
) -> tuple[
    dict[str, int],
    dict[str, int],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    required_headers = {
        "name",
        "projected start ns",
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
    start_index = header.index("projected start ns")
    duration_index = header.index("projected duration ns")
    level_index = header.index("lvl")
    children_index = header.index("numchild")
    projected_by_fingerprint: dict[str, int] = defaultdict(int)
    counts_by_fingerprint: dict[str, int] = defaultdict(int)
    invalid_ranges: list[dict[str, Any]] = []
    projected_ranges: list[dict[str, Any]] = []
    for row in rows:
        name = row[name_index]
        match = FINGERPRINT_RE.search(name)
        if match is None:
            continue
        fingerprint = match.group(1)
        try:
            projected_start_ns = int(row[start_index])
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
        projected_ranges.append(
            {
                "fingerprint": fingerprint,
                "projected_start_ns": projected_start_ns,
                "projected_end_ns": projected_start_ns + projected_ns,
                "projected_ns": projected_ns,
                "level": level,
                "child_count": child_count,
            }
        )
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
        projected_ranges,
    )


def sqlite_table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def read_nsight_graph_kernels(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"{path}: Nsight SQLite export does not exist")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        required = {
            "CUDA_GRAPH_NODE_EVENTS": {
                "graphNodeId",
                "originalGraphNodeId",
            },
            "CUPTI_ACTIVITY_KIND_KERNEL": {
                "start",
                "end",
                "graphNodeId",
                "demangledName",
            },
            "StringIds": {"id", "value"},
        }
        for table, columns in required.items():
            missing = columns - sqlite_table_columns(connection, table)
            require(
                not missing,
                f"{path}: {table} lacks columns {sorted(missing)}",
            )

        graph_origins = {
            int(graph_node_id): (
                int(original_graph_node_id)
                if original_graph_node_id is not None
                else int(graph_node_id)
            )
            for graph_node_id, original_graph_node_id in connection.execute(
                """
                SELECT graphNodeId, MAX(originalGraphNodeId)
                FROM CUDA_GRAPH_NODE_EVENTS
                GROUP BY graphNodeId
                """
            )
        }
        rows = []
        for start, end, graph_node_id, kernel_name in connection.execute(
            """
            SELECT kernel.start,
                   kernel.end,
                   kernel.graphNodeId,
                   strings.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            LEFT JOIN StringIds AS strings
              ON strings.id = kernel.demangledName
            WHERE kernel.graphNodeId IS NOT NULL
            ORDER BY kernel.start, kernel.end
            """
        ):
            require(
                isinstance(start, int)
                and isinstance(end, int)
                and end > start
                and isinstance(graph_node_id, int),
                f"{path}: invalid graph kernel row",
            )
            original_graph_node_id = graph_origins.get(
                graph_node_id, graph_node_id
            )
            rows.append(
                {
                    "start_ns": start,
                    "end_ns": end,
                    "elapsed_ns": end - start,
                    "graph_node_id": graph_node_id,
                    "graph_group": graph_node_id >> 32,
                    "original_graph_node_id": original_graph_node_id,
                    "original_graph_group": original_graph_node_id >> 32,
                    "original_graph_node_ordinal": (
                        original_graph_node_id & 0xFFFFFFFF
                    ),
                    "kernel_name": kernel_name or "<unnamed>",
                }
            )
        require(rows, f"{path}: no graph-backed CUDA kernel activity")
        return rows
    finally:
        connection.close()


def attribute_graph_kernels_to_owners(
    trace_path: Path,
    nsys_sqlite_path: Path,
    projected_ranges: list[dict[str, Any]],
    expected_kernel_work_ns: int,
) -> dict[str, Any]:
    layouts = read_replay_program_layouts(trace_path)
    kernels = read_nsight_graph_kernels(nsys_sqlite_path)
    kernel_starts = [row["start_ns"] for row in kernels]
    used_kernel_rows: set[int] = set()
    owner_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"elapsed_ns": 0, "instances": 0}
    )
    fingerprint_owner_totals: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"elapsed_ns": 0, "instances": 0}
    )
    owner_kernels: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"elapsed_ns": 0, "instances": 0}
    )
    command_totals: dict[tuple[str, int], dict[str, Any]] = {}
    mapped_range_count = 0

    for projected_range in sorted(
        projected_ranges, key=lambda row: row["projected_start_ns"]
    ):
        fingerprint = projected_range["fingerprint"]
        layout = layouts.get(fingerprint)
        if layout is None:
            continue
        range_start = projected_range["projected_start_ns"]
        range_end = projected_range["projected_end_ns"]
        index = bisect_left(kernel_starts, range_start)
        selected: list[tuple[int, dict[str, Any]]] = []
        while index < len(kernels) and kernels[index]["start_ns"] < range_end:
            kernel = kernels[index]
            require(
                kernel["end_ns"] <= range_end,
                f"{nsys_sqlite_path}: kernel crosses projected replay boundary",
            )
            selected.append((index, kernel))
            index += 1
        require(
            selected,
            f"{nsys_sqlite_path}: replay range {fingerprint} has no graph kernels",
        )
        graph_groups = {kernel["graph_group"] for _, kernel in selected}
        original_groups = {
            kernel["original_graph_group"] for _, kernel in selected
        }
        require(
            len(graph_groups) == 1 and len(original_groups) == 1,
            f"{nsys_sqlite_path}: replay range {fingerprint} mixes CUDA graphs",
        )

        for kernel_index, kernel in selected:
            require(
                kernel_index not in used_kernel_rows,
                f"{nsys_sqlite_path}: kernel belongs to overlapping replay ranges",
            )
            used_kernel_rows.add(kernel_index)
            ordinal = kernel["original_graph_node_ordinal"]
            command = next(
                (
                    candidate
                    for candidate in layout["commands"]
                    if candidate["graph_node_start"]
                    <= ordinal
                    < candidate["graph_node_end"]
                ),
                None,
            )
            require(
                command is not None,
                f"{nsys_sqlite_path}: graph node {ordinal} exceeds "
                f"{fingerprint} command layout",
            )
            native_op_id = command["native_op_id"]
            elapsed_ns = kernel["elapsed_ns"]
            owner_totals[native_op_id]["elapsed_ns"] += elapsed_ns
            owner_totals[native_op_id]["instances"] += 1
            fingerprint_owner = fingerprint_owner_totals[
                (fingerprint, native_op_id)
            ]
            fingerprint_owner["elapsed_ns"] += elapsed_ns
            fingerprint_owner["instances"] += 1
            owner_kernel = owner_kernels[(native_op_id, kernel["kernel_name"])]
            owner_kernel["elapsed_ns"] += elapsed_ns
            owner_kernel["instances"] += 1

            command_key = (fingerprint, command["ordinal"])
            command_total = command_totals.setdefault(
                command_key,
                {
                    **command,
                    "fingerprint": fingerprint,
                    "elapsed_ns": 0,
                    "kernel_instances": 0,
                },
            )
            command_total["elapsed_ns"] += elapsed_ns
            command_total["kernel_instances"] += 1
        mapped_range_count += 1

    mapped_kernel_work_ns = sum(
        values["elapsed_ns"] for values in owner_totals.values()
    )
    require(mapped_range_count > 0, "no projected ranges map to trace programs")
    require(
        mapped_kernel_work_ns == expected_kernel_work_ns,
        "raw graph kernel work differs from fingerprint kernel summary: "
        f"{mapped_kernel_work_ns} != {expected_kernel_work_ns}",
    )

    per_owner = [
        {
            "native_op_id": native_op_id,
            **values,
            "kernel_work_ratio": values["elapsed_ns"] / mapped_kernel_work_ns,
        }
        for native_op_id, values in owner_totals.items()
    ]
    per_owner.sort(key=lambda row: (-row["elapsed_ns"], row["native_op_id"]))
    per_owner_kernel = [
        {
            "native_op_id": native_op_id,
            "kernel_name": kernel_name,
            **values,
            "kernel_work_ratio": values["elapsed_ns"] / mapped_kernel_work_ns,
        }
        for (native_op_id, kernel_name), values in owner_kernels.items()
    ]
    per_owner_kernel.sort(
        key=lambda row: (
            -row["elapsed_ns"],
            row["native_op_id"],
            row["kernel_name"],
        )
    )
    fingerprint_work_ns: dict[str, int] = defaultdict(int)
    for (fingerprint, _), values in fingerprint_owner_totals.items():
        fingerprint_work_ns[fingerprint] += values["elapsed_ns"]
    per_fingerprint_owner = [
        {
            "fingerprint": fingerprint,
            "native_op_id": native_op_id,
            **values,
            "fingerprint_kernel_work_ratio": (
                values["elapsed_ns"] / fingerprint_work_ns[fingerprint]
            ),
        }
        for (fingerprint, native_op_id), values in fingerprint_owner_totals.items()
    ]
    per_fingerprint_owner.sort(
        key=lambda row: (
            row["fingerprint"],
            -row["elapsed_ns"],
            row["native_op_id"],
        )
    )
    per_command = list(command_totals.values())
    per_command.sort(
        key=lambda row: (-row["elapsed_ns"], row["fingerprint"], row["ordinal"])
    )
    return {
        "source": "nsight_sqlite_graph_nodes",
        "nsys_sqlite": str(nsys_sqlite_path),
        "mapped_projected_range_count": mapped_range_count,
        "mapped_kernel_instances": len(used_kernel_rows),
        "mapped_kernel_work_ns": mapped_kernel_work_ns,
        "program_layouts": [
            layouts[fingerprint] for fingerprint in sorted(layouts)
        ],
        "per_owner": per_owner,
        "per_fingerprint_owner": per_fingerprint_owner,
        "per_owner_kernel": per_owner_kernel,
        "per_command": per_command,
        "dominant_owner": per_owner[0],
    }


def analyze(
    trace_path: Path,
    nsys_kernel_path: Path,
    nsys_projection_path: Path,
    minimum_coverage: float,
    maximum_coverage: float,
    nsys_sqlite_path: Path | None = None,
) -> dict[str, Any]:
    require(
        0.0 < minimum_coverage <= 1.0,
        "minimum coverage must be within (0, 1]",
    )
    require(maximum_coverage >= 1.0, "maximum coverage must be at least 1")
    internal_elapsed, span_counts = read_replay_spans(trace_path)
    nsys_kernel_elapsed, kernels = parse_nsight_kernel_summary(nsys_kernel_path)
    (
        nsys_projected_elapsed,
        projection_counts,
        invalid_ranges,
        projected_ranges,
    ) = (
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

    owner_attribution = None
    owner_attribution_error = None
    if nsys_sqlite_path is not None:
        try:
            owner_attribution = attribute_graph_kernels_to_owners(
                trace_path,
                nsys_sqlite_path,
                projected_ranges,
                kernel_work_total,
            )
        except AttributionError as exc:
            owner_attribution_error = str(exc)

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
    if owner_attribution_error is not None:
        errors.append(f"graph-node owner attribution failed: {owner_attribution_error}")

    return {
        "schema_version": 3 if nsys_sqlite_path is not None else 2,
        "trace_jsonl": str(trace_path),
        "nsys_kernel_csv": str(nsys_kernel_path),
        "nsys_projection_csv": str(nsys_projection_path),
        "nsys_sqlite": str(nsys_sqlite_path) if nsys_sqlite_path else None,
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
        "owner_attribution": owner_attribution,
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
    submission = "b" * 64
    with tempfile.TemporaryDirectory(prefix="ferrum-replay-kernel-attribution-") as tmp:
        root = Path(tmp)
        trace = root / "trace.jsonl"
        trace.write_text(
            "\n".join(
                json.dumps(event)
                for event in [
                    {
                        "phase": "vnext.device_native_work",
                        "ts_unix_nanos": 1,
                        "shape": {
                            "command_index": 0,
                            "reusable_graph_node_count": 2,
                        },
                        "attributes": {
                            "execution_path": "replayed",
                            "physical_submission_fingerprint": submission,
                            "native_op_id": "owner.first",
                            "node_id": "node.first",
                        },
                    },
                    {
                        "phase": "vnext.device_native_work",
                        "ts_unix_nanos": 1,
                        "shape": {
                            "command_index": 1,
                            "reusable_graph_node_count": 1,
                        },
                        "attributes": {
                            "execution_path": "replayed",
                            "physical_submission_fingerprint": submission,
                            "native_op_id": "owner.second",
                            "node_id": "node.second",
                        },
                    },
                    {
                        "phase": "vnext.device_execution_span",
                        "ts_unix_nanos": 1,
                        "shape": {
                            "device_elapsed_ns": 1_000,
                            "start_command_index": 0,
                            "end_command_index": 2,
                        },
                        "attributes": {
                            "execution_path": "replayed",
                            "device_timing_span_kind": "reusable_executable",
                            "device_timing_status": "measured",
                            "physical_submission_fingerprint": submission,
                            "reusable_executable_fingerprint": fingerprint,
                        },
                    },
                ]
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

        sqlite_path = root / "nsys.sqlite"
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.executescript(
                """
                CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE CUDA_GRAPH_NODE_EVENTS(
                    graphNodeId INTEGER NOT NULL,
                    originalGraphNodeId INTEGER
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    graphNodeId INTEGER,
                    demangledName INTEGER NOT NULL
                );
                INSERT INTO StringIds VALUES(1, 'marlin'), (2, 'attention');
                INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES
                    (8589934592, 4294967296),
                    (8589934594, 4294967298);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                    (10, 710, 8589934592, 1),
                    (710, 980, 8589934594, 2);
                """
            )
            connection.commit()
        finally:
            connection.close()
        owner_summary = analyze(
            trace,
            nsys,
            projection,
            0.95,
            1.10,
            sqlite_path,
        )
        require(
            owner_summary["pass"],
            f"self-test graph owner fixture rejected: {owner_summary}",
        )
        owner = owner_summary["owner_attribution"]
        require(
            owner["mapped_kernel_work_ns"] == 970
            and owner["dominant_owner"]["native_op_id"] == "owner.first"
            and owner["dominant_owner"]["elapsed_ns"] == 700,
            f"self-test graph owner attribution mismatch: {owner}",
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
    parser.add_argument("--nsys-sqlite", type=Path)
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
            or args.nsys_sqlite
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
            args.nsys_sqlite,
        )
    except AttributionError as exc:
        summary = {
            "schema_version": 3 if args.nsys_sqlite else 2,
            "trace_jsonl": str(args.trace_jsonl),
            "nsys_kernel_csv": str(args.nsys_kernel_csv),
            "nsys_projection_csv": str(args.nsys_projection_csv),
            "nsys_sqlite": str(args.nsys_sqlite) if args.nsys_sqlite else None,
            "errors": [str(exc)],
            "pass": False,
        }
    write_json(args.out, summary)
    prefix = PASS_PREFIX if summary["pass"] else REJECT_PREFIX
    print(f"{prefix} {args.out}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
