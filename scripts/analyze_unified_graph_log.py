#!/usr/bin/env python3
"""
Summarize Ferrum unified-graph diagnostic logs.

This is intentionally a log parser, not a benchmark runner. It helps keep CUDA
graph validation scoped: run the product smoke/bench once, then use this script
to answer whether graph capture/replay actually happened and whether fatal CUDA
errors appeared in the log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ \n]+)")
UNIFIED_PROF_RE = re.compile(
    r"\[unified-prof\]\s+iter#(?P<iter>\d+)\s+items=(?P<items>\d+)\s+"
    r"prefill=(?P<prefill>\d+)\s+decode=(?P<decode>\d+)\s+"
    r"total=(?P<total>\d+)us\s+model=(?P<model>\d+)us\s+"
    r"decode_post=(?P<decode_post>\d+)us\s+\|\s+sample=(?P<sample>\d+)\s+"
    r"sched=(?P<sched>\d+)\s+stream=(?P<stream>\d+)\s+stop=(?P<stop>\d+)\s+"
    r"complete=(?P<complete>\d+)"
)
ITER_PROF_RE = re.compile(
    r"\[iter-prof\]\s+iter#(?P<iter>\d+)\s+total=(?P<total>\d+)us\s+"
    r"sched=(?P<sched>\d+)us\s+process=(?P<process>\d+)us\s+"
    r"batch_size=(?P<batch_size>\d+)"
)
BATCHED_OP_RE = re.compile(r"\[batched-op-profile\]\s+(?P<body>.*)")
FATAL_PATTERNS = (
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUDA_ERROR_STREAM_CAPTURE",
    "CUDA_ERROR_LAUNCH_FAILED",
    "illegal memory access",
    "out of memory",
    "panicked at",
    "thread '",
)


@dataclass
class TimingSummary:
    count: int = 0
    total_us: int = 0
    model_us: int = 0
    decode_post_us: int = 0
    sample_us: int = 0
    sched_us: int = 0
    stream_us: int = 0
    stop_us: int = 0
    complete_us: int = 0

    def add(self, fields: dict[str, int]) -> None:
        self.count += 1
        self.total_us += fields.get("total", 0)
        self.model_us += fields.get("model", 0)
        self.decode_post_us += fields.get("decode_post", 0)
        self.sample_us += fields.get("sample", 0)
        self.sched_us += fields.get("sched", 0)
        self.stream_us += fields.get("stream", 0)
        self.stop_us += fields.get("stop", 0)
        self.complete_us += fields.get("complete", 0)

    def avg(self, field_name: str) -> float:
        return getattr(self, field_name) / self.count if self.count else 0.0


@dataclass
class GraphLogSummary:
    files: list[str] = field(default_factory=list)
    captures: int = 0
    replays: int = 0
    eagers: int = 0
    skips: int = 0
    stats_lines: int = 0
    replay_origins: Counter[str] = field(default_factory=Counter)
    scopes: Counter[str] = field(default_factory=Counter)
    skip_reasons: Counter[str] = field(default_factory=Counter)
    keys: set[str] = field(default_factory=set)
    fatal_errors: Counter[str] = field(default_factory=Counter)
    graph_errors: list[str] = field(default_factory=list)
    profile_all: TimingSummary = field(default_factory=TimingSummary)
    profile_prefill: TimingSummary = field(default_factory=TimingSummary)
    profile_decode: TimingSummary = field(default_factory=TimingSummary)
    iter_profile_count: int = 0
    iter_total_us: int = 0
    iter_sched_us: int = 0
    iter_process_us: int = 0
    batch_sizes: Counter[int] = field(default_factory=Counter)
    op_profile_rows: int = 0
    op_profile_by_m: Counter[int] = field(default_factory=Counter)

    def to_jsonable(self) -> dict:
        data = asdict(self)
        data["replay_origins"] = dict(self.replay_origins)
        data["scopes"] = dict(self.scopes)
        data["skip_reasons"] = dict(self.skip_reasons)
        data["fatal_errors"] = dict(self.fatal_errors)
        data["batch_sizes"] = dict(sorted(self.batch_sizes.items()))
        data["op_profile_by_m"] = dict(sorted(self.op_profile_by_m.items()))
        data["keys"] = sorted(self.keys)
        return data


def parse_fields(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in FIELD_RE.finditer(line)}


def parse_int_fields(match: re.Match[str]) -> dict[str, int]:
    return {key: int(value) for key, value in match.groupdict().items()}


def classify_profile(summary: GraphLogSummary, fields: dict[str, int]) -> None:
    summary.profile_all.add(fields)
    if fields["prefill"] > 0 and fields["decode"] == 0:
        summary.profile_prefill.add(fields)
    if fields["decode"] > 0 and fields["prefill"] == 0:
        summary.profile_decode.add(fields)


def note_graph_line(summary: GraphLogSummary, line: str) -> None:
    fields = parse_fields(line)
    scope = fields.get("scope")
    key = fields.get("key")
    if scope:
        summary.scopes[scope] += 1
    if key:
        summary.keys.add(key)

    if "[unified-graph-capture]" in line:
        summary.captures += 1
    elif "[unified-graph-replay]" in line:
        summary.replays += 1
        summary.replay_origins[fields.get("origin", "unknown")] += 1
    elif "[unified-graph-skip]" in line:
        summary.skips += 1
        summary.skip_reasons[fields.get("reason", "unknown")] += 1
    elif "[unified-graph-stats]" in line:
        summary.stats_lines += 1
        if "replays" in fields:
            summary.replays = max(summary.replays, int(fields["replays"]))
        if "eagers" in fields:
            summary.eagers = max(summary.eagers, int(fields["eagers"]))
    elif "[unified-graph]" in line and "err" in line:
        summary.graph_errors.append(line.strip())


def parse_lines(paths: Iterable[Path]) -> GraphLogSummary:
    summary = GraphLogSummary(files=[str(path) for path in paths])
    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                if "[unified-graph" in line:
                    note_graph_line(summary, line)

                for pattern in FATAL_PATTERNS:
                    if pattern in line:
                        summary.fatal_errors[pattern] += 1

                if match := UNIFIED_PROF_RE.search(line):
                    classify_profile(summary, parse_int_fields(match))
                    continue

                if match := ITER_PROF_RE.search(line):
                    fields = parse_int_fields(match)
                    summary.iter_profile_count += 1
                    summary.iter_total_us += fields["total"]
                    summary.iter_sched_us += fields["sched"]
                    summary.iter_process_us += fields["process"]
                    summary.batch_sizes[fields["batch_size"]] += 1
                    continue

                if match := BATCHED_OP_RE.search(line):
                    fields = parse_fields(match.group("body"))
                    summary.op_profile_rows += 1
                    if "m" in fields:
                        summary.op_profile_by_m[int(fields["m"])] += 1
    return summary


def fmt_avg(summary: TimingSummary, field_name: str) -> str:
    return f"{summary.avg(field_name):.1f}" if summary.count else "0.0"


def render_markdown(summary: GraphLogSummary) -> str:
    lines: list[str] = []
    lines.append("# Unified Graph Log Summary")
    lines.append("")
    lines.append(f"- files: {len(summary.files)}")
    lines.append(
        f"- graph events: captures={summary.captures}, replays={summary.replays}, "
        f"eagers={summary.eagers}, skips={summary.skips}, keys={len(summary.keys)}"
    )
    if summary.scopes:
        scopes = ", ".join(f"{key}={value}" for key, value in summary.scopes.items())
        lines.append(f"- scopes: {scopes}")
    if summary.replay_origins:
        origins = ", ".join(f"{key}={value}" for key, value in summary.replay_origins.items())
        lines.append(f"- replay origins: {origins}")
    if summary.skip_reasons:
        reasons = ", ".join(f"{key}={value}" for key, value in summary.skip_reasons.items())
        lines.append(f"- skip reasons: {reasons}")
    if summary.fatal_errors:
        fatals = ", ".join(f"{key}={value}" for key, value in summary.fatal_errors.items())
        lines.append(f"- fatal/error markers: {fatals}")
    else:
        lines.append("- fatal/error markers: none parsed")

    lines.append("")
    lines.append("## Unified Profile")
    lines.append("")
    lines.append("| segment | rows | avg total us | avg model us | avg decode_post us |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, timing in (
        ("all", summary.profile_all),
        ("prefill-only", summary.profile_prefill),
        ("decode-only", summary.profile_decode),
    ):
        lines.append(
            f"| {name} | {timing.count} | {fmt_avg(timing, 'total_us')} | "
            f"{fmt_avg(timing, 'model_us')} | {fmt_avg(timing, 'decode_post_us')} |"
        )

    if summary.iter_profile_count:
        avg_total = summary.iter_total_us / summary.iter_profile_count
        avg_sched = summary.iter_sched_us / summary.iter_profile_count
        avg_process = summary.iter_process_us / summary.iter_profile_count
        top_batches = ", ".join(
            f"{batch}x{count}" for batch, count in summary.batch_sizes.most_common(8)
        )
        lines.append("")
        lines.append(
            f"iter-prof rows={summary.iter_profile_count}, avg_total_us={avg_total:.1f}, "
            f"avg_sched_us={avg_sched:.1f}, avg_process_us={avg_process:.1f}"
        )
        lines.append(f"batch sizes: {top_batches}")

    if summary.op_profile_rows:
        top_m = ", ".join(
            f"m={m}:{count}" for m, count in summary.op_profile_by_m.most_common(8)
        )
        lines.append("")
        lines.append(f"batched-op-profile rows={summary.op_profile_rows}; {top_m}")

    if summary.graph_errors:
        lines.append("")
        lines.append("## Graph Errors")
        lines.append("")
        for line in summary.graph_errors[:12]:
            lines.append(f"- `{line}`")

    lines.append("")
    if summary.fatal_errors:
        lines.append("DIAGNOSTIC VERDICT: graph log contains fatal/error markers.")
    elif summary.replays == 0:
        lines.append("DIAGNOSTIC VERDICT: no unified graph replay was parsed.")
    else:
        lines.append("DIAGNOSTIC VERDICT: unified graph replay was parsed without fatal markers.")
    return "\n".join(lines)


def run_self_test() -> None:
    sample = """\
[unified-graph-capture] count=1 scope=layers_only key=42 attention_key=7 m_total=16 num_seqs=16 max_kv_len=144
[unified-graph-replay] origin=post_capture count=1 scope=layers_only key=42 attention_key=7 m_total=16 num_seqs=16 max_kv_len=144
[unified-graph-stats] scope=layers_only key=42 replays=4 eagers=2 keys_seen=[42]
[unified-prof] iter#1 items=16 prefill=0 decode=16 total=28000us model=27600us decode_post=300us | sample=20 sched=0 stream=280 stop=0 complete=0 (us)
[iter-prof] iter#1 total=28200us sched=10us process=28190us batch_size=16
[batched-op-profile] m=16 total=27000us tail_mlp=14000us(124)
"""
    tmp = Path("/tmp/ferrum_unified_graph_log_selftest.log")
    tmp.write_text(sample)
    summary = parse_lines([tmp])
    assert summary.captures == 1
    assert summary.replays == 4
    assert summary.eagers == 2
    assert summary.profile_decode.count == 1
    assert summary.batch_sizes[16] == 1
    assert summary.op_profile_by_m[16] == 1
    tmp.unlink(missing_ok=True)
    print("SELFTEST OK: analyze_unified_graph_log")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="stderr/server log files to parse")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    parser.add_argument("--self-test", action="store_true", help="run parser self-test")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0
    if not args.logs:
        parser.error("at least one log path is required")
    missing = [path for path in args.logs if not path.exists()]
    if missing:
        parser.error(f"missing log path(s): {', '.join(str(path) for path in missing)}")

    summary = parse_lines(args.logs)
    if args.json:
        print(json.dumps(summary.to_jsonable(), indent=2, sort_keys=True))
    else:
        print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
