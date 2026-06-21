#!/usr/bin/env python3
"""Summarize Ferrum scheduler trace JSONL artifacts.

This is a diagnostic artifact helper, not a release gate. It turns
`--scheduler-trace-jsonl` output into stable JSON so W3 c32 runs can compare
effective decode cohorts, prefill chunks, and slow scheduler iterations without
hand-written jq per artifact.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def percentile(sorted_values: list[int], pct: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * pct)))
    return sorted_values[index]


def distribution(values: list[int]) -> dict[str, Any]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": values[-1],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return events


def phase_mix_key(plan: dict[str, Any]) -> tuple[int, int]:
    return (int(plan.get("decode_items") or 0), int(plan.get("prefill_items") or 0))


def analyze_events(events: list[dict[str, Any]], source: str) -> dict[str, Any]:
    some_events: list[dict[str, Any]] = []
    none_events: list[dict[str, Any]] = []
    error_events: list[dict[str, Any]] = []
    phase_process_us: dict[tuple[int, int], list[int]] = defaultdict(list)
    phase_counts: Counter[tuple[int, int]] = Counter()
    slow_events: list[dict[str, Any]] = []

    batch_sizes: list[int] = []
    scheduled_tokens: list[int] = []
    decode_items: list[int] = []
    prefill_items: list[int] = []
    prefill_chunk_tokens: list[int] = []
    prefill_remaining_before: list[int] = []
    decode_generated_tokens: list[int] = []
    final_prefill_chunks = 0
    nonfinal_prefill_chunks = 0
    request_detail_records = 0
    max_none_streak = 0

    for event in events:
        if event.get("event") != "scheduler_iteration":
            continue
        result = str(event.get("result") or "")
        if result == "none":
            none_events.append(event)
            max_none_streak = max(max_none_streak, int(event.get("none_streak") or 0))
            continue
        if not result.startswith("some"):
            continue
        some_events.append(event)
        if result == "some_error":
            error_events.append(event)

        plan = event.get("plan") or {}
        key = phase_mix_key(plan)
        phase_counts[key] += 1
        timing = event.get("timing_us") or {}
        process_us = timing.get("process")
        if isinstance(process_us, int):
            phase_process_us[key].append(process_us)
        batch_sizes.append(int(plan.get("batch_size") or 0))
        scheduled_tokens.append(int(plan.get("scheduled_tokens_total") or 0))
        decode_items.append(int(plan.get("decode_items") or 0))
        prefill_items.append(int(plan.get("prefill_items") or 0))

        slow_events.append(
            {
                "iteration": event.get("iteration"),
                "result": result,
                "process_us": process_us,
                "schedule_us": timing.get("schedule"),
                "decode_items": key[0],
                "prefill_items": key[1],
                "decode_tokens": plan.get("decode_tokens"),
                "prefill_tokens": plan.get("prefill_tokens"),
                "scheduled_tokens_total": plan.get("scheduled_tokens_total"),
            }
        )

        for request in plan.get("requests") or []:
            request_detail_records += 1
            phase = request.get("phase")
            if phase == "Prefilling":
                if isinstance(request.get("scheduled_tokens"), int):
                    prefill_chunk_tokens.append(request["scheduled_tokens"])
                if isinstance(request.get("prefill_tokens_remaining_before"), int):
                    prefill_remaining_before.append(request["prefill_tokens_remaining_before"])
                if request.get("is_final_prefill_chunk") is True:
                    final_prefill_chunks += 1
                elif request.get("is_final_prefill_chunk") is False:
                    nonfinal_prefill_chunks += 1
            elif phase == "Decoding" and isinstance(request.get("generated_tokens"), int):
                decode_generated_tokens.append(request["generated_tokens"])

    phase_mix = []
    for key, count in sorted(phase_counts.items(), key=lambda item: (-item[1], item[0])):
        process_values = phase_process_us.get(key, [])
        phase_mix.append(
            {
                "decode_items": key[0],
                "prefill_items": key[1],
                "count": count,
                "process_us": distribution(process_values),
            }
        )

    slow_events = [
        event
        for event in sorted(
            slow_events,
            key=lambda item: item["process_us"] if isinstance(item["process_us"], int) else -1,
            reverse=True,
        )
        if isinstance(event["process_us"], int)
    ][:10]

    return {
        "schema_version": 1,
        "source": source,
        "events": {
            "total": len(events),
            "scheduler_iteration": len(some_events) + len(none_events),
            "some": len(some_events),
            "none": len(none_events),
            "some_error": len(error_events),
            "max_none_streak": max_none_streak,
        },
        "plan_distributions": {
            "batch_size": distribution(batch_sizes),
            "scheduled_tokens_total": distribution(scheduled_tokens),
            "decode_items": distribution(decode_items),
            "prefill_items": distribution(prefill_items),
        },
        "phase_mix": phase_mix,
        "request_detail": {
            "present": request_detail_records > 0,
            "records": request_detail_records,
            "prefill_chunk_tokens": distribution(prefill_chunk_tokens),
            "prefill_remaining_before": distribution(prefill_remaining_before),
            "final_prefill_chunks": final_prefill_chunks,
            "nonfinal_prefill_chunks": nonfinal_prefill_chunks,
            "decode_generated_tokens": distribution(decode_generated_tokens),
        },
        "slowest_process_events": slow_events,
    }


def analyze_file(path: Path) -> dict[str, Any]:
    return analyze_events(load_jsonl(path), str(path))


def write_summary(summary: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_self_test() -> None:
    lines = [
        {
            "event": "scheduler_iteration",
            "iteration": 1,
            "result": "some_ok",
            "plan": {
                "batch_size": 2,
                "decode_items": 1,
                "prefill_items": 1,
                "decode_tokens": 1,
                "prefill_tokens": 64,
                "scheduled_tokens_total": 65,
                "requests": [
                    {
                        "request_id": "decode-a",
                        "phase": "Decoding",
                        "scheduled_tokens": 1,
                        "generated_tokens": 7,
                    },
                    {
                        "request_id": "prefill-b",
                        "phase": "Prefilling",
                        "scheduled_tokens": 64,
                        "prefill_tokens_remaining_before": 128,
                        "is_final_prefill_chunk": False,
                    },
                ],
            },
            "timing_us": {"schedule": 30, "process": 2500},
        },
        {
            "event": "scheduler_iteration",
            "iteration": 2,
            "result": "none",
            "none_streak": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        trace = Path(td) / "trace.jsonl"
        trace.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
        summary = analyze_file(trace)
    assert summary["events"]["some"] == 1
    assert summary["events"]["none"] == 1
    assert summary["request_detail"]["present"] is True
    assert summary["request_detail"]["nonfinal_prefill_chunks"] == 1
    assert summary["phase_mix"][0]["decode_items"] == 1
    assert summary["phase_mix"][0]["prefill_items"] == 1
    print("SCHEDULER TRACE ANALYSIS SELFTEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, help="Path to write summary JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.trace_jsonl is None or args.out is None:
        parser.error("trace_jsonl and --out are required unless --self-test is used")

    summary = analyze_file(args.trace_jsonl)
    write_summary(summary, args.out)
    print(f"SCHEDULER TRACE ANALYSIS PASS: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
