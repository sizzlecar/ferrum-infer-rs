#!/usr/bin/env python3
"""
Join two ferrum bench-serve outputs and emit a markdown ratio table.

Inputs are EITHER:
  - two single-cell `--output json` files (one BenchReport each)
  - two multi-cell `--output json` files (BenchReport[] each)
  - two `--output jsonl` files (one BenchReport per line)

The script joins by (model, backend, scenario, concurrency_or_rate). Cells
with mismatched env_hash are flagged but still compared (with a `⚠` mark).

A row's ratio is significant iff the CIs of A and B don't overlap. That's
the condition we use to call a perf change "real" — see PLAYBOOK § 4.A.

Usage:
  scripts/compare_bench.py a.json b.json
  scripts/compare_bench.py --label-a HEAD~1 --label-b HEAD a.json b.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_reports(path: Path):
    """Accept single-object, array, or JSONL."""
    text = path.read_text().strip()
    if not text:
        return []
    # JSONL: many lines, each a JSON object.
    if "\n" in text and text.lstrip().startswith("{"):
        first = text.lstrip()
        # Heuristic: if the whole file parses as one JSON, it's not JSONL.
        try:
            obj = json.loads(text)
            return [obj] if isinstance(obj, dict) else obj
        except json.JSONDecodeError:
            pass
        # Multi-line JSONL.
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    return [obj] if isinstance(obj, dict) else obj


def cell_key(r):
    """Stable identity for joining."""
    return (
        r.get("model"),
        r.get("backend"),
        r.get("scenario"),
        r.get("concurrency"),
        r.get("request_rate"),
    )


def cell_label(r):
    sc = r.get("scenario", "?")
    c = r.get("concurrency")
    rate = r.get("request_rate")
    if sc == "closed_loop" and c is not None:
        return f"c={c}"
    if sc == "open_loop" and rate is not None:
        return f"rate={rate}"
    return sc


def stat(r, path):
    """Walk a dotted path; return None if any step is missing.

    Examples:  stat(r, "ttft_ms.p50") → {mean, stddev?, ci95_hw?}
               stat(r, "output_throughput_tps") → same
    """
    cur = r
    for k in path.split("."):
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur


def significant(a, b):
    """True if CIs of a and b do not overlap."""
    if a is None or b is None:
        return False
    a_lo, a_hi = a["mean"] - a.get("ci95_hw", 0), a["mean"] + a.get("ci95_hw", 0)
    b_lo, b_hi = b["mean"] - b.get("ci95_hw", 0), b["mean"] + b.get("ci95_hw", 0)
    return a_hi < b_lo or b_hi < a_lo


def fmt_stat(s):
    if s is None:
        return "—"
    if s.get("ci95_hw"):
        return f"{s['mean']:.2f} ± {s['ci95_hw']:.2f}"
    return f"{s['mean']:.2f}"


def fmt_ratio(a, b):
    if a is None or b is None or a["mean"] == 0:
        return "—"
    r = b["mean"] / a["mean"]
    mark = " ✓" if significant(a, b) else " "
    arrow = ""
    if r < 0.95:
        arrow = " ↓"
    elif r > 1.05:
        arrow = " ↑"
    return f"{r:.3f}{arrow}{mark}"


def render(reports_a, reports_b, label_a, label_b):
    by_a = {cell_key(r): r for r in reports_a}
    by_b = {cell_key(r): r for r in reports_b}
    keys = sorted(set(by_a) | set(by_b), key=lambda t: (t[0] or "", t[2] or "", t[3] or 0))

    out = []
    out.append(f"# Bench compare — {label_a} → {label_b}\n")
    out.append("Significance marker (✓ = CIs don't overlap, real change at 95%); ↑/↓ = ≥ 5% drift.\n")
    out.append("")
    out.append("| cell | metric | " + label_a + " | " + label_b + " | ratio (B/A) |")
    out.append("|---|---|---|---|---|")

    metrics = [
        ("TTFT p50", "ttft_ms.p50"),
        ("TTFT p99", "ttft_ms.p99"),
        ("TPOT p50", "tpot_ms.p50"),
        ("TPOT p99", "tpot_ms.p99"),
        ("output_throughput", "output_throughput_tps"),
        ("goodput", "goodput_rps"),
    ]

    env_warnings = []
    for k in keys:
        a, b = by_a.get(k), by_b.get(k)
        label = cell_label(a or b)
        if a and b:
            eh_a, eh_b = a.get("env_hash"), b.get("env_hash")
            if eh_a != eh_b:
                env_warnings.append(
                    f"  cell {label}: env_hash differs ({eh_a} vs {eh_b})"
                )
        for mname, mpath in metrics:
            sa = stat(a, mpath) if a else None
            sb = stat(b, mpath) if b else None
            if sa is None and sb is None:
                continue
            out.append(
                f"| {label} | {mname} | {fmt_stat(sa)} | {fmt_stat(sb)} | {fmt_ratio(sa, sb)} |"
            )

    if env_warnings:
        out.append("\n## ⚠ env_hash mismatches\n")
        out.extend(env_warnings)
        out.append("\nApples-to-apples comparison invalid for the above cells.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    reports_a = load_reports(args.a)
    reports_b = load_reports(args.b)
    if not reports_a:
        sys.exit(f"{args.a}: no bench records found")
    if not reports_b:
        sys.exit(f"{args.b}: no bench records found")

    label_a = args.label_a or reports_a[0].get("env", {}).get("commit_sha", str(args.a))
    label_b = args.label_b or reports_b[0].get("env", {}).get("commit_sha", str(args.b))
    print(render(reports_a, reports_b, label_a, label_b))


if __name__ == "__main__":
    main()
