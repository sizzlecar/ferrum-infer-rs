#!/usr/bin/env python3
"""
Visualize a ferrum chrome trace as a stacked-bar layerwise breakdown.

Mirrors vLLM's `tools/profiler/visualize_layerwise_profile.py` — same
"group events by op category, stack the bars" pattern, applied to the
output of `ferrum_bench_core::trace::TraceWriter` (PLAYBOOK § Phase 1.5).

Usage:
  scripts/visualize_layerwise.py trace.json [-o decode.png]
  scripts/visualize_layerwise.py trace.json --separate prefill,decode

The trace is grouped by `cat` (category field — set by each
`FERRUM_*_PROF` probe to one of `attention`, `gemm`, `quant`, `norm`,
`comm`, `sampling`, `routing`, etc.). Bars are stacked per layer (`tid`)
and summed across all events of the same `(layer, cat)`.

Output: matplotlib PNG. Requires `matplotlib` + `numpy` only.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


CATEGORY_ORDER = [
    "attention",
    "gemm",
    "quant",
    "moe",
    "routing",
    "norm",
    "act",
    "comm",
    "sampling",
    "scheduling",
    "other",
]

CATEGORY_COLORS = {
    "attention": "#2E86AB",
    "gemm": "#A23B72",
    "quant": "#F18F01",
    "moe": "#C73E1D",
    "routing": "#FF8C42",
    "norm": "#7FB069",
    "act": "#4F6D7A",
    "comm": "#9CA3AF",
    "sampling": "#B5179E",
    "scheduling": "#6B7280",
    "other": "#374151",
}


def normalize_cat(c):
    """Bucket free-form cat strings into the known set."""
    c = (c or "").lower()
    for known in CATEGORY_ORDER:
        if known in c:
            return known
    return "other"


def load_trace(path):
    """Accept either {traceEvents: [...]} or a bare array."""
    text = Path(path).read_text()
    obj = json.loads(text)
    if isinstance(obj, dict):
        return obj.get("traceEvents", [])
    return obj


def aggregate(events):
    """Sum duration per (tid, category) — tid identifies the layer."""
    grid = defaultdict(lambda: defaultdict(float))  # grid[tid][cat] = dur_us
    for e in events:
        if e.get("ph") != "X":
            continue
        tid = int(e.get("tid", 0))
        cat = normalize_cat(e.get("cat", ""))
        grid[tid][cat] += float(e.get("dur", 0))
    return grid


def render(grid, out_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        sys.exit("matplotlib + numpy required. pip install matplotlib numpy")

    tids = sorted(grid.keys())
    cats = [c for c in CATEGORY_ORDER if any(grid[t].get(c, 0) > 0 for t in tids)]
    if not tids or not cats:
        sys.exit("trace has no usable events (need ph=X and tid+cat fields)")

    fig, ax = plt.subplots(figsize=(max(8, len(tids) * 0.4), 6))
    bottoms = np.zeros(len(tids))
    for cat in cats:
        vals = np.array([grid[t].get(cat, 0) / 1000.0 for t in tids])  # us → ms
        ax.bar(
            [str(t) for t in tids],
            vals,
            bottom=bottoms,
            label=cat,
            color=CATEGORY_COLORS.get(cat, "#374151"),
        )
        bottoms += vals

    ax.set_xlabel("layer (tid)")
    ax.set_ylabel("time (ms)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"→ wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument(
        "--separate",
        default="",
        help="comma-separated event-name prefixes to render as separate figures (e.g. 'prefill,decode')",
    )
    ap.add_argument("--title", default="ferrum layerwise breakdown")
    args = ap.parse_args()

    events = load_trace(args.trace)
    if not events:
        sys.exit(f"no events in {args.trace}")

    out = args.out or args.trace.with_suffix(".png")

    if args.separate:
        prefixes = [p.strip() for p in args.separate.split(",") if p.strip()]
        for prefix in prefixes:
            subset = [e for e in events if e.get("name", "").startswith(prefix)]
            if not subset:
                print(f"  no events match prefix '{prefix}', skipping", file=sys.stderr)
                continue
            grid = aggregate(subset)
            sub_out = out.with_suffix("").with_name(f"{out.stem}_{prefix}.png")
            render(grid, sub_out, f"{args.title} — {prefix}")
    else:
        grid = aggregate(events)
        render(grid, out, args.title)


if __name__ == "__main__":
    main()
