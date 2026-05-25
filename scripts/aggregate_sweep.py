#!/usr/bin/env python3
"""
Aggregate a sweep result directory into a markdown summary.

Reads docs/bench/sweep-<date>-<model>/{c1,c4,c16,c32}/{ferrum,vllm}_baseline.json
plus optional ferrum_trace.json + ferrum_nsys.csv, and emits the per-cell
gap table + per-cell category breakdown that drops directly into the
goal document.

Usage:
  scripts/aggregate_sweep.py docs/bench/sweep-2026-05-25-XXXX-qwen3-moe-30b-int4/
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def load_throughput(p):
    if not p.exists():
        return None
    with open(p) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    out = {}
    for k in ("output_throughput_tps", "request_throughput_rps"):
        if k in obj:
            out[k] = obj[k].get("mean") if isinstance(obj[k], dict) else obj[k]
    for k in ("ttft_ms", "tpot_ms"):
        if k in obj and "p50" in obj[k]:
            out[f"{k}_p50"] = obj[k]["p50"]["mean"]
    out["env_hash"] = obj.get("env_hash", "")
    out["commit"] = obj.get("env", {}).get("commit_sha", "")
    return out


def load_trace_categories(p):
    if not p.exists():
        return None
    try:
        data = json.load(open(p))
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    total = {}
    for e in data:
        c = e.get("cat", "?")
        total[c] = total.get(c, 0) + int(e.get("dur", 0))
    return total


def load_nsys_kernels(p, top_n=8):
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].replace('.', '').isdigit():
                continue
            rows.append(row)
    rows = rows[:top_n]
    out = []
    for r in rows:
        pct = float(r[0])
        name = r[-1].strip('"')
        # Shorten common patterns
        name = re.sub(r"void ", "", name)
        name = re.sub(r"_f16", "", name)
        m = re.match(r"^Marlin<\(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+),.*", name)
        if m:
            name = f"Marlin<{m.group(1)},{m.group(2)},{m.group(3)},{m.group(4)}>"
        if "cublasGemv" in name:
            name = "cublas gemv (lm_head)"
        if "cutlass" in name and "wmma" in name:
            name = "cutlass wmma"
        out.append((pct, name))
    return out


def render_cell(cell_dir):
    c = int(cell_dir.name.lstrip("c"))
    f = load_throughput(cell_dir / "ferrum_baseline.json")
    v = load_throughput(cell_dir / "vllm_baseline.json")
    trace = load_trace_categories(cell_dir / "ferrum_trace.json")
    nsys = load_nsys_kernels(cell_dir / "ferrum_nsys.csv")
    return {
        "c": c,
        "ferrum": f,
        "vllm": v,
        "trace": trace,
        "nsys": nsys,
    }


def render(cells, model, root):
    out = []
    out.append(f"# Sweep summary — {model}")
    out.append(f"\n_data: `{root}`_\n")

    # Headline ratio table
    out.append("## Throughput vs vLLM\n")
    out.append("| c | ferrum (tok/s) | vLLM (tok/s) | ratio | gap to 0.80 |")
    out.append("|---:|---:|---:|---:|---:|")
    for cell in cells:
        f, v = cell["ferrum"], cell["vllm"]
        ft = f["output_throughput_tps"] if f else None
        vt = v["output_throughput_tps"] if v else None
        if ft and vt:
            ratio = ft / vt
            gap = 0.80 - ratio
            need = (0.80 * vt) - ft  # tok/s needed to add
            out.append(
                f"| {cell['c']} | {ft:.1f} | {vt:.1f} | **{ratio:.3f}** | "
                f"{gap:+.3f} ({need:+.0f} tok/s) |"
            )
        else:
            out.append(f"| {cell['c']} | — | — | — | — |")

    # TTFT/TPOT table
    out.append("\n## Latency vs vLLM (p50)\n")
    out.append("| c | ferrum TTFT | vLLM TTFT | ratio | ferrum TPOT | vLLM TPOT | ratio |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|")
    for cell in cells:
        f, v = cell["ferrum"], cell["vllm"]
        if not (f and v):
            continue
        ft, vt = f.get("ttft_ms_p50"), v.get("ttft_ms_p50")
        fp, vp = f.get("tpot_ms_p50"), v.get("tpot_ms_p50")
        ttft_str = f"{ft/vt:.2f}x" if (ft and vt) else "—"
        tpot_str = f"{fp/vp:.2f}x" if (fp and vp) else "—"
        out.append(
            f"| {cell['c']} | {ft:.1f} ms | {vt:.1f} ms | {ttft_str} | "
            f"{fp:.1f} ms | {vp:.1f} ms | {tpot_str} |"
        )

    # Per-cell category breakdown from chrome trace
    out.append("\n## ferrum chrome-trace category split (Phase 1.5)\n")
    for cell in cells:
        if not cell["trace"]:
            continue
        out.append(f"\n### c={cell['c']}\n")
        tot = sum(cell["trace"].values())
        out.append("| category | total µs | % |")
        out.append("|---|---:|---:|")
        for k, v in sorted(cell["trace"].items(), key=lambda x: -x[1]):
            out.append(f"| {k} | {v:,} | {v/tot*100:.1f}% |")

    # nsys top kernels (only c=32 typically)
    for cell in cells:
        if cell["nsys"]:
            out.append(f"\n## nsys top kernels — c={cell['c']} (ground truth)\n")
            out.append("| % | kernel |")
            out.append("|---:|---|")
            for pct, name in cell["nsys"][:10]:
                out.append(f"| {pct:.1f} | `{name}` |")
            break

    # Env metadata
    if cells and cells[0]["ferrum"]:
        out.append("\n## Env metadata\n")
        f = cells[0]["ferrum"]
        out.append(f"- ferrum commit: `{f['commit']}`")
        out.append(f"- env_hash: `{f['env_hash']}`")

    return "\n".join(out)


def render_cell_detail(cell):
    """Per-cell bottleneck-cN.md content."""
    c = cell["c"]
    f, v = cell["ferrum"], cell["vllm"]
    out = [f"# Bottleneck — c={c} (Qwen3-30B-A3B-GPTQ-Int4 / RTX 4090)\n"]
    if f and v:
        ft = f["output_throughput_tps"]
        vt = v["output_throughput_tps"]
        ratio = ft / vt
        gap_to_80 = 0.80 - ratio
        out.append("## Headline\n")
        out.append(f"- **ferrum**: {ft:.1f} tok/s")
        out.append(f"- **vLLM**:   {vt:.1f} tok/s")
        out.append(f"- **ratio**:  {ratio:.3f}  ({'✓ above 0.80' if ratio >= 0.80 else f'gap {gap_to_80:+.3f}'})")
        out.append(f"- **TTFT p50**: ferrum {f.get('ttft_ms_p50',0):.1f} ms vs vLLM {v.get('ttft_ms_p50',0):.1f} ms")
        out.append(f"- **TPOT p50**: ferrum {f.get('tpot_ms_p50',0):.1f} ms vs vLLM {v.get('tpot_ms_p50',0):.1f} ms\n")

    if cell["trace"]:
        tot = sum(cell["trace"].values())
        out.append("## ferrum chrome-trace category split\n")
        out.append("| category | µs | % |")
        out.append("|---|---:|---:|")
        for k, val in sorted(cell["trace"].items(), key=lambda x: -x[1]):
            out.append(f"| {k} | {val:,} | {val/tot*100:.1f}% |")
        out.append("")
    else:
        out.append("_(chrome trace data unavailable for this cell)_\n")

    if cell["nsys"]:
        out.append("## nsys top kernels (ground truth)\n")
        out.append("| % | kernel |")
        out.append("|---:|---|")
        for pct, name in cell["nsys"]:
            out.append(f"| {pct:.1f} | `{name}` |")
        out.append("")
    else:
        out.append("_(no nsys profile captured for this cell — see c=32 for kernel-level data)_\n")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--model", default="Qwen3-30B-A3B-GPTQ-Int4")
    ap.add_argument(
        "--write-per-cell",
        type=Path,
        default=None,
        help="If set, write bottleneck-c{N}.md into this dir for each cell.",
    )
    args = ap.parse_args()

    cells = []
    for sub in sorted(args.sweep_dir.glob("c*"), key=lambda p: int(p.name.lstrip("c"))):
        if sub.is_dir():
            cells.append(render_cell(sub))

    if not cells:
        sys.exit(f"no c*/ subdirs found in {args.sweep_dir}")

    print(render(cells, args.model, args.sweep_dir))

    if args.write_per_cell:
        args.write_per_cell.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            outpath = args.write_per_cell / f"bottleneck-c{cell['c']}.md"
            outpath.write_text(render_cell_detail(cell))
            print(f"  → {outpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
