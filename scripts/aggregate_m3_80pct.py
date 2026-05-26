#!/usr/bin/env python3
"""
aggregate_m3_80pct.py — produce publication-grade summary table for the
M3 80% goal session.

Reads ferrum_baseline.json + vllm_baseline.json from each cell directory
(c1/, c4/, c16/, c32/) and emits a markdown table with CI95 derived
from n_repeats.

Usage:
    python3 scripts/aggregate_m3_80pct.py docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-26
"""

import json
import math
import os
import sys
from pathlib import Path


# Student t critical values for 95% CI, dof = n-1 (one-sided 0.025)
T_95 = {1: float('inf'), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def ci95_halfwidth(mean: float, std: float, n: int) -> float:
    if n <= 1 or std == 0:
        return float('nan')
    t = T_95.get(n, 2.0)  # ~2.0 for large n
    return t * std / math.sqrt(n)


def load_cell(cell_dir: Path, vllm_sweep_dir: Path = None):
    """Load ferrum + vllm baselines for one concurrency cell.

    ferrum_baseline.json uses ferrum bench-serve's nested schema
    (output_throughput_tps.mean / std). vllm_bench.json (from vllm bench
    serve) uses a flat schema (output_throughput, mean_ttft_ms, etc.).
    """
    c = int(cell_dir.name.lstrip('c'))
    out = {'c': c}
    f_path = cell_dir / 'ferrum_baseline.json'
    v_path = cell_dir / 'vllm_baseline.json'  # if sweep_bottleneck.sh ran vllm too
    if f_path.exists():
        f = json.loads(f_path.read_text())
        tps = f.get('output_throughput_tps', {})
        out['ferrum_mean'] = tps.get('mean', 0.0)
        out['ferrum_std'] = tps.get('std', 0.0)
        out['ferrum_n'] = f.get('n_repeats', 1)
        ttft = f.get('ttft_ms', {})
        out['ferrum_ttft_p50'] = ttft.get('p50') or ttft.get('mean') or 0.0
        tpot = f.get('tpot_ms', {})
        out['ferrum_tpot_p50'] = tpot.get('p50') or tpot.get('mean') or 0.0
    # vllm baseline: prefer vllm-only-sweep dir if present, else local
    if vllm_sweep_dir is not None:
        v_alt = vllm_sweep_dir / f'c{c}' / 'vllm_bench.json'
        if v_alt.exists():
            vd = json.loads(v_alt.read_text())
            # vllm bench serve flat schema
            out['vllm_mean'] = vd.get('output_throughput', 0.0)
            out['vllm_std'] = 0.0  # vllm bench is single-shot
            out['vllm_n'] = 1
            return out
    if v_path.exists():
        v = json.loads(v_path.read_text())
        tps = v.get('output_throughput_tps', {})
        out['vllm_mean'] = tps.get('mean', 0.0)
        out['vllm_std'] = tps.get('std', 0.0)
        out['vllm_n'] = v.get('n_repeats', 1)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    # Optional second arg: path to vllm-only-sweep directory
    vllm_sweep_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    cells = []
    for c in [1, 4, 16, 32]:
        cd = root / f'c{c}'
        if cd.is_dir():
            cells.append(load_cell(cd, vllm_sweep_dir))

    if not cells:
        print("ERROR: no cell dirs found under", root, file=sys.stderr)
        sys.exit(1)

    print(f"# M3 80% session results — {root.name}")
    print()
    print("Dataset: random in=256 out=128, num_prompts=128, n_repeats=5.")
    print("Hardware: Vast contract 37853406 RTX 4090 (sm_89), CUDA 13.0,")
    print("driver 580.159.04, locked 2520 MHz / 350 W.")
    print()

    # Main throughput table
    print("## Throughput (tok/s) and ratio to vLLM")
    print()
    print("| c | ferrum tok/s ± CI95 (n) | vLLM tok/s ± CI95 (n) | ratio | gap to 0.80 |")
    print("|--:|------------------------:|----------------------:|------:|------------:|")
    for cell in cells:
        c = cell['c']
        ft = cell.get('ferrum_mean', 0.0)
        fci = ci95_halfwidth(ft, cell.get('ferrum_std', 0.0), cell.get('ferrum_n', 1))
        fn = cell.get('ferrum_n', 1)
        vt = cell.get('vllm_mean', 0.0)
        vci = ci95_halfwidth(vt, cell.get('vllm_std', 0.0), cell.get('vllm_n', 1))
        vn = cell.get('vllm_n', 1)
        ratio = ft / vt if vt > 0 else 0.0
        gap = max(0.0, 0.80 - ratio) * 100
        f_str = f"{ft:.1f} ± {fci:.1f} (n={fn})" if not math.isnan(fci) else f"{ft:.1f} (n={fn})"
        v_str = f"{vt:.1f} ± {vci:.1f} (n={vn})" if not math.isnan(vci) and vt > 0 else (f"{vt:.1f} (n={vn})" if vt > 0 else "—")
        ratio_str = f"{ratio:.3f}" if vt > 0 else "—"
        gap_str = f"{gap:.1f} pp" if vt > 0 else "—"
        print(f"| {c} | {f_str} | {v_str} | {ratio_str} | {gap_str} |")
    print()

    # Latency table
    print("## Latency (ferrum)")
    print()
    print("| c | TTFT p50 (ms) | TPOT p50 (ms) |")
    print("|--:|--------------:|--------------:|")
    for cell in cells:
        c = cell['c']
        ttft = cell.get('ferrum_ttft_p50', 0.0)
        tpot = cell.get('ferrum_tpot_p50', 0.0)
        print(f"| {c} | {ttft:.1f} | {tpot:.2f} |")
    print()

    # Summary
    print("## Summary")
    print()
    achieved = sum(1 for cell in cells
                   if cell.get('vllm_mean', 0) > 0
                   and cell.get('ferrum_mean', 0) / cell.get('vllm_mean', 1e-9) >= 0.80)
    measured = sum(1 for cell in cells if cell.get('vllm_mean', 0) > 0)
    print(f"- Cells at ≥0.80 ratio: **{achieved}/{measured}** (target: 4/4)")
    if measured == 0:
        print("- vLLM not measured — ratios not computable; numbers above are ferrum-only.")
    print()


if __name__ == '__main__':
    main()
