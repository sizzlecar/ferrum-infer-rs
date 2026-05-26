#!/usr/bin/env python3
"""
Rigorous side-by-side kernel comparison between ferrum and vLLM nsys
CSVs. For each kernel category, reports:
  - total ms (sum of fires × per-call)
  - fires (call count)
  - µs per call (avg)
  - %GPU (within that bench's window)
Then sorts by absolute total-ms GAP so we identify where ferrum spends
MORE time than vLLM — independent of per-call vs launch-count source.

Usage:
    python3 scripts/compare_nsys_kernels.py \
        docs/.../m3-80pct-session-.../c32/ferrum_nsys.csv \
        docs/.../vllm-only-sweep/c32_nsys/vllm_nsys.kernels.csv
"""

import csv
import re
import sys
from collections import defaultdict


def parse_nsys_csv(path):
    """Return list of dicts: {pct, total_ns, fires, avg_ns, name}."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            # Format: % , total_ns , fires , avg_ns , med , min , max , stddev , "Name"
            parts = next(csv.reader([line]))
            if len(parts) < 9:
                continue
            try:
                rows.append({
                    'pct': float(parts[0]),
                    'total_ns': int(parts[1]),
                    'fires': int(parts[2]),
                    'avg_ns': float(parts[3]),
                    'name': parts[8],
                })
            except (ValueError, IndexError):
                continue
    return rows


def categorize(name):
    """Bucket a kernel by purpose, so we can compare apples-to-apples."""
    n = name.lower()
    if 'marlin_moe_wna16' in n or 'moe_marlin' in n:
        return 'moe-matmul'
    if 'marlin' in n and 'repack' in n:
        return 'marlin-repack'
    if 'marlin' in n:
        return 'dense-marlin (qkv/o)'
    if 'flash_fwd' in n or 'flash_decode' in n or 'paged_batched_flash' in n:
        return 'attention-decode-splitkv'
    if 'paged_batched_decode_attn' in n:
        return 'attention-decode-singlepass'
    if 'paged_varlen_attn' in n or 'flash_fwd_varlen' in n:
        return 'attention-prefill-varlen'
    if 'paged_attention_v2' in n:
        return 'attention-paged-v2'
    if 'cublasgemvparams' in n or 'cublas' in n and 'gemv' in n:
        return 'lm_head-gemv'
    if 'cutlass' in n and 'gemm' in n:
        return 'lm_head/dense-cutlass'
    if 'topk' in n and 'softmax' in n:
        return 'moe-route'
    if 'topkgating' in n:
        return 'moe-route'
    if 'moe_combine' in n:
        return 'moe-combine'
    if 'moe_align' in n:
        return 'moe-align'
    if 'rms_norm' in n or 'rmsnorm' in n:
        return 'rms-norm'
    if 'rope' in n or 'qk_norm' in n:
        return 'rope/qk-norm'
    if 'silu_mul' in n or 'act_and_mul' in n:
        return 'act-silu'
    if 'embedding' in n:
        return 'embedding'
    if 'residual' in n:
        return 'residual-add'
    if 'split_qkv' in n:
        return 'split-qkv'
    if 'kv_cache' in n or 'kvcache' in n:
        return 'kv-write'
    if 'argmax' in n:
        return 'sample-argmax'
    if 'softmax' in n:
        return 'softmax'
    if 'cu_index' in n or 'index_elementwise' in n or 'gather' in n:
        return 'gather/index'
    if 'elementwise' in n or 'direct_copy' in n:
        return 'elementwise/copy'
    if 'reduce' in n and 'flash' not in n:
        return 'reduce'
    return 'other'


def bucket_rows(rows):
    """Aggregate rows by category."""
    cats = defaultdict(lambda: {'total_ns': 0, 'fires': 0, 'kernels': []})
    for r in rows:
        c = categorize(r['name'])
        cats[c]['total_ns'] += r['total_ns']
        cats[c]['fires'] += r['fires']
        cats[c]['kernels'].append((r['name'][:80], r['fires'], r['avg_ns']))
    return cats


def main():
    if len(sys.argv) != 3:
        print("usage: compare_nsys_kernels.py <ferrum.csv> <vllm.csv>")
        sys.exit(1)
    ferrum = bucket_rows(parse_nsys_csv(sys.argv[1]))
    vllm = bucket_rows(parse_nsys_csv(sys.argv[2]))
    all_cats = sorted(set(ferrum.keys()) | set(vllm.keys()))

    # Compute gap per category
    gap_rows = []
    for c in all_cats:
        f_total_ms = ferrum.get(c, {}).get('total_ns', 0) / 1e6
        v_total_ms = vllm.get(c, {}).get('total_ns', 0) / 1e6
        f_fires = ferrum.get(c, {}).get('fires', 0)
        v_fires = vllm.get(c, {}).get('fires', 0)
        f_per_call_us = (f_total_ms * 1000 / f_fires) if f_fires else 0
        v_per_call_us = (v_total_ms * 1000 / v_fires) if v_fires else 0
        gap_rows.append({
            'cat': c,
            'f_total_ms': f_total_ms,
            'v_total_ms': v_total_ms,
            'gap_ms': f_total_ms - v_total_ms,
            'f_fires': f_fires,
            'v_fires': v_fires,
            'f_per_call_us': f_per_call_us,
            'v_per_call_us': v_per_call_us,
        })

    gap_rows.sort(key=lambda r: -abs(r['gap_ms']))

    print()
    print(f"{'category':30s} | {'ferrum':>22s} | {'vLLM':>22s} | {'gap':>10s}")
    print(f"{'':30s} | {'total_ms (fires× µs)':>22s} | {'total_ms (fires× µs)':>22s} | {'(ms ferrum-vllm)':>10s}")
    print("-" * 100)
    for r in gap_rows:
        f_lbl = f"{r['f_total_ms']:7.1f} ({r['f_fires']:>5d}× {r['f_per_call_us']:>5.1f})" if r['f_fires'] else "—"
        v_lbl = f"{r['v_total_ms']:7.1f} ({r['v_fires']:>5d}× {r['v_per_call_us']:>5.1f})" if r['v_fires'] else "—"
        gap = f"{r['gap_ms']:+7.1f}" if r['gap_ms'] != 0 else "—"
        print(f"{r['cat']:30s} | {f_lbl:>22s} | {v_lbl:>22s} | {gap:>10s}")

    # Totals
    f_total = sum(r['f_total_ms'] for r in gap_rows)
    v_total = sum(r['v_total_ms'] for r in gap_rows)
    print("-" * 100)
    print(f"{'TOTAL GPU time':30s} | {f_total:>7.1f} ms          | {v_total:>7.1f} ms          | {f_total - v_total:+7.1f}")
    print()
    print(f"ratio of GPU times: {f_total/v_total:.2f}× (ferrum/vllm)")
    print()

    # Per-category detail for top-3 gaps
    print()
    print("─── Top-3 gaps: kernel detail ───")
    for r in gap_rows[:3]:
        c = r['cat']
        print()
        print(f"[{c}]  gap = {r['gap_ms']:+.1f} ms")
        print(f"  ferrum: {r['f_total_ms']:.1f}ms total, {r['f_fires']} fires, {r['f_per_call_us']:.1f} µs/call")
        for n, fc, av in ferrum.get(c, {}).get('kernels', []):
            print(f"     {n:78s}  {fc:>5d}×  {av/1000:.1f} µs")
        print(f"  vLLM:   {r['v_total_ms']:.1f}ms total, {r['v_fires']} fires, {r['v_per_call_us']:.1f} µs/call")
        for n, fc, av in vllm.get(c, {}).get('kernels', []):
            print(f"     {n:78s}  {fc:>5d}×  {av/1000:.1f} µs")


if __name__ == '__main__':
    main()
