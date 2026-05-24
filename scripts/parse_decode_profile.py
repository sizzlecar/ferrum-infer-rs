#!/usr/bin/env python3
"""
Parse ferrum's FERRUM_DECODE_OP_PROFILE stderr output and render a
markdown bottleneck table.

The probe (qwen3_moe.rs etc.) emits lines like:

  [decode-profile] tokens=1 total=18 ms (55 t/s)
      attn:    7 ms ( 39%) over   24 calls
      moe:     8 ms ( 44%) over   24 calls
      ...

We aggregate over all decode iterations and produce a sorted
markdown table showing which op category dominates wall-clock time.

Usage:
  parse_decode_profile.py ferrum_profile.log ferrum.json vllm.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Matches per-bucket lines from two formats:
#   1. [decode-profile] style: "  attn:    7 ms ( 39%) over   24 calls"
#   2. [unified-op] style:     "[unified-op] paged_varlen_attention: 1792 calls 230 ms total avg=128us"
BUCKET_RE = re.compile(
    r"^\s+(\S+):\s+(\d+)\s+ms\s+\(\s*(\d+)%?\)\s+over\s+(\d+)\s+calls"
)
UNIFIED_OP_RE = re.compile(
    r"\[unified-op\]\s+([\w/+]+):\s+(\d+)\s+calls\s+(\d+)\s+ms\s+total"
)
TOTAL_RE = re.compile(r"\[\w+-profile\] tokens=(\d+) total=(\d+) ms")


def parse_log(path):
    """Return dict[op] = (total_ms, total_calls)."""
    totals = defaultdict(lambda: [0, 0])  # op → [ms, calls]
    overall_ms = 0
    overall_iters = 0
    with open(path) as f:
        for line in f:
            mt = TOTAL_RE.search(line)
            if mt:
                overall_ms += int(mt.group(2))
                overall_iters += 1
                continue
            mb = BUCKET_RE.match(line)
            if mb:
                op = mb.group(1)
                ms = int(mb.group(2))
                calls = int(mb.group(4))
                totals[op][0] += ms
                totals[op][1] += calls
                continue
            mu = UNIFIED_OP_RE.search(line)
            if mu:
                op = mu.group(1)
                calls = int(mu.group(2))
                ms = int(mu.group(3))
                totals[op][0] += ms
                totals[op][1] += calls
                overall_ms += ms  # unified-op output doesn't have separate "total" lines
                overall_iters += 1
    return totals, overall_ms, overall_iters


def load_throughput(path):
    """Extract output_throughput from a BenchReport JSON."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    return obj.get("output_throughput_tps", {}).get("mean")


def render(totals, overall_ms, overall_iters, ferrum_tps, vllm_tps):
    out = []
    out.append("# Bottleneck breakdown\n")
    if ferrum_tps and vllm_tps:
        ratio = ferrum_tps / vllm_tps
        out.append(
            f"**Throughput**: ferrum **{ferrum_tps:.1f}** tok/s vs vLLM **{vllm_tps:.1f}** tok/s "
            f"→ ratio **{ratio:.3f}** (gap {(1-ratio)*100:.1f}%)\n"
        )
    elif ferrum_tps:
        out.append(f"**ferrum throughput**: {ferrum_tps:.1f} tok/s (vLLM not available)\n")
    out.append(f"\n**ferrum decode iterations profiled**: {overall_iters} "
               f"(total wall {overall_ms} ms)\n")

    if not totals:
        out.append("\n⚠ No `[*-profile]` lines parsed — was FERRUM_DECODE_OP_PROFILE=1 set?\n")
        return "\n".join(out)

    out.append("\n## Per-op breakdown (sum across all profiled iterations)\n")
    out.append("| rank | op category | total ms | calls | avg µs/call | % of decode total |")
    out.append("|---|---|---:|---:|---:|---:|")

    sorted_ops = sorted(totals.items(), key=lambda kv: -kv[1][0])
    for rank, (op, (ms, calls)) in enumerate(sorted_ops, 1):
        pct = (ms / overall_ms * 100) if overall_ms > 0 else 0.0
        avg_us = (ms * 1000 / calls) if calls > 0 else 0.0
        out.append(
            f"| {rank} | `{op}` | {ms} | {calls} | {avg_us:.1f} | {pct:.1f}% |"
        )

    out.append("\n## Reading")
    out.append("")
    out.append("- Per-op times are CUDA-event accurate (BackendTimer probes, PLAYBOOK § 1.1+1.2).")
    out.append("  Pre-Phase 1.1 these were `Instant::now()` + `B::sync` — same accuracy but paying")
    out.append("  a per-probe stream sync; CUDA events let the kernel batch continue.")
    out.append("- The top op category in the table is where ferrum spends the most wall-clock during")
    out.append("  decode. If it's `gemm`, the dense Marlin / linear path dominates.")
    out.append("  If `attn`, paged decode attn is the cost. If `moe`, the MoE routing+combine.")
    out.append("- Compare against vLLM's published kernel times for the same model — vLLM emits")
    out.append("  no equivalent per-op summary, but `nsys profile` against vllm serve gives the")
    out.append("  kernel timeline you can read by hand.")
    return "\n".join(out)


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: parse_decode_profile.py ferrum_profile.log ferrum.json vllm.json")
    log = Path(sys.argv[1])
    ferrum_json = Path(sys.argv[2])
    vllm_json = Path(sys.argv[3])

    totals, overall_ms, overall_iters = parse_log(log)
    ferrum_tps = load_throughput(ferrum_json)
    vllm_tps = load_throughput(vllm_json)
    print(render(totals, overall_ms, overall_iters, ferrum_tps, vllm_tps))


if __name__ == "__main__":
    main()
