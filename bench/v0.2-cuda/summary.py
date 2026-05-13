#!/usr/bin/env python3
"""Pretty-print bench results from JSON files in this directory."""
import json
import glob
import os
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + "/results"
patterns = sys.argv[2:] if len(sys.argv) > 2 else ["*__r1.json"]
for pat in patterns:
    for f in sorted(glob.glob(os.path.join(results_dir, pat))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        name = os.path.basename(f).replace(".json", "")
        tps = d.get("output_throughput", 0)
        p50 = d.get("mean_tpot_ms", 0)
        p99 = d.get("p99_tpot_ms", 0)
        ttft = d.get("mean_ttft_ms", 0)
        ok = d.get("completed", 0)
        fail = d.get("failed", 0)
        print(f"{name:32s} out={tps:8.1f} tok/s  TPOT={p50:6.2f}/{p99:7.2f} ms  TTFT={ttft:6.0f} ms  ok={ok}/{ok+fail}")
