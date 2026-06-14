#!/usr/bin/env python3
"""Compare ferrum FERRUM_LAYER_DUMP output against the HF reference dump.

Usage: python scripts/gemma3_l1_compare.py <hf_dir> <ferrum_dir>

Reports per-layer max abs / mean abs diff and names the FIRST diverging
layer (threshold scaled to bf16 epsilon of the running max activation).
Exit 0 if logits argmax matches and no layer exceeds the threshold.
"""

import struct
import sys
from pathlib import Path


def read_f32(path: Path):
    data = path.read_bytes()
    return struct.unpack(f"<{len(data) // 4}f", data)


def stats(a, b):
    n = min(len(a), len(b))
    max_abs = 0.0
    sum_abs = 0.0
    peak = 0.0
    for x, y in zip(a[:n], b[:n]):
        d = abs(x - y)
        if d > max_abs:
            max_abs = d
        sum_abs += d
        m = max(abs(x), abs(y))
        if m > peak:
            peak = m
    return max_abs, sum_abs / n, peak, n


def main() -> None:
    hf_dir, fr_dir = Path(sys.argv[1]), Path(sys.argv[2])
    layers = sorted(p.name for p in hf_dir.glob("layer_*.bin"))
    # HF transformers stores the LAST hidden_states entry post-final-norm,
    # while ferrum dumps the pre-norm residual — not comparable directly.
    # The logits comparison covers final-norm + lm_head end-to-end.
    skipped_last = layers.pop() if layers else None
    if skipped_last:
        print(f"{skipped_last:14} skipped (HF stores it post-final-norm; logits cover it)")
    names = ["embed.bin", *layers, "logits.bin"]
    first_diverged = None
    for name in names:
        hf_p, fr_p = hf_dir / name, fr_dir / name
        if not fr_p.exists():
            print(f"{name:14} MISSING on ferrum side")
            first_diverged = first_diverged or name
            continue
        max_abs, mean_abs, peak, n = stats(read_f32(hf_p), read_f32(fr_p))
        # bf16 has ~8 mantissa bits: per-op rounding is ~peak/256; allow
        # accumulation headroom across a layer.
        threshold = max(peak / 256.0 * 4.0, 1e-4)
        flag = "" if max_abs <= threshold else "  <-- DIVERGED"
        if flag and first_diverged is None:
            first_diverged = name
        print(
            f"{name:14} n={n:9} max_abs={max_abs:.6f} mean_abs={mean_abs:.8f} "
            f"peak={peak:.3f} thr={threshold:.6f}{flag}"
        )

    hf_logits = read_f32(hf_dir / "logits.bin")
    fr_logits = read_f32(fr_dir / "logits.bin") if (fr_dir / "logits.bin").exists() else []
    if fr_logits:
        hf_top = max(range(len(hf_logits)), key=hf_logits.__getitem__)
        fr_top = max(range(len(fr_logits)), key=fr_logits.__getitem__)
        print(f"argmax: hf={hf_top} ferrum={fr_top} {'MATCH' if hf_top == fr_top else 'MISMATCH'}")
        if hf_top == fr_top and first_diverged is None:
            print("L1 DUMP COMPARE PASS")
            return
    print(f"first diverged: {first_diverged}")
    sys.exit(1)


if __name__ == "__main__":
    main()
