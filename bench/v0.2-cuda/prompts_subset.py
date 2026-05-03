#!/usr/bin/env python3
"""Build the deterministic 128-prompt subset for the v0.2 CUDA bench.

Reads ShareGPT_V3_unfiltered_cleaned_split.json, filters to single-turn
human prompts whose token length falls in [128, 512] (a well-defined
range for the W2/W3/W4 workloads at prompt_len=512), seeds an RNG with
the ferrum repo HEAD short hash, and samples 128 prompts. Writes
prompts.json with the exact prompts every run will use.

Usage:
    huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \\
        ShareGPT_V3_unfiltered_cleaned_split.json \\
        --local-dir /workspace/datasets
    python3 prompts_subset.py \\
        --input  /workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --output prompts.json \\
        --seed   "$(git -C $FERRUM_DIR rev-parse --short HEAD)"

Produces prompts.json:
    {
      "seed": "...",
      "count": 128,
      "tokenizer": "tiktoken/cl100k_base",
      "prompts": [
         {"id": 0, "text": "...", "approx_tokens": 234},
         ...
      ]
    }
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def approx_token_count(text: str) -> int:
    """Rough token estimate without loading a real tokenizer.

    cl100k_base averages ~4 bytes/token on English. Good enough for
    length filtering — we don't need exact bucketing here, just to
    reject obvious 50-token / 5000-token prompts.
    """
    return max(1, len(text.encode("utf-8")) // 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ShareGPT_V3_*.json path")
    ap.add_argument("--output", required=True, help="prompts.json output path")
    ap.add_argument("--seed", required=True, help="RNG seed (use repo HEAD short hash)")
    ap.add_argument("--count", type=int, default=128)
    ap.add_argument("--min-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        sys.exit(2)

    print(f"loading {src} (~640 MB)...", file=sys.stderr, flush=True)
    convos = json.loads(src.read_text())
    print(f"  {len(convos)} conversations loaded", file=sys.stderr)

    candidates: list[str] = []
    for c in convos:
        for turn in c.get("conversations", []):
            if turn.get("from") != "human":
                continue
            txt = turn.get("value", "").strip()
            n = approx_token_count(txt)
            if args.min_tokens <= n <= args.max_tokens:
                candidates.append(txt)
                break  # one prompt per conversation; avoids near-duplicates
    print(f"  {len(candidates)} prompts in [{args.min_tokens}, {args.max_tokens}] token range", file=sys.stderr)

    if len(candidates) < args.count:
        print(f"only {len(candidates)} candidates, need {args.count}", file=sys.stderr)
        sys.exit(3)

    # Seed: SHA256 of the user-supplied seed string → first 8 bytes as int.
    seed_int = int(hashlib.sha256(args.seed.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    sample = rng.sample(candidates, args.count)

    out = {
        "seed": args.seed,
        "count": args.count,
        "tokenizer_estimator": "approx (4 bytes/token)",
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "source": str(src.name),
        "prompts": [
            {"id": i, "text": p, "approx_tokens": approx_token_count(p)}
            for i, p in enumerate(sample)
        ],
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}: {args.count} prompts (seed={args.seed})", file=sys.stderr)


if __name__ == "__main__":
    main()
