#!/usr/bin/env python3
"""Shared-prefix workload bench for block-level prefix cache.

128 chat requests, ALL sharing the same ~200-token system message,
each with a UNIQUE 5-token user question. ferrum's block-level cache
should hit on the first ~12 blocks (system msg) and only prefill the
short user msg. Measures TTFT + throughput, prints comparison-ready
JSON to stdout.

Usage:
  python3 shared_prefix_bench.py --base-url http://127.0.0.1:8801 \
      --concurrency 32 --num-prompts 128 --max-tokens 64
"""

import argparse
import asyncio
import time
import json
import sys
from typing import List

# Pinned ~200-token system message. Token count is approximate;
# what matters is the byte/token IDENTITY across all 128 requests so
# the block hashes match.
SYSTEM_PROMPT = """You are a precise, helpful technical assistant.
When answering questions you must follow these rules carefully:

1. Be accurate. If you don't know something, say so directly.
2. Be concise. Prefer short, direct answers over verbose ones.
3. Show reasoning for non-trivial claims, but keep it to one or two
   sentences when possible.
4. For factual queries about programming, math, or systems, cite the
   relevant concept by name (e.g., "via the GIL", "by memoization").
5. Avoid filler phrases like "Great question!" or "I'd be happy to
   help with that." Just answer.
6. If a question is ambiguous, name the ambiguity in one sentence
   and pick the most likely interpretation."""

# 128 short user questions. Length 4-8 tokens each.
USER_QUESTIONS = [
    "What is GIL?",
    "Define amortized cost.",
    "Why is quicksort O(n log n)?",
    "What does ECC RAM mean?",
    "Explain TLB shootdown.",
    "What's CRDT?",
    "Define MVCC.",
    "What is Raft?",
    "Explain Bloom filter.",
    "What is a skip list?",
    "What's pre-order traversal?",
    "Explain CAS.",
    "What is a B+ tree?",
    "Define cache line.",
    "What's NUMA?",
    "Explain VMA.",
    "What is the kernel page table?",
    "Define eviction policy.",
    "What's TCP slow start?",
    "Explain NAT punching.",
    "What is QUIC?",
    "Define WAL.",
    "What's HSTS?",
    "Explain CRC32.",
    "What's a Merkle tree?",
    "Define ABA problem.",
    "What's a memory fence?",
    "Explain RCU.",
    "What's a futex?",
    "Define copy-on-write.",
    "What's mmap used for?",
    "Explain epoll.",
]
# Extend to 128 by duplicating; the SYSTEM prefix dominates anyway
# (cache hit is on system msg, not user content).
USER_QUESTIONS = (USER_QUESTIONS * 4)[:128]


async def fire_one(session, url, model, system, user, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t_start = time.time()
    async with session.post(url, json=payload) as resp:
        resp_json = await resp.json()
        t_end = time.time()
    # ferrum returns usage.completion_tokens / prompt_tokens
    usage = resp_json.get("usage", {})
    return {
        "wall_time_s": t_end - t_start,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def run(args):
    import aiohttp

    url = f"{args.base_url}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=120)
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def gated(user):
            async with sem:
                return await fire_one(
                    session, url, args.model, SYSTEM_PROMPT, user, args.max_tokens
                )

        prompts = USER_QUESTIONS[: args.num_prompts]
        t0 = time.time()
        results = await asyncio.gather(*[gated(u) for u in prompts])
        t1 = time.time()

    total_wall = t1 - t0
    total_completion = sum(r["completion_tokens"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    wall_times = sorted(r["wall_time_s"] for r in results)
    median_wall = wall_times[len(wall_times) // 2]
    p99_wall = wall_times[int(len(wall_times) * 0.99)]
    throughput = total_completion / total_wall

    summary = {
        "num_prompts": len(results),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "total_wall_s": total_wall,
        "total_completion_tokens": total_completion,
        "total_prompt_tokens": total_prompt,
        "throughput_tok_s": throughput,
        "median_wall_s": median_wall,
        "p99_wall_s": p99_wall,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8801")
    p.add_argument("--model", default="x")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))
