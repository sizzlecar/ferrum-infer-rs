#!/usr/bin/env python3
"""
Lightweight async OpenAI-chat bench client. Mimics `vllm bench serve`'s
protocol and metrics so we can run apples-to-apples against ferrum's
HTTP server WITHOUT installing the full vllm package on the pod.

Usage:
  python3 bench_serve_client.py \
      --base-url http://127.0.0.1:8800 \
      --model /workspace/.hf_home/.../snapshots/<sha>/ \
      --tokenizer /workspace/.hf_home/.../snapshots/<sha>/ \
      --random-input-len 256 \
      --random-output-len 128 \
      --num-prompts 128 \
      --max-concurrency 32 \
      --result-file out.json

Output JSON shape mirrors vllm's `BenchmarkServingMetrics`:
  output_throughput, mean_tpot_ms, mean_ttft_ms, p99_*

Streaming (`stream=true`) is required so we can measure TTFT (time to
first token) separately from the rest. Stops as soon as `--num-prompts`
have completed.

No external deps beyond httpx + transformers (for tokenizer to size the
random prompts). If transformers isn't available we fall back to
fixed-length whitespace-noise prompts of `--random-input-len` words —
not exact match to vllm's tokenized random generator but good enough
for relative comparisons across ferrum runs.
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx


def gen_random_prompt(input_len: int, vocab_words: list[str]) -> str:
    return " ".join(random.choices(vocab_words, k=input_len))


def load_vocab_words(tokenizer_path: Optional[str], fallback_size: int = 4000) -> list[str]:
    """Load tokenizer vocab to draw random words from. Falls back to a
    fixed-size dummy vocab if transformers unavailable."""
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(tokenizer_path)
            words = []
            for tid in range(min(tok.vocab_size, fallback_size)):
                t = tok.decode([tid]).strip()
                if t and len(t) >= 2 and len(t) <= 20 and t.isascii():
                    words.append(t)
            if len(words) >= 200:
                return words
        except Exception as e:
            print(f"[warn] transformers unavailable / load failed: {e}", file=sys.stderr)
    # Fallback: generic English-ish lowercase words
    return [
        "the", "of", "and", "to", "a", "in", "for", "is", "on", "that",
        "with", "as", "by", "this", "from", "or", "are", "be", "at", "an",
        "have", "you", "but", "not", "they", "their", "we", "all", "can", "if",
        "more", "one", "about", "would", "what", "so", "out", "up", "when", "into",
        "than", "any", "may", "use", "made", "system", "model", "data", "value", "result",
    ]


@dataclass
class RequestResult:
    success: bool
    ttft_ms: float
    e2e_ms: float
    output_tokens: int
    inter_token_times_ms: list[float]


async def stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestResult:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    first_token_time = None
    last_event_time = None
    inter_token_times: list[float] = []
    output_tokens = 0
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions",
                                 json=body, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode("utf-8", errors="replace")
                print(f"[err] HTTP {resp.status_code}: {err[:200]}", file=sys.stderr)
                return RequestResult(False, 0.0, 0.0, 0, [])
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content is None:
                    continue
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                else:
                    inter_token_times.append((now - last_event_time) * 1000.0)
                last_event_time = now
                output_tokens += 1
        e2e = (time.perf_counter() - start) * 1000.0
        ttft = ((first_token_time - start) * 1000.0) if first_token_time else e2e
        return RequestResult(True, ttft, e2e, output_tokens, inter_token_times)
    except httpx.HTTPError as e:
        print(f"[err] HTTP exception: {e}", file=sys.stderr)
        return RequestResult(False, 0.0, 0.0, 0, [])


async def run_bench(args: argparse.Namespace) -> dict:
    vocab_words = load_vocab_words(args.tokenizer)
    prompts = [gen_random_prompt(args.random_input_len, vocab_words)
               for _ in range(args.num_prompts)]

    # httpx connection limits; allow `--max-concurrency` parallel POSTs.
    limits = httpx.Limits(max_connections=args.max_concurrency * 2,
                          max_keepalive_connections=args.max_concurrency * 2)
    sem = asyncio.Semaphore(args.max_concurrency)
    results: list[RequestResult] = []

    bench_start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        async def one(p: str) -> RequestResult:
            async with sem:
                return await stream_one(client, args.base_url, args.model,
                                        p, args.random_output_len, args.timeout)
        tasks = [asyncio.create_task(one(p)) for p in prompts]
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)

    bench_wall_s = time.perf_counter() - bench_start

    successful = [r for r in results if r.success]
    total_output_tokens = sum(r.output_tokens for r in successful)
    output_throughput = total_output_tokens / bench_wall_s if bench_wall_s > 0 else 0.0
    ttfts = [r.ttft_ms for r in successful]
    e2es = [r.e2e_ms for r in successful]
    # TPOT per-request = (e2e - ttft) / max(0, output_tokens-1)
    tpots = []
    for r in successful:
        if r.output_tokens >= 2:
            tpots.append((r.e2e_ms - r.ttft_ms) / (r.output_tokens - 1))

    def pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]

    out = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "duration_s": round(bench_wall_s, 3),
        "total_output_tokens": total_output_tokens,
        "output_throughput": round(output_throughput, 2),
        "mean_ttft_ms": round(statistics.fmean(ttfts), 2) if ttfts else 0.0,
        "p99_ttft_ms": round(pct(ttfts, 0.99), 2),
        "mean_tpot_ms": round(statistics.fmean(tpots), 2) if tpots else 0.0,
        "p99_tpot_ms": round(pct(tpots, 0.99), 2),
        "mean_e2e_ms": round(statistics.fmean(e2es), 2) if e2es else 0.0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True,
                    help="model id sent in chat completion body")
    ap.add_argument("--tokenizer", default=None,
                    help="path/id for transformers tokenizer (used to draw random words)")
    ap.add_argument("--random-input-len", type=int, default=256)
    ap.add_argument("--random-output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--result-file", default=None)
    args = ap.parse_args()

    out = asyncio.run(run_bench(args))
    print(json.dumps(out, indent=2))
    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
