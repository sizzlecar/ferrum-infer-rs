#!/usr/bin/env python3
"""
Concurrent serving benchmark — vLLM benchmark_serving.py style.

Drives an OpenAI-compatible /v1/chat/completions endpoint with
configurable concurrency / request rate, captures per-request TTFT,
inter-token latency, and aggregate throughput. Output is both a
human-readable summary AND a JSON file for downstream comparison.

Usage:
    bench_serving.py \
        --base-url http://localhost:8000 \
        --model Qwen3-8B-Q4_K_M \
        --num-prompts 32 \
        --max-concurrency 8 \
        --request-rate inf \
        --max-tokens 128 \
        --result-file out.json

Standard metrics (matching vLLM's terminology):
  Request throughput  (req/s)
  Input  token throughput (tok/s)
  Output token throughput (tok/s)
  TTFT  mean / median / p99 (ms)
  TPOT  mean / median / p99 (ms) — time per output token, EXCLUDES TTFT
  ITL   mean / median / p99 (ms) — inter-token latency, distribution
                                    of gaps between successive tokens

Inputs (closer to realistic serving than naive identical prompts):
  - Default uses a small bundled set of varied prompts; --dataset can
    point at a JSONL file with `prompt` per line.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import aiohttp


# ── Default prompt set — varied length, varied topic. Replace via --dataset.
DEFAULT_PROMPTS: list[str] = [
    "Explain the difference between paged attention and vanilla attention in transformer inference.",
    "Write a 200-word essay about why Rust's ownership model makes it suitable for systems programming.",
    "Summarize the key innovations in the original Transformer paper (Vaswani et al. 2017) in three bullet points.",
    "Compare GPTQ and AWQ quantization. Which is better for INT4 on Apple Silicon GPUs and why?",
    "Describe how continuous batching differs from naive batching in LLM serving.",
    "What is FlashAttention's memory complexity? Walk through how it computes softmax in tiles.",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "Write Python code to compute the dot product of two large vectors using NumPy with explicit chunking.",
    "Explain why Apple Silicon has unified memory and how that affects LLM inference performance.",
    "What is the GGUF file format? Why did the llama.cpp project create it?",
    "Briefly: what does KV cache do, why does it grow with sequence length, and how does paged KV help?",
    "Outline an approach to detecting prompt-injection attacks at the LLM serving layer.",
    "Why is decode bottlenecked by memory bandwidth on Apple Silicon? Give the rough math.",
    "Describe what 'speculative decoding' achieves and the role of a draft model.",
    "Summarize the SwiGLU activation function and where it sits in a Llama-family transformer block.",
    "Walk through a single transformer decode step at the operator level: what reads/writes happen?",
]


@dataclass
class RequestInput:
    prompt: str
    max_tokens: int


@dataclass
class RequestResult:
    success: bool
    error: Optional[str] = None

    # All times in seconds, relative to send time.
    ttft: Optional[float] = None
    end: Optional[float] = None  # last-token timestamp
    arrival_times: list[float] = field(default_factory=list)
    # Token counts from the response (if reported); fall back to len of streamed deltas.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    streamed_token_count: int = 0

    @property
    def output_tokens(self) -> int:
        # Prefer the server-reported count; otherwise count deltas.
        return self.completion_tokens or self.streamed_token_count

    @property
    def itl_seconds(self) -> list[float]:
        # Per-token inter-arrival latencies, EXCLUDING TTFT (so first
        # interval is delta from token-0 to token-1, etc).
        if len(self.arrival_times) < 2:
            return []
        out: list[float] = []
        for a, b in zip(self.arrival_times[:-1], self.arrival_times[1:]):
            out.append(b - a)
        return out

    @property
    def tpot(self) -> Optional[float]:
        # Mean time per output token, excluding TTFT. Matches vLLM's
        # definition: (end - ttft) / (output_tokens - 1).
        if self.ttft is None or self.end is None or self.output_tokens < 2:
            return None
        return (self.end - self.ttft) / (self.output_tokens - 1)


async def stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    req: RequestInput,
) -> RequestResult:
    """Send one streaming request, capture per-token timings."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "stream": True,
        "temperature": 0.0,  # deterministic for reproducibility
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    result = RequestResult(success=False)
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=body, headers=headers, timeout=600) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return result
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = evt.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    # Count both `content` (final answer) and
                    # `reasoning_content` (Qwen3 thinking mode chain-of-thought)
                    # so engines that stream chain-of-thought (llama.cpp w/
                    # Qwen3) report comparable token counts.
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.perf_counter() - t0
                        if result.ttft is None:
                            result.ttft = now
                        result.arrival_times.append(now)
                        result.streamed_token_count += 1
                if "usage" in evt and evt["usage"]:
                    result.prompt_tokens = evt["usage"].get("prompt_tokens")
                    result.completion_tokens = evt["usage"].get("completion_tokens")
        result.end = time.perf_counter() - t0
        result.success = True
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


async def request_generator(
    inputs: list[RequestInput],
    request_rate: float,
) -> AsyncGenerator[RequestInput, None]:
    """Yield requests at the configured rate. Use rate=inf to release all immediately (burst)."""
    if request_rate == float("inf"):
        for req in inputs:
            yield req
        return
    # Poisson inter-arrival times
    for req in inputs:
        yield req
        # Sample interval ~ Exp(1/rate)
        interval = random.expovariate(request_rate)
        await asyncio.sleep(interval)


async def run_benchmark(
    base_url: str,
    model: str,
    inputs: list[RequestInput],
    max_concurrency: int,
    request_rate: float,
) -> tuple[list[RequestResult], float]:
    """Drive the full benchmark. Returns (per-request results, total wall time)."""
    sem = asyncio.Semaphore(max_concurrency)
    results: list[RequestResult] = []

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max_concurrency, force_close=False),
    ) as session:

        async def bounded_request(req: RequestInput) -> RequestResult:
            async with sem:
                return await stream_request(session, base_url, model, req)

        # Build tasks lazily as the generator yields (matters for rate-limited mode)
        tasks: list[asyncio.Task[RequestResult]] = []
        t0 = time.perf_counter()
        async for req in request_generator(inputs, request_rate):
            tasks.append(asyncio.create_task(bounded_request(req)))
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return results, wall


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def warmup_request(base_url: str, model: str) -> None:
    """One synchronous warmup so subsequent timings see hot pipelines / JIT."""
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": False,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with contextlib.suppress(Exception):
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()


def make_inputs(args: argparse.Namespace) -> list[RequestInput]:
    if args.dataset:
        prompts: list[str] = []
        with open(args.dataset, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj.get("prompt") or obj.get("text") or obj.get("input")
                if p:
                    prompts.append(p)
    else:
        prompts = DEFAULT_PROMPTS

    if not prompts:
        raise SystemExit("no prompts loaded")

    rng = random.Random(args.seed)
    inputs: list[RequestInput] = []
    for i in range(args.num_prompts):
        prompt = prompts[i % len(prompts)] if args.deterministic_prompts else rng.choice(prompts)
        inputs.append(RequestInput(prompt=prompt, max_tokens=args.max_tokens))
    return inputs


def summarize(
    results: list[RequestResult],
    wall_s: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    total_input = sum((r.prompt_tokens or 0) for r in ok)
    total_output = sum(r.output_tokens for r in ok)

    ttfts_ms = [r.ttft * 1000 for r in ok if r.ttft is not None]
    tpots_ms = [r.tpot * 1000 for r in ok if r.tpot is not None]
    itls_ms_all: list[float] = []
    for r in ok:
        itls_ms_all.extend(g * 1000 for g in r.itl_seconds)

    e2es_ms = [r.end * 1000 for r in ok if r.end is not None]

    summary: dict[str, Any] = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
            "max_tokens": args.max_tokens,
        },
        "completed": len(ok),
        "failed": len(fail),
        "wall_time_s": wall_s,
        "request_throughput_rps": len(ok) / wall_s if wall_s > 0 else 0,
        "input_throughput_tok_s": total_input / wall_s if wall_s > 0 else 0,
        "output_throughput_tok_s": total_output / wall_s if wall_s > 0 else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "ttft_ms": _stats(ttfts_ms),
        "tpot_ms": _stats(tpots_ms),
        "itl_ms": _stats(itls_ms_all),
        "e2e_ms": _stats(e2es_ms),
    }
    if fail:
        summary["error_samples"] = [r.error for r in fail[:5]]
    return summary


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p99": percentile(xs, 99),
        "max": max(xs),
    }


def print_human(s: dict[str, Any]) -> None:
    print("─" * 72)
    cfg = s["config"]
    print(
        f"BENCHMARK  base={cfg['base_url']}  model={cfg['model']}  "
        f"reqs={cfg['num_prompts']}  conc={cfg['max_concurrency']}  rate={cfg['request_rate']}"
    )
    print("─" * 72)
    print(f"  completed:               {s['completed']}/{s['completed'] + s['failed']}")
    print(f"  wall time:               {s['wall_time_s']:.2f}s")
    print(f"  request throughput:      {s['request_throughput_rps']:.2f} req/s")
    print(f"  input  token throughput: {s['input_throughput_tok_s']:.1f} tok/s")
    print(f"  output token throughput: {s['output_throughput_tok_s']:.1f} tok/s")
    print(f"  total input tokens:      {s['total_input_tokens']}")
    print(f"  total output tokens:     {s['total_output_tokens']}")
    print()
    for label, key in [
        ("TTFT", "ttft_ms"),
        ("TPOT", "tpot_ms"),
        ("ITL ", "itl_ms"),
        ("E2E ", "e2e_ms"),
    ]:
        st = s[key]
        if st["mean"] == st["mean"]:  # not NaN
            print(
                f"  {label}  mean {st['mean']:7.2f}ms  median {st['median']:7.2f}ms  "
                f"p99 {st['p99']:7.2f}ms  max {st['max']:7.2f}ms"
            )
    if "error_samples" in s:
        print()
        print("  errors (first 5):")
        for e in s["error_samples"]:
            print(f"    - {e}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="requests per second; inf = release all at once (burst). Otherwise Poisson.",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", help="JSONL file with one prompt per line under key 'prompt'")
    p.add_argument(
        "--deterministic-prompts",
        action="store_true",
        help="Round-robin through the prompt set instead of sampling, so every run sees same inputs",
    )
    p.add_argument("--result-file", help="Path to write per-run JSON summary")
    p.add_argument("--label", default="", help="Optional label echoed into the JSON")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    inputs = make_inputs(args)
    if not args.no_warmup:
        warmup_request(args.base_url, args.model)

    results, wall = asyncio.run(
        run_benchmark(
            args.base_url, args.model, inputs, args.max_concurrency, args.request_rate
        )
    )
    summary = summarize(results, wall, args)
    if args.label:
        summary["label"] = args.label
    print_human(summary)
    if args.result_file:
        Path(args.result_file).write_text(json.dumps(summary, indent=2))
        print(f"  saved → {args.result_file}")


if __name__ == "__main__":
    main()
