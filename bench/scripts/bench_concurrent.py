"""
Concurrent throughput bench harness — measures aggregate t/s under N
parallel requests for a single engine.

Designed to compare server-mode behaviour across ferrum / mistral.rs /
llama-server. Each engine has its own subcommand because their Python
APIs / HTTP surfaces differ:

  python bench_concurrent.py mistralrs <gguf> <max_seqs> <max_tokens>
  python bench_concurrent.py ferrum    <gguf> <tokenizer> <bin> <max_seqs> <max_tokens>
  python bench_concurrent.py llamacpp  <gguf> <max_seqs> <max_tokens>

Output: one JSON object on stdout per subcommand run, with:
  - engine
  - model
  - max_seqs (concurrency)
  - per-request {wall_s, prompt_tps, decode_tps, ttft_ms}
  - aggregate {wall_s, total_completion_tokens, total_tps}

Concurrency model:
  All N requests are submitted simultaneously. Aggregate throughput =
  sum(completion_tokens) / max(wall_per_request). This isolates the
  server's batching / scheduling efficiency from per-request overhead.

mistral.rs: uses `Runner(max_seqs=N, paged_attn=True)` and sends N
requests via threads. Their Python API supports concurrent
`send_completion_request` calls.

ferrum: uses the existing `ferrum bench --concurrency N` CLI surface;
we parse the throughput from its output.

llama-server: starts `llama-server` with `-np N` (parallel slots),
sends N HTTP requests, parses /v1/completions response.

Currently: the ferrum entry depends on Phase 4 (multi-seq paged KV +
scheduler fanout) landing — single-seq paged is in PRs #68/#69/#70
but doesn't yet enable real concurrent batching at the engine level.
mistral.rs and llama-server are already concurrency-capable.

Defaults: max_seqs=8, max_tokens=128, prompt = "Once upon a time" (4
tokens, identical to Group A baseline).
"""
import json
import os
import subprocess
import sys
import threading
import time

GGUF_DIR = os.environ.get("GGUF_DIR", "/Users/chejinxuan/ferrum-bench/models")
TOK_DIR = os.environ.get("TOK_DIR", "/Users/chejinxuan/ferrum-bench/tokenizers")
DEFAULT_PROMPT = "Once upon a time"


def bench_mistralrs(gguf_filename, max_seqs, max_tokens):
    from mistralrs import CompletionRequest, Runner, Which

    # Paged attention is required for true server-mode multi-seq batching.
    runner = Runner(
        which=Which.GGUF(
            quantized_model_id=GGUF_DIR,
            quantized_filename=gguf_filename,
        ),
        max_seqs=max_seqs,
        paged_attn=True,
    )

    results = [None] * max_seqs
    barrier = threading.Barrier(max_seqs)

    def worker(idx):
        req = CompletionRequest(
            prompt=DEFAULT_PROMPT,
            model="ignored",
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # Sync all workers before sending so timing measures real concurrency.
        barrier.wait()
        t0 = time.perf_counter()
        resp = runner.send_completion_request(req)
        wall = time.perf_counter() - t0
        u = resp.usage
        # ttft = time-to-first-token; mistral.rs's usage doesn't expose
        # this directly, fall back to total_prompt_time as a stand-in.
        ttft_ms = (u.total_prompt_time_sec or 0.0) * 1000.0
        results[idx] = {
            "request": idx,
            "wall_s": wall,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "prompt_tps": u.avg_prompt_tok_per_sec,
            "decode_tps": u.avg_compl_tok_per_sec,
            "ttft_ms": ttft_ms,
        }

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(max_seqs)]
    aggregate_t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    aggregate_wall = time.perf_counter() - aggregate_t0

    total_completion = sum(r["completion_tokens"] for r in results)
    return {
        "engine": "mistral.rs",
        "model": gguf_filename,
        "max_seqs": max_seqs,
        "max_tokens": max_tokens,
        "per_request": results,
        "aggregate": {
            "wall_s": aggregate_wall,
            "total_completion_tokens": total_completion,
            "total_tps": total_completion / aggregate_wall if aggregate_wall > 0 else 0,
        },
    }


def bench_ferrum(gguf_filename, tokenizer_filename, ferrum_bin, max_seqs, max_tokens):
    """Run `ferrum bench --concurrency N`. ferrum's bench command takes
    HF model aliases, not GGUF paths directly — so for now this entry
    is a placeholder. Phase 4 should add a `ferrum run --concurrent`
    or expose --gguf to the bench command.

    Workaround: for raw GGUF files, just run N parallel `ferrum run`
    processes with `--bench-mode` and aggregate. Each process is its
    own model load (not a server). Approximates concurrent request
    throughput, but pays N model-load cost.
    """
    raise NotImplementedError(
        "ferrum concurrent bench needs Phase 4 server-mode multi-seq integration. "
        "Use `ferrum bench` with HF model aliases for the existing per-request "
        "concurrency path (which scales poorly without paged KV — see "
        "bench/notes/2026-05-01-where-the-remaining-gap-lives.md)."
    )


def bench_llamacpp(gguf_filename, max_seqs, max_tokens):
    """Start `llama-server` with -np <max_seqs>, fire N parallel HTTP
    requests, parse responses. Simplest of the three: llama-server is
    already production-ready for multi-seq.
    """
    raise NotImplementedError(
        "llama-server entry: TODO. Wire this when running Group B for real. "
        "Use `llama-server -m <gguf> -np <max_seqs> -c <ctx_size>`, then POST "
        "to /v1/completions in N threads."
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    engine = sys.argv[1]

    if engine == "mistralrs":
        gguf = sys.argv[2]
        max_seqs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 128
        out = bench_mistralrs(gguf, max_seqs, max_tokens)
    elif engine == "ferrum":
        gguf = sys.argv[2]
        tok = sys.argv[3]
        ferrum_bin = sys.argv[4]
        max_seqs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 128
        out = bench_ferrum(gguf, tok, ferrum_bin, max_seqs, max_tokens)
    elif engine == "llamacpp":
        gguf = sys.argv[2]
        max_seqs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 128
        out = bench_llamacpp(gguf, max_seqs, max_tokens)
    else:
        print(f"unknown engine: {engine}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
