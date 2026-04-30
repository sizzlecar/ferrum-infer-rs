"""
mistral.rs single-shot bench harness — invoked once per (model, op) measurement.

The mistralrs Python wheel is installed via:
    uv venv --python python3.12 /tmp/mistral_bench
    uv pip install --python /tmp/mistral_bench/bin/python mistralrs-metal==0.8.0

Each invocation is a fresh process (clean memory, fresh model load) and
writes one JSON object on stdout summarising N trials. mistralrs INFO logs
go to stderr (caller redirects /dev/null).

Usage:
    python bench_mistral.py <gguf_filename> <op> <n_prompt> <n_gen> [trials]

    op ∈ {pp, tg}
    n_prompt: number of "the" tokens in the prompt (use 0 for "Once upon a time")
    n_gen:    max_tokens cap on generation
    trials:   default 3

Env:
    GGUF_DIR      — directory holding the GGUF files
                    (default /Users/chejinxuan/ferrum-bench/models)
"""
import json
import os
import sys
import time
from mistralrs import Runner, Which, CompletionRequest

GGUF_DIR = os.environ.get("GGUF_DIR", "/Users/chejinxuan/ferrum-bench/models")


def main():
    gguf_filename = sys.argv[1]
    op = sys.argv[2]  # "pp" or "tg"
    n_prompt = int(sys.argv[3])
    n_gen = int(sys.argv[4])
    trials = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    runner = Runner(
        which=Which.GGUF(
            quantized_model_id=GGUF_DIR,
            quantized_filename=gguf_filename,
        ),
        max_seqs=1,  # single-stream bench
        no_paged_attn=True,
        paged_attn=False,
    )

    if n_prompt == 0:
        prompt = "Once upon a time"
    else:
        prompt = " ".join(["the"] * n_prompt)

    runs = []
    for i in range(trials):
        req = CompletionRequest(
            prompt=prompt,
            model="ignored",
            max_tokens=n_gen,
            temperature=0.0,
        )
        t0 = time.perf_counter()
        resp = runner.send_completion_request(req)
        wall = time.perf_counter() - t0
        u = resp.usage
        runs.append(
            {
                "trial": i,
                "wall_s": wall,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_prompt_time_s": u.total_prompt_time_sec,
                "total_completion_time_s": u.total_completion_time_sec,
                "prompt_tps": u.avg_prompt_tok_per_sec,
                "decode_tps": u.avg_compl_tok_per_sec,
            }
        )

    out = {
        "engine": "mistral.rs",
        "model": gguf_filename,
        "op": op,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "trials": runs,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
