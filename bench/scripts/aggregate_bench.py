"""
Parse the per-(engine,model) bench output files and emit a markdown report.

Inputs (in $RESULTS_DIR):
  <model_tag>__llamacpp.txt    — llama-bench output (table)
  <model_tag>__ferrum.txt      — ferrum --bench-mode lines
  <model_tag>__mistralrs.txt   — mistral.rs JSON lines

Env:
  RESULTS_DIR — defaults to ../group-a-results relative to this script

Outputs: markdown to stdout. Pipe into the report file:
    python aggregate_bench.py > ../group-a-report.md
"""
import json
import os
import re
import statistics
import sys
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "group-a-results"
RESULTS = Path(os.environ.get("RESULTS_DIR", DEFAULT_RESULTS))
MODELS = [
    ("Qwen3-8B-Q4_K_M", "Qwen3-8B"),
    ("Meta-Llama-3.1-8B-Instruct-Q4_K_M", "Llama-3.1-8B"),
    ("Qwen3-30B-A3B-Q4_K_M", "Qwen3-30B-A3B"),
]


def median(xs):
    return statistics.median(xs) if xs else None


def parse_llamacpp(path):
    """Parse llama-bench markdown table. Returns {pp50, pp512, tg128} -> tok/s."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text()
    # llama-bench prints rows like:
    #   | model | size | params | backend | threads | test | t/s |
    #   | ...   | ...  | ...    | ...     | 8       | pp50 | 1234.5 ± 1.2 |
    for line in text.splitlines():
        m = re.search(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)\s*[±]?", line)
        if m:
            test, val = m.group(1), float(m.group(2))
            out[test] = val
    return out


def parse_ferrum(path):
    """Parse ferrum --bench-mode output, grouped by op (pp50/pp512/tg128)."""
    out = {"pp50": [], "pp512": [], "tg128": []}
    if not path.exists():
        return {k: None for k in out}
    current = None
    for line in path.read_text().splitlines():
        if "pp50 trial" in line:
            current = "pp50"
        elif "pp512 trial" in line:
            current = "pp512"
        elif "tg128 trial" in line:
            current = "tg128"
        if current:
            # prefill: NN tok in X.YZs (T tok/s)
            m = re.search(r"prefill:\s*(\d+)\s*tok\s+in\s+([\d.]+)s\s*\(([\d.]+)\s*tok/s\)", line)
            if m and current in ("pp50", "pp512"):
                out[current].append(float(m.group(3)))
            # throughput: T tok/s (decode only)
            m = re.search(r"throughput:\s*([\d.]+)\s*tok/s", line)
            if m and current == "tg128":
                out[current].append(float(m.group(1)))
    return {k: median(v) for k, v in out.items()}


def parse_mistralrs(path):
    """Parse mistral.rs JSON output blobs (one per op).

    mistralrs Usage reports `prompt_tps=0.0` when max_tokens=1 (the prefill
    time field stays at 0). Fall back to a wall-based estimate using the
    decode rate from the same model's tg128 trial:
        prefill_rate ≈ n_prompt / (wall - 1/decode_rate)
    """
    out = {"pp50": None, "pp512": None, "tg128": None, "_pp_walls": {}}
    if not path.exists():
        return out
    current_op = None
    for line in path.read_text().splitlines():
        if "--- pp50" in line:
            current_op = "pp50"
        elif "--- pp512" in line:
            current_op = "pp512"
        elif "--- tg128" in line:
            current_op = "tg128"
        elif line.startswith("{"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            trials = d.get("trials", [])
            n_prompt = d.get("n_prompt", 0)
            if current_op in ("pp50", "pp512"):
                rates = [t["prompt_tps"] for t in trials if t.get("prompt_tps")]
                if rates:
                    out[current_op] = median(rates)
                else:
                    # store walls + actual prompt_tokens for fallback
                    walls = [(t["wall_s"], t["prompt_tokens"]) for t in trials]
                    out["_pp_walls"][current_op] = (n_prompt, walls)
            elif current_op == "tg128":
                vals = [t["decode_tps"] for t in trials if t.get("decode_tps")]
                out[current_op] = median(vals)

    # Backfill pp rates from wall - 1/decode_rate.
    decode = out.get("tg128")
    if decode:
        for op in ("pp50", "pp512"):
            if out.get(op) is None and op in out["_pp_walls"]:
                _n_prompt_arg, walls = out["_pp_walls"][op]
                rates = []
                for wall, n_tokens in walls:
                    pure_pp = wall - 1.0 / decode
                    if pure_pp > 0 and n_tokens > 0:
                        rates.append(n_tokens / pure_pp)
                if rates:
                    out[op] = median(rates)
    out.pop("_pp_walls", None)
    return out


def fmt(v, prec=1):
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


def ttft_ms(pp50_tps, tg128_tps):
    if not pp50_tps or not tg128_tps:
        return None
    return 1000.0 * (50.0 / pp50_tps + 1.0 / tg128_tps)


def main():
    rows = []
    for tag, label in MODELS:
        ferrum = parse_ferrum(RESULTS / f"{tag}__ferrum.txt")
        mistral = parse_mistralrs(RESULTS / f"{tag}__mistralrs.txt")
        llamacpp = parse_llamacpp(RESULTS / f"{tag}__llamacpp.txt")
        rows.append((label, ferrum, mistral, llamacpp))

    print("# Group A 性能报告 — Apple Silicon (M1 Max 32 GB)\n")
    print("**生成时间**: 2026-04-30  ·  **GGUF Q4_K_M 同一文件**")
    print()
    print("## 单请求吞吐 (tok/s, 3 trials median)")
    print()
    print("| 模型 | 指标 | ferrum | mistral.rs | llama.cpp |")
    print("|---|---|---:|---:|---:|")

    for label, ferrum, mistral, llamacpp in rows:
        for op in ("pp50", "pp512", "tg128"):
            f = ferrum.get(op)
            m = mistral.get(op)
            l = llamacpp.get(op)
            print(
                f"| {label} | {op} | {fmt(f)} | {fmt(m)} | {fmt(l)} |"
            )
        print("|  |  |  |  |  |")

    print()
    print("## TTFT 估算 (50-token prompt, ms)")
    print()
    print("> TTFT(50) = 50 / pp50 + 1 / tg128 (ms)")
    print()
    print("| 模型 | ferrum | mistral.rs | llama.cpp |")
    print("|---|---:|---:|---:|")
    for label, ferrum, mistral, llamacpp in rows:
        f = ttft_ms(ferrum.get("pp50"), ferrum.get("tg128"))
        m = ttft_ms(mistral.get("pp50"), mistral.get("tg128"))
        l = ttft_ms(llamacpp.get("pp50"), llamacpp.get("tg128"))
        print(f"| {label} | {fmt(f)} | {fmt(m)} | {fmt(l)} |")

    print()
    print("## 方法论")
    print()
    print("- **硬件**: MacBook Pro M1 Max, 32 GB unified memory, macOS 15.1.1")
    print("- **后端**: Metal (所有引擎)")
    print("- **量化**: Q4_K_M（同一份 GGUF 文件三引擎共用）")
    print(
        "- **测法**: pp50/pp512 = N-token \"the\"-prompt + 1-token decode 测 prefill 速率; "
        "tg128 = `Once upon a time` 短 prompt + 128 tokens 测 decode 速率"
    )
    print(
        "- **复现命令**: 见 `/tmp/bench_one_model.sh` (ferrum + mistral.rs + llama-bench)，"
        "每模型 3 trials 取中位数"
    )
    print(
        "- **环境变量**: ferrum 默认 `FERRUM_KV_CAPACITY=4096` (PR #55); "
        "mistralrs paged_attn=False; llama-bench 默认配置"
    )


if __name__ == "__main__":
    main()
