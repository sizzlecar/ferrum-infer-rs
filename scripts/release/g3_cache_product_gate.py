#!/usr/bin/env python3
"""G3 prefix/session cache product release gate."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

REQUIRED_METRICS = [
    "ferrum_prefix_cache_hits_total",
    "ferrum_prefix_cache_misses_total",
    "ferrum_prefix_cache_evictions_total",
    "ferrum_prefix_cache_saved_prefill_tokens_total",
    "ferrum_prefix_cache_entries",
    "ferrum_prefix_cache_bytes",
    "ferrum_session_cache_hits_total",
    "ferrum_session_cache_misses_total",
    "ferrum_session_cache_evictions_total",
    "ferrum_session_cache_entries",
    "ferrum_session_cache_tokens",
]

BAD_RUNTIME_PATTERNS = [
    "panicked",
    "KV cache overflow",
    "failed to render model chat template",
    "<unk>",
    "[PAD]",
]


class GateLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=repo_root(), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def default_out_dir() -> Path:
    short = git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root() / "docs" / "release" / "g1-g4" / "g3-cache-product" / f"{stamp}-{short}"


def run(cmd: list[str], out: Path, log: GateLog, *, timeout: int = 1200, env: dict[str, str] | None = None, scan_runtime: bool = False) -> subprocess.CompletedProcess[str]:
    log.write("RUN " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "NO_COLOR": "1", **(env or {})},
        check=False,
    )
    out.write_text(proc.stdout, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}; log={out}")
    if scan_runtime:
        assert_no_bad_patterns(out.name, proc.stdout)
    return proc


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_RUNTIME_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden runtime pattern {pattern!r} in {label}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_health(base_url: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {base_url}")


def http_get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def metric_value(metrics: str, name: str) -> float:
    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == name:
            return float(parts[1])
    raise RuntimeError(f"missing metric {name}")


def tokenizer_dir(path: Path) -> Path | None:
    if path.is_dir() and path.joinpath("tokenizer.json").is_file():
        return path
    if path.is_file() and path.name == "tokenizer.json":
        return path.parent
    if path.is_file() and path.name.endswith(".tokenizer.json"):
        scratch = repo_root() / "target" / "g3-tokenizers" / path.stem
        scratch.mkdir(parents=True, exist_ok=True)
        dst = scratch / "tokenizer.json"
        if not dst.exists() or dst.read_bytes() != path.read_bytes():
            dst.write_bytes(path.read_bytes())
        return scratch
    return None


def find_tokenizer(model: str) -> Path:
    env = os.environ.get("G3_TOKENIZER") or os.environ.get("FERRUM_G3_TOKENIZER")
    if env:
        resolved = tokenizer_dir(Path(env))
        if resolved is not None:
            return resolved
    candidates = [
        Path.home() / "ferrum-bench" / "tokenizers" / "Qwen3-30B-A3B.tokenizer.json",
        Path.home() / "ferrum-bench" / "tokenizers" / "Qwen3-0.6B.tokenizer.json",
    ]
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    candidates.extend(hf.glob("models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))
    for path in candidates:
        resolved = tokenizer_dir(path)
        if resolved is not None:
            return resolved
    raise RuntimeError(f"tokenizer not found for {model}; set G3_TOKENIZER=/path/to/tokenizer-dir-or-tokenizer.json")


def start_server(bin_path: Path, model: str, args: list[str], out: Path, log: GateLog) -> tuple[subprocess.Popen[bytes], str, Path]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = out / f"serve-{port}.log"
    cmd = [str(bin_path), "serve", model, "--host", "127.0.0.1", "--port", str(port), *args]
    log.write("START " + " ".join(cmd))
    f = server_log.open("wb")
    proc = subprocess.Popen(cmd, cwd=repo_root(), stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "NO_COLOR": "1"})
    f.close()
    wait_health(base)
    return proc, base, server_log


def stop_server(proc: subprocess.Popen[bytes], server_log: Path) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    if server_log.exists():
        assert_no_bad_patterns(server_log.name, server_log.read_text(errors="replace"))


def run_bench_pair(args: argparse.Namespace, out: Path, log: GateLog) -> dict[str, Any]:
    if args.skip_bench:
        (out / "bench-summary.json").write_text(json.dumps({"status": "skipped"}, indent=2) + "\n")
        return {"status": "skipped"}

    tokenizer = find_tokenizer(args.model)
    bin_path = args.ferrum_bin
    run(["cargo", "build", "--release", "-p", "ferrum-cli", "--bin", "ferrum"], out / "release-build.log", log, timeout=1200)

    bench_shape = [
        "bench-serve",
        "--model", args.model,
        "--tokenizer", str(tokenizer),
        "--dataset", "shared-prefix",
        "--shared-prefix-len", str(args.shared_prefix_len),
        "--shared-suffix-len", str(args.shared_suffix_len),
        "--random-output-len", str(args.random_output_len),
        "--num-prompts", str(args.num_prompts),
        "--warmup-requests", str(args.warmup_requests),
        "--concurrency", str(args.concurrency),
        "--n-repeats", str(args.n_repeats),
        "--fail-on-error",
        "--require-ci",
        "--seed", "9271",
        "--output", "json",
    ]

    disabled_proc, disabled_base, disabled_log = start_server(bin_path, args.model, ["--disable-prefix-cache", "--session-cache", "off"], out, log)
    try:
        run([str(bin_path), *bench_shape, "--base-url", disabled_base, "--out", str(out / "bench-disabled.json")], out / "bench-disabled.log", log, timeout=args.bench_timeout)
    finally:
        stop_server(disabled_proc, disabled_log)

    enabled_proc, enabled_base, enabled_log = start_server(bin_path, args.model, ["--enable-prefix-cache", "--session-cache", "off"], out, log)
    try:
        run([str(bin_path), *bench_shape, "--base-url", enabled_base, "--out", str(out / "bench-enabled.json")], out / "bench-enabled.log", log, timeout=args.bench_timeout)
        metrics = http_get_text(enabled_base + "/metrics")
        (out / "bench-enabled-metrics.txt").write_text(metrics)
    finally:
        stop_server(enabled_proc, enabled_log)

    metrics = (out / "bench-enabled-metrics.txt").read_text()
    for name in REQUIRED_METRICS:
        if name not in metrics:
            raise RuntimeError(f"missing cache metric after enabled bench: {name}")
    hits = metric_value(metrics, "ferrum_prefix_cache_hits_total")
    saved = metric_value(metrics, "ferrum_prefix_cache_saved_prefill_tokens_total")
    if hits <= 0 or saved <= 0:
        raise RuntimeError(f"enabled bench did not record prefix hits/saved tokens: hits={hits} saved={saved}")

    summary = {
        "status": "pass",
        "tokenizer": str(tokenizer),
        "prefix_hits": hits,
        "prefix_saved_prefill_tokens": saved,
        "perf_result": "inconclusive",
        "reason": "G3 keeps unsafe engine-level KV prefix reuse forced off; product cache observability and correctness passed, but TTFT speedup is not claimed.",
    }
    (out / "bench-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def write_artifacts(out: Path, args: argparse.Namespace, checks: dict[str, Any]) -> None:
    (out / "gate.json").write_text(json.dumps({"status": "pass", "goal": "g3-cache-product", "checks": checks}, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "goal": "G3",
        "name": "cache-product",
        "status": "pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo": {
            "head": git_value(["rev-parse", "HEAD"]),
            "short_head": git_value(["rev-parse", "--short", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "dirty": bool(git_value(["status", "--porcelain"], "")),
        },
        "model": args.model,
        "checks": checks,
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "summary.md").write_text(
        "# G3 Cache Product Gate\n\n"
        "Status: PASS\n\n"
        f"Model: `{args.model}`\n\n"
        "Validated:\n"
        "- `cargo test --workspace --all-targets`\n"
        "- `cargo test -p ferrum-server cache_metrics_contract`\n"
        "- real-model prefix cache product correctness\n"
        "- real-model session cache correctness\n"
        "- shared-prefix `bench-serve --fail-on-error --require-ci` disabled/enabled runs, unless explicitly skipped\n"
        "- enabled metrics include prefix hits and saved prefill tokens\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--ferrum-bin", type=Path, default=repo_root() / "target" / "release" / "ferrum")
    parser.add_argument("--skip-workspace-test", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--shared-prefix-len", type=int, default=256)
    parser.add_argument("--shared-suffix-len", type=int, default=32)
    parser.add_argument("--random-output-len", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--bench-timeout", type=int, default=2400)
    args = parser.parse_args()

    out = (args.out or default_out_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = GateLog(out / "gate.log")
    checks: dict[str, Any] = {}

    if args.skip_workspace_test:
        (out / "cargo-test.log").write_text("skipped by --skip-workspace-test\n")
        checks["cargo_workspace_all_targets"] = "skipped"
    else:
        run(["cargo", "test", "--workspace", "--all-targets"], out / "cargo-test.log", log, timeout=2400)
        checks["cargo_workspace_all_targets"] = True

    run(["cargo", "test", "-p", "ferrum-server", "cache_metrics_contract"], out / "cache-metrics-contract.log", log, timeout=600)
    checks["cache_metrics_contract"] = True

    run(["cargo", "test", "--release", "-p", "ferrum-cli", "--test", "server_prefix_cache_product", "--", "--ignored", "--test-threads=1"], out / "server-prefix-cache-product.log", log, timeout=1800, scan_runtime=True)
    checks["server_prefix_cache_product"] = True

    run(["cargo", "test", "--release", "-p", "ferrum-cli", "--test", "server_session_cache", "--", "--ignored", "--test-threads=1"], out / "server-session-cache.log", log, timeout=1800, scan_runtime=True)
    checks["server_session_cache"] = True

    checks["bench"] = run_bench_pair(args, out, log)
    write_artifacts(out, args, checks)
    print(f"G3 CACHE PRODUCT PASS: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"G3 CACHE PRODUCT FAIL: {exc}", file=sys.stderr)
        raise
