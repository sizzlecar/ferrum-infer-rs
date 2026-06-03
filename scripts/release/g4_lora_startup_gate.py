#!/usr/bin/env python3
"""G4 startup LoRA serving release gate."""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from g1_g4_manifest import required_manifest_fields, utc_now


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
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = git_value(["rev-parse", "--short", "HEAD"])
    return repo_root() / "docs" / "release" / "g1-g4" / "g4-lora-startup" / f"{stamp}-{short}"


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


def run(cmd: list[str], out: Path, log: GateLog, *, timeout: int = 1200) -> None:
    log.write("RUN " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "NO_COLOR": "1"},
        timeout=timeout,
        check=False,
    )
    out.write_text(proc.stdout, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}; log={out}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")
        return err.code, json.loads(body)


def wait_health(base_url: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = http_json("GET", base_url + "/health")
            if code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {base_url}")


def start_server(bin_path: Path, model: str, out: Path, log: GateLog, extra: list[str]) -> tuple[subprocess.Popen[bytes], str, Path]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = out / f"serve-{port}.log"
    cmd = [str(bin_path), "serve", model, "--host", "127.0.0.1", "--port", str(port), *extra]
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
    text = server_log.read_text(errors="replace") if server_log.exists() else ""
    forbidden = ["panicked", "KV cache overflow", "<unk>", "[PAD]"]
    for pattern in forbidden:
        if pattern.lower() in text.lower():
            raise RuntimeError(f"forbidden runtime pattern {pattern!r} in {server_log}")


def f32_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def write_safetensors(path: Path) -> None:
    a = f32_bytes([0.0] * 6)
    b = f32_bytes([0.0] * 8)
    header = {
        "linear.lora_A.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, len(a)]},
        "linear.lora_B.weight": {"dtype": "F32", "shape": [4, 2], "data_offsets": [len(a), len(a) + len(b)]},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + a + b)


def write_lora_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["linear"],
        "base_model_name_or_path": "qwen3:0.6b",
    }, indent=2) + "\n")
    write_safetensors(path / "adapter_model.safetensors")


def tokenizer_dir() -> Path:
    env = os.environ.get("G4_TOKENIZER") or os.environ.get("FERRUM_G4_TOKENIZER")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path.home() / "ferrum-bench" / "tokenizers" / "Qwen3-0.6B",
        Path.home() / "ferrum-bench" / "tokenizers" / "Qwen3-30B-A3B.tokenizer.json",
        Path.home() / "ferrum-bench" / "tokenizers" / "Qwen3-0.6B.tokenizer.json",
    ])
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    candidates.extend(hf.glob("models--Qwen--Qwen3-0.6B/snapshots/*"))
    for path in candidates:
        if path.is_dir() and path.joinpath("tokenizer.json").is_file():
            return path
        if path.is_file() and path.name == "tokenizer.json":
            return path.parent
        if path.is_file() and path.name.endswith(".tokenizer.json"):
            scratch = repo_root() / "target" / "g4-tokenizers" / path.stem
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "tokenizer.json").write_bytes(path.read_bytes())
            return scratch
    raise RuntimeError("tokenizer not found; set G4_TOKENIZER=/path/to/tokenizer-dir-or-json")


def bench(bin_path: Path, model: str, base_url: str, out_json: Path, out_log: Path, log: GateLog, args: argparse.Namespace) -> None:
    cmd = [
        str(bin_path), "bench-serve",
        "--model", model,
        "--tokenizer", str(tokenizer_dir()),
        "--dataset", "random",
        "--random-input-len", str(args.random_input_len),
        "--random-output-len", str(args.random_output_len),
        "--concurrency", "1",
        "--num-prompts", str(args.num_prompts),
        "--warmup-requests", str(args.warmup_requests),
        "--n-repeats", "3",
        "--fail-on-error",
        "--require-ci",
        "--seed", "9271",
        "--output", "json",
        "--base-url", base_url,
        "--out", str(out_json),
    ]
    run(cmd, out_log, log, timeout=args.bench_timeout)


def throughput(path: Path) -> float:
    obj = json.loads(path.read_text())
    return float(obj["output_throughput_tps"]["mean"])


def safe_artifact_name(value: str) -> str:
    return value.replace("/", "--").replace(":", "-")


def model_ids(base_url: str) -> set[str]:
    code, models = http_json("GET", base_url + "/v1/models")
    if code != 200:
        raise RuntimeError(f"/v1/models failed: {code} {models}")
    return {item["id"] for item in models.get("data", [])}


def validate_lora_api(base_url: str, out: Path) -> tuple[str, str]:
    code, models = http_json("GET", base_url + "/v1/models")
    if code != 200:
        raise RuntimeError(f"/v1/models failed: {code} {models}")
    (out / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    ids = {item["id"] for item in models.get("data", [])}
    adapter_ids = sorted(model_id for model_id in ids if model_id.endswith(":sql"))
    if not adapter_ids:
        raise RuntimeError(f"adapter id missing from /v1/models: {ids}")
    adapter_id = adapter_ids[0]
    base_id = adapter_id.removesuffix(":sql")
    if base_id not in ids:
        raise RuntimeError(f"base model id {base_id} missing from /v1/models: {ids}")

    for model in [base_id, adapter_id]:
        code, body = http_json("POST", base_url + "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "不要展开推理，直接用一句话解释 LoRA。"}],
            "temperature": 0,
            "max_tokens": 256,
        })
        (out / f"chat-{safe_artifact_name(model)}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
        if code != 200:
            raise RuntimeError(f"chat failed for {model}: {code} {body}")
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or ""
        if not (content.strip() or reasoning.strip()):
            raise RuntimeError(f"empty chat output for {model}: {body}")

    code, body = http_json("POST", base_url + "/v1/chat/completions", {
        "model": f"{base_id}:missing",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    })
    (out / "chat-unknown-adapter.json").write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    if code != 400 or body.get("error", {}).get("param") != "model":
        raise RuntimeError(f"unknown adapter did not return OpenAI model error: code={code} body={body}")
    return base_id, adapter_id


def run_bench_gate(args: argparse.Namespace, out: Path, log: GateLog) -> dict[str, Any]:
    if args.skip_bench:
        summary = {"status": "skipped"}
        (out / "bench-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    bin_path = args.ferrum_bin
    fixture = out / "fixtures" / "sql-adapter"
    write_lora_fixture(fixture)
    run(["cargo", "build", "--release", "-p", "ferrum-cli", "--bin", "ferrum"], out / "release-build.log", log, timeout=1200)

    no_lora, no_lora_base, no_lora_log = start_server(bin_path, args.model, out, log, [])
    try:
        no_lora_ids = model_ids(no_lora_base)
        no_lora_model = sorted(no_lora_ids)[0] if no_lora_ids else args.model
        bench(bin_path, no_lora_model, no_lora_base, out / "bench-base-no-lora.json", out / "bench-base-no-lora.log", log, args)
    finally:
        stop_server(no_lora, no_lora_log)

    with_lora, with_lora_base, with_lora_log = start_server(
        bin_path,
        args.model,
        out,
        log,
        ["--lora", f"sql={fixture}", "--lora-model-id-template", "<base>:<name>"],
    )
    try:
        base_model_id, adapter_model_id = validate_lora_api(with_lora_base, out)
        bench(bin_path, base_model_id, with_lora_base, out / "bench-base-with-lora.json", out / "bench-base-with-lora.log", log, args)
        bench(bin_path, adapter_model_id, with_lora_base, out / "bench-adapter.json", out / "bench-adapter.log", log, args)
    finally:
        stop_server(with_lora, with_lora_log)

    base_no_lora = throughput(out / "bench-base-no-lora.json")
    base_with_lora = throughput(out / "bench-base-with-lora.json")
    adapter = throughput(out / "bench-adapter.json")
    base_regression_pct = 100.0 * max(0.0, base_no_lora - base_with_lora) / base_no_lora
    adapter_ratio = adapter / base_with_lora if base_with_lora > 0 else 0.0
    if base_regression_pct > args.max_base_regression_pct:
        raise RuntimeError(f"base path regression {base_regression_pct:.2f}% > {args.max_base_regression_pct:.2f}%")
    if adapter_ratio < args.min_adapter_ratio:
        raise RuntimeError(f"adapter throughput ratio {adapter_ratio:.3f} < {args.min_adapter_ratio:.3f}")
    summary = {
        "status": "pass",
        "base_no_lora_output_tps": base_no_lora,
        "base_with_lora_output_tps": base_with_lora,
        "adapter_output_tps": adapter,
        "base_regression_pct": base_regression_pct,
        "adapter_to_base_ratio": adapter_ratio,
        "fixture": str(fixture),
    }
    (out / "bench-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_manifest(
    out: Path,
    args: argparse.Namespace,
    checks: dict[str, Any],
    started_at_utc: str,
) -> None:
    manifest = {
        **required_manifest_fields(
            repo=repo_root(),
            goal="G4",
            name="lora-startup",
            models=[args.model],
            commands=[
                "cargo test --workspace --all-targets",
                "cargo test -p ferrum-quantization --test lora_linear_ref",
                "cargo test -p ferrum-models --test lora_loader",
                "cargo test -p ferrum-server lora",
                "cargo test -p ferrum-cli --test server_lora_startup",
                "cargo build --release -p ferrum-cli --bin ferrum",
                "ferrum bench-serve c=1 no-lora/base-with-lora/adapter",
            ],
            started_at_utc=started_at_utc,
            binary_path=args.ferrum_bin,
            features=[],
        ),
        "goal": "G4",
        "name": "lora-startup",
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
    (out / "gate.json").write_text(json.dumps({"status": "pass", "goal": "g4-lora-startup", "checks": checks}, indent=2) + "\n")
    (out / "summary.md").write_text(
        "# G4 LoRA Startup Gate\n\n"
        "Status: PASS\n\n"
        f"Model: `{args.model}`\n\n"
        "Validated:\n"
        "- `cargo test --workspace --all-targets`\n"
        "- LoRA f32 reference test\n"
        "- PEFT adapter loader tests\n"
        "- server adapter model routing tests\n"
        "- CLI startup LoRA tests\n"
        "- `/v1/models`, base chat, adapter chat, unknown adapter error\n"
        "- c=1 bench smoke for base/no-lora, base/with-lora, and adapter model id\n"
    )


def main() -> int:
    started_at_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--ferrum-bin", type=Path, default=repo_root() / "target" / "release" / "ferrum")
    parser.add_argument("--skip-workspace-test", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--random-input-len", type=int, default=64)
    parser.add_argument("--random-output-len", type=int, default=32)
    parser.add_argument("--bench-timeout", type=int, default=1800)
    parser.add_argument("--max-base-regression-pct", type=float, default=3.0)
    parser.add_argument("--min-adapter-ratio", type=float, default=0.50)
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
    run(["cargo", "test", "-p", "ferrum-quantization", "--test", "lora_linear_ref"], out / "lora-linear-ref.log", log)
    checks["lora_linear_ref"] = True
    run(["cargo", "test", "-p", "ferrum-models", "--test", "lora_loader"], out / "lora-loader.log", log)
    checks["lora_loader"] = True
    run(["cargo", "test", "-p", "ferrum-server", "lora"], out / "server-lora-routing.log", log)
    checks["server_lora_routing"] = True
    run(["cargo", "test", "-p", "ferrum-cli", "--test", "server_lora_startup"], out / "server-lora-startup.log", log)
    checks["server_lora_startup"] = True
    checks["bench"] = run_bench_gate(args, out, log)
    write_manifest(out, args, checks, started_at_utc)
    print(f"G4 LORA STARTUP PASS: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"G4 LORA STARTUP FAIL: {exc}", file=sys.stderr)
        raise
