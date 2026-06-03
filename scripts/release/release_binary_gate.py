#!/usr/bin/env python3
"""Release binary gates for official Ferrum assets and Homebrew formulae."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "https://github.com/sizzlecar/ferrum-infer-rs"
BAD_LOG_PATTERNS = [
    "panicked",
    "panic",
    "KV cache overflow",
    "failed to render model chat template",
    "command encoder",
    "failed assertion",
    "<unk>",
    "[PAD]",
]


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pat in BAD_LOG_PATTERNS:
        if pat.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pat!r} in {label}")


def run(cmd: list[str], *, cwd: Path | None = None, input: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, input=input, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def download(url: str, path: Path) -> None:
    last = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, path.open("wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f"download failed: {url}: {last}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def official_asset(version: str, asset: str) -> str:
    return f"{REPO}/releases/download/v{version}/{asset}"


def prepare_tarball(version: str, asset: str, out: Path, expected_sha: str | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / asset
    sha_path = out / f"{asset}.sha256"
    download(official_asset(version, asset), tar_path)
    if expected_sha is None:
        download(official_asset(version, f"{asset}.sha256"), sha_path)
        expected_sha = sha_path.read_text().split()[0]
    actual = sha256(tar_path)
    if actual != expected_sha:
        raise RuntimeError(f"sha256 mismatch for {asset}: actual={actual} expected={expected_sha}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(out)
    bin_path = out / "ferrum"
    if not bin_path.is_file():
        matches = list(out.rglob("ferrum"))
        if matches:
            bin_path = matches[0]
    if not bin_path.is_file():
        raise RuntimeError(f"ferrum binary not found after extracting {asset}")
    bin_path.chmod(bin_path.stat().st_mode | 0o111)
    return bin_path


def assert_version(bin_path: Path, version: str) -> None:
    p = run([str(bin_path), "--version"], timeout=20)
    if p.returncode != 0 or f"ferrum {version}" not in (p.stdout + p.stderr):
        raise RuntimeError(f"version check failed: rc={p.returncode} out={p.stdout} err={p.stderr}")


def cli_gate(bin_path: Path, model: str, out: Path) -> dict:
    text = "\n".join([
        "本轮句子是 ferrum-blue。只回答 OK",
        "上一条用户消息里的 ferrum 开头短语是什么？只输出短语，不要输出 OK",
        "/clear",
        "123+456 等于多少？只输出数字",
        "/bye",
        "",
    ])
    p = run([str(bin_path), "run", model], input=text, timeout=180)
    (out / "cli.stdout").write_text(p.stdout, errors="replace")
    (out / "cli.stderr").write_text(p.stderr, errors="replace")
    assert_no_bad_patterns("cli output", p.stdout + "\n" + p.stderr)
    combined = re.sub(r"<think>.*?</think>", "", p.stdout + "\n" + p.stderr, flags=re.S)
    ok = p.returncode == 0 and "ferrum-blue" in combined and "579" in combined
    if not ok:
        raise RuntimeError("CLI gate failed: expected ferrum-blue and 579")
    return {"passed": True, "has_context": True, "has_math": True}


def post(base: str, payload: dict, timeout: int = 120) -> tuple[int, str]:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def wait_health(port: int) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")


def serve_gate(bin_path: Path, model_path: str, model_name: str, out: Path, port: int, api_extra: bool) -> dict:
    log = out / "serve.log"
    with log.open("wb") as f:
        proc = subprocess.Popen([str(bin_path), "serve", model_path, "--host", "127.0.0.1", "--port", str(port)], stdout=f, stderr=subprocess.STDOUT)
    try:
        wait_health(port)
        common = {"model": model_name, "temperature": 0}
        s1, b1 = post(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "123+456 等于多少？只输出数字"}], "max_tokens": 256})
        c1 = json.loads(b1)["choices"][0]["message"].get("content", "") if s1 == 200 else b1
        s2, b2 = post(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "本轮短语是 ferrum-blue。只回答 OK"}, {"role": "assistant", "content": "OK"}, {"role": "user", "content": "第一条用户消息里的 ferrum 开头短语是什么？只输出短语，不要输出 OK"}], "max_tokens": 256})
        c2 = json.loads(b2)["choices"][0]["message"].get("content", "") if s2 == 200 else b2
        s3, b3 = post(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "写一个一万字介绍"}], "max_tokens": 10240})
        assert_no_bad_patterns("serve math response", c1)
        assert_no_bad_patterns("serve multiturn response", c2)
        assert_no_bad_patterns("serve boundary response", b3)
        result = {"math": [s1, c1], "multiturn": [s2, c2], "boundary_status": s3}
        if s1 != 200 or "579" not in c1:
            raise RuntimeError("serve math gate failed")
        if s2 != 200 or "ferrum-blue" not in c2:
            raise RuntimeError("serve multi-turn gate failed")
        if s3 != 400:
            raise RuntimeError("serve boundary gate did not return 400")
        if api_extra:
            schema = {"type": "json_schema", "json_schema": {"name": "Answer", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}
            s4, b4 = post(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "计算 123+456。最终答案必须只用 JSON 对象表示，格式为 {\"answer\":579}，不要 Markdown。"}], "response_format": schema, "max_tokens": 1024})
            msg = json.loads(b4)["choices"][0]["message"].get("content", "") if s4 == 200 else b4
            assert_no_bad_patterns("serve strict-json response", msg)
            if s4 != 200 or json.loads(msg).get("answer") != 579:
                raise RuntimeError("strict JSON gate failed")
            tools = [{"type": "function", "function": {"name": "calc", "description": "calculate expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
            s5, b5 = post(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "调用工具 calc 计算 123+456"}], "tools": tools, "tool_choice": {"type": "function", "function": {"name": "calc"}}, "max_tokens": 256})
            choice = json.loads(b5)["choices"][0] if s5 == 200 else {}
            assert_no_bad_patterns("serve tool-call response", b5)
            if s5 != 200 or choice.get("finish_reason") != "tool_calls" or "123+456" not in json.dumps(choice, ensure_ascii=False):
                raise RuntimeError("tool call gate failed")
            s6, b6 = post(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "请用一句话解释 String::from"}], "stream": True, "max_tokens": 256})
            assert_no_bad_patterns("serve stream response", b6)
            if s6 != 200 or b6.count("data: [DONE]") != 1 or '"content"' not in b6:
                raise RuntimeError("stream gate failed")
            result.update({"strict_json": [s4, msg], "tool_call": [s5, choice.get("finish_reason")], "stream": [s6, b6.count("data: [DONE]")]})
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)
        text = log.read_text(errors="replace") if log.exists() else ""
        assert_no_bad_patterns(str(log), text)


def check_ldd(bin_path: Path, out: Path) -> None:
    p = run(["ldd", str(bin_path)], timeout=30)
    (out / "ldd.txt").write_text(p.stdout + p.stderr, errors="replace")
    text = p.stdout + p.stderr
    if p.returncode != 0 or "not found" in text or re.search(r"torch|python|vllm", text, re.I):
        raise RuntimeError("ldd dependency gate failed")


def write_gate(out: Path, mode: str, version: str, checks: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps({"status": "pass", "mode": mode, "version": version, "checks": checks}, ensure_ascii=False, indent=2) + "\n")


def gate_tarball(args, *, asset: str, default_model: str, model_name: str, cuda: bool) -> None:
    bin_path = prepare_tarball(args.version, asset, args.out, args.sha256)
    assert_version(bin_path, args.version)
    if cuda:
        check_ldd(bin_path, args.out)
    model = args.model or default_model
    checks = {"version": True, "cli": cli_gate(bin_path, model, args.out), "serve": serve_gate(bin_path, model, args.model_name or model_name, args.out, args.port, True)}
    write_gate(args.out, "cuda-tarball" if cuda else "metal-tarball", args.version, checks)
    print(f"{'CUDA' if cuda else 'METAL'} TARBALL GATE PASS: {args.out}")


def homebrew_metal(args) -> None:
    p = run(["brew", "reinstall", "sizzlecar/ferrum/ferrum"], timeout=300)
    (args.out / "brew_reinstall.log").write_text(p.stdout + p.stderr, errors="replace")
    if p.returncode != 0:
        raise RuntimeError("brew reinstall failed")
    bin_path = Path(shutil.which("ferrum") or "")
    assert_version(bin_path, args.version)
    model = args.model or "/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    checks = {"version": True, "cli": cli_gate(bin_path, model, args.out), "serve": serve_gate(bin_path, model, args.model_name or "Qwen3-30B-A3B-Q4_K_M", args.out, args.port, True)}
    write_gate(args.out, "homebrew-metal", args.version, checks)
    print(f"HOMEBREW METAL GATE PASS: {args.out}")


def homebrew_cuda_fetch(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    p = run(["brew", "fetch", "--force", "sizzlecar/ferrum/ferrum-cuda"], timeout=300)
    (args.out / "brew_fetch.log").write_text(p.stdout + p.stderr, errors="replace")
    if p.returncode != 0:
        raise RuntimeError("brew fetch ferrum-cuda failed")
    info = run(["brew", "info", "--json=v2", "sizzlecar/ferrum/ferrum-cuda"], timeout=60)
    (args.out / "brew_info.json").write_text(info.stdout + info.stderr, errors="replace")
    if f"v{args.version}/ferrum-linux-x86_64-cuda-sm89.tar.gz" not in (info.stdout + info.stderr):
        raise RuntimeError("ferrum-cuda formula does not point at requested version asset")
    write_gate(args.out, "homebrew-cuda-fetch", args.version, {"fetch": True, "formula_version": args.version})
    print(f"HOMEBREW CUDA FETCH GATE PASS: {args.out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ["metal-tarball", "cuda-tarball", "homebrew-metal", "homebrew-cuda-fetch"]:
        p = sub.add_parser(mode)
        p.add_argument("--version", required=True)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--sha256")
        p.add_argument("--model")
        p.add_argument("--model-name")
        p.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "metal-tarball":
            gate_tarball(args, asset="ferrum-macos-aarch64.tar.gz", default_model="/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf", model_name="Qwen3-30B-A3B-Q4_K_M", cuda=False)
        elif args.mode == "cuda-tarball":
            gate_tarball(args, asset="ferrum-linux-x86_64-cuda-sm89.tar.gz", default_model="/workspace/hf-cache/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441", model_name="Qwen3-30B-A3B-GPTQ-Int4", cuda=True)
        elif args.mode == "homebrew-metal":
            homebrew_metal(args)
        elif args.mode == "homebrew-cuda-fetch":
            homebrew_cuda_fetch(args)
        return 0
    except Exception as e:
        (args.out / "gate.json").write_text(json.dumps({"status": "fail", "mode": args.mode, "version": args.version, "error": str(e)}, ensure_ascii=False, indent=2) + "\n")
        print(f"RELEASE BINARY GATE FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
