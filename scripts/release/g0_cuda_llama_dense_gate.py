#!/usr/bin/env python3
"""Supplemental CUDA release gate for a Llama 8B-class dense model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_concurrency_quality_regression import run_concurrency_quality_regression
from openai_tool_call_regression import run_tool_call_regression

BAD_PATTERNS = [
    "panic",
    "panicked",
    "KV cache overflow",
    "failed to render model chat template",
    "command encoder",
    "failed assertion",
    "<unk>",
    "[PAD]",
    "Internal Server Error",
]
BENCH_QUALITY_COUNT_FIELDS = (
    "bad_output_per_run",
    "malformed_stream_per_run",
    "missing_done_per_run",
    "duplicate_done_per_run",
    "zero_output_tokens_per_run",
    "stream_bulk_flush_per_run",
    "http_500_per_run",
    "panic_per_run",
)
RELEASE_CONCURRENCY_CELLS = {1, 4, 16, 32}


def run(
    cmd: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env=env,
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pat in BAD_PATTERNS:
        if pat.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pat!r} in {label}")


def validate_bench_quality(report: dict[str, Any], *, label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field in BENCH_QUALITY_COUNT_FIELDS:
        values = report.get(field)
        if not isinstance(values, list):
            raise RuntimeError(f"{label}: missing {field}")
        total = 0
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(f"{label}: {field} contains non-integer {value!r}")
            total += value
        if total != 0:
            raise RuntimeError(f"{label}: {field} total={total} values={values!r}")
        counts[field.removesuffix("_per_run")] = total
    return counts


def validate_artifact(root: Path, *, expected_git_sha: str | None = None) -> dict[str, Any]:
    """Read-only validation of a copied dense CUDA G0 artifact."""

    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"dense artifact root is invalid: {root}")
    gate_path = root / "gate.json"
    metadata_path = root / "metadata.json"
    if not gate_path.is_file() or gate_path.is_symlink() or not metadata_path.is_file() or metadata_path.is_symlink():
        raise RuntimeError("dense gate/metadata artifact is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if gate.get("status") != "pass" or gate.get("lane") != "g0_cuda4090_llama_dense":
        raise RuntimeError("dense gate status/lane differs")
    if metadata.get("model_architecture") != "llama_dense" or metadata.get("git_dirty") is not False:
        raise RuntimeError("dense metadata architecture/dirty state differs")
    git_sha = metadata.get("git_sha")
    if not isinstance(git_sha, str) or re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise RuntimeError("dense metadata git SHA differs")
    if expected_git_sha is not None and git_sha != expected_git_sha:
        raise RuntimeError("dense metadata candidate differs")
    binary_sha = metadata.get("binary_sha256")
    if not isinstance(binary_sha, str) or re.fullmatch(r"[0-9a-f]{64}", binary_sha) is None:
        raise RuntimeError("dense metadata binary SHA differs")
    checks = gate.get("checks")
    required_checks = {"run", "serve_health", "serve_capacity", "serve", "tool_call_regression", "concurrency_quality_regression", "bench_serve"}
    if not isinstance(checks, dict) or set(checks) != required_checks:
        raise RuntimeError("dense gate check denominator differs")
    if checks["run"] != {"passed": True, "rc": 0, "has_context": True}:
        raise RuntimeError("dense run receipt differs")
    if checks["serve_health"].get("status") != 200 or checks["serve_health"].get("passed") is not True:
        raise RuntimeError("dense health receipt differs")
    capacity = checks["serve_capacity"]
    if capacity.get("passed") is not True or capacity.get("max_sequences", 0) < capacity.get("required_concurrency", 1) or capacity.get("target_concurrency", 0) < capacity.get("required_concurrency", 1):
        raise RuntimeError("dense serve capacity differs")
    serve = checks["serve"]
    if serve.get("math") != {"status": 200, "passed": True} or serve.get("multi_turn") != {"status": 200, "passed": True} or serve.get("stream_usage") != {"status": 200, "done_count": 1, "passed": True}:
        raise RuntimeError("dense serve correctness differs")
    if checks["tool_call_regression"].get("status") != "pass":
        raise RuntimeError("dense tool-call regression differs")
    cells = checks["concurrency_quality_regression"].get("cells")
    required_cells = RELEASE_CONCURRENCY_CELLS
    configured_cells = metadata.get("config", {}).get("concurrency_cells")
    if not isinstance(configured_cells, list) or {int(value) for value in configured_cells} != required_cells or len(configured_cells) != len(required_cells):
        raise RuntimeError("dense configured release matrix differs")
    if not isinstance(cells, list) or {row.get("concurrency") for row in cells if isinstance(row, dict)} != required_cells or any(row.get("passed") is not True or row.get("crosstalk") != 0 or row.get("length_finishes") != 0 for row in cells):
        raise RuntimeError("dense concurrency quality differs")
    bench_rows = checks["bench_serve"].get("rows")
    if not isinstance(bench_rows, list) or {row.get("concurrency") for row in bench_rows if isinstance(row, dict)} != required_cells:
        raise RuntimeError("dense bench cell denominator differs")
    for row in bench_rows:
        if row.get("completed", 0) <= 0 or row.get("errored") != 0 or row.get("output_token_count_source") != "usage" or row.get("throughput_tok_s", 0) <= 0:
            raise RuntimeError("dense bench correctness/performance receipt differs")
        if any(row.get(field.removesuffix("_per_run"), 1) != 0 for field in BENCH_QUALITY_COUNT_FIELDS):
            raise RuntimeError("dense bench quality count differs")
    required_files = {
        "run.command.json", "run.stdin", "run.stdout", "run.stderr",
        "serve.command.json", "serve.log", "serve.health.json",
        "serve.math.response.json", "serve.multiturn.response.json", "serve.stream.sse",
        "serve.effective_config.json", "serve.decision_trace.jsonl",
        "bench-serve.command.json", "bench-serve.stdout", "bench-serve.stderr", "bench-serve.json",
    }
    for name in required_files:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"dense evidence file is missing: {name}")
    run_command = json.loads((root / "run.command.json").read_text())
    serve_command = json.loads((root / "serve.command.json").read_text())
    bench_command = json.loads((root / "bench-serve.command.json").read_text())
    if "run" not in run_command or run_command.count("--disable-thinking") != 1:
        raise RuntimeError("dense run command differs")
    if "serve" not in serve_command or serve_command.count("--disable-thinking") != 1:
        raise RuntimeError("dense serve command differs")
    for option in ("--fail-on-error", "--require-ci", "--seed", "--n-repeats"):
        if option not in bench_command:
            raise RuntimeError(f"dense bench command lacks {option}")
    def command_value(option: str) -> str:
        positions = [index for index, value in enumerate(bench_command) if value == option]
        if len(positions) != 1 or positions[0] + 1 >= len(bench_command):
            raise RuntimeError(f"dense bench command {option} differs")
        return str(bench_command[positions[0] + 1])
    if command_value("--seed") != "9271":
        raise RuntimeError("dense bench seed differs")
    try:
        repeats = int(command_value("--n-repeats"))
    except ValueError as error:
        raise RuntimeError("dense bench repeat count differs") from error
    if repeats < 3:
        raise RuntimeError("dense bench repeat count differs")
    raw_reports = json.loads((root / "bench-serve.json").read_text())
    if not isinstance(raw_reports, list) or len(raw_reports) != len(required_cells):
        raise RuntimeError("dense raw bench report denominator differs")
    raw_by_cell = {row.get("concurrency"): row for row in raw_reports if isinstance(row, dict)}
    if set(raw_by_cell) != required_cells:
        raise RuntimeError("dense raw bench release matrix differs")
    gate_by_cell = {row["concurrency"]: row for row in bench_rows}
    for concurrency in sorted(required_cells):
        raw_row = raw_by_cell[concurrency]
        gate_row = gate_by_cell[concurrency]
        completed = raw_row.get("completed_per_run")
        errored = raw_row.get("errored_per_run")
        throughput = raw_row.get("output_throughput_tps")
        if not isinstance(completed, list) or len(completed) != repeats or not all(type(value) is int and value >= 0 for value in completed):
            raise RuntimeError("dense raw completed counts differ")
        if not isinstance(errored, list) or len(errored) != repeats or not all(type(value) is int and value >= 0 for value in errored):
            raise RuntimeError("dense raw errored counts differ")
        if not isinstance(throughput, dict) or not isinstance(throughput.get("mean"), (int, float)):
            raise RuntimeError("dense raw throughput differs")
        if gate_row.get("completed") != sum(completed) or gate_row.get("errored") != sum(errored) or gate_row.get("throughput_tok_s") != throughput["mean"] or gate_row.get("output_token_count_source") != raw_row.get("output_token_count_source"):
            raise RuntimeError("dense gate/raw bench summary differs")
        for field in BENCH_QUALITY_COUNT_FIELDS:
            raw_values = raw_row.get(field)
            if not isinstance(raw_values, list) or len(raw_values) != repeats or not all(type(value) is int and value >= 0 for value in raw_values):
                raise RuntimeError(f"dense raw {field} differs")
            if gate_row.get(field.removesuffix("_per_run")) != sum(raw_values):
                raise RuntimeError(f"dense gate/raw {field} differs")
    for name in ("run.stdout", "run.stderr", "serve.log", "bench-serve.stdout", "bench-serve.stderr"):
        assert_no_bad_patterns(name, (root / name).read_text(encoding="utf-8", errors="strict"))
    return {"status": "pass", "git_sha": git_sha, "binary_sha256": binary_sha, "cells": sorted(required_cells)}


def self_test() -> None:
    source = Path(__file__).resolve().parents[2] / "docs/release/g0/0.7.7/cuda-llama-dense"
    with tempfile.TemporaryDirectory(prefix="ferrum-dense-artifact-selftest-") as raw:
        root = Path(raw) / "artifact"
        shutil.copytree(source, root)
        candidate = "a" * 40
        metadata = json.loads((root / "metadata.json").read_text())
        metadata["git_dirty"] = False
        metadata["git_sha"] = candidate
        write(root / "metadata.json", json.dumps(metadata, indent=2) + "\n")
        for name in ("run.command.json", "serve.command.json"):
            command = json.loads((root / name).read_text())
            command.append("--disable-thinking")
            write(root / name, json.dumps(command, indent=2) + "\n")
        validate_artifact(root, expected_git_sha=candidate)
        metadata = json.loads((root / "metadata.json").read_text())
        original_cells = metadata["config"]["concurrency_cells"]
        metadata["config"]["concurrency_cells"] = [1]
        write(root / "metadata.json", json.dumps(metadata, indent=2) + "\n")
        try:
            validate_artifact(root, expected_git_sha=candidate)
        except RuntimeError as error:
            if "configured release matrix" not in str(error):
                raise
        else:
            raise RuntimeError("dense c1-only matrix unexpectedly passed")
        metadata["config"]["concurrency_cells"] = original_cells
        write(root / "metadata.json", json.dumps(metadata, indent=2) + "\n")
        bench_command_path = root / "bench-serve.command.json"
        bench_command = json.loads(bench_command_path.read_text())
        seed_position = bench_command.index("--seed") + 1
        bench_command[seed_position] = "42"
        write(bench_command_path, json.dumps(bench_command, indent=2) + "\n")
        try:
            validate_artifact(root, expected_git_sha=candidate)
        except RuntimeError as error:
            if "bench seed differs" not in str(error):
                raise
        else:
            raise RuntimeError("dense wrong seed unexpectedly passed")
        bench_command[seed_position] = "9271"
        repeats_position = bench_command.index("--n-repeats") + 1
        bench_command[repeats_position] = "2"
        write(bench_command_path, json.dumps(bench_command, indent=2) + "\n")
        try:
            validate_artifact(root, expected_git_sha=candidate)
        except RuntimeError as error:
            if "repeat count differs" not in str(error):
                raise
        else:
            raise RuntimeError("dense insufficient repeats unexpectedly passed")
        bench_command[repeats_position] = "3"
        write(bench_command_path, json.dumps(bench_command, indent=2) + "\n")
        gate = json.loads((root / "gate.json").read_text())
        gate["checks"]["bench_serve"]["rows"][0]["output_token_count_source"] = "estimated"
        write(root / "gate.json", json.dumps(gate, indent=2) + "\n")
        try:
            validate_artifact(root, expected_git_sha=candidate)
        except RuntimeError as error:
            if "bench correctness/performance" not in str(error):
                raise
        else:
            raise RuntimeError("dense tampered bench evidence unexpectedly passed")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def post(base: str, payload: dict[str, Any], timeout: int = 180) -> tuple[int, str]:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def get(url: str, timeout: int = 30) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def wait_health(port: int, timeout_sec: int = 600) -> None:
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
                last = f"status={r.status}"
        except Exception as e:
            last = str(e)
            time.sleep(2)
    raise RuntimeError(f"server did not become healthy within {timeout_sec}s: {last}")


def hf_cache_dir(model: str) -> Path | None:
    if "/" not in model:
        return None
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    repo_dir = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    if repo_dir.is_dir():
        return repo_dir
    return None


def resolve_tokenizer(tokenizer: str, model: str) -> Path:
    if tokenizer != "auto":
        path = Path(tokenizer)
        if path.is_file() and path.name == "tokenizer.json":
            return path.parent
        if (path / "tokenizer.json").is_file():
            return path
        raise RuntimeError(f"tokenizer not found: {path}")
    model_path = Path(model)
    if model_path.is_dir() and (model_path / "tokenizer.json").is_file():
        return model_path
    repo = hf_cache_dir(model)
    if repo is not None:
        snapshots = sorted((repo / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snapshots:
            if (snap / "tokenizer.json").is_file():
                return snap
    raise RuntimeError(
        "could not auto-resolve tokenizer.json; pass --tokenizer or pre-download the HF model"
    )


def assistant_text_from_jsonl(stdout: str) -> str:
    chunks: list[str] = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "assistant":
            chunks.append(str(row.get("content", "")))
    if chunks:
        return "\n".join(chunks)
    return stdout


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def run_cli_gate(root: Path, ferrum_bin: Path, model: str, repo: Path) -> dict[str, Any]:
    input_text = "\n".join(
        [
            "请记住短语 ferrum-blue。只回答 OK。",
            "第一条用户消息里的 ferrum 开头短语是什么？只输出短语。",
            "/bye",
            "",
        ]
    )
    cmd = [
        str(ferrum_bin),
        "run",
        model,
        "--backend",
        "cuda",
        "--max-tokens",
        "512",
        "--output-format",
        "jsonl",
        "--disable-thinking",
    ]
    p = run(cmd, cwd=repo, input_text=input_text, timeout=900)
    write(root / "run.command.json", json.dumps(cmd, indent=2) + "\n")
    write(root / "run.stdin", input_text)
    write(root / "run.stdout", p.stdout)
    write(root / "run.stderr", p.stderr)
    text = p.stdout + "\n" + p.stderr
    assert_no_bad_patterns("ferrum run", text)
    answer = strip_think(assistant_text_from_jsonl(p.stdout))
    if p.returncode != 0 or "ferrum-blue" not in answer:
        raise RuntimeError("ferrum run multi-turn gate failed")
    return {"passed": True, "rc": p.returncode, "has_context": True}


def serve_correctness(root: Path, model: str, port: int) -> dict[str, Any]:
    base = f"http://127.0.0.1:{port}"
    common = {
        "model": model,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    checks: dict[str, Any] = {}

    s1, b1 = post(
        base,
        {
            **common,
            "messages": [{"role": "user", "content": "123+456 等于多少？只输出数字。"}],
            "max_tokens": 128,
        },
    )
    write(root / "serve.math.response.json", b1)
    c1 = json.loads(b1)["choices"][0]["message"].get("content", "") if s1 == 200 else b1
    assert_no_bad_patterns("serve math", c1)
    if s1 != 200 or "579" not in strip_think(c1):
        raise RuntimeError("serve math gate failed")
    checks["math"] = {"status": s1, "passed": True}

    s2, b2 = post(
        base,
        {
            **common,
            "messages": [
                {"role": "user", "content": "本轮短语是 ferrum-blue。只回答 OK。"},
                {"role": "assistant", "content": "OK"},
                {
                    "role": "user",
                    "content": "第一条用户消息里的 ferrum 开头短语是什么？只输出短语。",
                },
            ],
            "max_tokens": 128,
        },
    )
    write(root / "serve.multiturn.response.json", b2)
    c2 = json.loads(b2)["choices"][0]["message"].get("content", "") if s2 == 200 else b2
    assert_no_bad_patterns("serve multi-turn", c2)
    if s2 != 200 or "ferrum-blue" not in strip_think(c2):
        raise RuntimeError("serve multi-turn gate failed")
    checks["multi_turn"] = {"status": s2, "passed": True}

    s3, b3 = post(
        base,
        {
            **common,
            "messages": [{"role": "user", "content": "请用一句话解释 String::from。"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 128,
        },
    )
    write(root / "serve.stream.sse", b3)
    assert_no_bad_patterns("serve stream", b3)
    if s3 != 200 or b3.count("data: [DONE]") != 1 or '"usage"' not in b3:
        raise RuntimeError("serve stream usage gate failed")
    checks["stream_usage"] = {"status": s3, "done_count": b3.count("data: [DONE]"), "passed": True}
    return checks


def capture_health(root: Path, port: int) -> dict[str, Any]:
    status, body = get(f"http://127.0.0.1:{port}/health", timeout=30)
    write(root / "serve.health.json", body)
    if status != 200:
        raise RuntimeError(f"health endpoint returned status={status}")
    data = json.loads(body)
    assert_no_bad_patterns("serve health", body)
    return {"status": status, "passed": True, "version": data.get("version")}


def decision_selected_int(data: dict[str, Any], selection: str) -> int | None:
    for decision in data.get("decisions", []):
        if decision.get("selection") != selection:
            continue
        try:
            return int(decision.get("selected"))
        except (TypeError, ValueError):
            return None
    return None


def validate_serve_capacity(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    data = json.loads((root / "serve.effective_config.json").read_text())
    max_concurrency = max(int(c) for c in cfg.get("concurrency_cells", [1, 4, 16, 32]))
    max_sequences = decision_selected_int(data, "max_sequences")
    workload_profile = data.get("workload_profile")
    if not isinstance(workload_profile, dict):
        workload_profile = {}
    target_concurrency = workload_profile.get("target_concurrency")
    if max_sequences is None or max_sequences < max_concurrency:
        raise RuntimeError(
            f"serve max_sequences {max_sequences!r} < required concurrency {max_concurrency}"
        )
    if not isinstance(target_concurrency, int) or target_concurrency < max_concurrency:
        raise RuntimeError(
            f"serve target_concurrency {target_concurrency!r} < required concurrency "
            f"{max_concurrency}"
        )
    return {
        "passed": True,
        "required_concurrency": max_concurrency,
        "max_sequences": max_sequences,
        "target_concurrency": target_concurrency,
    }


def run_bench_gate(
    root: Path,
    ferrum_bin: Path,
    model: str,
    tokenizer_dir: Path,
    port: int,
    cfg: dict[str, Any],
    repo: Path,
) -> dict[str, Any]:
    cells = ",".join(str(c) for c in cfg.get("concurrency_cells", [1, 4, 16, 32]))
    out = root / "bench-serve.json"
    cmd = [
        str(ferrum_bin),
        "bench-serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        model,
        "--tokenizer",
        str(tokenizer_dir),
        "--dataset",
        "random",
        "--random-input-len",
        str(cfg.get("random_input_len", 256)),
        "--random-output-len",
        str(cfg.get("random_output_len", 128)),
        "--num-prompts",
        str(cfg.get("num_prompts", 96)),
        "--warmup-requests",
        str(cfg.get("warmup_requests", 10)),
        "--n-repeats",
        str(cfg.get("n_repeats", 3)),
        "--concurrency-sweep",
        cells,
        "--fail-on-error",
        "--require-ci",
        "--seed",
        str(cfg.get("seed", 9271)),
        "--output",
        "json",
        "--out",
        str(out),
        "--tag",
        "cuda-llama-dense",
        "--enable-thinking",
        "false",
    ]
    p = run(cmd, cwd=repo, timeout=3600)
    write(root / "bench-serve.command.json", json.dumps(cmd, indent=2) + "\n")
    write(root / "bench-serve.stdout", p.stdout)
    write(root / "bench-serve.stderr", p.stderr)
    assert_no_bad_patterns("bench-serve output", p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"bench-serve failed rc={p.returncode}")
    data = json.loads(out.read_text())
    reports = data if isinstance(data, list) else [data]
    required = {int(c) for c in cfg.get("concurrency_cells", [1, 4, 16, 32])}
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for report in reports:
        concurrency = int(report.get("concurrency") or 0)
        seen.add(concurrency)
        completed = sum(int(v) for v in report.get("completed_per_run", []))
        errored = sum(int(v) for v in report.get("errored_per_run", []))
        throughput = (report.get("output_throughput_tps") or {}).get("mean")
        output_source = report.get("output_token_count_source")
        if report.get("n_repeats") < 3:
            raise RuntimeError(f"c={concurrency}: n_repeats < 3")
        if errored != 0 or completed <= 0:
            raise RuntimeError(f"c={concurrency}: completed={completed} errored={errored}")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            raise RuntimeError(f"c={concurrency}: invalid throughput={throughput}")
        if output_source != "usage":
            raise RuntimeError(f"c={concurrency}: output_token_count_source={output_source!r}")
        quality_counts = validate_bench_quality(report, label=f"c={concurrency}")
        rows.append(
            {
                "concurrency": concurrency,
                "completed": completed,
                "errored": errored,
                "throughput_tok_s": throughput,
                "output_token_count_source": output_source,
                **quality_counts,
            }
        )
    missing = sorted(required - seen)
    if missing:
        raise RuntimeError(f"bench missing concurrency cells: {missing}")
    return {"passed": True, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--validate-artifact", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ferrum-bin", type=Path, default=Path("./target/release/ferrum"))
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("G0 CUDA LLAMA DENSE ARTIFACT SELFTEST PASS")
        return 0
    if args.validate_artifact is not None:
        try:
            validate_artifact(args.validate_artifact)
        except Exception as error:
            print(f"G0 CUDA LLAMA DENSE ARTIFACT FAIL: {error}", file=sys.stderr)
            return 1
        print(f"G0 CUDA LLAMA DENSE ARTIFACT PASS: {args.validate_artifact}")
        return 0
    if args.config is None or args.out is None:
        ap.error("--config and --out are required unless --validate-artifact is used")

    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads(args.config.read_text())
    model = args.model or cfg["model"]
    tokenizer_cfg = args.tokenizer or cfg.get("tokenizer", "auto")
    port = args.port or int(cfg.get("port", 19300))
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    ferrum_bin = args.ferrum_bin

    checks: dict[str, Any] = {}
    proc: subprocess.Popen[str] | None = None
    serve_log = root / "serve.log"
    try:
        metadata = {
            "model": model,
            "model_architecture": cfg.get("model_architecture", "llama_dense"),
            "git_sha": run(["git", "rev-parse", "HEAD"], cwd=repo, timeout=30).stdout.strip(),
            "git_dirty": bool(run(["git", "status", "--short"], cwd=repo, timeout=30).stdout.strip()),
            "binary_sha256": sha256(ferrum_bin),
            "config": cfg,
        }
        write(root / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")

        checks["run"] = run_cli_gate(root, ferrum_bin, model, repo)

        with serve_log.open("w", encoding="utf-8") as log:
            serve_cmd = [
                str(ferrum_bin),
                "serve",
                "--model",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--effective-config-json",
                str(root / "serve.effective_config.json"),
                "--decision-trace-jsonl",
                str(root / "serve.decision_trace.jsonl"),
                "--disable-thinking",
            ]
            if cfg.get("max_model_len") is not None:
                serve_cmd.extend(["--max-model-len", str(cfg["max_model_len"])])
            if cfg.get("kv_capacity") is not None:
                serve_cmd.extend(["--kv-capacity", str(cfg["kv_capacity"])])
            if cfg.get("max_num_seqs") is not None:
                serve_cmd.extend(["--max-num-seqs", str(cfg["max_num_seqs"])])
            write(root / "serve.command.json", json.dumps(serve_cmd, indent=2) + "\n")
            proc = subprocess.Popen(
                serve_cmd,
                cwd=repo,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
        wait_health(port)
        tokenizer_dir = resolve_tokenizer(tokenizer_cfg, model)
        checks["serve_health"] = capture_health(root, port)
        checks["serve_capacity"] = validate_serve_capacity(root, cfg)
        checks["serve"] = serve_correctness(root, model, port)
        checks["tool_call_regression"] = run_tool_call_regression(
            f"http://127.0.0.1:{port}",
            model,
            root / "tool-call-regression",
            enable_thinking=False,
        )
        checks["concurrency_quality_regression"] = run_concurrency_quality_regression(
            f"http://127.0.0.1:{port}",
            model,
            root / "concurrency-quality-regression",
            [int(c) for c in cfg.get("concurrency_cells", [1, 4, 16, 32])],
            enable_thinking=False,
        )
        checks["bench_serve"] = run_bench_gate(root, ferrum_bin, model, tokenizer_dir, port, cfg, repo)

        if proc is not None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            proc = None
        log_text = serve_log.read_text(errors="replace") if serve_log.exists() else ""
        assert_no_bad_patterns(str(serve_log), log_text)
        gate = {
            "status": "pass",
            "lane": "g0_cuda4090_llama_dense",
            "model": model,
            "checks": checks,
        }
        write(root / "gate.json", json.dumps(gate, ensure_ascii=False, indent=2) + "\n")
        print(f"G0 CUDA LLAMA DENSE GATE PASS: {root}")
        return 0
    except Exception as e:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=10)
        write(
            root / "gate.json",
            json.dumps(
                {
                    "status": "fail",
                    "lane": "g0_cuda4090_llama_dense",
                    "model": model,
                    "error": str(e),
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        print(f"G0 CUDA LLAMA DENSE GATE FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
