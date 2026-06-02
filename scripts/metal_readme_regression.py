#!/usr/bin/env python3
"""Metal README regression runner.

Runs the README Apple Silicon rows with content correctness, multi-turn,
concurrency throughput, and swap-state evidence. Unlike the old
docs/bench/macos-2026-05-02 shell scripts, this runner does not stop after the
first failed correctness check; it records the failure and continues collecting
performance evidence for the remaining rows.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cell:
    concurrency: int
    prompts: int
    baseline_tps: float


@dataclass(frozen=True)
class ModelCase:
    key: str
    label: str
    gguf: str
    moe: bool
    cells: tuple[Cell, ...]


CASES = (
    ModelCase(
        key="llama31_8b",
        label="Llama-3.1-8B",
        gguf="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        moe=False,
        cells=(
            Cell(concurrency=1, prompts=8, baseline_tps=29.1),
            Cell(concurrency=8, prompts=24, baseline_tps=51.3),
            Cell(concurrency=16, prompts=32, baseline_tps=96.7),
        ),
    ),
    ModelCase(
        key="qwen3_8b",
        label="Qwen3-8B",
        gguf="Qwen3-8B-Q4_K_M.gguf",
        moe=False,
        cells=(Cell(concurrency=16, prompts=32, baseline_tps=93.2),),
    ),
    ModelCase(
        key="qwen3_30b_a3b",
        label="Qwen3-30B-A3B",
        gguf="Qwen3-30B-A3B-Q4_K_M.gguf",
        moe=True,
        cells=(Cell(concurrency=16, prompts=32, baseline_tps=72.5),),
    ),
)


def run_text(cmd: list[str], *, timeout: int = 30) -> str:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    ).stdout


def swapusage() -> str:
    return run_text(["sysctl", "vm.swapusage"], timeout=5).strip()


def vm_stat_head() -> str:
    out = run_text(["vm_stat"], timeout=5)
    return "\n".join(out.splitlines()[:12])


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def get_url(url: str, timeout: int = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def start_server(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    port: int,
    out_dir: Path,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(model_env(case))
    stdout = (out_dir / f"{case.key}.server.stdout").open("w", encoding="utf-8")
    stderr = (out_dir / f"{case.key}.server.stderr").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(ferrum_bin),
            "serve",
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    return proc


def model_env(case: ModelCase) -> dict[str, str]:
    env = {
        "FERRUM_METAL_PAGED_KV": "1",
        "FERRUM_PAGED_MAX_SEQS": "32",
        "FERRUM_KV_CAPACITY": "512",
        "FERRUM_MAX_BATCH": "16",
        "FERRUM_GREEDY_ARGMAX": "1",
    }
    if case.moe:
        env.update(
            {
                "FERRUM_MAX_BATCHED_TOKENS": "3072",
                "FERRUM_MOE_BATCHED": "1",
                "FERRUM_MOE_BATCHED_DECODE": "1",
                "FERRUM_MOE_BATCH_THRESHOLD": "2",
            }
        )
    return env


def run_cli_check(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    out_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    prefix = f"{case.key}.run"
    env = os.environ.copy()
    env.update(model_env(case))
    input_text = "\n".join(
        [
            "请记住暗号是蓝色月亮。只回答：已记住。",
            "刚才暗号是什么？只回答暗号。",
            "/bye",
            "",
        ]
    )
    cmd = [
        str(ferrum_bin),
        "run",
        str(model_path),
        "--backend",
        "metal",
        "--max-tokens",
        "64",
        "--system",
        "请用中文回答，不要输出代码。",
        "--output-format",
        "jsonl",
    ]
    result: dict[str, Any] = {"passed": False, "rc": None, "assistant": []}
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        write(out_dir / f"{prefix}.stdout", stdout)
        write(out_dir / f"{prefix}.stderr", stderr)
        result.update({"rc": "timeout", "error": f"run timed out after {timeout_sec}s"})
        return result

    write(out_dir / f"{prefix}.stdin", input_text)
    write(out_dir / f"{prefix}.stdout", proc.stdout)
    write(out_dir / f"{prefix}.stderr", proc.stderr)
    result["rc"] = proc.returncode

    assistant: list[str] = []
    for line in proc.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "assistant":
            assistant.append(str(row.get("content", "")))
    stderr_lower = proc.stderr.lower()
    joined = "\n".join(assistant).lower()
    no_abort = (
        "failed assertion" not in stderr_lower
        and "command encoder" not in stderr_lower
        and "panic" not in stderr_lower
    )
    first_ok = len(assistant) >= 1 and "记住" in assistant[0]
    second_ok = len(assistant) >= 2 and "蓝色月亮" in assistant[1]
    passed = proc.returncode == 0 and no_abort and first_ok and second_ok
    result.update(
        {
            "assistant": assistant,
            "no_abort": no_abort,
            "first_turn_ok": first_ok,
            "second_turn_ok": second_ok,
            "contains_secret": "蓝色月亮" in "\n".join(assistant),
            "passed": passed,
        }
    )
    write(out_dir / f"{prefix}.verdict.txt", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def wait_ready(port: int, proc: subprocess.Popen[str], out_path: Path) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + 180
    last_status = 0
    last_body = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            write(out_path, f"server exited rc={proc.returncode}\n{last_body}\n")
            return False
        last_status, last_body = get_url(url)
        if last_status == 200:
            write(out_path, last_body)
            return True
        time.sleep(1)
    write(out_path, f"timeout status={last_status}\n{last_body}\n")
    return False


def chat_check(port: int, case: ModelCase, out_dir: Path) -> dict[str, Any]:
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    result: dict[str, Any] = {}
    paris_payload = {
        "model": case.label,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Reply with just the city name.",
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    status, body = post_json(base, paris_payload)
    write(out_dir / f"{case.key}.paris_payload.json", json.dumps(paris_payload, indent=2))
    write(out_dir / f"{case.key}.paris_response.json", body)
    content = ""
    try:
        content = json.loads(body)["choices"][0]["message"]["content"]
    except Exception:
        content = body[:300]
    paris_pass = status == 200 and "paris" in content.lower()
    result["paris"] = {
        "status": status,
        "passed": paris_pass,
        "content": content,
    }
    write(
        out_dir / f"{case.key}.paris_verdict.txt",
        f"content={content}\npassed={str(paris_pass).lower()}\n",
    )

    multiturn_payload = {
        "model": case.label,
        "messages": [
            {
                "role": "system",
                "content": "请用中文回答，不要输出代码。",
            },
            {
                "role": "user",
                "content": "请记住暗号是蓝色月亮。只回答：已记住。",
            },
            {"role": "assistant", "content": "已记住。"},
            {
                "role": "user",
                "content": "刚才暗号是什么？只回答暗号。",
            },
        ],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
    }
    status, body = post_json(base, multiturn_payload)
    write(
        out_dir / f"{case.key}.multiturn_payload.json",
        json.dumps(multiturn_payload, indent=2),
    )
    write(out_dir / f"{case.key}.multiturn_response.json", body)
    content = ""
    try:
        content = json.loads(body)["choices"][0]["message"]["content"]
    except Exception:
        content = body[:300]
    multiturn_pass = status == 200 and "蓝色月亮" in content
    result["multiturn"] = {
        "status": status,
        "passed": multiturn_pass,
        "content": content,
    }
    write(
        out_dir / f"{case.key}.multiturn_verdict.txt",
        f"content={content}\npassed={str(multiturn_pass).lower()}\n",
    )
    return result


def run_bench_cell(
    bench_py: Path,
    port: int,
    case: ModelCase,
    cell: Cell,
    out_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    prefix = f"{case.key}.c{cell.concurrency}"
    result_file = out_dir / f"{prefix}.json"
    log_file = out_dir / f"{prefix}.bench.log"
    write(out_dir / f"{prefix}.swap_before.txt", swapusage() + "\n")
    cmd = [
        sys.executable,
        str(bench_py),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        case.label,
        "--num-prompts",
        str(cell.prompts),
        "--max-concurrency",
        str(cell.concurrency),
        "--max-tokens",
        "64",
        "--ignore-eos",
        "--deterministic-prompts",
        "--result-file",
        str(result_file),
    ]
    row: dict[str, Any] = {
        "concurrency": cell.concurrency,
        "prompts": cell.prompts,
        "baseline_tps": cell.baseline_tps,
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
        write(log_file, proc.stdout)
        row["runner_rc"] = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        write(
            log_file,
            stdout
            + f"\n[metal-readme-regression] timed out after {timeout_sec}s\n",
        )
        row.update(
            {
                "runner_rc": "timeout",
                "error": f"bench timed out after {timeout_sec}s",
                "completed": None,
                "failed": None,
                "output_throughput_tok_s": None,
                "ratio_to_readme": None,
                "not_regressed_90pct": False,
            }
        )
        return row
    finally:
        write(out_dir / f"{prefix}.swap_after.txt", swapusage() + "\n")
    if result_file.is_file():
        data = json.loads(result_file.read_text(encoding="utf-8"))
        tps = data.get("output_throughput_tok_s")
        completed = data.get("completed")
        failed = data.get("failed")
        ratio = tps / cell.baseline_tps if isinstance(tps, (int, float)) else None
        row.update(
            {
                "completed": completed,
                "failed": failed,
                "output_throughput_tok_s": tps,
                "ratio_to_readme": ratio,
                "not_regressed_90pct": bool(
                    isinstance(ratio, float)
                    and ratio >= 0.90
                    and completed == cell.prompts
                    and failed == 0
                ),
                "ttft_median_ms": data.get("ttft_ms", {}).get("median"),
                "tpot_median_ms": data.get("tpot_ms", {}).get("median"),
            }
        )
    return row


def markdown_summary(report: dict[str, Any]) -> str:
    out = ["# Metal README Regression - 2026-06-02", ""]
    out.append("Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.")
    out.append("")
    out.append(f"Ferrum: `{report['ferrum_version'].strip()}`")
    out.append(f"Swap at start: `{report['swap_start']}`")
    out.append(f"Swap at end: `{report['swap_end']}`")
    out.append("")
    out.append("| Model | Serve correctness | Serve multi-turn | Run REPL multi-turn | c | README tok/s | Current tok/s | Ratio | Completed | Gate |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for model in report["models"]:
        correctness = "pass" if model["chat"]["paris"]["passed"] else "FAIL"
        multiturn = "pass" if model["chat"]["multiturn"]["passed"] else "FAIL"
        run_gate = "pass" if model.get("run", {}).get("passed") else "FAIL"
        for cell in model["cells"]:
            tps = cell.get("output_throughput_tok_s")
            ratio = cell.get("ratio_to_readme")
            gate = "pass" if cell.get("not_regressed_90pct") else "FAIL"
            completed = f"{cell.get('completed')}/{cell.get('prompts')}"
            if isinstance(tps, (int, float)):
                out.append(
                    f"| {model['label']} | {correctness} | {multiturn} | {run_gate} | {cell['concurrency']} | "
                    f"{cell['baseline_tps']:.1f} | {tps:.1f} | {ratio:.3f} | {completed} | {gate} |"
                )
            else:
                out.append(
                    f"| {model['label']} | {correctness} | {multiturn} | {run_gate} | {cell['concurrency']} | "
                    f"{cell['baseline_tps']:.1f} | n/a | n/a | {completed} | {gate} |"
                )
    out.append("")
    out.append("Correctness prompts:")
    for model in report["models"]:
        out.append(
            f"- `{model['label']}` Paris: `{model['chat']['paris']['content']}`; "
            f"serve multi-turn: `{model['chat']['multiturn']['content']}`; "
            f"run multi-turn: `{model.get('run', {}).get('assistant', [])}`"
        )
    out.append("")
    out.append("Notes:")
    out.append("- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.")
    out.append("- Throughput cells request `ignore_eos=true` and streaming usage so runs are max-token and token-count comparable to the README harness.")
    out.append("- This runner records correctness failures and still collects performance data.")
    out.append("- Active swap means results are release-regression evidence, not clean marketing numbers.")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ferrum-bin", type=Path, default=ROOT / "target/release/ferrum")
    parser.add_argument("--models-dir", type=Path, default=Path("/Users/chejinxuan/ferrum-bench/models"))
    parser.add_argument("--bench-py", type=Path, default=ROOT / "bench/scripts/bench_serving.py")
    parser.add_argument("--port", type=int, default=18181)
    parser.add_argument("--bench-timeout-sec", type=int, default=300)
    parser.add_argument("--run-timeout-sec", type=int, default=300)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only matching model key(s), e.g. --only qwen3_30b_a3b. May be repeated.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ferrum_version": run_text([str(args.ferrum_bin), "--version"], timeout=10),
        "swap_start": swapusage(),
        "vm_stat_start": vm_stat_head(),
        "models": [],
    }

    selected = [case for case in CASES if not args.only or case.key in set(args.only)]
    for case in selected:
        model_path = args.models_dir / case.gguf
        model_report: dict[str, Any] = {
            "key": case.key,
            "label": case.label,
            "gguf": str(model_path),
            "moe": case.moe,
            "chat": {},
            "run": {},
            "cells": [],
        }
        proc: subprocess.Popen[str] | None = None
        try:
            write(args.out / f"{case.key}.swap_before_server.txt", swapusage() + "\n")
            proc = start_server(args.ferrum_bin, model_path, case, args.port, args.out)
            ready = wait_ready(args.port, proc, args.out / f"{case.key}.models.json")
            model_report["server_ready"] = ready
            if ready:
                model_report["chat"] = chat_check(args.port, case, args.out)
                for cell in case.cells:
                    model_report["cells"].append(
                        run_bench_cell(
                            args.bench_py,
                            args.port,
                            case,
                            cell,
                            args.out,
                            args.bench_timeout_sec,
                        )
                    )
            else:
                model_report["chat"] = {
                    "paris": {"passed": False, "content": "server not ready"},
                    "multiturn": {"passed": False, "content": "server not ready"},
                }
        except Exception as e:
            model_report["error"] = repr(e)
            if not model_report["chat"]:
                model_report["chat"] = {
                    "paris": {"passed": False, "content": repr(e)},
                    "multiturn": {"passed": False, "content": repr(e)},
                }
        finally:
            stop_server(proc)
            write(args.out / f"{case.key}.swap_after_server.txt", swapusage() + "\n")
            time.sleep(8)
        model_report["run"] = run_cli_check(
            args.ferrum_bin,
            model_path,
            case,
            args.out,
            args.run_timeout_sec,
        )
        report["models"].append(model_report)
        write(args.out / "summary.json", json.dumps(report, indent=2, ensure_ascii=False))

    report["swap_end"] = swapusage()
    report["vm_stat_end"] = vm_stat_head()
    write(args.out / "summary.json", json.dumps(report, indent=2, ensure_ascii=False))
    write(args.out / "summary.md", markdown_summary(report))
    print(args.out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
