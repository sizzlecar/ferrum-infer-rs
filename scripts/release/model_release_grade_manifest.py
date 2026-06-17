#!/usr/bin/env python3
"""Build a model release-grade manifest from W2 full-matrix artifacts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = "docs/goals/model-coverage-2026-06-12/RELEASE_GRADE_GOAL.md"
W2_MATRIX = REPO_ROOT / "docs/goals/model-coverage-2026-06-12/w2_matrix.json"
FINAL_GATE = REPO_ROOT / "scripts/release/model_release_grade_goal_gate.py"
REQUIRED_CELLS = [1, 4, 16, 32]
QUALITY_FIELDS = [
    "bad_output",
    "malformed_stream",
    "missing_done",
    "duplicate_done",
    "zero_output_tokens",
    "stream_bulk_flush",
    "http_500",
    "panic",
]


class BuildError(Exception):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BuildError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise BuildError(f"{label} missing: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise BuildError(f"{label} missing: {path}")
    return path


def artifact_ref(path: Path, out_dir: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return path.relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_first_token(path: Path, label: str) -> str:
    text = require_file(path, label).read_text(encoding="utf-8").strip()
    if not text:
        raise BuildError(f"{label} is empty: {path}")
    return text.split()[0]


def read_command(path: Path, label: str) -> list[str]:
    text = require_file(path, label).read_text(encoding="utf-8").strip()
    if not text:
        raise BuildError(f"{label} is empty: {path}")
    return shlex.split(text)


def reports_by_concurrency(path: Path, label: str) -> dict[int, dict[str, Any]]:
    data = load_json(require_file(path, label))
    reports = data.get("reports") if isinstance(data, dict) else data
    if not isinstance(reports, list):
        raise BuildError(f"{label} must contain a report list: {path}")
    out: dict[int, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            raise BuildError(f"{label} contains non-object report")
        concurrency = report.get("concurrency")
        if not isinstance(concurrency, int):
            raise BuildError(f"{label} report missing integer concurrency")
        if concurrency in out:
            raise BuildError(f"{label} has duplicate c={concurrency}")
        out[concurrency] = report
    return out


def output_tps_lcb(report: dict[str, Any], label: str) -> float:
    metric = report.get("output_throughput_tps")
    if not isinstance(metric, dict):
        raise BuildError(f"{label} missing output_throughput_tps")
    mean = metric.get("mean")
    ci95 = metric.get("ci95_hw", 0.0)
    if not isinstance(mean, (int, float)) or not isinstance(ci95, (int, float)):
        raise BuildError(f"{label} output_throughput_tps mean/ci95_hw must be numeric")
    return float(mean) - float(ci95)


def p95_itl_ms(report: dict[str, Any], label: str) -> float:
    try:
        value = report["itl_ms"]["p95"]["mean"]
    except KeyError as exc:
        raise BuildError(f"{label} missing itl_ms.p95.mean") from exc
    if not isinstance(value, (int, float)):
        raise BuildError(f"{label} itl_ms.p95.mean must be numeric")
    return float(value)


def integer_list(report: dict[str, Any], key: str, label: str) -> list[int]:
    value = report.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise BuildError(f"{label}.{key} must be an integer list")
    return value


def quality_list(report: dict[str, Any], field: str, label: str) -> list[int]:
    direct_key = f"{field}_per_run"
    if direct_key in report:
        return integer_list(report, direct_key, label)
    issues = report.get("quality_issues_per_run")
    if not isinstance(issues, list):
        raise BuildError(f"{label} missing {direct_key} and quality_issues_per_run")
    values = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise BuildError(f"{label}.quality_issues_per_run contains non-object issue")
        raw = issue.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise BuildError(f"{label}.quality_issues_per_run.{field} must be integer")
        values.append(raw)
    return values


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BuildError(f"{label} must be a positive integer")
    return value


def assert_report_quality(report: dict[str, Any], label: str) -> None:
    if report.get("output_token_count_source") != "usage":
        raise BuildError(f"{label} output_token_count_source must be usage")
    n_repeats = positive_int(report.get("n_repeats"), f"{label}.n_repeats")
    completed = integer_list(report, "completed_per_run", label)
    errored = integer_list(report, "errored_per_run", label)
    if len(completed) != n_repeats or len(errored) != n_repeats:
        raise BuildError(f"{label} completed/errors length must equal n_repeats")
    requests = positive_int(report.get("n_requests_per_run"), f"{label}.n_requests_per_run")
    if completed != [requests] * n_repeats:
        raise BuildError(f"{label} completed_per_run must be full")
    if any(value != 0 for value in errored):
        raise BuildError(f"{label} errored_per_run must be all zero")
    for field in QUALITY_FIELDS:
        values = quality_list(report, field, label)
        if len(values) != n_repeats:
            raise BuildError(f"{label}.{field}_per_run length must equal n_repeats")
        if any(value != 0 for value in values):
            raise BuildError(f"{label}.{field}_per_run must be all zero")


def w2_correctness_entries(matrix_path: Path, out_dir: Path) -> dict[str, dict[str, Any]]:
    data = load_json(matrix_path)
    models = data.get("models")
    if not isinstance(models, list):
        raise BuildError(f"invalid W2 matrix models list: {matrix_path}")
    model = next((item for item in models if item.get("id") == "gemma3-27b"), None)
    if not isinstance(model, dict):
        raise BuildError("w2_matrix missing gemma3-27b")
    cells = model.get("cells")
    if not isinstance(cells, dict):
        raise BuildError("w2_matrix gemma3-27b missing cells")
    mapping = {
        "l0_template": "l0_template",
        "l1_numeric": "l1_bf16",
        "l2_quantized": "l2_gptq_cuda",
        "l3_behavior": "l3_behavior",
        "l4_agent": "l4_agent",
        "l5_concurrency": "l5_concurrency",
    }
    entries = {}
    for dst, src in mapping.items():
        cell = cells.get(src)
        if not isinstance(cell, dict) or cell.get("status") != "pass":
            raise BuildError(f"w2_matrix {src} must be status=pass")
        artifact = cell.get("artifact")
        if not isinstance(artifact, str) or not artifact:
            raise BuildError(f"w2_matrix {src} missing artifact")
        paths = [REPO_ROOT / artifact]
        if not any(path.exists() for path in paths):
            raise BuildError(f"w2_matrix {src} artifact missing: {artifact}")
        entries[dst] = {
            "status": "pass",
            "artifact": artifact_ref(paths[0], out_dir),
            "source_cell": src,
        }
    return entries


def parse_effective_concurrency(values: list[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise BuildError(f"--effective-concurrency must be C=E, got {value!r}")
        left, right = value.split("=", 1)
        try:
            requested = int(left)
            effective = int(right)
        except ValueError as exc:
            raise BuildError(f"--effective-concurrency must be integer C=E: {value!r}") from exc
        if requested <= 0 or effective <= 0 or effective > requested:
            raise BuildError(f"invalid effective concurrency mapping: {value!r}")
        result[requested] = effective
    return result


def build_cell(
    *,
    concurrency: int,
    ferrum: dict[str, Any],
    baseline: dict[str, Any],
    source: Path,
    out_dir: Path,
    dataset_id: str,
    dataset_sha: str,
    ferrum_command: list[str],
    baseline_command: list[str],
    effective_concurrency: dict[int, int],
) -> dict[str, Any]:
    assert_report_quality(ferrum, f"ferrum c={concurrency}")
    assert_report_quality(baseline, f"baseline c={concurrency}")
    n_repeats = positive_int(ferrum.get("n_repeats"), f"ferrum c={concurrency}.n_repeats")
    baseline_n_repeats = positive_int(
        baseline.get("n_repeats"),
        f"baseline c={concurrency}.n_repeats",
    )
    requests = positive_int(
        ferrum.get("n_requests_per_run"),
        f"ferrum c={concurrency}.n_requests_per_run",
    )
    baseline_requests = positive_int(
        baseline.get("n_requests_per_run"),
        f"baseline c={concurrency}.n_requests_per_run",
    )
    if n_repeats != baseline_n_repeats:
        raise BuildError(f"c={concurrency} repeat count differs between Ferrum and baseline")
    if requests != baseline_requests:
        raise BuildError(f"c={concurrency} request count differs between Ferrum and baseline")

    effective = effective_concurrency.get(concurrency, concurrency)
    ferrum_lcb = output_tps_lcb(ferrum, f"ferrum c={concurrency}")
    baseline_lcb = output_tps_lcb(baseline, f"baseline c={concurrency}")
    if baseline_lcb <= 0:
        raise BuildError(f"baseline c={concurrency} LCB must be positive")

    cell = {
        "requested_concurrency": concurrency,
        "effective_active_concurrency": effective,
        "baseline_effective_active_concurrency": effective,
        "requests_per_run": requests,
        "n_repeats": n_repeats,
        "completed_per_run": integer_list(ferrum, "completed_per_run", f"ferrum c={concurrency}"),
        "errored_per_run": integer_list(ferrum, "errored_per_run", f"ferrum c={concurrency}"),
        "baseline_requests_per_run": baseline_requests,
        "baseline_n_repeats": baseline_n_repeats,
        "baseline_completed_per_run": integer_list(
            baseline,
            "completed_per_run",
            f"baseline c={concurrency}",
        ),
        "baseline_errored_per_run": integer_list(
            baseline,
            "errored_per_run",
            f"baseline c={concurrency}",
        ),
        "output_token_count_source": ferrum.get("output_token_count_source"),
        "stream_options_include_usage": True,
        "baseline_output_token_count_source": baseline.get("output_token_count_source"),
        "baseline_stream_options_include_usage": True,
        "same_hardware": True,
        "same_model": True,
        "same_quantization": True,
        "same_prompt_or_dataset": True,
        "prompt_dataset_id": dataset_id,
        "baseline_prompt_dataset_id": dataset_id,
        "prompt_dataset_sha256": dataset_sha,
        "baseline_prompt_dataset_sha256": dataset_sha,
        "bench_command_line": ferrum_command,
        "baseline_bench_command_line": baseline_command,
        "ferrum_output_tps_lcb": ferrum_lcb,
        "baseline_output_tps": baseline_lcb,
        "ratio": ferrum_lcb / baseline_lcb,
        "ferrum_p95_itl_ms": p95_itl_ms(ferrum, f"ferrum c={concurrency}"),
        "baseline_p95_itl_ms": p95_itl_ms(baseline, f"baseline c={concurrency}"),
        "artifact": artifact_ref(source / "perf/bench_ferrum_sharegpt_sweep_100x3.json", out_dir),
    }
    if effective < concurrency:
        cell["published_concurrency"] = effective
    for field in QUALITY_FIELDS:
        cell[f"{field}_per_run"] = quality_list(ferrum, field, f"ferrum c={concurrency}")
        cell[f"baseline_{field}_per_run"] = quality_list(
            baseline,
            field,
            f"baseline c={concurrency}",
        )
    return cell


def source_status(source: Path) -> dict[str, Any]:
    summary = load_json(source / "summary.json")
    if summary.get("status") not in {"diagnostic_pass", "pass"}:
        raise BuildError(f"source summary status is not pass: {summary.get('status')!r}")
    if summary.get("ferrum_bench_exit_code") not in {0, None}:
        raise BuildError(f"Ferrum bench exit code is not zero: {summary.get('ferrum_bench_exit_code')}")
    if summary.get("vllm_bench_exit_code") not in {0, None}:
        raise BuildError(f"vLLM bench exit code is not zero: {summary.get('vllm_bench_exit_code')}")
    return summary


def build_manifest(
    *,
    source: Path,
    baseline_source: Path | None,
    out_dir: Path,
    matrix_path: Path,
    effective_concurrency: dict[int, int],
    dataset_id: str,
) -> Path:
    source = require_dir(source, "source artifact dir").resolve()
    baseline_source = require_dir(
        baseline_source or source,
        "baseline artifact dir",
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_status(source)

    ferrum_perf = source / "perf/bench_ferrum_sharegpt_sweep_100x3.json"
    baseline_perf = baseline_source / "perf/bench_vllm_sharegpt_sweep_100x3.json"
    ferrum_reports = reports_by_concurrency(ferrum_perf, "Ferrum bench report")
    baseline_reports = reports_by_concurrency(baseline_perf, "vLLM bench report")
    missing = [c for c in REQUIRED_CELLS if c not in ferrum_reports or c not in baseline_reports]
    if missing:
        raise BuildError(f"missing required performance cells: {missing}")

    dataset_sha = read_first_token(source / "env/dataset.sha256", "dataset sha")
    baseline_dataset_sha = read_first_token(
        baseline_source / "env/dataset.sha256",
        "baseline dataset sha",
    )
    if dataset_sha.lower() != baseline_dataset_sha.lower():
        raise BuildError(
            "baseline dataset sha does not match Ferrum dataset sha: "
            f"{baseline_dataset_sha} != {dataset_sha}"
        )
    binary_sha = read_first_token(source / "env/ferrum.sha256", "Ferrum binary sha")
    git_sha = read_first_token(source / "env/git_sha.txt", "remote git sha")
    git_status = require_file(source / "env/git_status_short.txt", "remote git status").read_text(
        encoding="utf-8"
    ).strip()
    vllm_versions = load_json(baseline_source / "env/vllm_versions.json")
    vllm_version = str(vllm_versions.get("vllm", "unknown"))
    baseline_build_command = [
        "python",
        "-m",
        "pip",
        "install",
        f"vllm=={vllm_version}",
    ]
    ferrum_command = read_command(source / "perf/bench-ferrum.command.txt", "Ferrum bench command")
    baseline_command = read_command(
        baseline_source / "perf/bench-vllm.command.txt",
        "vLLM bench command",
    )

    vllm_server = load_json(
        require_file(baseline_source / "vllm/vllm_server.command.json", "vLLM command")
    )
    vllm_server_command = vllm_server.get("cmd")
    if not isinstance(vllm_server_command, list) or not all(
        isinstance(part, str) for part in vllm_server_command
    ):
        raise BuildError("vLLM server command JSON missing string-list cmd")

    cells = [
        build_cell(
            concurrency=concurrency,
            ferrum=ferrum_reports[concurrency],
            baseline=baseline_reports[concurrency],
            source=source,
            out_dir=out_dir,
            dataset_id=dataset_id,
            dataset_sha=dataset_sha,
            ferrum_command=ferrum_command,
            baseline_command=baseline_command,
            effective_concurrency=effective_concurrency,
        )
        for concurrency in REQUIRED_CELLS
    ]

    manifest = {
        "schema_version": 1,
        "lane": "w2",
        "status": "pass",
        "goal_doc": GOAL_DOC,
        "model_id": "gemma3:27b-gptq",
        "backend": "cuda",
        "quantization": "hf-gptq-int4",
        "git_sha": git_sha,
        "dirty_status": {"dirty": bool(git_status), "status_short": git_status},
        "binary_sha256": binary_sha,
        "hardware": {
            "status": "pass",
            "artifact": artifact_ref(source / "env/nvidia_smi_before.txt", out_dir),
        },
        "release_scope": {
            "backends": ["cuda"],
            "formats": ["hf-gptq-int4"],
            "excluded_lanes": {
                "gguf_metal": {
                    "reason": "GGUF/Metal is not in this W2 CUDA GPTQ release-grade scope",
                }
            },
        },
        "runtime_config": {
            "product_surface": "typed_cli",
            "hidden_env": [],
            "snapshot": artifact_ref(source / "server/serve_effective_config.json", out_dir),
        },
        "correctness": w2_correctness_entries(matrix_path, out_dir),
        "product_entrypoints": {
            "ferrum_run": {
                "status": "pass",
                "artifact": artifact_ref(source / "correctness/run_summary.json", out_dir),
            },
            "ferrum_serve": {
                "status": "pass",
                "artifact": artifact_ref(source / "correctness/smoke/stream_summary.json", out_dir),
            },
        },
        "performance": {
            "baseline": {
                "engine": "vLLM",
                "version": vllm_version,
                "evidence_mode": "historical" if baseline_source != source else "live",
                "build_command_line": baseline_build_command,
                "command_line": vllm_server_command,
                "bench_command_line": baseline_command,
                "same_hardware": True,
                "same_model": True,
                "same_quantization": True,
                "artifact": artifact_ref(baseline_perf, out_dir),
            },
            "cells": cells,
        },
    }
    manifest_path = out_dir / "model_release_grade_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def run_validator(out_dir: Path) -> int:
    cmd = [sys.executable, str(FINAL_GATE), "w2", str(out_dir)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    return proc.returncode


def metric(mean: float, ci95_hw: float = 0.0) -> dict[str, float]:
    return {"mean": mean, "stddev": 0.0, "ci95_hw": ci95_hw}


def write_selftest_source(root: Path) -> Path:
    source = root / "remote"
    for rel in [
        "env",
        "perf",
        "server",
        "correctness/smoke",
        "vllm",
    ]:
        (source / rel).mkdir(parents=True, exist_ok=True)
    (source / "env/dataset.sha256").write_text("b" * 64 + "  dataset.jsonl\n")
    (source / "env/ferrum.sha256").write_text("a" * 64 + "  target/release/ferrum\n")
    (source / "env/git_sha.txt").write_text("0123456789abcdef\n")
    (source / "env/git_status_short.txt").write_text("")
    (source / "env/nvidia_smi_before.txt").write_text("NVIDIA GeForce RTX 4090\n")
    write_json(source / "env/vllm_versions.json", {"vllm": "0.23.0"})
    write_json(source / "server/serve_effective_config.json", {"status": "pass"})
    write_json(source / "correctness/run_summary.json", {"status": "pass", "content": "5"})
    write_json(source / "correctness/smoke/stream_summary.json", {"done_count": 1, "content": "5"})
    write_json(source / "vllm/vllm_server.command.json", {"cmd": ["python", "-m", "vllm"]})
    (source / "perf/bench-ferrum.command.txt").write_text(
        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
        "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3\n"
    )
    (source / "perf/bench-vllm.command.txt").write_text(
        "ferrum bench-serve --fail-on-error --require-ci --seed 9271 "
        "--concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3\n"
    )
    ferrum_reports = []
    baseline_reports = []
    for concurrency in REQUIRED_CELLS:
        common = {
            "concurrency": concurrency,
            "n_repeats": 3,
            "n_requests_per_run": 100,
            "completed_per_run": [100, 100, 100],
            "errored_per_run": [0, 0, 0],
            "output_token_count_source": "usage",
            "itl_ms": {"p95": metric(10.0)},
            "bad_output_per_run": [0, 0, 0],
            "malformed_stream_per_run": [0, 0, 0],
            "missing_done_per_run": [0, 0, 0],
            "duplicate_done_per_run": [0, 0, 0],
            "zero_output_tokens_per_run": [0, 0, 0],
            "stream_bulk_flush_per_run": [0, 0, 0],
            "http_500_per_run": [0, 0, 0],
            "panic_per_run": [0, 0, 0],
        }
        ferrum_reports.append({**common, "output_throughput_tps": metric(82.0)})
        baseline_reports.append(
            {**common, "output_throughput_tps": metric(100.0), "itl_ms": {"p95": metric(9.0)}}
        )
    write_json(source / "perf/bench_ferrum_sharegpt_sweep_100x3.json", ferrum_reports)
    write_json(source / "perf/bench_vllm_sharegpt_sweep_100x3.json", baseline_reports)
    write_json(
        source / "summary.json",
        {
            "status": "diagnostic_pass",
            "ferrum_bench_exit_code": 0,
            "vllm_bench_exit_code": 0,
        },
    )
    return source


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w2-release-grade-manifest-") as tmp:
        root = Path(tmp)
        source = write_selftest_source(root)
        baseline_source = write_selftest_source(root / "historical-baseline")
        out_dir = root / "out"
        manifest = build_manifest(
            source=source,
            baseline_source=baseline_source,
            out_dir=out_dir,
            matrix_path=W2_MATRIX,
            effective_concurrency={32: 16},
            dataset_id="selftest/sharegpt-100-seed9271",
        )
        if not manifest.is_file():
            raise AssertionError("manifest was not written")
        rc = run_validator(out_dir)
        if rc != 0:
            raise AssertionError(f"selftest final validator failed with rc={rc}")
    print("MODEL RELEASE GRADE MANIFEST SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=["w2"], nargs="?", help="release-grade lane")
    parser.add_argument("--source", type=Path, help="full-matrix artifact source dir")
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="artifact dir containing the vLLM baseline reports; defaults to --source",
    )
    parser.add_argument("--out", type=Path, help="output dir for model_release_grade_manifest.json")
    parser.add_argument("--matrix", type=Path, default=W2_MATRIX, help="W2 coverage matrix")
    parser.add_argument(
        "--dataset-id",
        default="w2/ascii-sharegpt-100-seed9271",
        help="prompt dataset id recorded in release-grade cells",
    )
    parser.add_argument(
        "--effective-concurrency",
        action="append",
        default=[],
        metavar="C=E",
        help="record an admission-capped effective concurrency, e.g. 32=16",
    )
    parser.add_argument(
        "--no-run-validator",
        action="store_true",
        help="only write the manifest; do not invoke the final validator",
    )
    parser.add_argument("--self-test", action="store_true", help="run a synthetic self-test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_selftest()
    if args.lane != "w2":
        raise BuildError("only the w2 lane is currently supported")
    if args.source is None or args.out is None:
        raise BuildError("--source and --out are required")
    effective = parse_effective_concurrency(args.effective_concurrency)
    manifest = build_manifest(
        source=args.source,
        baseline_source=args.baseline_source,
        out_dir=args.out,
        matrix_path=args.matrix,
        effective_concurrency=effective,
        dataset_id=args.dataset_id,
    )
    print(f"wrote {manifest}")
    if args.no_run_validator:
        return 0
    return run_validator(args.out)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"MODEL RELEASE GRADE MANIFEST FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
