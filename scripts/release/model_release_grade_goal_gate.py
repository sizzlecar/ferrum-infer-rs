#!/usr/bin/env python3
"""Final validator for the model release-grade W2/W3 goal.

The coverage validators under scripts/w1_goal_validator.py and
scripts/w2_goal_validator.py prove "can run" coverage.  This gate proves the
stronger RELEASE_GRADE_GOAL.md contract: correctness artifacts first, both
product entrypoints, same-hardware baseline evidence, and >=80% performance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = REPO_ROOT / "docs/goals/model-coverage-2026-06-12/RELEASE_GRADE_GOAL.md"
MANIFEST_NAME = "model_release_grade_manifest.json"
RESULT_MANIFEST_NAME = "model_release_grade_goal_gate.manifest.json"
PASS_LINES = {
    "w2": "MODEL_RELEASE_GRADE_W2 PASS",
    "w3": "MODEL_RELEASE_GRADE_W3 PASS",
}
REQUIRED_CONCURRENCY = {1, 4, 16, 32}
REQUIRED_CORRECTNESS = {
    "l0_template",
    "l1_numeric",
    "l2_quantized",
    "l3_behavior",
    "l4_agent",
    "l5_concurrency",
}
REQUIRED_PRODUCT_ENTRYPOINTS = {"ferrum_run", "ferrum_serve"}
W3_REQUIRED_CORRECTNESS = {
    "w3_s0_design",
    "w3_s0_microbench",
    "w3_s1_single_layer",
    "w3_s2_whole_model_product_path",
}
REQUIRED_ZERO_RUN_COUNT_FIELDS = [
    "bad_output_per_run",
    "malformed_stream_per_run",
    "missing_done_per_run",
    "duplicate_done_per_run",
    "zero_output_tokens_per_run",
    "stream_bulk_flush_per_run",
    "http_500_per_run",
    "panic_per_run",
]
MIN_RATIO = 0.8
MAX_ITL_MULTIPLE = 1.25
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
DIRTY_CLEAN_STRINGS = {"", "clean", "false", "0", "no tracked changes"}


class ValidationError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_object(value: Any, label: str, problems: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        problems.append(f"{label} must be a JSON object")
        return {}
    return value


def as_list(value: Any, label: str, problems: list[str]) -> list[Any]:
    if not isinstance(value, list):
        problems.append(f"{label} must be a JSON list")
        return []
    return value


def number(value: Any, label: str, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{label} must be numeric")
        return None
    if not math.isfinite(float(value)):
        problems.append(f"{label} must be finite")
        return None
    return float(value)


def positive_int(value: Any, label: str, problems: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        problems.append(f"{label} must be a positive integer")
        return None
    return value


def require_true(value: Any, label: str, problems: list[str]) -> None:
    if value is not True:
        problems.append(f"{label} must be true")


def artifact_candidates(raw: str, out_dir: Path) -> list[Path]:
    path = Path(raw)
    if path.is_absolute():
        return [path]
    return [REPO_ROOT / path, out_dir / path]


def require_artifact(raw: Any, label: str, out_dir: Path, problems: list[str]) -> None:
    if not isinstance(raw, str) or not raw:
        problems.append(f"{label} artifact path must be a non-empty string")
        return
    if not any(candidate.exists() for candidate in artifact_candidates(raw, out_dir)):
        candidates = ", ".join(str(candidate) for candidate in artifact_candidates(raw, out_dir))
        problems.append(f"{label} artifact missing: {raw} (checked {candidates})")


def validate_evidence_entry(
    entry: Any,
    label: str,
    out_dir: Path,
    problems: list[str],
) -> None:
    if isinstance(entry, str):
        require_artifact(entry, label, out_dir, problems)
        return
    obj = as_object(entry, label, problems)
    if not obj:
        return
    status = obj.get("status", "pass")
    if status != "pass":
        problems.append(f"{label} status must be pass, got {status!r}")
    paths: list[Any] = []
    if "artifact" in obj:
        paths.append(obj["artifact"])
    if "path" in obj:
        paths.append(obj["path"])
    if "artifacts" in obj:
        artifacts = obj["artifacts"]
        if isinstance(artifacts, list):
            paths.extend(artifacts)
        else:
            problems.append(f"{label}.artifacts must be a list")
    if not paths:
        problems.append(f"{label} must reference at least one artifact")
    for idx, raw in enumerate(paths):
        require_artifact(raw, f"{label}[{idx}]", out_dir, problems)


def command_parts(value: Any, label: str, problems: list[str]) -> list[str]:
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return value
    problems.append(f"{label} must be a string list")
    return []


def has_flag(parts: list[str], flag: str) -> bool:
    return flag in parts


def flag_value(parts: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, part in enumerate(parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
        if part == flag and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def validate_bench_command(parts: list[str], n_repeats: int, label: str, problems: list[str]) -> None:
    if not any(part == "bench-serve" or part.endswith("/bench-serve") for part in parts):
        problems.append(f"{label} command must invoke ferrum bench-serve")
    for flag in ["--fail-on-error", "--require-ci"]:
        if not has_flag(parts, flag):
            problems.append(f"{label} command missing {flag}")
    if flag_value(parts, "--seed") != "9271":
        problems.append(f"{label} command must include --seed 9271")
    repeat_value = flag_value(parts, "--n-repeats")
    if repeat_value is None:
        problems.append(f"{label} command missing --n-repeats")
    elif repeat_value != str(n_repeats):
        problems.append(
            f"{label} command --n-repeats={repeat_value}, expected manifest n_repeats={n_repeats}"
        )
    for part in parts:
        if re.match(r"^FERRUM_[A-Z0-9_]+=", part):
            problems.append(f"{label} command uses hidden env override: {part.split('=', 1)[0]}")


def validate_release_scope(scope: dict[str, Any], lane: str, problems: list[str]) -> None:
    backends = scope.get("backends")
    formats = scope.get("formats")
    if not isinstance(backends, list) or not all(isinstance(item, str) for item in backends):
        problems.append("release_scope.backends must be a string list")
        backends = []
    if not isinstance(formats, list) or not all(isinstance(item, str) for item in formats):
        problems.append("release_scope.formats must be a string list")
        formats = []
    if lane == "w2":
        if "cuda" not in backends:
            problems.append("W2 release scope must include cuda")
        if not any("gptq" in item.lower() for item in formats):
            problems.append("W2 release scope must include a GPTQ format")
        if "metal" not in backends:
            excluded = as_object(scope.get("excluded_lanes"), "release_scope.excluded_lanes", problems)
            reason = ""
            for key in ["gguf_metal", "metal_gguf", "gguf/metal"]:
                value = excluded.get(key)
                if isinstance(value, dict):
                    reason = str(value.get("reason", ""))
                elif isinstance(value, str):
                    reason = value
                if reason:
                    break
            if not reason:
                problems.append(
                    "W2 CUDA-only scope must explicitly exclude GGUF/Metal with a reason"
                )


def validate_dirty_status(value: Any, problems: list[str]) -> None:
    if isinstance(value, dict):
        dirty = value.get("dirty")
        if dirty is True:
            problems.append("dirty_status.dirty must be false for release-grade evidence")
        files = value.get("files", value.get("dirty_files", []))
        if files:
            problems.append("dirty_status files must be empty for release-grade evidence")
        if dirty is None and not any(key in value for key in ["clean", "status_short"]):
            problems.append("dirty_status must explicitly record clean/dirty state")
        if value.get("clean") is False:
            problems.append("dirty_status.clean must not be false for release-grade evidence")
        status_short = value.get("status_short")
        if isinstance(status_short, str) and status_short.strip():
            problems.append("dirty_status.status_short must be empty for release-grade evidence")
        return
    if isinstance(value, list):
        if value:
            problems.append("dirty_status list must be empty for release-grade evidence")
        return
    if isinstance(value, str):
        if value.strip().lower() not in DIRTY_CLEAN_STRINGS:
            problems.append("dirty_status string must indicate a clean worktree")
        return
    problems.append("dirty_status must be a dict, list, or string")


def validate_runtime_config(runtime: dict[str, Any], out_dir: Path, problems: list[str]) -> None:
    product_surface = runtime.get("product_surface")
    allowed = {"typed_cli", "typed_config", "typed_defaults", "model_defaults"}
    if product_surface not in allowed:
        problems.append(
            "runtime_config.product_surface must be one of "
            + ", ".join(sorted(allowed))
        )
    hidden_env = runtime.get("hidden_env", [])
    if hidden_env:
        problems.append("runtime_config.hidden_env must be empty for release-grade evidence")
    snapshot = runtime.get("snapshot") or runtime.get("artifact")
    if snapshot is None:
        problems.append("runtime_config must include a snapshot artifact")
    else:
        require_artifact(snapshot, "runtime_config.snapshot", out_dir, problems)


def validate_top_level(manifest: dict[str, Any], lane: str, out_dir: Path, problems: list[str]) -> None:
    if manifest.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    if manifest.get("lane") != lane:
        problems.append(f"lane must be {lane!r}")
    if manifest.get("status") != "pass":
        problems.append("status must be pass")
    if manifest.get("goal_doc") != str(GOAL_DOC.relative_to(REPO_ROOT)):
        problems.append(
            "goal_doc must be "
            f"{GOAL_DOC.relative_to(REPO_ROOT)}"
        )
    if not GOAL_DOC.is_file():
        problems.append(f"goal doc is missing on disk: {GOAL_DOC}")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"]:
        problems.append("model_id must be a non-empty string")
    if not isinstance(manifest.get("backend"), str) or not manifest["backend"]:
        problems.append("backend must be a non-empty string")
    if not isinstance(manifest.get("quantization"), str) or not manifest["quantization"]:
        problems.append("quantization must be a non-empty string")
    git_sha = manifest.get("git_sha")
    if not isinstance(git_sha, str) or not re.match(r"^[0-9a-fA-F]{7,40}$", git_sha):
        problems.append("git_sha must be a 7-40 character hex string")
    validate_dirty_status(manifest.get("dirty_status"), problems)
    digest = manifest.get("binary_sha256")
    if not isinstance(digest, str) or not HEX64.match(digest):
        problems.append("binary_sha256 must be a 64-character hex digest")
    validate_evidence_entry(manifest.get("hardware"), "hardware", out_dir, problems)
    validate_release_scope(
        as_object(manifest.get("release_scope"), "release_scope", problems),
        lane,
        problems,
    )
    validate_runtime_config(
        as_object(manifest.get("runtime_config"), "runtime_config", problems),
        out_dir,
        problems,
    )


def validate_correctness(
    manifest: dict[str, Any],
    lane: str,
    out_dir: Path,
    problems: list[str],
) -> None:
    correctness = as_object(manifest.get("correctness"), "correctness", problems)
    required = set(REQUIRED_CORRECTNESS)
    if lane == "w3":
        required |= W3_REQUIRED_CORRECTNESS
    for key in sorted(required):
        if key not in correctness:
            problems.append(f"correctness missing {key}")
            continue
        validate_evidence_entry(correctness[key], f"correctness.{key}", out_dir, problems)
    product = as_object(manifest.get("product_entrypoints"), "product_entrypoints", problems)
    for key in sorted(REQUIRED_PRODUCT_ENTRYPOINTS):
        if key not in product:
            problems.append(f"product_entrypoints missing {key}")
            continue
        validate_evidence_entry(product[key], f"product_entrypoints.{key}", out_dir, problems)


def metric_from_cell(cell: dict[str, Any], problems: list[str], label: str) -> float | None:
    lcb = cell.get("ferrum_output_tps_lcb", cell.get("ferrum_throughput_lcb"))
    if lcb is not None:
        return number(lcb, f"{label}.ferrum_output_tps_lcb", problems)
    mean = cell.get("ferrum_output_tps_mean", cell.get("ferrum_throughput_mean"))
    value = number(mean, f"{label}.ferrum_output_tps_mean", problems)
    n_repeats = cell.get("n_repeats")
    if isinstance(n_repeats, int) and n_repeats < 5:
        problems.append(
            f"{label} uses mean throughput without lower CI bound; n_repeats must be >=5"
        )
    return value


def baseline_tps_from_cell(cell: dict[str, Any], baseline: dict[str, Any], problems: list[str], label: str) -> float | None:
    value = cell.get("baseline_output_tps", cell.get("baseline_throughput_tps"))
    if value is None:
        value = baseline.get("output_tps", baseline.get("throughput_tps"))
    return number(value, f"{label}.baseline_output_tps", problems)


def validate_tail_latency(
    cell: dict[str, Any],
    perf: dict[str, Any],
    label: str,
    problems: list[str],
) -> None:
    if cell.get("offline_throughput_only") is True or perf.get("offline_throughput_only") is True:
        return
    ferrum = number(cell.get("ferrum_p95_itl_ms"), f"{label}.ferrum_p95_itl_ms", problems)
    baseline = number(cell.get("baseline_p95_itl_ms"), f"{label}.baseline_p95_itl_ms", problems)
    if ferrum is None or baseline is None:
        return
    if baseline <= 0:
        problems.append(f"{label}.baseline_p95_itl_ms must be > 0")
        return
    if ferrum > baseline * MAX_ITL_MULTIPLE:
        problems.append(
            f"{label} p95 ITL {ferrum:.3f}ms exceeds {MAX_ITL_MULTIPLE:.2f}x "
            f"baseline {baseline:.3f}ms"
        )


def validate_performance_cell(
    cell: dict[str, Any],
    baseline: dict[str, Any],
    perf: dict[str, Any],
    out_dir: Path,
    problems: list[str],
) -> int | None:
    concurrency = positive_int(
        cell.get("requested_concurrency", cell.get("concurrency")),
        "performance cell requested_concurrency",
        problems,
    )
    label = f"performance.c{concurrency}" if concurrency is not None else "performance.cell"
    if concurrency is None:
        return None
    effective = positive_int(
        cell.get("effective_active_concurrency", concurrency),
        f"{label}.effective_active_concurrency",
        problems,
    )
    baseline_effective = positive_int(
        cell.get("baseline_effective_active_concurrency", effective),
        f"{label}.baseline_effective_active_concurrency",
        problems,
    )
    if effective is not None and effective > concurrency:
        problems.append(f"{label}.effective_active_concurrency cannot exceed requested concurrency")
    if effective != baseline_effective:
        problems.append(f"{label} effective active concurrency must match baseline")
    if effective is not None and effective < concurrency:
        published = cell.get("published_concurrency")
        if published != effective:
            problems.append(
                f"{label} is admission-capped; published_concurrency must be {effective}"
            )

    n_repeats = positive_int(cell.get("n_repeats"), f"{label}.n_repeats", problems)
    requests = positive_int(cell.get("requests_per_run"), f"{label}.requests_per_run", problems)
    if n_repeats is not None and n_repeats < 3:
        problems.append(f"{label}.n_repeats must be >= 3")
    completed = as_list(cell.get("completed_per_run"), f"{label}.completed_per_run", problems)
    errored = as_list(cell.get("errored_per_run"), f"{label}.errored_per_run", problems)
    if n_repeats is not None:
        if len(completed) != n_repeats:
            problems.append(f"{label}.completed_per_run length must equal n_repeats")
        if len(errored) != n_repeats:
            problems.append(f"{label}.errored_per_run length must equal n_repeats")
    if requests is not None and completed and completed != [requests] * len(completed):
        problems.append(f"{label}.completed_per_run must be full for every repeat")
    if errored and any(value != 0 for value in errored):
        problems.append(f"{label}.errored_per_run must be all zero")
    for field in REQUIRED_ZERO_RUN_COUNT_FIELDS:
        values = as_list(cell.get(field), f"{label}.{field}", problems)
        if n_repeats is not None and len(values) != n_repeats:
            problems.append(f"{label}.{field} length must equal n_repeats")
        non_zero = [
            value
            for value in values
            if not isinstance(value, int) or isinstance(value, bool) or value != 0
        ]
        if non_zero:
            problems.append(f"{label}.{field} must be all zero")
    if cell.get("output_token_count_source") != "usage":
        problems.append(f"{label}.output_token_count_source must be usage")
    require_true(cell.get("stream_options_include_usage"), f"{label}.stream_options_include_usage", problems)
    for key in ["same_hardware", "same_model", "same_quantization", "same_prompt_or_dataset"]:
        require_true(cell.get(key, baseline.get(key)), f"{label}.{key}", problems)

    command = command_parts(
        cell.get("bench_command_line", perf.get("bench_command_line")),
        f"{label}.bench_command_line",
        problems,
    )
    if command and n_repeats is not None:
        validate_bench_command(command, n_repeats, label, problems)

    ferrum_metric = metric_from_cell(cell, problems, label)
    baseline_metric = baseline_tps_from_cell(cell, baseline, problems, label)
    if ferrum_metric is not None and baseline_metric is not None:
        if baseline_metric <= 0:
            problems.append(f"{label}.baseline_output_tps must be > 0")
        else:
            ratio = ferrum_metric / baseline_metric
            reported = cell.get("ratio")
            if reported is not None:
                reported_number = number(reported, f"{label}.ratio", problems)
                if reported_number is not None and abs(reported_number - ratio) > 0.001:
                    problems.append(
                        f"{label}.ratio {reported_number:.6f} does not match computed {ratio:.6f}"
                    )
            if ratio + 1e-12 < MIN_RATIO:
                problems.append(
                    f"{label} ratio {ratio:.6f} < required {MIN_RATIO:.3f}"
                )
    validate_tail_latency(cell, perf, label, problems)
    if "artifact" not in cell:
        problems.append(f"{label} must include a Ferrum performance artifact")
    else:
        validate_evidence_entry(cell["artifact"], f"{label}.artifact", out_dir, problems)
    return concurrency


def validate_performance(manifest: dict[str, Any], out_dir: Path, problems: list[str]) -> None:
    perf = as_object(manifest.get("performance"), "performance", problems)
    baseline = as_object(perf.get("baseline"), "performance.baseline", problems)
    for key in ["engine", "version", "build_command_line", "command_line"]:
        if key not in baseline:
            problems.append(f"performance.baseline missing {key}")
    if "artifact" not in baseline:
        problems.append("performance.baseline must include an artifact")
    else:
        validate_evidence_entry(baseline["artifact"], "performance.baseline.artifact", out_dir, problems)
    command_parts(baseline.get("command_line"), "performance.baseline.command_line", problems)
    command_parts(baseline.get("build_command_line"), "performance.baseline.build_command_line", problems)
    for key in ["same_hardware", "same_model", "same_quantization"]:
        require_true(baseline.get(key), f"performance.baseline.{key}", problems)

    cells = as_list(perf.get("cells"), "performance.cells", problems)
    seen: set[int] = set()
    for idx, raw_cell in enumerate(cells):
        cell = as_object(raw_cell, f"performance.cells[{idx}]", problems)
        if not cell:
            continue
        concurrency = validate_performance_cell(cell, baseline, perf, out_dir, problems)
        if concurrency is not None:
            if concurrency in seen:
                problems.append(f"performance has duplicate c={concurrency} cell")
            seen.add(concurrency)
    missing = sorted(REQUIRED_CONCURRENCY - seen)
    if missing:
        problems.append(f"performance missing required concurrency cells: {missing}")


def validate_manifest(data: dict[str, Any], lane: str, out_dir: Path) -> list[str]:
    problems: list[str] = []
    validate_top_level(data, lane, out_dir, problems)
    validate_correctness(data, lane, out_dir, problems)
    validate_performance(data, out_dir, problems)
    return problems


def result_manifest(
    *,
    lane: str,
    out_dir: Path,
    status: str,
    problems: list[str],
    manifest_path: Path,
) -> dict[str, Any]:
    pass_line = f"{PASS_LINES[lane]}: {out_dir}"
    return {
        "schema_version": 1,
        "lane": lane,
        "status": status,
        "goal_doc": str(GOAL_DOC.relative_to(REPO_ROOT)),
        "manifest": str(manifest_path),
        "artifact_dir": str(out_dir),
        "validated_at": iso_now(),
        "pass_line": pass_line if status == "pass" else None,
        "problems": problems,
    }


def run_gate(lane: str, out_dir: Path, manifest_path: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str]
    try:
        data = as_object(load_json(manifest_path), str(manifest_path), [])
        problems = validate_manifest(data, lane, out_dir)
    except ValidationError as exc:
        problems = [str(exc)]

    status = "pass" if not problems else "fail"
    write_json(
        out_dir / RESULT_MANIFEST_NAME,
        result_manifest(
            lane=lane,
            out_dir=out_dir,
            status=status,
            problems=problems,
            manifest_path=manifest_path,
        ),
    )
    if problems:
        print(f"MODEL_RELEASE_GRADE_{lane.upper()} FAIL ({len(problems)} problems)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{PASS_LINES[lane]}: {out_dir}")
    return 0


def write_selftest_manifest(root: Path, *, ratio: float = 0.82) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel in [
        "hardware.json",
        "runtime.json",
        "l0.json",
        "l1.json",
        "l2.json",
        "l3.json",
        "l4.json",
        "l5.json",
        "run.json",
        "serve.json",
        "baseline.json",
        "c1.json",
        "c4.json",
        "c16.json",
        "c32.json",
    ]:
        write_json(root / rel, {"status": "pass", "name": rel})
    cells = []
    for concurrency in sorted(REQUIRED_CONCURRENCY):
        effective = 16 if concurrency == 32 else concurrency
        cells.append(
            {
                "requested_concurrency": concurrency,
                "effective_active_concurrency": effective,
                "baseline_effective_active_concurrency": effective,
                "published_concurrency": effective,
                "requests_per_run": 100,
                "n_repeats": 3,
                "completed_per_run": [100, 100, 100],
                "errored_per_run": [0, 0, 0],
                "bad_output_per_run": [0, 0, 0],
                "malformed_stream_per_run": [0, 0, 0],
                "missing_done_per_run": [0, 0, 0],
                "duplicate_done_per_run": [0, 0, 0],
                "zero_output_tokens_per_run": [0, 0, 0],
                "stream_bulk_flush_per_run": [0, 0, 0],
                "http_500_per_run": [0, 0, 0],
                "panic_per_run": [0, 0, 0],
                "output_token_count_source": "usage",
                "stream_options_include_usage": True,
                "same_hardware": True,
                "same_model": True,
                "same_quantization": True,
                "same_prompt_or_dataset": True,
                "bench_command_line": [
                    "ferrum",
                    "bench-serve",
                    "--fail-on-error",
                    "--require-ci",
                    "--seed",
                    "9271",
                    "--n-repeats",
                    "3",
                ],
                "ferrum_output_tps_lcb": 100.0 * ratio,
                "baseline_output_tps": 100.0,
                "ratio": ratio,
                "ferrum_p95_itl_ms": 10.0,
                "baseline_p95_itl_ms": 9.0,
                "artifact": f"c{concurrency}.json",
            }
        )
    manifest = {
        "schema_version": 1,
        "lane": "w2",
        "status": "pass",
        "goal_doc": str(GOAL_DOC.relative_to(REPO_ROOT)),
        "model_id": "gemma3-27b",
        "backend": "cuda",
        "quantization": "gptq-int4",
        "git_sha": "0123456789abcdef",
        "dirty_status": {"dirty": False},
        "binary_sha256": "a" * 64,
        "hardware": {"status": "pass", "artifact": "hardware.json"},
        "release_scope": {
            "backends": ["cuda"],
            "formats": ["gptq-int4"],
            "excluded_lanes": {
                "gguf_metal": {
                    "reason": "not in this release-grade W2 scope",
                }
            },
        },
        "runtime_config": {
            "product_surface": "typed_cli",
            "hidden_env": [],
            "snapshot": "runtime.json",
        },
        "correctness": {
            "l0_template": {"status": "pass", "artifact": "l0.json"},
            "l1_numeric": {"status": "pass", "artifact": "l1.json"},
            "l2_quantized": {"status": "pass", "artifact": "l2.json"},
            "l3_behavior": {"status": "pass", "artifact": "l3.json"},
            "l4_agent": {"status": "pass", "artifact": "l4.json"},
            "l5_concurrency": {"status": "pass", "artifact": "l5.json"},
        },
        "product_entrypoints": {
            "ferrum_run": {"status": "pass", "artifact": "run.json"},
            "ferrum_serve": {"status": "pass", "artifact": "serve.json"},
        },
        "performance": {
            "baseline": {
                "engine": "llama.cpp",
                "version": "selftest",
                "build_command_line": ["cmake", "--build", "build"],
                "command_line": ["llama-bench", "-ngl", "999"],
                "same_hardware": True,
                "same_model": True,
                "same_quantization": True,
                "artifact": "baseline.json",
            },
            "cells": cells,
        },
    }
    path = root / MANIFEST_NAME
    write_json(path, manifest)
    return path


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-model-release-grade-") as tmp:
        tmp_root = Path(tmp)
        good = tmp_root / "good"
        good_manifest = write_selftest_manifest(good, ratio=0.82)
        good_problems = validate_manifest(load_json(good_manifest), "w2", good)
        if good_problems:
            raise AssertionError("good selftest manifest failed: " + "; ".join(good_problems))

        bad = tmp_root / "bad-ratio"
        bad_manifest = write_selftest_manifest(bad, ratio=0.79)
        bad_problems = validate_manifest(load_json(bad_manifest), "w2", bad)
        if not any("ratio" in problem and "< required" in problem for problem in bad_problems):
            raise AssertionError("bad ratio selftest did not fail as expected")

        hidden = tmp_root / "hidden-env"
        hidden_manifest = write_selftest_manifest(hidden, ratio=0.82)
        data = load_json(hidden_manifest)
        data["runtime_config"]["hidden_env"] = ["FERRUM_FORCE_FAST_PATH=1"]
        write_json(hidden_manifest, data)
        hidden_problems = validate_manifest(data, "w2", hidden)
        if not any("hidden_env" in problem for problem in hidden_problems):
            raise AssertionError("hidden env selftest did not fail as expected")

        dirty = tmp_root / "dirty"
        dirty_manifest = write_selftest_manifest(dirty, ratio=0.82)
        data = load_json(dirty_manifest)
        data["dirty_status"] = {"dirty": True, "files": ["crates/ferrum-models/src/lib.rs"]}
        write_json(dirty_manifest, data)
        dirty_problems = validate_manifest(data, "w2", dirty)
        if not any("dirty_status" in problem for problem in dirty_problems):
            raise AssertionError("dirty status selftest did not fail as expected")

        missing_baseline = tmp_root / "missing-baseline-artifact"
        missing_baseline_manifest = write_selftest_manifest(missing_baseline, ratio=0.82)
        data = load_json(missing_baseline_manifest)
        del data["performance"]["baseline"]["artifact"]
        write_json(missing_baseline_manifest, data)
        missing_baseline_problems = validate_manifest(data, "w2", missing_baseline)
        if not any(
            "performance.baseline must include an artifact" in problem
            for problem in missing_baseline_problems
        ):
            raise AssertionError("missing baseline artifact selftest did not fail as expected")

        missing_cell = tmp_root / "missing-cell-artifact"
        missing_cell_manifest = write_selftest_manifest(missing_cell, ratio=0.82)
        data = load_json(missing_cell_manifest)
        del data["performance"]["cells"][0]["artifact"]
        write_json(missing_cell_manifest, data)
        missing_cell_problems = validate_manifest(data, "w2", missing_cell)
        if not any(
            "must include a Ferrum performance artifact" in problem
            for problem in missing_cell_problems
        ):
            raise AssertionError("missing cell artifact selftest did not fail as expected")

        bad_quality = tmp_root / "bad-quality-count"
        bad_quality_manifest = write_selftest_manifest(bad_quality, ratio=0.82)
        data = load_json(bad_quality_manifest)
        data["performance"]["cells"][0]["bad_output_per_run"] = [0, 1, 0]
        write_json(bad_quality_manifest, data)
        bad_quality_problems = validate_manifest(data, "w2", bad_quality)
        if not any("bad_output_per_run must be all zero" in problem for problem in bad_quality_problems):
            raise AssertionError("bad quality count selftest did not fail as expected")

    print("MODEL RELEASE GRADE GOAL SELFTEST PASS")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", nargs="?", choices=sorted(PASS_LINES))
    parser.add_argument("out_dir", nargs="?")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.lane is None or args.out_dir is None:
        parser.error("lane and out_dir are required unless --self-test is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return run_selftest()
    out_dir = Path(args.out_dir)
    manifest = args.manifest or out_dir / MANIFEST_NAME
    return run_gate(args.lane, out_dir, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
