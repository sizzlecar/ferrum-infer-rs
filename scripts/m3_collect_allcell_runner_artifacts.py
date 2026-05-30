#!/usr/bin/env python3
"""Collect per-concurrency M3 runner artifacts into one all-cell artifact."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_CELLS = [1, 4, 16, 32]
SUMMARY_METRICS = [
    "throughput_mean",
    "throughput_stddev",
    "throughput_ci95_hw",
    "ttft_p50",
    "tpot_p50",
    "itl_p95",
    "completed",
    "errored",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def publishability_for_verdict(verdict: str) -> tuple[bool, str | None]:
    if verdict == "pass":
        return False, None
    if verdict == "diagnostic-only":
        return True, "diagnostic-only artifact"
    return True, "artifact verdict is fail"


def cell_from_dir(path: Path) -> int:
    name = path.name
    if not name.startswith("c"):
        raise SystemExit(f"cell directory must be named c<N>: {path}")
    try:
        value = int(name[1:])
    except ValueError as exc:
        raise SystemExit(f"cell directory must be named c<N>: {path}") from exc
    if value <= 0:
        raise SystemExit(f"cell concurrency must be positive: {path}")
    return value


def sorted_cell_dirs(root: Path) -> list[Path]:
    cells = [path for path in root.glob("c*") if path.is_dir() and (path / "manifest.json").exists()]
    return sorted(cells, key=cell_from_dir)


def aggregate_checklist(
    child_manifests: list[dict[str, Any]],
    case_manifests: list[dict[str, Any]],
    *,
    change_type: str | None,
    baseline_case: str | None,
    performance_required: bool,
) -> dict[str, Any]:
    touched_areas: set[str] = set()
    required_gates: set[str] = set()
    local_gates: list[dict[str, Any]] = []
    skipped_gates: list[dict[str, Any]] = []
    benchmark_impacts: list[dict[str, Any]] = []
    for manifest in child_manifests:
        checklist = manifest.get("validation_checklist") or {}
        touched_areas.update(str(item) for item in checklist.get("touched_areas", []) if item)
        required_gates.update(
            str(item) for item in checklist.get("required_correctness_gates", []) if item
        )
        local_gates.extend(item for item in checklist.get("local_gates", []) if isinstance(item, dict))
        skipped_gates.extend(
            item for item in checklist.get("skipped_gates", []) if isinstance(item, dict)
        )
        impact = checklist.get("benchmark_impact")
        if isinstance(impact, dict):
            benchmark_impacts.append(impact)
    observed_by_name: dict[str, dict[str, Any]] = {}
    for manifest in case_manifests:
        for gate in manifest.get("correctness_gates", []) or []:
            if not isinstance(gate, dict) or not gate.get("name"):
                continue
            name = str(gate["name"])
            current = observed_by_name.get(name)
            ok = bool(gate.get("ok"))
            if current is None:
                observed_by_name[name] = {
                    "name": name,
                    "ok": ok,
                    "evidence": "all-cell child case manifests",
                }
            else:
                current["ok"] = bool(current.get("ok")) and ok

    completed = sum(int((manifest.get("metrics") or {}).get("completed") or 0) for manifest in case_manifests)
    errored = sum(int((manifest.get("metrics") or {}).get("errored") or 0) for manifest in case_manifests)
    bench_required = "bench_completion" in required_gates
    checklist = {
        "schema_version": 1,
        "change_type": change_type or str(
            (child_manifests[0].get("validation_checklist") or {}).get(
                "change_type", "opt_in_experiment"
            )
        ),
        "touched_areas": sorted(touched_areas) or ["benchmark_harness"],
        "required_correctness_gates": sorted(required_gates) or ["bench_completion"],
        "observed_correctness_gates": sorted(
            observed_by_name.values(), key=lambda item: item["name"]
        ),
        "bench_completion": {
            "required": bench_required,
            "ok": completed > 0 and errored == 0,
            "completed": completed,
            "errored": errored,
        },
        "local_gates": local_gates,
        "skipped_gates": skipped_gates,
        "performance_regression_required": performance_required,
        "baseline_case": baseline_case,
        "case": None,
    }
    if benchmark_impacts:
        exercised = any(bool(impact.get("m3_benchmark_exercised")) for impact in benchmark_impacts)
        checklist["benchmark_impact"] = {
            "m3_benchmark_exercised": exercised,
            "reason": "aggregated from child runner artifacts",
            "evidence": "; ".join(
                str(impact.get("evidence") or impact.get("reason") or "child artifact")
                for impact in benchmark_impacts
            ),
        }
    return checklist


def aggregate_performance_gates(
    cell_summaries: list[tuple[int, dict[str, Any]]],
    *,
    baseline_case: str,
    candidates: list[str],
    required_cells: list[int],
) -> dict[str, Any]:
    first_gates = cell_summaries[0][1].get("performance_regression_gates") or {}
    thresholds = first_gates.get("thresholds") or {}
    observed_cells = sorted({cell for cell, _summary in cell_summaries})
    result: dict[str, Any] = {
        "schema_version": 1,
        "enabled": bool(first_gates.get("enabled", False)),
        "baseline_case": baseline_case,
        "thresholds": thresholds,
        "cases": {},
        "required_concurrency_cells": required_cells,
        "observed_concurrency_cells": observed_cells,
        "concurrency_cells_ok": all(cell in observed_cells for cell in required_cells),
    }
    for candidate in candidates:
        metrics: list[dict[str, Any]] = []
        ok = True
        for cell, summary in cell_summaries:
            gates = summary.get("performance_regression_gates") or {}
            case_gate = (gates.get("cases") or {}).get(candidate)
            if not isinstance(case_gate, dict):
                ok = False
                metrics.append(
                    {
                        "metric": f"c{cell}.missing_performance_gate",
                        "baseline": None,
                        "candidate": None,
                        "delta_pct": None,
                        "threshold": {"type": "required", "value": 1},
                        "ok": False,
                        "reason": f"missing child performance gate for {candidate}",
                    }
                )
                continue
            ok = ok and bool(case_gate.get("ok"))
            for metric in case_gate.get("metrics", []) or []:
                if not isinstance(metric, dict):
                    ok = False
                    continue
                item = dict(metric)
                item["metric"] = f"c{cell}.{item.get('metric', 'unknown')}"
                item["concurrency"] = cell
                metrics.append(item)
                ok = ok and bool(item.get("ok"))
        result["cases"][candidate] = {
            "baseline_case": baseline_case,
            "ok": ok,
            "metrics": metrics,
        }
    return result


def aggregate(
    root: Path,
    *,
    baseline_case: str,
    candidates: list[str],
    change_type: str | None,
    required_cells: list[int],
) -> dict[str, Any]:
    root = root.resolve()
    cells = sorted_cell_dirs(root)
    if not cells:
        raise SystemExit(f"no child runner artifacts found under {root}/c*/manifest.json")

    rows: list[dict[str, Any]] = []
    root_cases: list[dict[str, Any]] = []
    child_manifests: list[dict[str, Any]] = []
    case_manifests: list[dict[str, Any]] = []
    cell_summaries: list[tuple[int, dict[str, Any]]] = []
    child_verdicts: list[str] = []

    for cell_root in cells:
        cell = cell_from_dir(cell_root)
        child_manifest = load_json(cell_root / "manifest.json")
        child_summary = load_json(resolve(str(child_manifest["summary_json"]), cell_root))
        child_manifests.append(child_manifest)
        cell_summaries.append((cell, child_summary))
        child_verdicts.append(str(child_manifest.get("artifact_verdict", "fail")))

        for row in child_summary.get("rows", []) or []:
            item = dict(row)
            item["concurrency"] = int(item.get("concurrency", cell))
            item["cell"] = f"c{cell}"
            for metric in SUMMARY_METRICS:
                item.setdefault(metric, None)
            rows.append(item)

        for case_ref in child_manifest.get("cases", []) or []:
            manifest_path = resolve(str(case_ref["manifest"]), cell_root)
            case_manifest = load_json(manifest_path)
            case_manifests.append(case_manifest)
            root_cases.append(
                {
                    "name": f"c{cell}_{case_ref.get('name', case_manifest.get('name', 'case'))}",
                    "manifest": str(manifest_path),
                }
            )

    if any(verdict == "fail" for verdict in child_verdicts):
        verdict = "fail"
    elif any(verdict == "diagnostic-only" for verdict in child_verdicts):
        verdict = "diagnostic-only"
    else:
        verdict = "pass"
    not_publishable, not_publishable_reason = publishability_for_verdict(verdict)

    performance_required = any(
        bool((manifest.get("validation_checklist") or {}).get("performance_regression_required"))
        for manifest in child_manifests
    )
    summary = {
        "rows": rows,
        "performance_regression_gates": aggregate_performance_gates(
            cell_summaries,
            baseline_case=baseline_case,
            candidates=candidates,
            required_cells=required_cells,
        ),
    }
    write_json(root / "summary.json", summary)

    first = child_manifests[0]
    manifest = {
        "runner": "scripts/m3_collect_allcell_runner_artifacts.py",
        "schema_version": 1,
        "name": f"{first.get('name', 'm3-runner')}-allcells",
        "created_at": now_iso(),
        "artifact_verdict": verdict,
        "not_publishable": not_publishable,
        "not_publishable_reason": not_publishable_reason,
        "validation_checklist": aggregate_checklist(
            child_manifests,
            case_manifests,
            change_type=change_type,
            baseline_case=baseline_case,
            performance_required=performance_required,
        ),
        "preflight": {
            "children": [
                {
                    "cell": f"c{cell_from_dir(path)}",
                    "manifest": str(path / "manifest.json"),
                    "summary_json": str(path / "summary.json"),
                }
                for path in cells
            ],
            "first_child_preflight": first.get("preflight", {}),
        },
        "runtime_preset": first.get("runtime_preset"),
        "cases": root_cases,
        "summary_json": str(root / "summary.json"),
    }
    write_json(root / "manifest.json", manifest)
    return {"root": str(root), "cells": [cell_from_dir(path) for path in cells], "cases": len(root_cases)}


def self_test() -> None:
    from m3_validate_runner_artifact import REQUIRED_DECISIONS, validate_artifact

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        thresholds = {
            "enabled": True,
            "throughput_min_delta_pct": -3.0,
            "ttft_max_regression_pct": 10.0,
            "tpot_max_regression_pct": 5.0,
            "itl_p95_max_regression_pct": 10.0,
        }
        checklist = {
            "schema_version": 1,
            "change_type": "opt_in_experiment",
            "touched_areas": ["attention_prefill_mixed_path", "fa2_runtime_path"],
            "required_correctness_gates": ["paris", "bench_completion"],
            "observed_correctness_gates": [{"name": "paris", "ok": True, "evidence": "fixture"}],
            "bench_completion": {"required": True, "ok": True, "completed": 2, "errored": 0},
            "local_gates": [],
            "skipped_gates": [],
            "performance_regression_required": True,
            "baseline_case": "fa_layout",
            "case": None,
            "benchmark_impact": {
                "m3_benchmark_exercised": True,
                "reason": "FA2 source changes are exercised by M3 all-cell benches",
                "evidence": "self-test child artifact",
            },
        }
        decisions = [
            {
                "schema_version": 1,
                "selection": selection,
                "selected": "fixture",
                "source": "default",
                "source_key": None,
                "candidates": ["fixture"],
                "rejected": [],
                "affects": ["performance"],
            }
            for selection in sorted(REQUIRED_DECISIONS)
        ]

        def write_case_manifest(path: Path, *, name: str, cell: int, throughput: float) -> None:
            case_dir = path.parent
            bench_json = case_dir / "bench.json"
            bench_log = case_dir / "bench.log"
            server_log = case_dir / "server.log"
            effective_config_json = case_dir / "effective_config.json"
            decision_trace_jsonl = case_dir / "decision_trace.jsonl"
            profile_jsonl = case_dir / "profile.jsonl"
            env_hash = f"sha256:{name}-{cell}"
            snapshot = {
                "schema_version": 1,
                "preset": "m3_qwen3_30b_a3b_int4",
                "env_hash": env_hash,
                "entries": [
                    {
                        "key": "FERRUM_FA_LAYOUT_VARLEN",
                        "effective_value": "1",
                        "source": "script_case",
                        "effect": "performance",
                        "affects": ["performance"],
                    }
                ],
            }
            write_json(bench_json, {})
            bench_log.write_text("bench fixture\n")
            server_log.write_text("server fixture\n")
            effective_entries = [
                {
                    "key": entry["key"],
                    "effective_value": entry["effective_value"],
                    "source": entry["source"],
                    "affects": entry["affects"],
                }
                for entry in snapshot["entries"]
            ]
            auto_config_inputs = {
                "model_capabilities": {
                    "architecture": "qwen3_moe",
                    "quantization": "gptq_int4",
                    "moe": {
                        "num_experts": 128,
                        "experts_per_token": 8,
                        "moe_intermediate_size": 768,
                    },
                    "max_context_len": 40960,
                    "num_hidden_layers": 48,
                    "head_dim": 128,
                    "kv_heads": 4,
                    "estimated_weight_bytes": 19327352832,
                    "supported_dtypes": ["fp16"],
                    "graph_safe_moe": True,
                },
                "hardware_capabilities": {
                    "backend": "cuda",
                    "cuda_runtime": "12.8",
                    "compute_capability": "8.9",
                    "vram_bytes": 25753026560,
                    "sm_count": 128,
                    "supported_dtypes": ["fp16", "fp32"],
                    "supported_kv_dtypes": ["fp16", "int8"],
                    "graph_support": True,
                    "compiled_features": {
                        "cuda": True,
                        "vllm_paged_attn": True,
                        "vllm_moe_marlin": True,
                        "cuda_graph": True,
                        "greedy_argmax": True,
                        "fa2_source": False,
                        "fa2_direct_ffi": False,
                    },
                },
                "workload_profile": {
                    "preset": "m3_qwen3_30b_a3b_int4",
                    "serving_mode": "bench_serve",
                    "target_concurrency": 32,
                    "prompt_length_class": "random_256",
                    "output_length_class": "random_128",
                    "priority": "throughput",
                },
            }
            write_json(
                effective_config_json,
                {
                    "schema_version": 1,
                    "preset": snapshot["preset"],
                    "env_hash": env_hash,
                    "entries": effective_entries,
                    **auto_config_inputs,
                    "decisions": decisions,
                },
            )
            decision_trace_jsonl.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in decisions)
            )
            profile_jsonl.write_text("")
            case_checklist = dict(checklist)
            case_checklist["case"] = name
            case_checklist["bench_completion"] = {
                "required": True,
                "ok": True,
                "completed": 2,
                "errored": 0,
            }
            write_json(
                path,
                {
                    "schema_version": 1,
                    "name": name,
                    "started_at": now_iso(),
                    "artifact_verdict": "pass",
                    "not_publishable": False,
                    "not_publishable_reason": None,
                    "port": 18000 + cell,
                    "git_head": "abc",
                    "git_status_short": [],
                    "binary_sha256": "sha256:bin",
                    "features": "cuda",
                    "runtime_preset": "m3_qwen3_30b_a3b_int4",
                    "env_hash": env_hash,
                    "preset_env": {},
                    "base_env": {},
                    "effective_env": {},
                    "case_env": {"FERRUM_FA_LAYOUT_VARLEN": "1"},
                    "runtime_config_snapshot": snapshot,
                    "model_dir": "/model",
                    "server_log": str(server_log),
                    "bench_json": str(bench_json),
                    "bench_log": str(bench_log),
                    "effective_config_json": str(effective_config_json),
                    "decision_trace_jsonl": str(decision_trace_jsonl),
                    "auto_config_decision_count": len(decisions),
                    "profile_jsonl": str(profile_jsonl),
                    "correctness_gates": [{"name": "paris", "ok": True}],
                    "validation_checklist": case_checklist,
                    "cleanup_status": {
                        "sent_kill": False,
                        "returncode": 0,
                        "process_leak_ok": True,
                        "process_leaks": [],
                        "global_process_hygiene_ok": True,
                        "global_process_findings": [],
                    },
                    "metrics": {
                        "completed": 2,
                        "errored": 0,
                        "throughput_mean": throughput,
                        "throughput_stddev": 1.0,
                        "throughput_ci95_hw": 2.0,
                        "ttft_p50": 1.0,
                        "tpot_p50": 1.0,
                        "itl_p95": 1.0,
                    },
                    "status": "pass",
                },
            )

        for cell in DEFAULT_REQUIRED_CELLS:
            cell_root = root / f"c{cell}"
            source_case = cell_root / f"fa2_source_c{cell}_n3" / "manifest.json"
            layout_case = cell_root / f"fa_layout_c{cell}_n3" / "manifest.json"
            for path, name, throughput in (
                (source_case, "fa2_source", 120.0 + cell),
                (layout_case, "fa_layout", 100.0 + cell),
            ):
                write_case_manifest(path, name=name, cell=cell, throughput=throughput)
            write_json(
                cell_root / "summary.json",
                {
                    "rows": [
                        {
                            "name": "fa2_source",
                            "status": "pass",
                            "artifact_verdict": "pass",
                            "not_publishable": False,
                            "concurrency": cell,
                            "throughput_mean": 120.0 + cell,
                            "throughput_stddev": 1.0,
                            "throughput_ci95_hw": 2.0,
                            "ttft_p50": 1.0,
                            "tpot_p50": 1.0,
                            "itl_p95": 1.0,
                            "completed": 2,
                            "errored": 0,
                        },
                        {
                            "name": "fa_layout",
                            "status": "pass",
                            "artifact_verdict": "pass",
                            "not_publishable": False,
                            "concurrency": cell,
                            "throughput_mean": 100.0 + cell,
                            "throughput_stddev": 1.0,
                            "throughput_ci95_hw": 2.0,
                            "ttft_p50": 1.0,
                            "tpot_p50": 1.0,
                            "itl_p95": 1.0,
                            "completed": 2,
                            "errored": 0,
                        },
                    ],
                    "performance_regression_gates": {
                        "schema_version": 1,
                        "enabled": True,
                        "baseline_case": "fa_layout",
                        "thresholds": thresholds,
                        "cases": {
                            "fa2_source": {
                                "baseline_case": "fa_layout",
                                "ok": True,
                                "metrics": [
                                    {
                                        "metric": "throughput_mean",
                                        "baseline": 100.0 + cell,
                                        "candidate": 120.0 + cell,
                                        "delta_pct": 10.0,
                                        "threshold": {"type": "min_delta_pct", "value": -3.0},
                                        "ok": True,
                                        "reason": "ok",
                                    }
                                ],
                            }
                        },
                        "required_concurrency_cells": [],
                        "observed_concurrency_cells": [cell],
                        "concurrency_cells_ok": True,
                    },
                },
            )
            write_json(
                cell_root / "manifest.json",
                {
                    "runner": "scripts/m3_ab_runner.py",
                    "schema_version": 1,
                    "name": "m3-fa2-direct-ffi-ab",
                    "created_at": now_iso(),
                    "artifact_verdict": "pass",
                    "not_publishable": False,
                    "not_publishable_reason": None,
                    "validation_checklist": checklist,
                    "preflight": {},
                    "runtime_preset": "m3_qwen3_30b_a3b_int4",
                    "summary_json": str(cell_root / "summary.json"),
                    "cases": [
                        {"name": "fa2_source", "manifest": str(source_case)},
                        {"name": "fa_layout", "manifest": str(layout_case)},
                    ],
                },
            )
        result = aggregate(
            root,
            baseline_case="fa_layout",
            candidates=["fa2_source"],
            change_type="default_path",
            required_cells=DEFAULT_REQUIRED_CELLS,
        )
        assert result["cells"] == DEFAULT_REQUIRED_CELLS
        summary = load_json(root / "summary.json")
        gates = summary["performance_regression_gates"]
        assert gates["observed_concurrency_cells"] == DEFAULT_REQUIRED_CELLS
        assert gates["concurrency_cells_ok"]
        assert gates["cases"]["fa2_source"]["ok"]
        assert len(gates["cases"]["fa2_source"]["metrics"]) == 4
        manifest = load_json(root / "manifest.json")
        assert manifest["validation_checklist"]["change_type"] == "default_path"
        assert manifest["validation_checklist"]["benchmark_impact"] == {
            "m3_benchmark_exercised": True,
            "reason": "aggregated from child runner artifacts",
            "evidence": "; ".join(["self-test child artifact"] * 4),
        }
        assert len(manifest["cases"]) == 8
        result = validate_artifact(root, require_bench=True, require_profile_events=False)
        assert result["ok"]
    print("m3_collect_allcell_runner_artifacts self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--baseline-case", default="fa_layout")
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--change-type")
    parser.add_argument(
        "--required-cells",
        default=",".join(str(cell) for cell in DEFAULT_REQUIRED_CELLS),
        help="comma-separated concurrency cells required by the aggregate gate",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is set")
    required_cells = [int(item) for item in args.required_cells.replace(",", " ").split()]
    result = aggregate(
        args.root,
        baseline_case=args.baseline_case,
        candidates=args.candidate or ["fa2_source"],
        change_type=args.change_type,
        required_cells=required_cells,
    )
    print(
        "ALLCELL_ARTIFACT root={root} cells={cells} cases={cases}".format(
            root=result["root"], cells=",".join(str(cell) for cell in result["cells"]), cases=result["cases"]
        )
    )


if __name__ == "__main__":
    main()
