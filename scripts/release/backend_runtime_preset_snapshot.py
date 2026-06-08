#!/usr/bin/env python3
"""Generate and validate backend runtime preset snapshots without loading weights."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_DIR = Path("scripts/release/snapshots/backend_runtime_preset")
REQUIRED_CASES = {
    "metal_llama_8b_dense_gguf",
    "metal_qwen3_30b_a3b_moe_gguf",
    "cuda_llama_8b_dense_gptq",
    "cuda_qwen3_30b_a3b_moe_gptq",
}
REQUIRED_GROUPS = {
    "attention_impls",
    "graph_modes",
    "kv_layouts",
    "kv_dtypes",
    "moe_decode_paths",
    "cache_modes",
}


class SnapshotError(Exception):
    pass


def run_generator(root: Path) -> list[dict[str, Any]]:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "ferrum-types",
        "--example",
        "backend_runtime_preset_snapshot",
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SnapshotError(
            f"snapshot generator failed rc={proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"snapshot generator emitted invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise SnapshotError("snapshot generator output must be a list")
    return data


def decision_map(effective: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions = effective.get("decisions")
    if not isinstance(decisions, list):
        raise SnapshotError("effective_config.decisions must be a list")
    out: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        selection = decision.get("selection")
        if isinstance(selection, str):
            out[selection] = decision
    return out


def rejected_candidates(decision: dict[str, Any]) -> list[dict[str, str]]:
    selected = decision.get("selected")
    rejected = decision.get("rejected")
    if isinstance(rejected, list) and rejected:
        return [
            {
                "value": str(item.get("value")),
                "reason": str(item.get("reason")),
            }
            for item in rejected
            if isinstance(item, dict)
        ]
    candidates = decision.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        {"value": str(candidate), "reason": "candidate not selected"}
        for candidate in candidates
        if candidate != selected
    ]


def candidate_group(
    name: str,
    decision: dict[str, Any],
    *,
    selected_override: str | None = None,
    candidates_override: list[str] | None = None,
    rejected_override: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    selected = selected_override or str(decision.get("selected"))
    candidates = candidates_override or [str(item) for item in decision.get("candidates", [])]
    rejected = rejected_override if rejected_override is not None else rejected_candidates(decision)
    return {
        "name": name,
        "selected": selected,
        "candidates": candidates,
        "rejected": rejected,
    }


def synthesize_kv_layout_group(decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decode = decisions.get("attention_decode_backend", {})
    selected_decode = str(decode.get("selected", "legacy_paged_decode"))
    selected = (
        "paged_vllm"
        if selected_decode.startswith("vllm_paged_attn")
        else "paged_legacy"
    )
    candidates = ["paged_vllm", "paged_legacy"]
    rejected = [
        {
            "value": candidate,
            "reason": "attention decode backend selected a different KV layout",
        }
        for candidate in candidates
        if candidate != selected
    ]
    return {
        "name": "kv_layouts",
        "selected": selected,
        "candidates": candidates,
        "rejected": rejected,
    }


def synthesize_kv_dtype_group(effective: dict[str, Any]) -> dict[str, Any]:
    hardware = effective.get("hardware_capabilities") or {}
    runtime_entries = {
        entry.get("key"): entry.get("effective_value")
        for entry in effective.get("entries", [])
        if isinstance(entry, dict)
    }
    candidates = hardware.get("supported_kv_dtypes")
    if not isinstance(candidates, list) or not candidates:
        candidates = ["fp16"]
    candidates = [str(candidate) for candidate in candidates]
    selected = str(runtime_entries.get("FERRUM_KV_DTYPE") or candidates[0])
    if selected not in candidates:
        candidates = [selected, *candidates]
    return {
        "name": "kv_dtypes",
        "selected": selected,
        "candidates": candidates,
        "rejected": [
            {"value": candidate, "reason": "KV dtype not selected"}
            for candidate in candidates
            if candidate != selected
        ],
    }


def build_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    name = case.get("name")
    if not isinstance(name, str) or not name:
        raise SnapshotError("case missing non-empty name")
    effective = case.get("effective_config")
    if not isinstance(effective, dict):
        raise SnapshotError(f"{name}: effective_config must be object")
    decisions = decision_map(effective)
    required_decisions = {
        "attention_decode_backend",
        "moe_graph_policy",
        "moe_implementation",
        "prefix_cache_policy",
        "scheduler_admission_policy",
        "max_sequences",
        "max_batched_tokens",
    }
    missing = sorted(required_decisions - set(decisions))
    if missing:
        raise SnapshotError(f"{name}: missing decisions: {', '.join(missing)}")
    groups = [
        candidate_group("attention_impls", decisions["attention_decode_backend"]),
        candidate_group("graph_modes", decisions["moe_graph_policy"]),
        synthesize_kv_layout_group(decisions),
        synthesize_kv_dtype_group(effective),
        candidate_group("moe_decode_paths", decisions["moe_implementation"]),
        candidate_group("cache_modes", decisions["prefix_cache_policy"]),
    ]
    snapshot = {
        "schema_version": 1,
        "name": name,
        "description": case.get("description"),
        "entries": effective.get("entries"),
        "admission": effective.get("admission"),
        "selected": {
            "backend": effective.get("hardware_capabilities", {}).get("backend"),
            "model_architecture": effective.get("model_capabilities", {}).get("architecture"),
            "model_quantization": effective.get("model_capabilities", {}).get("quantization"),
            "scheduler": decisions["scheduler_admission_policy"].get("selected"),
            "kv_layout": next(g for g in groups if g["name"] == "kv_layouts")["selected"],
            "kv_dtype": next(g for g in groups if g["name"] == "kv_dtypes")["selected"],
            "max_sequences": decisions["max_sequences"].get("selected"),
            "max_batched_tokens": decisions["max_batched_tokens"].get("selected"),
            "attention_impl": decisions["attention_decode_backend"].get("selected"),
            "graph_mode": decisions["moe_graph_policy"].get("selected"),
            "moe_decode_path": decisions["moe_implementation"].get("selected"),
            "cache_mode": decisions["prefix_cache_policy"].get("selected"),
        },
        "model_capabilities": effective.get("model_capabilities"),
        "hardware_capabilities": effective.get("hardware_capabilities"),
        "workload_profile": effective.get("workload_profile"),
        "candidate_groups": groups,
        "decisions": effective.get("decisions"),
    }
    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    name = snapshot.get("name", "<unknown>")
    groups = snapshot.get("candidate_groups")
    if not isinstance(groups, list):
        raise SnapshotError(f"{name}: candidate_groups must be a list")
    group_names = {group.get("name") for group in groups if isinstance(group, dict)}
    missing = sorted(REQUIRED_GROUPS - group_names)
    if missing:
        raise SnapshotError(f"{name}: missing candidate groups: {', '.join(missing)}")
    for group in groups:
        if not isinstance(group, dict):
            raise SnapshotError(f"{name}: candidate group must be object")
        selected = group.get("selected")
        candidates = group.get("candidates")
        rejected = group.get("rejected")
        if not isinstance(selected, str) or not selected:
            raise SnapshotError(f"{name}.{group.get('name')}: selected must be non-empty")
        if not isinstance(candidates, list) or candidates.count(selected) != 1:
            raise SnapshotError(
                f"{name}.{group.get('name')}: candidates must contain selected exactly once"
            )
        non_selected = [candidate for candidate in candidates if candidate != selected]
        if len(non_selected) != len(rejected or []):
            raise SnapshotError(f"{name}.{group.get('name')}: rejected count mismatch")
        for item in rejected:
            if not isinstance(item, dict) or not str(item.get("reason", "")).strip():
                raise SnapshotError(f"{name}.{group.get('name')}: rejected reason missing")


def canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def diff_values(expected: Any, actual: Any, path: str = "$") -> list[str]:
    if type(expected) is not type(actual):
        return [f"{path}: type {type(expected).__name__} -> {type(actual).__name__}"]
    if isinstance(expected, dict):
        diffs: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                diffs.append(f"{path}.{key}: added {actual[key]!r}")
            elif key not in actual:
                diffs.append(f"{path}.{key}: removed {expected[key]!r}")
            else:
                diffs.extend(diff_values(expected[key], actual[key], f"{path}.{key}"))
        return diffs
    if isinstance(expected, list):
        diffs = []
        for idx in range(max(len(expected), len(actual))):
            if idx >= len(expected):
                diffs.append(f"{path}[{idx}]: added {actual[idx]!r}")
            elif idx >= len(actual):
                diffs.append(f"{path}[{idx}]: removed {expected[idx]!r}")
            else:
                diffs.extend(diff_values(expected[idx], actual[idx], f"{path}[{idx}]"))
        return diffs
    if expected != actual:
        return [f"{path}: {expected!r} -> {actual!r}"]
    return []


def write_actual(out_dir: Path, snapshots: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in snapshots:
        (out_dir / f"{snapshot['name']}.json").write_text(canonical(snapshot))
    summary = {
        "schema_version": 1,
        "status": "pass",
        "cases": [snapshot["name"] for snapshot in snapshots],
    }
    (out_dir / "summary.json").write_text(canonical(summary))


def compare_or_update(
    root: Path,
    snapshots: list[dict[str, Any]],
    *,
    update: bool,
) -> list[str]:
    expected_dir = root / EXPECTED_DIR
    expected_dir.mkdir(parents=True, exist_ok=True)
    diffs: list[str] = []
    for snapshot in snapshots:
        expected_path = expected_dir / f"{snapshot['name']}.json"
        actual_text = canonical(snapshot)
        if update or not expected_path.exists():
            expected_path.write_text(actual_text)
            continue
        expected = json.loads(expected_path.read_text())
        diffs.extend(
            f"{snapshot['name']} {line}" for line in diff_values(expected, snapshot)
        )
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    try:
        cases = run_generator(root)
        snapshots = [build_snapshot(case) for case in cases]
        seen = {snapshot["name"] for snapshot in snapshots}
        missing = sorted(REQUIRED_CASES - seen)
        extra = sorted(seen - REQUIRED_CASES)
        if missing or extra:
            raise SnapshotError(f"case set mismatch missing={missing} extra={extra}")
        snapshots.sort(key=lambda item: item["name"])
        write_actual(out_dir, snapshots)
        diffs = compare_or_update(root, snapshots, update=args.update)
        if diffs:
            (out_dir / "snapshot_diffs.txt").write_text("\n".join(diffs) + "\n")
            raise SnapshotError(
                "backend runtime preset snapshots changed; see "
                f"{out_dir / 'snapshot_diffs.txt'}"
            )
    except SnapshotError as exc:
        print(f"BACKEND PRESET SNAPSHOT FAIL: {out_dir}: {exc}", file=sys.stderr)
        return 1

    print(f"BACKEND PRESET SNAPSHOT PASS: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
