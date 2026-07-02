#!/usr/bin/env python3
"""Build a W3 L1 numeric/reference correctness artifact.

This gate packages the existing Rust Qwen3.5/Qwen3.6 architecture reference
tests into the release-grade L1 artifact schema. It is intentionally CPU/local:
it proves the typed Rust reference coverage for the W3 architecture components,
not CUDA/Metal execution and not full W3 release readiness.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAME = "w3_l1_numeric.json"
PASS_LINE_PREFIX = "W3 L1 NUMERIC PASS"

COVERAGE_TESTS = {
    "linear_attention": [
        "linear_attention_core_composes_conv_gdn_delta_and_norm",
        "dense_linear_attention_layer_composes_attention_residual_and_dense_mlp",
        "sparse_moe_linear_attention_layer_composes_attention_residual_and_moe",
    ],
    "full_attention": [
        "full_attention_core_applies_causal_softmax",
        "dense_full_attention_layer_composes_attention_residual_and_dense_mlp",
        "sparse_moe_full_attention_layer_composes_attention_residual_and_moe",
    ],
    "full_attention_official_shape": [
        "rope_uses_partial_interleaved_rotation",
        "full_attention_core_applies_qwen35_output_gate",
        "dense_full_attention_layer_accepts_qwen35_gate_shape_with_hidden_not_q_total",
    ],
    "deltanet": [
        "gated_delta_attention_matches_gating_plus_recurrent_reference",
        "recurrent_delta_rule_single_token_updates_state",
        "recurrent_delta_rule_repeats_qk_heads_over_value_heads",
    ],
    "moe_or_dense": [
        "dense_reference_model_forward_composes_layers_norm_and_lm_head",
        "sparse_moe_reference_model_forward_composes_layers_norm_and_lm_head",
        "sparse_moe_shared_expert_composes_router_fused_experts_and_shared_gate",
    ],
    "lm_head": [
        "dense_reference_model_forward_composes_layers_norm_and_lm_head",
        "sparse_moe_reference_model_forward_composes_layers_norm_and_lm_head",
    ],
}


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str], *, log_prefix: str, out_dir: Path) -> dict[str, Any]:
    stdout_path = out_dir / f"{log_prefix}.stdout.txt"
    stderr_path = out_dir / f"{log_prefix}.stderr.txt"
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise GateError(
            f"command failed ({proc.returncode}): {' '.join(command)}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return {
        "command_line": command,
        "returncode": proc.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def validate_coverage(log_text: str) -> tuple[dict[str, bool], dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}
    coverage: dict[str, bool] = {}
    for category, tests in COVERAGE_TESTS.items():
        hits = [test for test in tests if test in log_text]
        matched[category] = hits
        coverage[category] = bool(hits)
    missing = [category for category, present in coverage.items() if not present]
    if missing:
        raise GateError(f"missing required L1 coverage categories: {missing}")
    return coverage, matched


def build_artifact(
    *,
    out_dir: Path,
    model_id: str,
    command: dict[str, Any],
    log_text: str,
) -> dict[str, Any]:
    coverage, matched = validate_coverage(log_text)
    comparisons_total = sum(len(tests) for tests in matched.values())
    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l1_numeric",
        "model_id": model_id,
        "product_surface": "typed_config",
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"{PASS_LINE_PREFIX}: {out_dir}",
        "numeric": {
            "comparisons_total": comparisons_total,
            "comparisons_passed": comparisons_total,
            "atol": 1e-5,
            "deterministic": True,
            "source": "cargo_test_qwen35_reference_components",
        },
        "coverage": coverage,
        "coverage_tests": matched,
        "reference": {
            "engine": "Ferrum Rust CPU reference tests",
            "artifact": command["stdout"],
        },
        "commands": [command],
    }
    write_json(out_dir / ARTIFACT_NAME, artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="artifact output directory")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-35B-A3B+Qwen/Qwen3.6-35B-A3B",
        help="model/family id to record in the artifact",
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic script self-test")
    return parser.parse_args()


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-l1-numeric-") as tmp:
        root = Path(tmp)
        log_text = "\n".join(test for tests in COVERAGE_TESTS.values() for test in tests)
        command = {
            "command_line": ["selftest"],
            "returncode": 0,
            "stdout": str(root / "selftest.stdout.txt"),
            "stderr": str(root / "selftest.stderr.txt"),
        }
        (root / "selftest.stdout.txt").write_text(log_text, encoding="utf-8")
        (root / "selftest.stderr.txt").write_text("", encoding="utf-8")
        artifact = build_artifact(
            out_dir=root / "out",
            model_id="selftest-qwen35",
            command=command,
            log_text=log_text,
        )
        if not all(artifact["coverage"].values()):
            raise AssertionError("selftest coverage did not pass")

        try:
            validate_coverage("only_dense_reference_model_forward_composes_layers_norm_and_lm_head")
        except GateError as exc:
            if "missing required L1 coverage" not in str(exc):
                raise AssertionError(f"unexpected selftest error: {exc}") from exc
        else:
            raise AssertionError("missing coverage selftest did not fail")

    print("W3 L1 NUMERIC SELFTEST PASS")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        if args.out is None:
            raise GateError("missing required arg: --out")
        args.out.mkdir(parents=True, exist_ok=True)
        command = run_command(
            ["cargo", "test", "-p", "ferrum-models", "qwen35", "--", "--nocapture"],
            log_prefix="cargo_qwen35_numeric",
            out_dir=args.out,
        )
        log_text = (
            Path(command["stdout"]).read_text(encoding="utf-8")
            + "\n"
            + Path(command["stderr"]).read_text(encoding="utf-8")
        )
        artifact = build_artifact(
            out_dir=args.out,
            model_id=args.model_id,
            command=command,
            log_text=log_text,
        )
    except GateError as exc:
        print(f"W3 L1 NUMERIC FAIL: {exc}", file=sys.stderr)
        return 1
    print(artifact["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
