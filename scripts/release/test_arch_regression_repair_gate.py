#!/usr/bin/env python3
"""Final validator for the test-architecture regression repair goal.

Goal doc:
  docs/goals/test-architecture-regression-repair-2026-06-11/GOAL.md

Required final PASS line:
  TEST_ARCH_REGRESSION_REPAIR PASS: <out_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

PASS_PREFIX = "TEST_ARCH_REGRESSION_REPAIR PASS"
SELFTEST_PASS = "TEST_ARCH_REGRESSION_REPAIR SELFTEST PASS"

REQUIRED_CUDA_FEATURES = {
    "cuda",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
    "fa2-source",
}
REQUIRED_REPRO_COVERAGE = {
    ("metal", "run"),
    ("metal", "serve"),
    ("cuda", "run"),
    ("cuda", "serve"),
    ("cuda", "decode_over_context"),
}
REQUIRED_PRODUCT_ENTRYPOINTS = {
    ("metal", "run"),
    ("metal", "serve"),
    ("cuda", "run"),
    ("cuda", "serve"),
}


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    if not path.exists():
        raise ValidationError(f"missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid json in {path}: {exc}") from exc


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def resolve_artifact_path(root: Path, out_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [out_dir / path, root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def require_sha256(value: Any, failures: list[str], label: str) -> None:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        failures,
        f"{label}: expected lowercase SHA256 hex",
    )


def matrix_all_pass(path: Path, failures: list[str], label: str) -> None:
    data = load_json(path)
    models = data.get("models")
    require(isinstance(models, list) and models, failures, f"{label}: no models")
    for row in models or []:
        model_id = row.get("id", "<unknown>")
        platforms = row.get("platforms") or {}
        require(
            isinstance(platforms, dict) and bool(platforms),
            failures,
            f"{label}: {model_id} no platforms",
        )
        for platform, status in platforms.items():
            require(
                status == "PASS",
                failures,
                f"{label}: {model_id} {platform} status {status!r} != PASS",
            )
        detail = str(row.get("detail", "")).lower()
        require("degenerate" not in detail, failures, f"{label}: {model_id} degeneration detail")


def validate_source(out_dir: Path, failures: list[str]) -> None:
    source = load_json(out_dir / "source.json")
    require(source.get("schema_version") == 1, failures, "source.json: schema_version must be 1")
    require(isinstance(source.get("git_sha"), str) and len(source["git_sha"]) >= 8, failures, "source.json: missing git_sha")
    require(source.get("contains_origin_main_0db64121") is True, failures, "source.json: must confirm origin/main 0db64121 is included")
    require(source.get("contains_handoff_ed4c8d87") is True, failures, "source.json: must confirm handoff ed4c8d87 is included")
    require(isinstance(source.get("git_status_short"), list), failures, "source.json: git_status_short must list dirty files")
    hashes = source.get("binary_sha256") or {}
    require(isinstance(hashes, dict), failures, "source.json: binary_sha256 must be an object")
    for key in ("metal", "cuda"):
        require_sha256(hashes.get(key), failures, f"source.json binary_sha256.{key}")


def validate_repro(out_dir: Path, failures: list[str]) -> None:
    repro = load_json(out_dir / "repro.json")
    require(repro.get("schema_version") == 1, failures, "repro.json: schema_version must be 1")
    entries = repro.get("entries")
    require(isinstance(entries, list) and entries, failures, "repro.json: entries required")
    coverage: set[tuple[str, str]] = set()
    for entry in entries or []:
        backend = entry.get("backend")
        entrypoint = entry.get("entrypoint")
        if isinstance(backend, str) and isinstance(entrypoint, str):
            coverage.add((backend, entrypoint))
        require(entry.get("status") in {"reproduced", "fixed", "not_reproduced_with_evidence"}, failures, f"repro.json: bad status for {entry.get('id')}")
        require(entry.get("artifact"), failures, f"repro.json: {entry.get('id')} missing artifact")
        require(entry.get("reason"), failures, f"repro.json: {entry.get('id')} missing reason")
    missing = sorted(REQUIRED_REPRO_COVERAGE - coverage)
    require(not missing, failures, f"repro.json: missing required coverage {missing}")


def validate_metal(root: Path, out_dir: Path, failures: list[str]) -> None:
    metal = load_json(out_dir / "metal.json")
    require(metal.get("schema_version") == 1, failures, "metal.json: schema_version must be 1")
    pass_line = metal.get("l1_metal_pass_line", "")
    require(
        isinstance(pass_line, str) and pass_line.startswith("TEST_ARCH L1_METAL PASS:"),
        failures,
        "metal.json: missing TEST_ARCH L1_METAL PASS line",
    )
    require_sha256(metal.get("binary_sha256"), failures, "metal.json binary_sha256")
    matrix_path = resolve_artifact_path(root, out_dir, metal.get("matrix_json"))
    require(matrix_path is not None and matrix_path.exists(), failures, "metal.json: matrix_json missing")
    if matrix_path is not None and matrix_path.exists():
        matrix_all_pass(matrix_path, failures, "metal matrix")
    require(metal.get("fresh_after_repair") is True, failures, "metal.json: fresh_after_repair must be true")
    require(metal.get("degeneration_detector_passed") is True, failures, "metal.json: degeneration detector must pass")
    entrypoints = metal.get("entrypoints") or {}
    require(entrypoints.get("run") is True, failures, "metal.json: ferrum run not proven")
    require(entrypoints.get("serve") is True, failures, "metal.json: ferrum serve not proven")


def validate_cuda(root: Path, out_dir: Path, failures: list[str]) -> None:
    cuda = load_json(out_dir / "cuda.json")
    require(cuda.get("schema_version") == 1, failures, "cuda.json: schema_version must be 1")
    require_sha256(cuda.get("binary_sha256"), failures, "cuda.json binary_sha256")
    features = set(cuda.get("compiled_features") or [])
    missing_features = sorted(REQUIRED_CUDA_FEATURES - features)
    require(not missing_features, failures, f"cuda.json: missing compiled features {missing_features}")
    require(cuda.get("full_feature_build") is True, failures, "cuda.json: full_feature_build must be true")
    pass_line = cuda.get("l1_cuda_pass_line", "")
    require(
        isinstance(pass_line, str) and pass_line.startswith("TEST_ARCH L1_CUDA PASS:"),
        failures,
        "cuda.json: missing TEST_ARCH L1_CUDA PASS line",
    )
    require(cuda.get("hb09_verify_live") is True, failures, "cuda.json: hb-09 real verify-live missing")
    require(cuda.get("hb11_verify_live") is True, failures, "cuda.json: hb-11 real verify-live missing")
    require(cuda.get("placeholder_verify_live") is not True, failures, "cuda.json: placeholder verify-live is not valid")
    hardware = cuda.get("hardware") or {}
    require("4090" in str(hardware.get("gpu_name", "")), failures, "cuda.json: expected RTX 4090 hardware")
    require(hardware.get("nvidia_smi"), failures, "cuda.json: nvidia_smi evidence missing")
    require(hardware.get("nvcc_version"), failures, "cuda.json: nvcc_version evidence missing")
    entrypoints = cuda.get("entrypoints") or {}
    require(entrypoints.get("run") is True, failures, "cuda.json: ferrum run not proven")
    require(entrypoints.get("serve") is True, failures, "cuda.json: ferrum serve not proven")
    require(cuda.get("moe_fast_path_selected") is True, failures, "cuda.json: MoE fast path not selected")
    require(cuda.get("vpa_status") in {"selected", "available_disabled_for_ab"}, failures, "cuda.json: VPA status not proven")
    require(cuda.get("vpa_ab") is True, failures, "cuda.json: VPA on/off A/B missing")
    matrix_path = resolve_artifact_path(root, out_dir, cuda.get("matrix_json"))
    require(matrix_path is not None and matrix_path.exists(), failures, "cuda.json: matrix_json missing")
    if matrix_path is not None and matrix_path.exists():
        matrix_all_pass(matrix_path, failures, "cuda matrix")
    decode = cuda.get("decode_over_context") or {}
    turns = decode.get("turns")
    require(isinstance(turns, list) and len(turns) >= 6, failures, "cuda.json: decode probe needs >=6 turns")
    for index, turn in enumerate(turns or [], 1):
        for key in ("input_tokens", "output_tokens", "tok_s"):
            require(isinstance(turn.get(key), (int, float)) and turn[key] > 0, failures, f"cuda.json: turn {index} missing positive {key}")
        require(turn.get("coherent") is True, failures, f"cuda.json: turn {index} not coherent")
    bench = cuda.get("bench_serve") or {}
    require(bench.get("fail_on_error") is True, failures, "cuda.json: bench-serve must use --fail-on-error")
    require(bench.get("require_ci") is True, failures, "cuda.json: bench-serve must use --require-ci")
    require(bench.get("seed") == 9271, failures, "cuda.json: bench-serve seed must be 9271")
    require(bench.get("n_repeats") == 3, failures, "cuda.json: bench-serve n_repeats must be 3")


def validate_product_entrypoints(out_dir: Path, failures: list[str]) -> None:
    data = load_json(out_dir / "product_entrypoints.json")
    require(data.get("schema_version") == 1, failures, "product_entrypoints.json: schema_version must be 1")
    entries = data.get("entries")
    require(isinstance(entries, list) and entries, failures, "product_entrypoints.json: entries required")
    coverage: set[tuple[str, str]] = set()
    for entry in entries or []:
        backend = entry.get("backend")
        entrypoint = entry.get("entrypoint")
        if isinstance(backend, str) and isinstance(entrypoint, str) and entry.get("passed") is True:
            coverage.add((backend, entrypoint))
        require(entry.get("artifact"), failures, f"product_entrypoints.json: {backend}/{entrypoint} missing artifact")
    missing = sorted(REQUIRED_PRODUCT_ENTRYPOINTS - coverage)
    require(not missing, failures, f"product_entrypoints.json: missing passed coverage {missing}")


def run_validate(repo_root: Path, out_dir: Path) -> None:
    if not out_dir.is_dir():
        raise ValidationError(f"out_dir does not exist: {out_dir}")
    failures: list[str] = []
    validate_source(out_dir, failures)
    validate_repro(out_dir, failures)
    validate_metal(repo_root, out_dir, failures)
    validate_cuda(repo_root, out_dir, failures)
    validate_product_entrypoints(out_dir, failures)
    caveats = out_dir / "known_caveats.md"
    require(caveats.exists() and caveats.read_text(encoding="utf-8").strip(), failures, "known_caveats.md missing or empty")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise ValidationError(f"{len(failures)} check(s) failed")
    print(f"{PASS_PREFIX}: {out_dir}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out = root / "final"
        out.mkdir()
        fake_sha = "a" * 64
        write_json(
            out / "source.json",
            {
                "schema_version": 1,
                "git_sha": "0db6412142fcb6a759e1f4a1f928936beec65488",
                "contains_origin_main_0db64121": True,
                "contains_handoff_ed4c8d87": True,
                "git_status_short": [],
                "binary_sha256": {"metal": fake_sha, "cuda": fake_sha},
            },
        )
        write_json(
            out / "repro.json",
            {
                "schema_version": 1,
                "entries": [
                    {"id": "m-run", "backend": "metal", "entrypoint": "run", "status": "fixed", "artifact": "a", "reason": "covered"},
                    {"id": "m-serve", "backend": "metal", "entrypoint": "serve", "status": "fixed", "artifact": "b", "reason": "covered"},
                    {"id": "c-run", "backend": "cuda", "entrypoint": "run", "status": "fixed", "artifact": "c", "reason": "covered"},
                    {"id": "c-serve", "backend": "cuda", "entrypoint": "serve", "status": "fixed", "artifact": "d", "reason": "covered"},
                    {"id": "c-decode", "backend": "cuda", "entrypoint": "decode_over_context", "status": "fixed", "artifact": "e", "reason": "covered"},
                ],
            },
        )
        matrix = {
            "schema_version": 1,
            "models": [{"id": "qwen3-moe", "platforms": {"cuda": "PASS", "metal": "PASS"}, "detail": ""}],
        }
        write_json(out / "metal_matrix.json", matrix)
        write_json(out / "cuda_matrix.json", matrix)
        write_json(
            out / "metal.json",
            {
                "schema_version": 1,
                "l1_metal_pass_line": "TEST_ARCH L1_METAL PASS: out/metal",
                "binary_sha256": fake_sha,
                "matrix_json": "metal_matrix.json",
                "fresh_after_repair": True,
                "degeneration_detector_passed": True,
                "entrypoints": {"run": True, "serve": True},
            },
        )
        write_json(
            out / "cuda.json",
            {
                "schema_version": 1,
                "binary_sha256": fake_sha,
                "compiled_features": sorted(REQUIRED_CUDA_FEATURES),
                "full_feature_build": True,
                "l1_cuda_pass_line": "TEST_ARCH L1_CUDA PASS: out/cuda",
                "hb09_verify_live": True,
                "hb11_verify_live": True,
                "placeholder_verify_live": False,
                "hardware": {
                    "gpu_name": "NVIDIA GeForce RTX 4090",
                    "nvidia_smi": "ok",
                    "nvcc_version": "ok",
                },
                "entrypoints": {"run": True, "serve": True},
                "moe_fast_path_selected": True,
                "vpa_status": "selected",
                "vpa_ab": True,
                "matrix_json": "cuda_matrix.json",
                "decode_over_context": {
                    "turns": [
                        {"input_tokens": 100 + i, "output_tokens": 120, "tok_s": 20.0, "coherent": True}
                        for i in range(6)
                    ]
                },
                "bench_serve": {
                    "fail_on_error": True,
                    "require_ci": True,
                    "seed": 9271,
                    "n_repeats": 3,
                },
            },
        )
        write_json(
            out / "product_entrypoints.json",
            {
                "schema_version": 1,
                "entries": [
                    {"backend": "metal", "entrypoint": "run", "passed": True, "artifact": "a"},
                    {"backend": "metal", "entrypoint": "serve", "passed": True, "artifact": "b"},
                    {"backend": "cuda", "entrypoint": "run", "passed": True, "artifact": "c"},
                    {"backend": "cuda", "entrypoint": "serve", "passed": True, "artifact": "d"},
                ],
            },
        )
        (out / "known_caveats.md").write_text("M3 vLLM parity remains separate.\n", encoding="utf-8")
        run_validate(root, out)
    print(SELFTEST_PASS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
            return 0
        if args.validate:
            run_validate(args.repo_root.resolve(), args.validate.resolve())
            return 0
        parser.error("choose --self-test or --validate OUT_DIR")
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
