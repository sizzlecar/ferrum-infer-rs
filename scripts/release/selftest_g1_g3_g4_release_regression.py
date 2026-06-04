#!/usr/bin/env python3
"""Self-test for G1/G3/G4 release regression artifact validators."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from g1_g3_g4_release_regression import validate_cpu_root, validate_cuda_root


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_cpu_root(root: Path) -> Path:
    cpu = root / "cpu"
    cpu.mkdir(parents=True)
    (cpu / "cpu-cli.log").write_text("cpu cli ok\n")
    (cpu / "cpu-serve.log").write_text("cpu serve ok\n")
    write_json(
        cpu / "cpu-correctness.json",
        {
            "status": "pass",
            "backend": "cpu",
            "ferrum_run_one_shot": True,
            "ferrum_serve_chat": True,
            "openai_nonstream": True,
            "openai_stream": True,
            "context_limit_400": True,
            "stream_done_count": 1,
        },
    )
    return cpu


def make_cuda_root(root: Path) -> Path:
    cuda = root / "cuda"
    cuda.mkdir(parents=True)
    (cuda / "cuda-cli.log").write_text("cuda cli ok\n")
    (cuda / "cuda-serve.log").write_text("cuda serve ok\n")
    write_json(
        cuda / "cuda-correctness.json",
        {
            "status": "pass",
            "backend": "cuda",
            "ferrum_run_one_shot": True,
            "ferrum_serve_chat": True,
            "openai_nonstream": True,
            "openai_stream": True,
            "context_limit_400": True,
            "stream_done_count": 1,
            "g1_vllm_migration": True,
            "g3_cache_product": True,
            "g4_lora_inference": True,
        },
    )
    write_json(
        cuda / "cuda-performance.json",
        {
            "status": "pass",
            "backend": "cuda",
            "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            "concurrency_cells": [1, 4, 16, 32],
        },
    )
    for rel in [
        "g1-vllm-migration/gate.json",
        "g3-cache-product-small/gate.json",
        "g4-lora-inference-small/gate.json",
        "g0-cuda-smoke/g0_cuda4090_smoke.gate.json",
        "g0-cuda-full/g0_cuda4090_full.gate.json",
    ]:
        write_json(cuda / rel, {"status": "pass"})
    write_json(cuda / "g0-cuda-full/manifest.json", {"artifact_verdict": "pass"})
    write_json(
        cuda / "g0-cuda-full/summary.json",
        {
            "performance_regression_gates": {
                "observed_concurrency_cells": [1, 4, 16, 32],
                "concurrency_cells_ok": True,
                "cases": {"candidate": {"ok": True}},
            }
        },
    )
    return cuda


def expect_failure(fn, needle: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r} in {exc!r}") from exc
    else:
        raise AssertionError("expected validation failure")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-g1g3g4-selftest-") as td:
        root = Path(td)
        cpu = make_cpu_root(root)
        cpu_check = validate_cpu_root(cpu)
        assert cpu_check["correctness"]["backend"] == "cpu"
        cpu_bad = make_cpu_root(root / "cpu-bad-root")
        write_json(
            cpu_bad / "cpu-correctness.json",
            {
                **json.loads((cpu_bad / "cpu-correctness.json").read_text()),
                "stream_done_count": 2,
            },
        )
        expect_failure(lambda: validate_cpu_root(cpu_bad), "exactly one stream")

        cuda = make_cuda_root(root)
        cuda_check = validate_cuda_root(cuda)
        assert len(cuda_check["required_gates"]) == 5
        cuda_bad = make_cuda_root(root / "cuda-bad-root")
        write_json(
            cuda_bad / "g0-cuda-full/summary.json",
            {
                "performance_regression_gates": {
                    "observed_concurrency_cells": [1, 32],
                    "concurrency_cells_ok": False,
                    "cases": {"candidate": {"ok": True}},
                }
            },
        )
        expect_failure(lambda: validate_cuda_root(cuda_bad), "missing full CUDA concurrency cells")
    print("G1/G3/G4 RELEASE REGRESSION SELFTEST PASS")


if __name__ == "__main__":
    main()
