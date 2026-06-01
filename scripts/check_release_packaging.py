#!/usr/bin/env python3
"""Static consistency checks for release assets and Homebrew packaging.

This guard intentionally avoids network calls and does not prove that a tag
workflow has succeeded. It catches local release-packaging drift before the
tagged release path runs.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_ASSET_SM89 = "ferrum-linux-x86_64-cuda-sm89"


def _read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing required file: {path}")
        return ""


def _require(text: str, needle: str, path: Path, errors: list[str], label: str) -> None:
    if needle not in text:
        errors.append(f"{path}: missing {label}: {needle!r}")


def check(root: Path) -> list[str]:
    errors: list[str] = []

    release_cuda = _read(root / ".github/workflows/release-cuda.yml", errors)
    release = _read(root / ".github/workflows/release.yml", errors)
    release_sh = _read(root / "scripts/release.sh", errors)
    readme = _read(root / "README.md", errors)
    readme_zh = _read(root / "README_zh.md", errors)
    candidate = _read(
        root
        / "docs/bench/dev-loop-product-api-goal-progress-20260601/release-candidate-0.7.3-20260601.md",
        errors,
    )
    readiness = _read(
        root
        / "docs/bench/dev-loop-product-api-goal-progress-20260601/release-readiness-20260601.md",
        errors,
    )

    _require(
        release_cuda,
        'echo "asset=ferrum-linux-x86_64-cuda-sm${cuda_compute_cap}"',
        root / ".github/workflows/release-cuda.yml",
        errors,
        "CUDA release asset output",
    )
    _require(
        release_cuda,
        "--features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source",
        root / ".github/workflows/release-cuda.yml",
        errors,
        "CUDA release feature set",
    )
    _require(
        release_cuda,
        "torch|python|vllm",
        root / ".github/workflows/release-cuda.yml",
        errors,
        "Torch/Python/vLLM link guard",
    )
    _require(
        release,
        "release-cuda.yml",
        root / ".github/workflows/release.yml",
        errors,
        "main release workflow CUDA handoff note",
    )

    _require(
        release_sh,
        'CUDA_BREW_COMPUTE_CAP="${CUDA_BREW_COMPUTE_CAP:-89}"',
        root / "scripts/release.sh",
        errors,
        "default Homebrew CUDA compute capability",
    )
    _require(
        release_sh,
        'CUDA_BREW_ASSET="ferrum-linux-x86_64-cuda-sm${CUDA_BREW_COMPUTE_CAP}"',
        root / "scripts/release.sh",
        errors,
        "Homebrew CUDA asset name",
    )
    _require(
        release_sh,
        'wait_workflow_success "release-cuda.yml"',
        root / "scripts/release.sh",
        errors,
        "release-cuda workflow wait",
    )
    _require(
        release_sh,
        "SHA_LINUX_CUDA",
        root / "scripts/release.sh",
        errors,
        "Linux CUDA sha256 fetch",
    )
    _require(
        release_sh,
        "Formula/ferrum-cuda.rb",
        root / "scripts/release.sh",
        errors,
        "ferrum-cuda formula write",
    )
    _require(
        release_sh,
        "class FerrumCuda < Formula",
        root / "scripts/release.sh",
        errors,
        "ferrum-cuda formula class",
    )
    _require(
        release_sh,
        'conflicts_with "ferrum"',
        root / "scripts/release.sh",
        errors,
        "ferrum-cuda conflict guard",
    )
    _require(
        release_sh,
        "brew install ferrum-cuda",
        root / "scripts/release.sh",
        errors,
        "release completion CUDA brew instruction",
    )

    for path, text in [
        (root / "README.md", readme),
        (root / "README_zh.md", readme_zh),
        (
            root
            / "docs/bench/dev-loop-product-api-goal-progress-20260601/release-candidate-0.7.3-20260601.md",
            candidate,
        ),
        (
            root
            / "docs/bench/dev-loop-product-api-goal-progress-20260601/release-readiness-20260601.md",
            readiness,
        ),
    ]:
        _require(text, "ferrum-cuda", path, errors, "CUDA Homebrew package mention")
        _require(text, CUDA_ASSET_SM89, path, errors, "CUDA sm89 release asset mention")

    return errors


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write(
            root / ".github/workflows/release-cuda.yml",
            'echo "asset=ferrum-linux-x86_64-cuda-sm${cuda_compute_cap}"\n'
            "cargo build --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source\n"
            "grep -Ei 'torch|python|vllm' file\n",
        )
        _write(root / ".github/workflows/release.yml", "release-cuda.yml\n")
        _write(
            root / "scripts/release.sh",
            'CUDA_BREW_COMPUTE_CAP="${CUDA_BREW_COMPUTE_CAP:-89}"\n'
            'CUDA_BREW_ASSET="ferrum-linux-x86_64-cuda-sm${CUDA_BREW_COMPUTE_CAP}"\n'
            'wait_workflow_success "release-cuda.yml"\n'
            "SHA_LINUX_CUDA=abc\n"
            "Formula/ferrum-cuda.rb\n"
            "class FerrumCuda < Formula\n"
            'conflicts_with "ferrum"\n'
            "brew install ferrum-cuda\n",
        )
        for path in [
            root / "README.md",
            root / "README_zh.md",
            root
            / "docs/bench/dev-loop-product-api-goal-progress-20260601/release-candidate-0.7.3-20260601.md",
            root
            / "docs/bench/dev-loop-product-api-goal-progress-20260601/release-readiness-20260601.md",
        ]:
            _write(path, f"brew install ferrum-cuda\n{CUDA_ASSET_SM89}\n")
        assert check(root) == []

        bad = root / "README.md"
        bad.write_text("brew install ferrum-cuda\n", encoding="utf-8")
        assert any(CUDA_ASSET_SM89 in err for err in check(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return 0

    errors = check(args.root.resolve())
    if errors:
        for err in errors:
            print(err)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
