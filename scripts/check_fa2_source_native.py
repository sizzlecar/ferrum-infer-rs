#!/usr/bin/env python3
"""Check that the product fa2-source path is native and repo-owned.

This guard is intentionally narrow: legacy microbench scripts may still build
historical FlashAttention-source shims, but the ferrum-kernels `fa2-source`
feature must not require external FlashAttention/CUTLASS source trees.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


FORBIDDEN_BUILD_RS_TOKENS = [
    "FERRUM_FA2_SRC_DIR",
    "FA_SRC_DIR",
    "FERRUM_CUTLASS_INCLUDE_DIR",
    "CUTLASS_INCLUDE_DIR",
    "vllm-flash-attention",
    "flash_fwd_split_hdim128",
    "fa2_source/stubs",
]

REQUIRED_BUILD_RS_TOKENS = [
    "kernels/fa2_source/ferrum_fa2_paged_varlen.cu",
    "compile_fa2_source",
    "fa2_source",
]

REQUIRED_KERNEL_TOKENS = [
    "ferrum_fa2_paged_varlen_fwd",
    "ferrum_fa2_paged_varlen_kernel_f16",
    "warp-partition online softmax",
    "partial_out",
    "partial_m",
    "partial_l",
    "local_m",
    "local_l",
]

FORBIDDEN_KERNEL_TOKENS = [
    "s_scores[kv_pos]",
    "static_cast<size_t>(max_kv_len) * sizeof(float)",
]

FORBIDDEN_RUNTIME_TOKENS = [
    'vars.contains_key("FERRUM_FA2_DIRECT_FFI_SHIM") || fa2_source',
    "FERRUM_FA2_DIRECT_FFI_SHIM\") || fa2_source",
]

REQUIRED_RUNTIME_TOKENS = [
    "pub(crate) fa2_source: bool",
    'let fa2_source = trueish(vars.get("FERRUM_FA2_SOURCE"));',
    'None => vars.contains_key("FERRUM_FA2_DIRECT_FFI_SHIM"),',
    "fa2_source,",
]

REQUIRED_UNIFIED_LAYER_TOKENS = [
    "let use_fa2_c_abi = self.runtime_env.fa2_source || self.runtime_env.fa2_direct_ffi;",
    "self.runtime_env.fa_layout_varlen || use_fa2_c_abi",
    "if use_fa2_c_abi {",
]

REQUIRED_KV_TOKENS = [
    "|| self.runtime_env.fa2_source",
]


class CheckError(Exception):
    pass


def require_file(path: Path) -> str:
    if not path.is_file():
        raise CheckError(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def check_repo(root: Path) -> None:
    build_rs = require_file(root / "crates/ferrum-kernels/build.rs")
    kernel = require_file(
        root / "crates/ferrum-kernels/kernels/fa2_source/ferrum_fa2_paged_varlen.cu"
    )
    runtime = require_file(root / "crates/ferrum-models/src/models/qwen3_moe_runtime.rs")
    unified_layer = require_file(
        root / "crates/ferrum-models/src/models/qwen3_moe_forward_unified_layer.rs"
    )
    kv = require_file(root / "crates/ferrum-models/src/models/qwen3_moe/kv.rs")

    for token in FORBIDDEN_BUILD_RS_TOKENS:
        if token in build_rs:
            raise CheckError(f"forbidden product fa2-source build token in build.rs: {token}")

    for token in REQUIRED_BUILD_RS_TOKENS:
        if token not in build_rs:
            raise CheckError(f"missing product fa2-source build token in build.rs: {token}")

    for token in REQUIRED_KERNEL_TOKENS:
        if token not in kernel:
            raise CheckError(f"missing native FA2 kernel token: {token}")

    for token in FORBIDDEN_KERNEL_TOKENS:
        if token in kernel:
            raise CheckError(f"forbidden full-score reader token in native FA2 kernel: {token}")

    for token in FORBIDDEN_RUNTIME_TOKENS:
        if token in runtime:
            raise CheckError(f"forbidden runtime source/direct-FFI conflation token: {token}")

    for token in REQUIRED_RUNTIME_TOKENS:
        if token not in runtime:
            raise CheckError(f"missing runtime source/direct-FFI separation token: {token}")

    for token in REQUIRED_UNIFIED_LAYER_TOKENS:
        if token not in unified_layer:
            raise CheckError(f"missing unified-layer FA2 C ABI dispatch token: {token}")

    for token in REQUIRED_KV_TOKENS:
        if token not in kv:
            raise CheckError(f"missing FA2 source FA-layout pool allocation token: {token}")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_dir = root / "crates/ferrum-kernels"
        kernel_dir = build_dir / "kernels/fa2_source"
        kernel_dir.mkdir(parents=True)
        model_dir = root / "crates/ferrum-models/src/models"
        (model_dir / "qwen3_moe").mkdir(parents=True)
        (build_dir / "build.rs").write_text(
            'fn compile_fa2_source() { let _ = "kernels/fa2_source/ferrum_fa2_paged_varlen.cu"; let _ = "fa2_source"; }\n',
            encoding="utf-8",
        )
        (kernel_dir / "ferrum_fa2_paged_varlen.cu").write_text(
            "// warp-partition online softmax\n"
            "void ferrum_fa2_paged_varlen_fwd(); __global__ void ferrum_fa2_paged_varlen_kernel_f16() {}\n"
            "float *partial_out; float *partial_m; float *partial_l; float local_m; float local_l;\n",
            encoding="utf-8",
        )
        (model_dir / "qwen3_moe_runtime.rs").write_text(
            'pub(crate) fa2_source: bool\n'
            'let fa2_source = trueish(vars.get("FERRUM_FA2_SOURCE"));\n'
            'None => vars.contains_key("FERRUM_FA2_DIRECT_FFI_SHIM"),\n'
            'fa2_source,\n',
            encoding="utf-8",
        )
        (model_dir / "qwen3_moe_forward_unified_layer.rs").write_text(
            "let use_fa2_c_abi = self.runtime_env.fa2_source || self.runtime_env.fa2_direct_ffi;\n"
            "self.runtime_env.fa_layout_varlen || use_fa2_c_abi\n"
            "if use_fa2_c_abi {\n",
            encoding="utf-8",
        )
        (model_dir / "qwen3_moe/kv.rs").write_text(
            "|| self.runtime_env.fa2_source\n",
            encoding="utf-8",
        )
        check_repo(root)

        (build_dir / "build.rs").write_text(
            'fn compile_fa2_source() { let _ = "FERRUM_FA2_SRC_DIR"; }\n',
            encoding="utf-8",
        )
        try:
            check_repo(root)
        except CheckError:
            pass
        else:
            raise AssertionError("self-test expected forbidden token failure")

        (build_dir / "build.rs").write_text(
            'fn compile_fa2_source() { let _ = "kernels/fa2_source/ferrum_fa2_paged_varlen.cu"; let _ = "fa2_source"; }\n',
            encoding="utf-8",
        )
        (model_dir / "qwen3_moe_runtime.rs").write_text(
            'pub(crate) fa2_source: bool\n'
            'let fa2_source = trueish(vars.get("FERRUM_FA2_SOURCE"));\n'
            'None => vars.contains_key("FERRUM_FA2_DIRECT_FFI_SHIM") || fa2_source,\n'
            'fa2_source,\n',
            encoding="utf-8",
        )
        try:
            check_repo(root)
        except CheckError:
            pass
        else:
            raise AssertionError("self-test expected source/direct-FFI conflation failure")
    print("check_fa2_source_native self-test ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    check_repo(args.repo.resolve())
    print("fa2-source native boundary ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
