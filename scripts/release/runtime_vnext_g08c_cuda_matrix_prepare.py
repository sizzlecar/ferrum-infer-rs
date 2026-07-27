#!/usr/bin/env python3
"""Prepare source-bound G08C Qwen3-30B-A3B model-matrix inputs."""

from pathlib import Path

from runtime_vnext_g08b_cuda_matrix_prepare import BackendSpec, main


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
MODEL_KEY = "m3-qwen3-30b-a3b"
SERVED_MODEL_NAME = "m3-qwen3-30b-a3b"
SHARED_PREPARE_PATH = (
    "scripts/release/runtime_vnext_g08b_cuda_matrix_prepare.py"
)
SCRIPT_REPO_PATH = "scripts/release/runtime_vnext_g08c_cuda_matrix_prepare.py"

CUDA_SPEC = BackendSpec(
    backend="cuda",
    model_key=MODEL_KEY,
    model_label="G08C M3",
    model_lock_path=SCRIPT_DIR
    / "configs/runtime_vnext_g08c_m3_cuda.models.lock.json",
    lock_id="runtime-vnext-g08c-m3-cuda-v1",
    weight_revision="9b534e4318b7ebc3c961a839f13eb18b1833f441",
    weight_format="gptq_int4",
    weight_file_count=6,
    semantic_file_count=5,
    build_ready_prefix="FERRUM RUNTIME VNEXT G08C CUDA BUILD READY",
    manifest_ready_prefix="FERRUM RUNTIME VNEXT G08C CUDA MANIFEST READY",
    prepare_selftest_pass_line=(
        "FERRUM RUNTIME VNEXT G08C CUDA PREPARE SELFTEST PASS"
    ),
    probe_commands={
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version", "--verbose"],
        "nvcc": ["nvcc", "--version"],
        "nvidia_smi": ["nvidia-smi"],
    },
    typed_run_config={
        "backend": "cuda",
        "gpu_devices": [0],
        "gpu_memory_utilization": 0.9,
    },
    typed_serve_config={
        "backend": "cuda",
        "gpu_devices": [0],
        "gpu_memory_utilization": 0.9,
        "served_model_name": SERVED_MODEL_NAME,
    },
    run_extra_args=(
        "--gpu-devices",
        "0",
        "--gpu-memory-utilization",
        "0.90",
    ),
    serve_extra_args=(
        "--gpu-devices",
        "0",
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ),
    source_repo_paths=(
        SHARED_PREPARE_PATH,
        SCRIPT_REPO_PATH,
    ),
    selftest_temp_prefix="ferrum-g08c-cuda-prepare-",
)

METAL_SPEC = BackendSpec(
    backend="metal",
    model_key=MODEL_KEY,
    model_label="G08C M3",
    model_lock_path=SCRIPT_DIR
    / "configs/runtime_vnext_g08c_m3_metal.models.lock.json",
    lock_id="runtime-vnext-g08c-m3-metal-v1",
    weight_revision="e4d4bafdfb96a411a163846265362aceb0b9c63a",
    weight_format="gguf_q4_k_m",
    weight_file_count=1,
    semantic_file_count=5,
    build_ready_prefix="FERRUM RUNTIME VNEXT G08C METAL BUILD READY",
    manifest_ready_prefix="FERRUM RUNTIME VNEXT G08C METAL MANIFEST READY",
    prepare_selftest_pass_line=(
        "FERRUM RUNTIME VNEXT G08C METAL PREPARE SELFTEST PASS"
    ),
    probe_commands={
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version", "--verbose"],
        "xcodebuild": ["xcodebuild", "-version"],
        "system_profiler": [
            "system_profiler",
            "SPDisplaysDataType",
        ],
    },
    typed_run_config={
        "backend": "metal",
        "gpu_memory_utilization": 0.9,
    },
    typed_serve_config={
        "backend": "metal",
        "gpu_memory_utilization": 0.9,
        "served_model_name": SERVED_MODEL_NAME,
    },
    run_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
    ),
    serve_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ),
    source_repo_paths=(
        SHARED_PREPARE_PATH,
        SCRIPT_REPO_PATH,
        "scripts/release/runtime_vnext_g08c_metal_matrix_prepare.py",
    ),
    selftest_temp_prefix="ferrum-g08c-metal-prepare-",
)

BACKEND_SPECS = {
    CUDA_SPEC.backend: CUDA_SPEC,
    METAL_SPEC.backend: METAL_SPEC,
}


if __name__ == "__main__":
    raise SystemExit(
        main(
            backend_specs=BACKEND_SPECS,
            error_label="runtime_vnext_g08c_matrix_prepare",
        )
    )
