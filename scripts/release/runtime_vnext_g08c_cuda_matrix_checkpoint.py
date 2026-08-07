#!/usr/bin/env python3
"""Validate canonical G08C Qwen3-30B-A3B C01-C21 model matrices."""

from pathlib import Path

from runtime_vnext_g08b_cuda_matrix_checkpoint import CheckpointSpec, main


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_KEY = "m3-qwen3-30b-a3b"

CUDA_SPEC = CheckpointSpec(
    backend="cuda",
    model_key=MODEL_KEY,
    model_label="G08C M3",
    model_lock_path=SCRIPT_DIR
    / "configs/runtime_vnext_g08c_m3_cuda.models.lock.json",
    weight_revision="9b534e4318b7ebc3c961a839f13eb18b1833f441",
    weight_file_count=6,
    semantic_file_count=5,
    checkpoint_id="runtime-vnext-g08c-cuda-model-matrix",
    checkpoint_label="G08C-CUDA-MATRIX",
    expected_case_count=120,
    required_client_concurrency=32,
    required_active_floor=32,
    required_active_duty_cycle=0.80,
    concurrency_cells=(1, 4, 16, 32),
    artifact_type_prefix="runtime_vnext_g08c_cuda_model_matrix",
    pass_prefix="FERRUM RUNTIME VNEXT G08C CUDA MODEL MATRIX PASS",
    fail_prefix="FERRUM RUNTIME VNEXT G08C CUDA MODEL MATRIX FAIL",
    selftest_pass_line=(
        "FERRUM RUNTIME VNEXT G08C CUDA MODEL MATRIX SELFTEST PASS"
    ),
    does_not_prove=(
        "G08C Metal Q4_K_M product path",
        "G08C legacy/reference parity",
        "G08C mutation and legacy-deletion acceptance",
        "G08C CUDA/Metal performance smoke",
        "G08C final PASS",
        "G09 formal performance",
        "G10 release readiness",
    ),
)

METAL_SPEC = CheckpointSpec(
    backend="metal",
    model_key=MODEL_KEY,
    model_label="G08C M3",
    model_lock_path=SCRIPT_DIR
    / "configs/runtime_vnext_g08c_m3_metal.models.lock.json",
    weight_revision="e4d4bafdfb96a411a163846265362aceb0b9c63a",
    weight_file_count=1,
    semantic_file_count=5,
    checkpoint_id="runtime-vnext-g08c-metal-model-matrix",
    checkpoint_label="G08C-METAL-MATRIX",
    expected_case_count=119,
    required_client_concurrency=16,
    required_active_floor=16,
    required_active_duty_cycle=0.80,
    concurrency_cells=(1, 4, 16),
    artifact_type_prefix="runtime_vnext_g08c_metal_model_matrix",
    pass_prefix="FERRUM RUNTIME VNEXT G08C METAL MODEL MATRIX PASS",
    fail_prefix="FERRUM RUNTIME VNEXT G08C METAL MODEL MATRIX FAIL",
    selftest_pass_line=(
        "FERRUM RUNTIME VNEXT G08C METAL MODEL MATRIX SELFTEST PASS"
    ),
    does_not_prove=(
        "current-HEAD G08C CUDA GPTQ-Int4 product path",
        "G08C legacy/reference parity",
        "G08C mutation and legacy-deletion acceptance",
        "G08C CUDA/Metal performance smoke",
        "G08C final PASS",
        "G09 formal performance",
        "G10 release readiness",
    ),
)

CHECKPOINT_SPECS = {
    CUDA_SPEC.backend: CUDA_SPEC,
    METAL_SPEC.backend: METAL_SPEC,
}


if __name__ == "__main__":
    raise SystemExit(main(checkpoint_specs=CHECKPOINT_SPECS))
