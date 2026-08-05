#!/usr/bin/env python3
"""Typed M1 specifications for the shared G08 model-matrix machinery."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from runtime_vnext_g08b_cuda_matrix_checkpoint import CheckpointSpec
from runtime_vnext_g08b_cuda_matrix_prepare import BackendSpec


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_KEY = "m1-qwen35-4b"
SERVED_MODEL_NAME = MODEL_KEY

CUDA_LOCK = SCRIPT_DIR / "configs/runtime_vnext_g08a_m1_cuda.models.lock.json"
METAL_LOCK = SCRIPT_DIR / "configs/runtime_vnext_g08a_m1_metal.models.lock.json"
CATALOG = SCRIPT_DIR / "configs/runtime_vnext_models.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class G08AModelLockError(RuntimeError):
    """The checked-in M1 lock lost required provenance or exact file identity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G08AModelLockError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == keys, f"{label} fields differ: {sorted(value)}")
    return value


SEMANTIC_FILES = {
    "README.md": (
        "1406be1b6b8fd8a6545870da516912804756593628a1d0fb0a7965211e82a7bb",
        77661,
    ),
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        7756,
    ),
    "config.json": (
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
        3161,
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        12807982,
    ),
    "tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        16710,
    ),
}

WEIGHT_FILES = {
    "cuda": {
        "config.json": SEMANTIC_FILES["config.json"],
        "model.safetensors-00001-of-00002.safetensors": (
            "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
            5329398688,
        ),
        "model.safetensors-00002-of-00002.safetensors": (
            "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
            3990429408,
        ),
        "model.safetensors.index.json": (
            "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
            76196,
        ),
        "tokenizer.json": SEMANTIC_FILES["tokenizer.json"],
        "tokenizer_config.json": SEMANTIC_FILES["tokenizer_config.json"],
    },
    "metal": {
        "Qwen3.5-4B-Q4_K_M.gguf": (
            "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
            2740937888,
        )
    },
}


def exact_files(value: Any, expected: dict[str, tuple[str, int]], label: str) -> None:
    require(isinstance(value, list), f"{label} must be an array")
    observed: dict[str, tuple[str, int]] = {}
    for index, raw in enumerate(value):
        row = exact_object(raw, {"path", "sha256", "size_bytes"}, f"{label}[{index}]")
        path = row["path"]
        require(isinstance(path, str) and path not in observed, f"{label} path is invalid")
        digest = row["sha256"]
        size = row["size_bytes"]
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest), f"{label} SHA invalid")
        require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label} size invalid")
        observed[path] = (digest, size)
    require(observed == expected, f"{label} identity differs")


def validate_model_lock_contract(backend: str) -> None:
    require(backend in {"cuda", "metal"}, f"unsupported G08A backend: {backend}")
    path = CUDA_LOCK if backend == "cuda" else METAL_LOCK
    document = exact_object(
        json.loads(path.read_text(encoding="utf-8")),
        {"schema_version", "lock_id", "provenance", "models"},
        f"{backend} lock",
    )
    require(document["schema_version"] == 1, f"{backend} lock schema differs")
    require(
        document["lock_id"] == f"runtime-vnext-g08a-m1-{backend}-v1",
        f"{backend} lock id differs",
    )
    provenance = exact_object(
        document["provenance"],
        {"source_git_sha", "source_catalog_sha256", "source_lock_sha256", "extraction"},
        f"{backend} lock provenance",
    )
    require(GIT_SHA_RE.fullmatch(provenance["source_git_sha"]) is not None, "source Git SHA invalid")
    require(provenance["source_catalog_sha256"] == sha256(CATALOG), "source catalog SHA differs")
    require(SHA256_RE.fullmatch(provenance["source_lock_sha256"]) is not None, "source lock SHA invalid")
    require(isinstance(provenance["extraction"], str) and provenance["extraction"], "extraction is absent")

    models = document["models"]
    require(isinstance(models, list) and len(models) == 1, f"{backend} lock model count differs")
    model = exact_object(
        models[0],
        {"key", "official_model_id", "role", "lanes"},
        f"{backend} model",
    )
    require(model["key"] == MODEL_KEY, f"{backend} model key differs")
    require(model["official_model_id"] == "Qwen/Qwen3.5-4B", "official model id differs")
    require(model["role"] == "primary_dense_hybrid_canary", "model role differs")
    lanes = exact_object(model["lanes"], {backend}, f"{backend} lanes")
    lane = exact_object(
        lanes[backend],
        {
            "catalog_lane_id",
            "repo",
            "revision",
            "format",
            "files",
            "semantic_source",
            "chat_template",
            "generation_config",
            "hardware_policy",
            "license",
        },
        f"{backend} lane",
    )
    expected_repo = "Qwen/Qwen3.5-4B" if backend == "cuda" else "unsloth/Qwen3.5-4B-GGUF"
    expected_revision = (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        if backend == "cuda"
        else "e87f176479d0855a907a41277aca2f8ee7a09523"
    )
    require(lane["catalog_lane_id"] == f"M1-{backend.upper()}", f"{backend} lane id differs")
    require(lane["repo"] == expected_repo and lane["revision"] == expected_revision, f"{backend} source differs")
    require(lane["format"] == ("bf16_safetensors" if backend == "cuda" else "gguf_q4_k_m"), f"{backend} format differs")
    exact_files(lane["files"], WEIGHT_FILES[backend], f"{backend} weight files")

    semantic = exact_object(lane["semantic_source"], {"repo", "revision", "files"}, f"{backend} semantic source")
    require(
        semantic["repo"] == "Qwen/Qwen3.5-4B"
        and semantic["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        f"{backend} semantic identity differs",
    )
    exact_files(semantic["files"], SEMANTIC_FILES, f"{backend} semantic files")
    require(
        lane["chat_template"]
        == {
            "source": "semantic_source",
            "repo": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "path": "tokenizer_config.json",
            "json_pointer": "/chat_template",
            "container_sha256": SEMANTIC_FILES["tokenizer_config.json"][0],
            "content_sha256": SEMANTIC_FILES["chat_template.jinja"][0],
            "content_bytes": SEMANTIC_FILES["chat_template.jinja"][1],
        },
        f"{backend} chat template differs",
    )
    require(
        lane["generation_config"]
        == {
            "source": "semantic_source",
            "repo": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "path": "generation_config.json",
            "present": False,
            "policy": "absent",
        },
        f"{backend} generation config differs",
    )
    require(
        lane["hardware_policy"]
        == ("cuda-g0-1x-rtx4090" if backend == "cuda" else "metal-reference-m1-max-32gb"),
        f"{backend} hardware policy differs",
    )
    license_row = exact_object(lane["license"], {"spdx", "source"}, f"{backend} license")
    require(license_row["spdx"] == "apache-2.0", f"{backend} license differs")
    require(
        license_row["source"] == f"https://huggingface.co/{expected_repo}/tree/{expected_revision}",
        f"{backend} license source differs",
    )

CUDA_PREPARE_SPEC = BackendSpec(
    backend="cuda",
    model_key=MODEL_KEY,
    model_label="G08A M1",
    model_lock_path=CUDA_LOCK,
    lock_id="runtime-vnext-g08a-m1-cuda-v1",
    weight_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    weight_format="bf16_safetensors",
    weight_file_count=6,
    semantic_file_count=5,
    build_ready_prefix="FERRUM RUNTIME VNEXT G08A CUDA BUILD READY",
    manifest_ready_prefix="FERRUM RUNTIME VNEXT G08A CUDA MANIFEST READY",
    prepare_selftest_pass_line="FERRUM RUNTIME VNEXT G08A CUDA PREPARE SELFTEST PASS",
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
    run_extra_args=("--gpu-devices", "0", "--gpu-memory-utilization", "0.90"),
    serve_extra_args=(
        "--gpu-devices",
        "0",
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ),
    source_repo_paths=(
        "scripts/release/runtime_vnext_g08a_matrix_specs.py",
        "scripts/release/runtime_vnext_g08a_cuda_matrix_prepare.py",
    ),
    selftest_temp_prefix="ferrum-g08a-cuda-prepare-",
)

METAL_PREPARE_SPEC = BackendSpec(
    backend="metal",
    model_key=MODEL_KEY,
    model_label="G08A M1",
    model_lock_path=METAL_LOCK,
    lock_id="runtime-vnext-g08a-m1-metal-v1",
    weight_revision="e87f176479d0855a907a41277aca2f8ee7a09523",
    weight_format="gguf_q4_k_m",
    weight_file_count=1,
    semantic_file_count=5,
    build_ready_prefix="FERRUM RUNTIME VNEXT G08A METAL BUILD READY",
    manifest_ready_prefix="FERRUM RUNTIME VNEXT G08A METAL MANIFEST READY",
    prepare_selftest_pass_line="FERRUM RUNTIME VNEXT G08A METAL PREPARE SELFTEST PASS",
    probe_commands={
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version", "--verbose"],
        "xcodebuild": ["xcodebuild", "-version"],
        "system_profiler": ["system_profiler", "SPDisplaysDataType"],
    },
    typed_run_config={"backend": "metal", "gpu_memory_utilization": 0.9},
    typed_serve_config={
        "backend": "metal",
        "gpu_memory_utilization": 0.9,
        "served_model_name": SERVED_MODEL_NAME,
    },
    run_extra_args=("--gpu-memory-utilization", "0.90"),
    serve_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ),
    source_repo_paths=(
        "scripts/release/runtime_vnext_g08a_matrix_specs.py",
        "scripts/release/runtime_vnext_g08a_metal_matrix_prepare.py",
    ),
    selftest_temp_prefix="ferrum-g08a-metal-prepare-",
)

PREPARE_SPECS = {
    CUDA_PREPARE_SPEC.backend: CUDA_PREPARE_SPEC,
    METAL_PREPARE_SPEC.backend: METAL_PREPARE_SPEC,
}

CUDA_CHECKPOINT_SPEC = CheckpointSpec(
    backend="cuda",
    model_key=MODEL_KEY,
    model_label="G08A M1",
    model_lock_path=CUDA_LOCK,
    weight_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    weight_file_count=6,
    semantic_file_count=5,
    checkpoint_id="runtime-vnext-g08a-cuda-model-matrix",
    checkpoint_label="G08A-CUDA-MATRIX",
    expected_case_count=703,
    required_client_concurrency=32,
    required_active_floor=32,
    required_active_duty_cycle=0.80,
    concurrency_cells=(1, 4, 16, 32),
    artifact_type_prefix="runtime_vnext_g08a_cuda_model_matrix",
    pass_prefix="FERRUM RUNTIME VNEXT G08A CUDA MODEL MATRIX PASS",
    fail_prefix="FERRUM RUNTIME VNEXT G08A CUDA MODEL MATRIX FAIL",
    selftest_pass_line="FERRUM RUNTIME VNEXT G08A CUDA MODEL MATRIX SELFTEST PASS",
    does_not_prove=(
        "G08A Metal Q4_K_M product path",
        "G08A numerical reference",
        "G08A mutation and legacy-deletion acceptance",
        "G08A CUDA/Metal performance smoke",
        "G08A final PASS",
        "G09 formal performance",
        "G10 release readiness",
    ),
)

METAL_CHECKPOINT_SPEC = CheckpointSpec(
    backend="metal",
    model_key=MODEL_KEY,
    model_label="G08A M1",
    model_lock_path=METAL_LOCK,
    weight_revision="e87f176479d0855a907a41277aca2f8ee7a09523",
    weight_file_count=1,
    semantic_file_count=5,
    checkpoint_id="runtime-vnext-g08a-metal-model-matrix",
    checkpoint_label="G08A-METAL-MATRIX",
    expected_case_count=702,
    required_client_concurrency=16,
    required_active_floor=16,
    required_active_duty_cycle=0.80,
    concurrency_cells=(1, 4, 16),
    artifact_type_prefix="runtime_vnext_g08a_metal_model_matrix",
    pass_prefix="FERRUM RUNTIME VNEXT G08A METAL MODEL MATRIX PASS",
    fail_prefix="FERRUM RUNTIME VNEXT G08A METAL MODEL MATRIX FAIL",
    selftest_pass_line="FERRUM RUNTIME VNEXT G08A METAL MODEL MATRIX SELFTEST PASS",
    does_not_prove=(
        "current-HEAD G08A CUDA BF16 product path",
        "G08A numerical reference",
        "G08A mutation and legacy-deletion acceptance",
        "G08A CUDA/Metal performance smoke",
        "G08A final PASS",
        "G09 formal performance",
        "G10 release readiness",
    ),
)

CHECKPOINT_SPECS = {
    CUDA_CHECKPOINT_SPEC.backend: CUDA_CHECKPOINT_SPEC,
    METAL_CHECKPOINT_SPEC.backend: METAL_CHECKPOINT_SPEC,
}
