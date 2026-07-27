#!/usr/bin/env python3
"""Prepare source-bound G08C Qwen3-30B-A3B Metal matrix inputs."""

from runtime_vnext_g08b_cuda_matrix_prepare import main
from runtime_vnext_g08c_cuda_matrix_prepare import BACKEND_SPECS


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_backend="metal",
            fixed_backend=True,
            backend_specs=BACKEND_SPECS,
            error_label="runtime_vnext_g08c_matrix_prepare",
        )
    )
