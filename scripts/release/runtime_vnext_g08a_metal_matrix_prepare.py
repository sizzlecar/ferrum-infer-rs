#!/usr/bin/env python3
"""Prepare source-bound inputs for the G08A Metal model-matrix gate."""

from runtime_vnext_g08a_matrix_specs import PREPARE_SPECS, validate_model_lock_contract
from runtime_vnext_g08b_cuda_matrix_prepare import main


if __name__ == "__main__":
    validate_model_lock_contract("metal")
    raise SystemExit(
        main(
            default_backend="metal",
            fixed_backend=True,
            backend_specs=PREPARE_SPECS,
            error_label="runtime_vnext_g08a_metal_matrix_prepare",
        )
    )
