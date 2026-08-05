#!/usr/bin/env python3
"""Validate the canonical G08A Qwen3.5-4B Metal C01-C21 report."""

from runtime_vnext_g08a_matrix_specs import CHECKPOINT_SPECS, validate_model_lock_contract
from runtime_vnext_g08b_cuda_matrix_checkpoint import main


if __name__ == "__main__":
    validate_model_lock_contract("metal")
    raise SystemExit(
        main(
            default_backend="metal",
            fixed_backend=True,
            checkpoint_specs=CHECKPOINT_SPECS,
        )
    )
