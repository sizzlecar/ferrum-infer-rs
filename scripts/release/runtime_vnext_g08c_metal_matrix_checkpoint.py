#!/usr/bin/env python3
"""Validate the canonical G08C Qwen3-30B-A3B Metal C01-C21 report."""

from runtime_vnext_g08b_cuda_matrix_checkpoint import main
from runtime_vnext_g08c_cuda_matrix_checkpoint import CHECKPOINT_SPECS


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_backend="metal",
            fixed_backend=True,
            checkpoint_specs=CHECKPOINT_SPECS,
        )
    )
