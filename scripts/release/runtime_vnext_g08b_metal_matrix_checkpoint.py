#!/usr/bin/env python3
"""Validate the canonical G08B Qwen3.5-35B Metal C01-C21 candidate report."""

from runtime_vnext_g08b_cuda_matrix_checkpoint import main


if __name__ == "__main__":
    raise SystemExit(main(default_backend="metal", fixed_backend=True))
