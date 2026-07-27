#!/usr/bin/env python3
"""Prepare source-bound inputs for the G08B Metal model-matrix gate."""

from runtime_vnext_g08b_cuda_matrix_prepare import main


if __name__ == "__main__":
    raise SystemExit(main(default_backend="metal", fixed_backend=True))
