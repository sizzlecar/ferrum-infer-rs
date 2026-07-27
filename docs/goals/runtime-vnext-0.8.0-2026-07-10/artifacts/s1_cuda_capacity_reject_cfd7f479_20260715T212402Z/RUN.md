# S1 CUDA capacity diagnostic: REJECT

- Lane: `vnext-s1-cuda-capacity`
- Source: `cfd7f479f9614510667e3059fb9a91565ad663de` (clean)
- Binary SHA256: `517ad790f6c016c46e7ff633da0b34257f37b4514a994c31ada60764cad0e50c`
- Hardware: 1x RTX 4090, 23028 MiB, driver 595.45.04
- Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Product entrypoint: `ferrum run`
- Result: `REJECT physical_backing_deferral_self_stale_livelock`

The model loaded successfully in 37.3 seconds. Request-level backing maintenance then
allocated 496,816 bytes once. The next sequence-level deferral observed epochs `1/2`,
but maintenance saw `2/4` because returning the deferral released its temporary parent
request authority. This repeated 38,499 times without a prefill submission before the
bounded process stop.

The strict stale check remains required. The source fix must retain the parent request
authority across maintenance, rather than ignoring release or capacity epoch changes.
The exact next-run stop condition is recorded in `reject-summary.json`.

The canonical external artifact is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-s1/s1_cuda_capacity_reject_cfd7f479_20260715T212402Z.tar.gz`
(`afbb8b01ae616898f78e315e060b509d14b004a83d6a7093d32aee662b2823ae`, 1,317,193 bytes).
It contains the collector's machine-readable REJECT and the 161,922,071-byte trace as a
gzip with both uncompressed and compressed SHA256 values. The repository retains only
this conclusion and its compact machine-readable summary.
