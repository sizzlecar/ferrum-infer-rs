# W2 typed-config ShareGPT prefill profile retry failure

- status: FAIL
- failure: retry after f352ff3f still hit CUDA stream capture unsupported panic
- run_profile_rc: 143
- bench_rc: 143
- chat_smoke_pass: True
- panic_present: True
- prefill_profile_present: True
- batched_op_profile_present: True

## Panic

- CudaBackend: stream sync: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")
