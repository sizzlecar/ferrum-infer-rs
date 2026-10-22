//! Hardware-independent checks for the native Torch oracle's configuration,
//! binary protocol, managed Host capacity, and Linux process-group cleanup.
//! No CUDA feature, LibTorch installation, compiler, or GPU is required.
//! Run with `cargo test -p ferrum-models --test torch_native_oracle_protocol`.
//! The ignored subprocess worker is invoked by its Rust timeout test only.

#[path = "support/torch_native_oracle.rs"]
mod native;
