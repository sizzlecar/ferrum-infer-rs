//! Real backend checkpoint copies and numerical continuation through the
//! shared driver's product acknowledgement boundary.

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "prefix/metal.rs"]
mod metal_token_ack;

#[cfg(feature = "cuda")]
#[path = "prefix/cuda.rs"]
mod cuda_token_ack;
