//! Re-export of `Linear` trait (canonical home: ferrum-kernels).
//!
//! The trait itself lives in `ferrum-kernels::linear` so that Backend-level
//! helpers (`layer_forward_fused`) can reference it without ferrum-kernels
//! depending on this crate (which would be circular).

pub use ferrum_kernels::Linear;
