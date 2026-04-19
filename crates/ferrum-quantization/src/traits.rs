//! Re-export of `Linear` trait (canonical home: ferrum-kernels) plus
//! `LinearFactory` for weight-loader-side Linear construction.
//!
//! The trait itself lives in `ferrum-kernels::linear` so that Backend-level
//! helpers (`layer_forward_fused`) can reference it without ferrum-kernels
//! depending on this crate (which would be circular).

use ferrum_kernels::backend::Backend;

pub use ferrum_kernels::Linear;

/// Factory for constructing `Linear<B>` from raw loaded tensors.
///
/// A `WeightLoader` will typically hold a `LinearFactory<B>` and call it
/// for each layer weight. This indirection keeps the loader independent
/// of any specific `Backend`.
pub trait LinearFactory<B: Backend>: Send + Sync {
    /// Build a dense `Linear` from an already-materialised weight buffer.
    ///
    /// `weight_row_major`: `[out_features, in_features]` laid out contiguously.
    fn dense(
        &self,
        weight_row_major: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Box<dyn Linear<B>>;
}
