//! `DefaultLinearFactory` — materialises dense f32 weights into
//! `DenseLinear<B>`. Used by any `WeightLoader` implementation that wants
//! delegate the "f32 slice → Linear" step without tying itself to a
//! particular backend.

use ferrum_kernels::backend::Backend;

use crate::dense::DenseLinear;
use crate::traits::{Linear, LinearFactory};

/// The baseline factory: produces `DenseLinear<B>` from row-major f32 weights.
///
/// Loaders handling GPTQ / AWQ / GGUF bypass this factory and build their
/// own `Linear` variants directly. `DefaultLinearFactory` only exists so
/// non-quantized paths have one well-known implementation.
pub struct DefaultLinearFactory;

impl<B: Backend> LinearFactory<B> for DefaultLinearFactory {
    fn dense(
        &self,
        weight_row_major: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Box<dyn Linear<B>> {
        Box::new(DenseLinear::<B>::from_rows(
            weight_row_major,
            out_features,
            in_features,
        ))
    }
}
