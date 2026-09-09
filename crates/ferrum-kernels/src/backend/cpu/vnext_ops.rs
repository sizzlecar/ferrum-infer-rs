//! CPU operators over admitted host buffers and original physical weights.
//! Operator registration is supplied by the CPU composition, not the legacy
//! backend or the test reference device.

mod bindings;
mod causal_attention;
mod causal_attention_launch;
mod composition;
mod elementwise;
mod gated_delta;
mod gated_delta_launch;
mod launch;
mod lowering;
mod matrix;
mod provider;
mod scalar;
mod weights;

pub use composition::CpuVNextComposition;

#[cfg(test)]
mod tests;
