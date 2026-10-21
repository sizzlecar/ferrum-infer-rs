//! Portable vNext reference backend.
//!
//! This backend is a production contract oracle, not a product fallback. It is
//! registered explicitly as [`DeviceClass::Reference`] and advertises only the
//! operations it really executes.

mod composition;
mod dense_linear;
mod layer_norm;
mod pointwise;
mod runtime;
mod transcendental;

pub use composition::{
    reference_vnext_capabilities, reference_vnext_dense_linear_bf16_provider,
    reference_vnext_dense_linear_provider, reference_vnext_gated_residual_bf16_provider,
    reference_vnext_layer_norm_bf16_provider, reference_vnext_modulation_scale_bf16_provider,
    reference_vnext_operation_registry, reference_vnext_silu_bf16_provider,
    reference_vnext_silu_mul_bf16_provider, reference_vnext_tanh_bf16_provider,
    ReferenceVNextComposition, REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
};
pub use runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceDeviceRuntimeSnapshot, ReferenceDeviceStream,
};
