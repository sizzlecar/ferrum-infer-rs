//! Portable vNext reference backend.
//!
//! This backend is a production contract oracle, not a product fallback. It is
//! registered explicitly as [`DeviceClass::Reference`] and advertises only the
//! operations it really executes.

mod adjacent_rope;
mod composition;
mod dense_linear;
mod gelu_tanh;
mod head_rms_norm;
mod layer_norm;
mod pointwise;
mod runtime;
mod structured_attention;
mod transcendental;
mod zero_centered_rms_norm;

pub use adjacent_rope::REFERENCE_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID;
pub use composition::{
    reference_vnext_adjacent_rope_bf16_provider, reference_vnext_capabilities,
    reference_vnext_dense_linear_bf16_provider, reference_vnext_dense_linear_provider,
    reference_vnext_gated_residual_bf16_provider, reference_vnext_gelu_tanh_bf16_provider,
    reference_vnext_head_rms_norm_bf16_provider, reference_vnext_layer_norm_bf16_provider,
    reference_vnext_modulation_scale_bf16_provider, reference_vnext_operation_registry,
    reference_vnext_silu_bf16_provider, reference_vnext_silu_mul_bf16_provider,
    reference_vnext_structured_attention_bf16_provider, reference_vnext_tanh_bf16_provider,
    reference_vnext_zero_centered_rms_norm_bf16_provider, ReferenceVNextComposition,
    REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
};
pub use gelu_tanh::REFERENCE_GELU_TANH_BF16_NUMERICAL_PROFILE_ID;
pub use head_rms_norm::REFERENCE_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID;
pub use runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceDeviceRuntimeSnapshot, ReferenceDeviceStream,
};
pub use structured_attention::REFERENCE_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID;
pub use zero_centered_rms_norm::REFERENCE_ZERO_CENTERED_RMS_NORM_BF16_NUMERICAL_PROFILE_ID;
