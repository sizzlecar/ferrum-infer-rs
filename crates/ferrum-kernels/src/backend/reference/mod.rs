//! Portable vNext reference backend.
//!
//! This backend is a production contract oracle, not a product fallback. It is
//! registered explicitly as [`DeviceClass::Reference`] and advertises only the
//! operations it really executes.

mod composition;
mod dense_linear;
mod half_head_oracle;
mod last_token_f16_operands;
mod runtime;

pub use composition::{
    reference_vnext_capabilities, reference_vnext_operation_registry, ReferenceVNextComposition,
    REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
};
pub use half_head_oracle::{
    q6_half_operands_oracle_inputs, reference_half_operands_head_oracle_registry,
    ReferenceHalfOperandsHeadOracle,
};
pub use runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceDeviceRuntimeSnapshot, ReferenceDeviceStream,
};
