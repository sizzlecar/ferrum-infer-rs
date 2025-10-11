//! Metal backend error types
//!
//! Metal-specific error handling using the unified FerrumError type.

use ferrum_types::{DataType, FerrumError};

/// Helper functions to create Metal-specific FerrumError instances
pub struct MetalError;

impl MetalError {
    pub fn initialization_failed(msg: impl Into<String>) -> FerrumError {
        FerrumError::backend(format!("Metal initialization failed: {}", msg.into()))
    }

    pub fn device_not_available() -> FerrumError {
        FerrumError::device("Metal device not available")
    }

    pub fn compilation_failed(msg: impl Into<String>) -> FerrumError {
        FerrumError::backend(format!("Metal library compilation failed: {}", msg.into()))
    }

    pub fn kernel_execution_failed(msg: impl Into<String>) -> FerrumError {
        FerrumError::backend(format!("Metal kernel execution failed: {}", msg.into()))
    }

    pub fn memory_allocation_failed(msg: impl Into<String>) -> FerrumError {
        FerrumError::device(format!("Memory allocation failed: {}", msg.into()))
    }

    pub fn unsupported_data_type(dtype: DataType) -> FerrumError {
        FerrumError::unsupported(format!("Unsupported data type: {:?}", dtype))
    }

    pub fn invalid_tensor_shape(shape: Vec<usize>) -> FerrumError {
        FerrumError::backend(format!("Invalid tensor shape: {:?}", shape))
    }

    pub fn invalid_argument(msg: impl Into<String>) -> FerrumError {
        FerrumError::backend(format!("Invalid argument: {}", msg.into()))
    }

    pub fn generic(msg: impl Into<String>) -> FerrumError {
        FerrumError::backend(format!("Metal error: {}", msg.into()))
    }
}
