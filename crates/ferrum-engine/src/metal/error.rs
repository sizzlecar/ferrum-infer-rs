//! Metal backend error types

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Metal device not available")]
    DeviceNotAvailable,
    
    #[error("Metal library compilation failed: {0}")]
    CompilationFailed(String),
    
    #[error("Metal kernel execution failed: {0}")]
    KernelExecutionFailed(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(ferrum_core::DataType),
    
    #[error("Invalid tensor shape: {0:?}")]
    InvalidTensorShape(Vec<usize>),
    
    #[error("Generic Metal error: {0}")]
    Generic(String),
}

impl From<MetalError> for ferrum_core::Error {
    fn from(err: MetalError) -> Self {
        ferrum_core::Error::internal(err.to_string())
    }
}
