//! Device and computation types

use serde::{Deserialize, Serialize};

/// Device type for computation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU device
    CPU,
    /// NVIDIA CUDA device with device index
    CUDA(usize),
    /// AMD ROCm device with device index
    ROCm(usize),
    /// Apple GPU using Metal Performance Shaders
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal,
}

impl Device {
    /// Check if device is GPU-based
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::CUDA(_) | Device::ROCm(_))
            || {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                {
                    matches!(self, Device::Metal)
                }
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                {
                    false
                }
            }
    }

    /// Get device index for GPU devices
    pub fn index(&self) -> Option<usize> {
        match self {
            Device::CUDA(idx) | Device::ROCm(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(idx) => write!(f, "cuda:{}", idx),
            Device::ROCm(idx) => write!(f, "rocm:{}", idx),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => write!(f, "metal"),
        }
    }
}

/// Data type for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point
    FP32,
    /// 16-bit floating point (IEEE 754)
    FP16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit floating point (E5M2 or E4M3)
    FP8,
    /// 32-bit signed integer
    INT32,
    /// 16-bit signed integer
    INT16,
    /// 8-bit signed integer
    INT8,
    /// 4-bit signed integer
    INT4,
    /// 32-bit unsigned integer
    UINT32,
    /// 16-bit unsigned integer  
    UINT16,
    /// 8-bit unsigned integer
    UINT8,
    /// 4-bit unsigned integer
    UINT4,
    /// Boolean
    BOOL,
}

impl DataType {
    /// Get size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::FP32 | DataType::INT32 | DataType::UINT32 => 4,
            DataType::FP16 | DataType::BF16 | DataType::INT16 | DataType::UINT16 => 2,
            DataType::FP8 | DataType::INT8 | DataType::UINT8 | DataType::BOOL => 1,
            DataType::INT4 | DataType::UINT4 => 1, // Packed, but minimum 1 byte
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::FP32 | DataType::FP16 | DataType::BF16 | DataType::FP8)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::INT32
                | DataType::INT16
                | DataType::INT8
                | DataType::INT4
                | DataType::UINT32
                | DataType::UINT16
                | DataType::UINT8
                | DataType::UINT4
        )
    }

    /// Check if this is a quantized type (reduced precision)
    pub fn is_quantized(&self) -> bool {
        matches!(self, DataType::FP8 | DataType::INT8 | DataType::INT4 | DataType::UINT4)
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            DataType::FP32 => "fp32",
            DataType::FP16 => "fp16",
            DataType::BF16 => "bf16",
            DataType::FP8 => "fp8",
            DataType::INT32 => "int32",
            DataType::INT16 => "int16",
            DataType::INT8 => "int8",
            DataType::INT4 => "int4",
            DataType::UINT32 => "uint32",
            DataType::UINT16 => "uint16",
            DataType::UINT8 => "uint8",
            DataType::UINT4 => "uint4",
            DataType::BOOL => "bool",
        };
        write!(f, "{}", name)
    }
}
