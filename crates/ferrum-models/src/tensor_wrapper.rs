//! Candle Tensor wrapper implementing TensorLike

use candle_core::Tensor;
use ferrum_interfaces::TensorLike;
use ferrum_types::{DataType, Device, FerrumError, Result};
use std::any::Any;

/// Wrapper for Candle Tensor to implement TensorLike
#[derive(Debug, Clone)]
pub struct CandleTensorWrapper {
    tensor: Tensor,
}

impl CandleTensorWrapper {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub fn inner(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_inner(self) -> Tensor {
        self.tensor
    }

    /// Safe extraction from Arc<dyn TensorLike>
    pub fn from_tensorref(tensor_ref: &ferrum_interfaces::TensorRef) -> Option<Tensor> {
        // Try to extract by getting raw data and reconstructing
        // This is safe because we only read immutable data
        let _ = tensor_ref;

        // For now, return None if not our wrapper
        // A better approach would be to add a method to TensorLike to extract data
        None
    }
}

impl TensorLike for CandleTensorWrapper {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.tensor.dims()
    }

    fn dtype(&self) -> DataType {
        match self.tensor.dtype() {
            candle_core::DType::F32 => DataType::FP32,
            candle_core::DType::F16 => DataType::FP16,
            candle_core::DType::BF16 => DataType::BF16,
            _ => DataType::FP32,
        }
    }

    fn device(&self) -> Device {
        match self.tensor.device() {
            candle_core::Device::Cpu => Device::CPU,
            candle_core::Device::Cuda(_) => Device::CUDA(0),
            candle_core::Device::Metal(_) => {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                return Device::Metal;
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                Device::CPU
            }
        }
    }

    fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    fn view(&self, _start: &[usize], _end: &[usize]) -> Result<ferrum_interfaces::TensorRef> {
        // TODO: Implement tensor slicing
        Err(FerrumError::model("Tensor view not yet implemented"))
    }

    fn reshape(&self, shape: &[usize]) -> Result<ferrum_interfaces::TensorRef> {
        let reshaped = self
            .tensor
            .reshape(shape)
            .map_err(|e| FerrumError::model(format!("Reshape failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(reshaped)))
    }

    fn to_cpu(&self) -> Result<ferrum_interfaces::TensorRef> {
        if matches!(self.tensor.device(), candle_core::Device::Cpu) {
            return Ok(std::sync::Arc::new(self.clone()));
        }

        let cpu_tensor = self
            .tensor
            .to_device(&candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("to_cpu failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(cpu_tensor)))
    }

    fn to_device(&self, device: &Device) -> Result<ferrum_interfaces::TensorRef> {
        let candle_device = match device {
            Device::CPU => candle_core::Device::Cpu,
            Device::CUDA(id) => candle_core::Device::new_cuda(*id)
                .map_err(|e| FerrumError::device(format!("CUDA device error: {}", e)))?,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => candle_core::Device::new_metal(0)
                .map_err(|e| FerrumError::device(format!("Metal device error: {}", e)))?,
            Device::ROCm(_) => {
                return Err(FerrumError::device("ROCm not supported yet"));
            }
        };

        let device_tensor = self
            .tensor
            .to_device(&candle_device)
            .map_err(|e| FerrumError::model(format!("to_device failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(device_tensor)))
    }

    fn to_dtype(&self, dtype: DataType) -> Result<ferrum_interfaces::TensorRef> {
        let candle_dtype = match &dtype {
            DataType::FP32 => candle_core::DType::F32,
            DataType::FP16 => candle_core::DType::F16,
            DataType::BF16 => candle_core::DType::BF16,
            _ => {
                return Err(FerrumError::model(format!(
                    "Unsupported dtype: {:?}",
                    dtype
                )))
            }
        };

        let converted = self
            .tensor
            .to_dtype(candle_dtype)
            .map_err(|e| FerrumError::model(format!("to_dtype failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(converted)))
    }

    /// Extract tensor data as Vec<f32> - Candle implementation
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        // Handle different tensor dimensions
        match self.tensor.dims().len() {
            1 => self
                .tensor
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::model(format!("to_vec1 failed: {}", e))),
            2 => {
                // Take first batch: [batch, vocab] -> [vocab]
                let batch = self
                    .tensor
                    .to_vec2::<f32>()
                    .map_err(|e| FerrumError::model(format!("to_vec2 failed: {}", e)))?;
                Ok(batch.into_iter().next().unwrap_or_default())
            }
            3 => {
                // Take last token of first batch: [batch, seq, vocab] -> [vocab]
                let all = self
                    .tensor
                    .to_vec3::<f32>()
                    .map_err(|e| FerrumError::model(format!("to_vec3 failed: {}", e)))?;
                Ok(all
                    .into_iter()
                    .next()
                    .and_then(|seq| seq.into_iter().last())
                    .unwrap_or_default())
            }
            4 => {
                // Handle [batch, seq, extra, vocab] - squeeze and take last
                // First squeeze to 3D by selecting first element of extra dim
                let squeezed = self
                    .tensor
                    .squeeze(2)
                    .map_err(|e| FerrumError::model(format!("Squeeze dim 2 failed: {}", e)))?;

                // Now extract as 3D: [batch, seq, vocab]
                let all = squeezed
                    .to_vec3::<f32>()
                    .map_err(|e| FerrumError::model(format!("to_vec3 (from 4D) failed: {}", e)))?;
                Ok(all
                    .into_iter()
                    .next()
                    .and_then(|seq| seq.into_iter().last())
                    .unwrap_or_default())
            }
            _ => Err(FerrumError::model(format!(
                "Unsupported dims: {:?}",
                self.tensor.dims()
            ))),
        }
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        // Handle different tensor dimensions for token IDs
        match self.tensor.dims().len() {
            1 => self
                .tensor
                .to_vec1::<u32>()
                .map_err(|e| FerrumError::model(format!("to_vec1<u32> failed: {}", e))),
            2 => {
                // Take first batch: [batch, seq] -> [seq]
                let batch = self
                    .tensor
                    .to_vec2::<u32>()
                    .map_err(|e| FerrumError::model(format!("to_vec2<u32> failed: {}", e)))?;
                Ok(batch.into_iter().next().unwrap_or_default())
            }
            _ => Err(FerrumError::model(format!(
                "Unsupported dims for token extraction: {:?}",
                self.tensor.dims()
            ))),
        }
    }

    fn argmax_last_dim_u32(&self) -> Result<u32> {
        // Same strategy as runtime CandleTensor: argmax on-device, read back a scalar.
        use candle_core::{IndexOp, D};

        let dims = self.tensor.dims();
        let logits_1d = match dims.len() {
            1 => self.tensor.clone(),
            2 => self
                .tensor
                .i(0)
                .map_err(|e| FerrumError::model(format!("Index batch failed: {}", e)))?,
            3 => {
                let seq_len = dims[1];
                self.tensor
                    .i((0, seq_len.saturating_sub(1)))
                    .map_err(|e| FerrumError::model(format!("Index last token failed: {}", e)))?
            }
            4 => {
                // [batch, seq, extra, vocab] -> take batch 0, last seq, extra 0 -> [vocab]
                let seq_len = dims[1];
                self.tensor
                    .i((0, seq_len.saturating_sub(1), 0))
                    .map_err(|e| {
                        FerrumError::model(format!("Index last token (4D) failed: {}", e))
                    })?
            }
            _ => {
                return Err(FerrumError::model(format!(
                    "argmax_last_dim_u32 unsupported dims: {:?}",
                    dims
                )))
            }
        };

        let idx = logits_1d
            .argmax(D::Minus1)
            .map_err(|e| FerrumError::model(format!("Argmax failed: {}", e)))?
            .to_device(&candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("Argmax to CPU failed: {}", e)))?
            .to_vec0::<u32>()
            .map_err(|e| FerrumError::model(format!("Argmax readback failed: {}", e)))?;

        Ok(idx)
    }
}
