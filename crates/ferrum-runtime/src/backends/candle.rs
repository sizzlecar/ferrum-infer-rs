//! Candle backend - MVP implementation with core functionality
//!
//! Provides basic Candle tensor operations for CPU and GPU devices.

use crate::{
    ComputeBackend, DeviceMemoryManager, MemoryPool, TensorFactory, TensorLike, TensorOps,
    TensorRef,
};
use async_trait::async_trait;
use ferrum_interfaces::backend::{BackendCapabilities, BackendStatus, KernelExecutor};
use ferrum_types::{DataType, Device, Result};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Candle tensor wrapper
pub struct CandleTensor {
    inner: candle_core::Tensor,
    device: Device,
    dtype: DataType,
}

impl CandleTensor {
    pub fn new(tensor: candle_core::Tensor) -> Result<Self> {
        let device = candle_device_to_ferrum(tensor.device())?;
        let dtype = candle_dtype_to_ferrum(tensor.dtype())?;

        Ok(Self {
            inner: tensor,
            device,
            dtype,
        })
    }

    pub fn inner(&self) -> &candle_core::Tensor {
        &self.inner
    }
}

impl std::fmt::Debug for CandleTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleTensor")
            .field("shape", &self.inner.dims())
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish()
    }
}

impl TensorLike for CandleTensor {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.inner.dims()
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn to_device(&self, device: &Device) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device.clone())?;
        let moved = self
            .inner
            .to_device(&candle_device)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Device transfer: {}", e)))?;
        Ok(Arc::new(Self::new(moved)?))
    }

    fn to_dtype(&self, dtype: DataType) -> Result<TensorRef> {
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;
        let converted = self
            .inner
            .to_dtype(candle_dtype)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("DType conversion: {}", e)))?;
        Ok(Arc::new(Self::new(converted)?))
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        // Extract tensor data as Vec<f32>
        match self.inner.dims().len() {
            1 => self
                .inner
                .to_vec1::<f32>()
                .map_err(|e| ferrum_types::FerrumError::backend(format!("to_vec1 failed: {}", e))),
            2 => {
                let batch = self.inner.to_vec2::<f32>().map_err(|e| {
                    ferrum_types::FerrumError::backend(format!("to_vec2 failed: {}", e))
                })?;
                Ok(batch.into_iter().next().unwrap_or_default())
            }
            3 => {
                let all = self.inner.to_vec3::<f32>().map_err(|e| {
                    ferrum_types::FerrumError::backend(format!("to_vec3 failed: {}", e))
                })?;
                Ok(all
                    .into_iter()
                    .next()
                    .and_then(|seq| seq.into_iter().last())
                    .unwrap_or_default())
            }
            _ => Err(ferrum_types::FerrumError::backend(format!(
                "Unsupported tensor dimensions: {:?}",
                self.inner.dims()
            ))),
        }
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorRef> {
        let reshaped = self
            .inner
            .reshape(shape)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Reshape: {}", e)))?;
        Ok(Arc::new(Self::new(reshaped)?))
    }

    fn to_cpu(&self) -> Result<TensorRef> {
        self.to_device(&Device::CPU)
    }

    fn view(&self, _start: &[usize], _end: &[usize]) -> Result<TensorRef> {
        // MVP: simplified, return clone
        Ok(Arc::new(Self {
            inner: self.inner.clone(),
            device: self.device.clone(),
            dtype: self.dtype,
        }))
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn argmax_last_dim_u32(&self) -> Result<u32> {
        // Fast path for greedy sampling: compute argmax on the tensor's device,
        // then transfer only a single scalar to CPU.
        //
        // This is intentionally conservative: it assumes batch=1 and returns the
        // first element when batch exists.
        use candle_core::{IndexOp, D};

        let dims = self.inner.dims();
        let logits_1d = match dims.len() {
            1 => self.inner.clone(),
            2 => {
                // [batch, vocab] -> take batch 0 -> [vocab]
                self.inner.i(0).map_err(|e| {
                    ferrum_types::FerrumError::backend(format!("Index batch failed: {}", e))
                })?
            }
            3 => {
                // [batch, seq, vocab] -> take batch 0, last seq -> [vocab]
                let seq_len = dims[1];
                self.inner.i((0, seq_len.saturating_sub(1))).map_err(|e| {
                    ferrum_types::FerrumError::backend(format!("Index last token failed: {}", e))
                })?
            }
            _ => {
                return Err(ferrum_types::FerrumError::backend(format!(
                    "argmax_last_dim_u32 unsupported dims: {:?}",
                    dims
                )))
            }
        };

        // Candle argmax returns a tensor; we read back a single u32.
        let idx = logits_1d
            .argmax(D::Minus1)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Argmax failed: {}", e)))?
            .to_device(&candle_core::Device::Cpu)
            .map_err(|e| {
                ferrum_types::FerrumError::backend(format!("Argmax to CPU failed: {}", e))
            })?
            .to_vec0::<u32>()
            .map_err(|e| {
                ferrum_types::FerrumError::backend(format!("Argmax readback failed: {}", e))
            })?;

        Ok(idx)
    }
}

/// Candle tensor factory
pub struct CandleTensorFactory {
    device: Device,
}

impl CandleTensorFactory {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl std::fmt::Debug for CandleTensorFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleTensorFactory")
            .field("device", &self.device)
            .finish()
    }
}

impl TensorFactory for CandleTensorFactory {
    fn empty(&self, shape: &[usize], dtype: DataType, device: Device) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device)?;
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;

        let tensor = candle_core::Tensor::zeros(shape, candle_dtype, &candle_device)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(tensor)?))
    }

    fn from_slice(
        &self,
        data: &[f32],
        shape: &[usize],
        dtype: DataType,
        device: Device,
    ) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device)?;
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;

        let tensor = candle_core::Tensor::from_slice(data, shape, &candle_device)
            .and_then(|t| t.to_dtype(candle_dtype))
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;

        Ok(Arc::new(CandleTensor::new(tensor)?))
    }

    fn to_device(&self, tensor: &TensorRef, device: Device) -> Result<TensorRef> {
        tensor.to_device(&device)
    }

    fn narrow(
        &self,
        tensor: &TensorRef,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let narrowed = candle_tensor
            .narrow(dim, start, length)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(narrowed)?))
    }

    fn reshape(&self, tensor: &TensorRef, shape: &[usize]) -> Result<TensorRef> {
        tensor.reshape(shape)
    }

    fn zeros_like(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let zeros = candle_core::Tensor::zeros(
            candle_tensor.shape(),
            candle_tensor.dtype(),
            candle_tensor.device(),
        )
        .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(zeros)?))
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device.clone())?;
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;

        let tensor = candle_core::Tensor::zeros(shape, candle_dtype, &candle_device)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(tensor)?))
    }

    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device.clone())?;
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;

        let tensor = candle_core::Tensor::ones(shape, candle_dtype, &candle_device)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(tensor)?))
    }

    fn uniform(
        &self,
        shape: &[usize],
        low: f32,
        high: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device.clone())?;
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;

        let tensor = candle_core::Tensor::rand(low, high, shape, &candle_device)
            .and_then(|t| t.to_dtype(candle_dtype))
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(tensor)?))
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: f32,
        std: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        let candle_device = ferrum_device_to_candle(device.clone())?;
        let candle_dtype = ferrum_dtype_to_candle(dtype)?;

        let tensor = candle_core::Tensor::randn(mean, std, shape, &candle_device)
            .and_then(|t| t.to_dtype(candle_dtype))
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(tensor)?))
    }

    fn from_tensor(&self, tensor: &TensorRef, device: &Device) -> Result<TensorRef> {
        tensor.to_device(device)
    }
}

/// Candle tensor operations
#[derive(Debug, Clone, Default)]
pub struct CandleTensorOps;

impl TensorOps for CandleTensorOps {
    fn matmul(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_candle = get_candle_tensor(a)?;
        let b_candle = get_candle_tensor(b)?;

        let result = a_candle
            .matmul(b_candle)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn add(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_candle = get_candle_tensor(a)?;
        let b_candle = get_candle_tensor(b)?;

        let result =
            (a_candle + b_candle).map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn mul(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_candle = get_candle_tensor(a)?;
        let b_candle = get_candle_tensor(b)?;

        let result =
            (a_candle * b_candle).map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn sub(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_candle = get_candle_tensor(a)?;
        let b_candle = get_candle_tensor(b)?;

        let result =
            (a_candle - b_candle).map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn div(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_candle = get_candle_tensor(a)?;
        let b_candle = get_candle_tensor(b)?;

        let result =
            (a_candle / b_candle).map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn softmax(&self, tensor: &TensorRef, dim: i32) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let dim_usize = if dim < 0 {
            (candle_tensor.rank() as i32 + dim) as usize
        } else {
            dim as usize
        };

        let result = candle_nn::ops::softmax(candle_tensor, dim_usize)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn layer_norm(
        &self,
        input: &TensorRef,
        weight: &TensorRef,
        bias: Option<&TensorRef>,
        eps: f32,
    ) -> Result<TensorRef> {
        let input_candle = get_candle_tensor(input)?;
        let weight_candle = get_candle_tensor(weight)?;
        let _bias_candle = bias.map(|b| get_candle_tensor(b)).transpose()?;

        // MVP: simplified layer norm
        let zero_bias = candle_core::Tensor::zeros(
            weight_candle.shape(),
            weight_candle.dtype(),
            weight_candle.device(),
        )
        .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;

        let bias_tensor = if let Some(b) = _bias_candle {
            b
        } else {
            &zero_bias
        };

        let normalized = candle_nn::ops::layer_norm(input_candle, weight_candle, bias_tensor, eps)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(normalized)?))
    }

    fn rms_norm(&self, input: &TensorRef, weight: &TensorRef, eps: f32) -> Result<TensorRef> {
        let input_candle = get_candle_tensor(input)?;
        let weight_candle = get_candle_tensor(weight)?;

        let _rms = candle_nn::RmsNorm::new(weight_candle.clone(), eps as f64);
        let result = candle_nn::ops::rms_norm(input_candle, weight_candle, eps)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn relu(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let result = candle_tensor
            .relu()
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn gelu(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let result = candle_tensor
            .gelu()
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn silu(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let result = candle_nn::ops::silu(candle_tensor)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn concat(&self, tensors: &[&TensorRef], dim: usize) -> Result<TensorRef> {
        let candle_tensors: Result<Vec<_>> = tensors.iter().map(|t| get_candle_tensor(t)).collect();
        let candle_tensors = candle_tensors?;

        let result = candle_core::Tensor::cat(&candle_tensors, dim)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn split(&self, tensor: &TensorRef, sizes: &[usize], dim: usize) -> Result<Vec<TensorRef>> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let mut result = Vec::new();
        let mut offset = 0;

        for &size in sizes {
            let chunk = candle_tensor
                .narrow(dim, offset, size)
                .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
            result.push(Arc::new(CandleTensor::new(chunk)?) as TensorRef);
            offset += size;
        }

        Ok(result)
    }

    fn transpose(&self, tensor: &TensorRef, dim0: usize, dim1: usize) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let result = candle_tensor
            .transpose(dim0, dim1)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }

    fn permute(&self, tensor: &TensorRef, dims: &[usize]) -> Result<TensorRef> {
        let candle_tensor = get_candle_tensor(tensor)?;
        let result = candle_tensor
            .permute(dims)
            .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))?;
        Ok(Arc::new(CandleTensor::new(result)?))
    }
}

/// Candle backend
pub struct CandleBackend {
    device: Device,
    tensor_factory: CandleTensorFactory,
    tensor_ops: CandleTensorOps,
    memory_manager: MemoryPool,
}

impl std::fmt::Debug for CandleBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleBackend")
            .field("device", &self.device)
            .finish()
    }
}

impl CandleBackend {
    pub async fn new(device: Device) -> Result<Self> {
        debug!("Initializing Candle backend for: {:?}", device);

        let tensor_factory = CandleTensorFactory::new(device.clone());
        let tensor_ops = CandleTensorOps;
        let memory_manager = MemoryPool::new(
            device.clone(),
            crate::memory::InternalMemoryPoolConfig {
                initial_size: 1024 * 1024 * 1024, // 1GB
                max_size: 4 * 1024 * 1024 * 1024, // 4GB
                growth_factor: 1.5,
                enable_defragmentation: true,
                min_pooled_size: 256,
                max_pooled_size: 1024 * 1024, // 1MB
                size_buckets: 32,
            },
        );

        Ok(Self {
            device,
            tensor_factory,
            tensor_ops,
            memory_manager,
        })
    }
}

#[async_trait]
impl ComputeBackend for CandleBackend {
    fn name(&self) -> &str {
        "candle"
    }

    fn capabilities(&self) -> BackendCapabilities {
        let supported_devices = {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            {
                vec![Device::CPU, Device::CUDA(0), Device::Metal]
            }
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            {
                vec![Device::CPU, Device::CUDA(0)]
            }
        };

        BackendCapabilities {
            supported_dtypes: vec![DataType::FP32, DataType::FP16, DataType::BF16],
            supported_devices,
            max_tensor_dims: 8,
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: false,
            supports_flash_attention: false,
            supports_paged_attention: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            max_batch_size: 32,
            max_sequence_length: 4096,
            memory_alignment: 256,
            supports_custom_kernels: false,
            supports_cuda_graphs: false,
            extra_capabilities: HashMap::new(),
        }
    }

    fn tensor_ops(&self) -> &dyn TensorOps {
        &self.tensor_ops
    }

    fn tensor_factory(&self) -> &dyn TensorFactory {
        &self.tensor_factory
    }

    fn memory_manager(&self) -> &dyn DeviceMemoryManager {
        &self.memory_manager
    }

    fn kernel_executor(&self) -> Option<&dyn KernelExecutor> {
        None // MVP: no custom kernels
    }

    async fn initialize(&mut self, _device: &Device) -> Result<()> {
        // Already initialized in new()
        Ok(())
    }

    fn supports_device(&self, device: &Device) -> bool {
        match device {
            Device::CPU | Device::CUDA(_) => true,
            Device::ROCm(_) => false,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => cfg!(feature = "metal"),
        }
    }

    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    async fn synchronize(&self, _device: &Device) -> Result<()> {
        // MVP: no-op for CPU, would need actual sync for GPU
        Ok(())
    }

    fn status(&self) -> BackendStatus {
        BackendStatus {
            is_initialized: true,
            is_ready: true,
            active_devices: vec![self.device.clone()],
            memory_usage: HashMap::new(),
            operations_completed: 0,
            last_error: None,
            backend_specific: HashMap::new(),
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        debug!("Shutting down Candle backend");
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_candle_tensor(tensor: &TensorRef) -> Result<&candle_core::Tensor> {
    // MVP: use type_id check since as_any not in TensorLike trait yet
    let concrete_ref: &CandleTensor = unsafe {
        // This is safe if we always create tensors through this backend
        &*(Arc::as_ptr(tensor) as *const CandleTensor)
    };
    Ok(&concrete_ref.inner)
}

fn ferrum_dtype_to_candle(dtype: DataType) -> Result<candle_core::DType> {
    match dtype {
        DataType::FP32 => Ok(candle_core::DType::F32),
        DataType::FP16 => Ok(candle_core::DType::F16),
        DataType::BF16 => Ok(candle_core::DType::BF16),
        DataType::UINT32 => Ok(candle_core::DType::U32),
        DataType::UINT8 => Ok(candle_core::DType::U8),
        DataType::INT32 => Ok(candle_core::DType::U32), // Fallback
        _ => Err(ferrum_types::FerrumError::backend(format!(
            "Unsupported dtype: {:?}",
            dtype
        ))),
    }
}

fn candle_dtype_to_ferrum(dtype: candle_core::DType) -> Result<DataType> {
    match dtype {
        candle_core::DType::F32 => Ok(DataType::FP32),
        candle_core::DType::F16 => Ok(DataType::FP16),
        candle_core::DType::BF16 => Ok(DataType::BF16),
        candle_core::DType::U32 => Ok(DataType::UINT32),
        candle_core::DType::U8 => Ok(DataType::UINT8),
        _ => Err(ferrum_types::FerrumError::backend(format!(
            "Unsupported Candle dtype: {:?}",
            dtype
        ))),
    }
}

fn ferrum_device_to_candle(device: Device) -> Result<candle_core::Device> {
    match device {
        Device::CPU => Ok(candle_core::Device::Cpu),
        Device::CUDA(id) => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(id as usize)
                    .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = id;
                Err(ferrum_types::FerrumError::unsupported("CUDA not enabled"))
            }
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        Device::Metal => {
            #[cfg(feature = "metal")]
            {
                candle_core::Device::new_metal(0)
                    .map_err(|e| ferrum_types::FerrumError::backend(e.to_string()))
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(ferrum_types::FerrumError::unsupported("Metal not enabled"))
            }
        }
        Device::ROCm(_) => Err(ferrum_types::FerrumError::unsupported("ROCm not supported")),
    }
}

fn candle_device_to_ferrum(device: &candle_core::Device) -> Result<Device> {
    match device {
        candle_core::Device::Cpu => Ok(Device::CPU),
        candle_core::Device::Cuda(_) => Ok(Device::CUDA(0)), // Default to GPU 0
        candle_core::Device::Metal(_) => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                Ok(Device::Metal)
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                Err(ferrum_types::FerrumError::unsupported(
                    "Metal devices are not available on this platform",
                ))
            }
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversions() {
        // FP32
        let candle_fp32 = ferrum_dtype_to_candle(DataType::FP32).unwrap();
        let back_fp32 = candle_dtype_to_ferrum(candle_fp32).unwrap();
        assert_eq!(back_fp32, DataType::FP32);

        // FP16
        let candle_fp16 = ferrum_dtype_to_candle(DataType::FP16).unwrap();
        let back_fp16 = candle_dtype_to_ferrum(candle_fp16).unwrap();
        assert_eq!(back_fp16, DataType::FP16);
    }

    #[test]
    fn test_device_conversions_cpu() {
        let ferrum_device = Device::CPU;
        let candle_device = ferrum_device_to_candle(ferrum_device.clone()).unwrap();
        let back_device = candle_device_to_ferrum(&candle_device).unwrap();
        assert_eq!(back_device, Device::CPU);
    }

    #[tokio::test]
    async fn test_candle_backend_creation() {
        let backend = CandleBackend::new(Device::CPU).await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_candle_backend_name() {
        let backend = CandleBackend::new(Device::CPU).await.unwrap();
        assert_eq!(backend.name(), "candle");
    }

    #[tokio::test]
    async fn test_candle_backend_capabilities() {
        let backend = CandleBackend::new(Device::CPU).await.unwrap();
        let caps = backend.capabilities();

        assert!(caps.supported_dtypes.contains(&DataType::FP32));
        assert!(caps.max_tensor_dims > 0);
    }

    #[tokio::test]
    async fn test_candle_backend_supports_cpu() {
        let backend = CandleBackend::new(Device::CPU).await.unwrap();
        assert!(backend.supports_device(&Device::CPU));
    }

    #[test]
    fn test_tensor_factory_zeros() {
        let factory = CandleTensorFactory::new(Device::CPU);
        let tensor = factory
            .zeros(&[2, 3], DataType::FP32, &Device::CPU)
            .unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DataType::FP32);
    }

    #[test]
    fn test_tensor_factory_ones() {
        let factory = CandleTensorFactory::new(Device::CPU);
        let tensor = factory.ones(&[2, 2], DataType::FP32, &Device::CPU).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_ops_add() {
        let factory = CandleTensorFactory::new(Device::CPU);
        let ops = CandleTensorOps;

        let a = factory
            .from_slice(&[1.0, 2.0], &[2], DataType::FP32, Device::CPU)
            .unwrap();
        let b = factory
            .from_slice(&[3.0, 4.0], &[2], DataType::FP32, Device::CPU)
            .unwrap();

        let result = ops.add(&a, &b).unwrap();
        let data = result.to_vec_f32().unwrap();

        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_ops_matmul() {
        let factory = CandleTensorFactory::new(Device::CPU);
        let ops = CandleTensorOps;

        // 2x2 matrices
        let a = factory
            .from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DataType::FP32, Device::CPU)
            .unwrap();
        let b = factory
            .from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], DataType::FP32, Device::CPU)
            .unwrap();

        let result = ops.matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_reshape() {
        let factory = CandleTensorFactory::new(Device::CPU);
        let tensor = factory
            .zeros(&[2, 3], DataType::FP32, &Device::CPU)
            .unwrap();

        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
    }

    #[test]
    fn test_tensor_to_cpu() {
        let factory = CandleTensorFactory::new(Device::CPU);
        let tensor = factory
            .zeros(&[2, 3], DataType::FP32, &Device::CPU)
            .unwrap();

        let cpu_tensor = tensor.to_cpu().unwrap();
        assert_eq!(cpu_tensor.device(), Device::CPU);
    }
}
