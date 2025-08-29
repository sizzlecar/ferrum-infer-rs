//! Model executor implementation with Candle backend

use async_trait::async_trait;
use ferrum_core::{
    Executor, ExecutionTask, ExecutionResult, ExecutorCapabilities,
    ExecutorStatus, Result, Error, TaskType,
};
use candle_core::{Device, Tensor as CandleTensor};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug};

/// Executor configuration
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Device to use for execution
    pub device: Device,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Enable Flash Attention
    pub enable_flash_attention: bool,
    
    /// Enable Paged Attention
    pub enable_paged_attention: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            max_batch_size: 256,
            enable_flash_attention: true,
            enable_paged_attention: true,
        }
    }
}

/// Candle-based executor
pub struct CandleExecutor {
    config: ExecutorConfig,
    status: Arc<RwLock<ExecutorStatus>>,
}

impl CandleExecutor {
    /// Create a new Candle executor
    pub fn new(config: ExecutorConfig) -> Self {
        info!("Initializing CandleExecutor with device: {:?}", config.device);
        
        Self {
            config,
            status: Arc::new(RwLock::new(ExecutorStatus::Idle)),
        }
    }
    
    /// Convert Ferrum tensor to Candle tensor
    fn to_candle_tensor(&self, tensor: &ferrum_core::Tensor) -> Result<CandleTensor> {
        let shape: Vec<usize> = tensor.shape.clone();
        
        CandleTensor::from_vec(
            tensor.data.clone(),
            shape,
            &self.config.device,
        ).map_err(|e| Error::internal(format!("Failed to create Candle tensor: {}", e)))
    }
    
    /// Convert Candle tensor to Ferrum tensor
    fn from_candle_tensor(&self, tensor: &CandleTensor) -> Result<ferrum_core::Tensor> {
        let shape = tensor.dims().to_vec();
        let data = tensor.flatten_all()
            .map_err(|e| Error::internal(format!("Failed to flatten tensor: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| Error::internal(format!("Failed to convert tensor to vec: {}", e)))?;
        
        Ok(ferrum_core::Tensor::new(data, shape))
    }
}

#[async_trait]
impl Executor for CandleExecutor {
    async fn execute(&self, task: ExecutionTask) -> Result<ExecutionResult> {
        // Update status
        {
            *self.status.write() = ExecutorStatus::Running;
        }
        
        let start_time = std::time::Instant::now();
        
        debug!("Executing task {:?} of type {:?}", task.task_id, task.task_type);
        
        // Convert input tensor
        let input_tensor = self.to_candle_tensor(&task.input)?;
        
        // Execute based on task type
        let output_tensor = match task.task_type {
            TaskType::Prefill => {
                // Prefill phase - process entire prompt
                debug!("Executing prefill for {} tokens", input_tensor.dims()[0]);
                
                // Simplified - actual implementation would run through model layers
                input_tensor.clone()
            }
            TaskType::Decode => {
                // Decode phase - generate next token
                debug!("Executing decode step");
                
                // Simplified - actual implementation would run through model with KV cache
                input_tensor.clone()
            }
        };
        
        // Convert output back
        let output = self.from_candle_tensor(&output_tensor)?;
        
        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Update status
        {
            *self.status.write() = ExecutorStatus::Idle;
        }
        
        Ok(ExecutionResult {
            task_id: task.task_id,
            output,
            execution_time_ms,
        })
    }
    
    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: self.config.max_batch_size,
            supports_flash_attention: self.config.enable_flash_attention,
            supports_paged_attention: self.config.enable_paged_attention,
            supports_continuous_batching: true,
        }
    }
    
    fn status(&self) -> ExecutorStatus {
        *self.status.read()
    }
}
