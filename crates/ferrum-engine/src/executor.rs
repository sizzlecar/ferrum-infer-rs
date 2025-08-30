//! Abstract executor implementation that delegates to backend
//!
//! This module provides a backend-agnostic executor that uses the Backend trait
//! to perform actual computation, allowing different ML frameworks to be plugged in.

use async_trait::async_trait;
use ferrum_core::{
    Executor, ExecutionTask, ExecutionResult, ExecutorCapabilities,
    ExecutorStatus, ExecutorState, Result, TaskType, Backend,
};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug};

/// Executor configuration
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Enable Flash Attention
    pub enable_flash_attention: bool,
    
    /// Enable Paged Attention
    pub enable_paged_attention: bool,
    
    /// Device configuration
    pub device_config: DeviceConfig,
}

/// Device configuration
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Device type (CPU, CUDA, etc.)
    pub device_type: String,
    
    /// Device ID (for multi-GPU)
    pub device_id: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            enable_flash_attention: true,
            enable_paged_attention: true,
            device_config: DeviceConfig {
                device_type: "cuda".to_string(),
                device_id: 0,
            },
        }
    }
}

/// Generic executor that uses a backend for computation
pub struct GenericExecutor {
    config: ExecutorConfig,
    backend: Arc<dyn Backend>,
    state: Arc<RwLock<ExecutorState>>,
}

impl GenericExecutor {
    /// Create a new executor with the given backend
    pub fn new(config: ExecutorConfig, backend: Arc<dyn Backend>) -> Self {
        info!("Initializing GenericExecutor with backend: {}", backend.name());
        
        let state = ExecutorState {
            status: ExecutorStatus::Idle,
            is_ready: true,
            current_batch_size: 0,
            queued_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
        };
        
        Self {
            config,
            backend,
            state: Arc::new(RwLock::new(state)),
        }
    }
    
    /// Get the backend
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

#[async_trait]
impl Executor for GenericExecutor {
    async fn execute(&self, task: ExecutionTask) -> Result<ExecutionResult> {
        debug!("Executing task: {:?}", task.task_id);
        
        // Update state
        {
            let mut state = self.state.write();
            state.current_batch_size = task.batch_size;
            state.status = ExecutorStatus::Running;
        }
        
        // Delegate to backend for actual execution
        match task.task_type {
            TaskType::Prefill | TaskType::Decode | TaskType::ModelForward => {
                // Backend handles the actual model forward pass
                let result = ExecutionResult {
                    task_id: task.task_id,
                    output: task.input.clone(), // Placeholder - backend would transform
                    execution_time_ms: 0,
                };
                
                // Update state
                {
                    let mut state = self.state.write();
                    state.completed_tasks += 1;
                }
                
                Ok(result)
            }
            TaskType::Sampling => {
                // Backend handles sampling
                let result = ExecutionResult {
                    task_id: task.task_id,
                    output: task.input.clone(), // Placeholder
                    execution_time_ms: 0,
                };
                
                Ok(result)
            }
            TaskType::Preprocessing => {
                // Backend handles preprocessing
                let result = ExecutionResult {
                    task_id: task.task_id,
                    output: task.input.clone(), // Placeholder
                    execution_time_ms: 0,
                };
                
                Ok(result)
            }
            TaskType::Postprocessing => {
                // Backend handles postprocessing
                let result = ExecutionResult {
                    task_id: task.task_id,
                    output: task.input.clone(), // Placeholder
                    execution_time_ms: 0,
                };
                
                Ok(result)
            }
        }
    }
    
    fn capabilities(&self) -> ExecutorCapabilities {
        let backend_caps = self.backend.capabilities();
        
        ExecutorCapabilities {
            max_batch_size: backend_caps.max_batch_size.min(self.config.max_batch_size),
            max_sequence_length: backend_caps.max_sequence_length,
            supports_flash_attention: backend_caps.supports_flash_attention && self.config.enable_flash_attention,
            supports_paged_attention: backend_caps.supports_paged_attention && self.config.enable_paged_attention,
            supports_continuous_batching: true, // Always supported
            supports_tensor_parallelism: backend_caps.supports_tensor_parallelism,
            supports_pipeline_parallelism: false, // Not yet implemented
            supported_dtypes: vec![
                ferrum_core::DataType::FP32,
                ferrum_core::DataType::FP16,
                ferrum_core::DataType::BF16,
            ],
        }
    }
    
    fn status(&self) -> ExecutorStatus {
        self.state.read().status
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Tests would use a mock backend
}