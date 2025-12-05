//! Metal compute pipeline management for quantized operations
//! 
//! This module provides high-level interfaces for executing quantized 
//! matrix operations on Apple GPU using Metal compute shaders.

use crate::metal::{MetalContext, MetalError};
use crate::metal::quantization::{MatvecParams, pack_matrix_q4_0_for_gpu, QK4_0};
use ferrum_types::FerrumError;
use metal::{Buffer, ComputePipelineDescriptor, ComputePipelineState, MTLSize};
use std::sync::Arc;
use tracing::{debug, info};

/// Metal compute pipeline for Q4_0 matrix-vector operations
pub struct Q4_0MatvecPipeline {
    pipeline_state: ComputePipelineState,
    context: Arc<MetalContext>,
}

impl Q4_0MatvecPipeline {
    /// Create a new Q4_0 matvec pipeline
    pub fn new(context: Arc<MetalContext>) -> Result<Self, FerrumError> {
        let library = context.library()
            .ok_or_else(|| MetalError::generic("Metal library not loaded"))?;
        
        let function = library.get_function("q4_0_matvec_main", None)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to find q4_0_matvec_main function: {}", e)))?;
        
        let pipeline_descriptor = ComputePipelineDescriptor::new();
        pipeline_descriptor.set_compute_function(Some(&function));
        pipeline_descriptor.set_label("Q4_0 MatVec Pipeline");
        
        let pipeline_state = context.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to create pipeline state: {}", e)))?;
        
        info!("Q4_0 matvec pipeline created successfully");
        
        Ok(Self {
            pipeline_state,
            context,
        })
    }
    
    /// Execute Q4_0 quantized matrix-vector multiplication
    /// 
    /// # Arguments
    /// * `weights` - Quantized weight matrix data (packed format)
    /// * `input` - Input vector (f32)
    /// * `nrows` - Number of matrix rows
    /// * `ncols` - Number of matrix columns (must be multiple of 32)
    /// 
    /// # Returns
    /// Result vector of length `nrows`
    pub fn execute_matvec(
        &self,
        weights: &[u32],
        input: &[f32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Vec<f32>, FerrumError> {
        // Validate input dimensions
        if ncols % QK4_0 != 0 {
            return Err(MetalError::invalid_argument(
                format!("ncols ({}) must be divisible by QK4_0 ({})", ncols, QK4_0)
            ));
        }
        
        if input.len() != ncols {
            return Err(MetalError::invalid_argument(
                format!("Input vector length ({}) must equal ncols ({})", input.len(), ncols)
            ));
        }
        
        let blocks_per_row = ncols / QK4_0;
        let expected_weight_size = nrows * blocks_per_row * 5; // 5 u32 per block
        
        if weights.len() != expected_weight_size {
            return Err(MetalError::invalid_argument(
                format!("Weight data size ({}) doesn't match expected ({})", weights.len(), expected_weight_size)
            ));
        }
        
        debug!("Executing Q4_0 matvec: {}×{} matrix", nrows, ncols);
        
        // Create parameters
        let params = MatvecParams {
            nrows: nrows as u32,
            ncols: ncols as u32,
            blocks_per_row: blocks_per_row as u32,
            _pad: 0,
        };
        
        // Create Metal buffers
        let params_buffer = self.create_buffer_with_data(
            bytemuck::bytes_of(&params),
            "Q4_0 Params"
        )?;
        
        let weights_buffer = self.create_buffer_with_data(
            bytemuck::cast_slice(weights),
            "Q4_0 Weights"
        )?;
        
        let input_buffer = self.create_buffer_with_data(
            bytemuck::cast_slice(input),
            "Q4_0 Input"
        )?;
        
        let output_size = nrows * std::mem::size_of::<f32>();
        let output_buffer = self.context.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared
        );
        output_buffer.set_label("Q4_0 Output");
        
        // Create and execute compute command
        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline_state);
        encoder.set_buffer(0, Some(&params_buffer), 0);
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        
        // Dispatch: one workgroup per output row
        let threadgroup_size = MTLSize::new(32, 1, 1); // 32 threads per workgroup
        let grid_size = MTLSize::new(1, nrows as u64, 1); // nrows workgroups
        
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read back results
        let output_ptr = output_buffer.contents() as *const f32;
        let output_slice = unsafe {
            std::slice::from_raw_parts(output_ptr, nrows)
        };
        
        Ok(output_slice.to_vec())
    }
    
    /// Create a Metal buffer with data
    fn create_buffer_with_data(&self, data: &[u8], label: &str) -> Result<Buffer, FerrumError> {
        let buffer = self.context.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared
        );
        buffer.set_label(label);
        Ok(buffer)
    }
}

/// High-level Q4_0 matrix operations interface
pub struct Q4_0MatrixOps {
    pipeline: Q4_0MatvecPipeline,
}

impl Q4_0MatrixOps {
    /// Create Q4_0 matrix operations with Metal acceleration
    pub fn new(context: Arc<MetalContext>) -> Result<Self, FerrumError> {
        let pipeline = Q4_0MatvecPipeline::new(context)?;
        Ok(Self { pipeline })
    }
    
    /// Quantize and execute matrix-vector multiplication
    /// 
    /// This is the high-level interface that handles quantization internally
    pub fn quantized_matvec(
        &self,
        weights: &[f32], // Full precision weights [nrows, ncols]
        input: &[f32],   // Input vector [ncols]
        nrows: usize,
        ncols: usize,
    ) -> Result<Vec<f32>, FerrumError> {
        debug!("Quantizing {}×{} matrix to Q4_0 format", nrows, ncols);
        
        // Quantize weights to Q4_0 format
        let quantized_weights = pack_matrix_q4_0_for_gpu(weights, nrows, ncols);
        
        // Execute on GPU
        self.pipeline.execute_matvec(&quantized_weights, input, nrows, ncols)
    }
    
    /// Execute pre-quantized matrix-vector multiplication
    /// 
    /// Use this when weights are already quantized to avoid repeated quantization
    pub fn execute_quantized_matvec(
        &self,
        quantized_weights: &[u32],
        input: &[f32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Vec<f32>, FerrumError> {
        self.pipeline.execute_matvec(quantized_weights, input, nrows, ncols)
    }
}

// ============================================================================
// RMS Normalization Pipeline
// ============================================================================

/// Parameters for RMS normalization
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    pub batch_size: u32,
    pub hidden_size: u32,
    pub epsilon: f32,
    pub _pad: u32,
}

/// Metal compute pipeline for RMS Normalization
pub struct RmsNormPipeline {
    single_pipeline: ComputePipelineState,
    batched_pipeline: ComputePipelineState,
    residual_pipeline: ComputePipelineState,
    inplace_pipeline: ComputePipelineState,
    context: Arc<MetalContext>,
}

impl RmsNormPipeline {
    /// Create a new RMS Norm pipeline
    pub fn new(context: Arc<MetalContext>) -> Result<Self, FerrumError> {
        let library = context
            .library()
            .ok_or_else(|| MetalError::generic("Metal library not loaded"))?;

        // Load all kernel variants
        let single_fn = library
            .get_function("rms_norm_single", None)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to find rms_norm_single: {}", e)))?;

        let batched_fn = library
            .get_function("rms_norm_batched", None)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to find rms_norm_batched: {}", e)))?;

        let residual_fn = library
            .get_function("rms_norm_residual", None)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to find rms_norm_residual: {}", e)))?;

        let inplace_fn = library
            .get_function("rms_norm_inplace", None)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to find rms_norm_inplace: {}", e)))?;

        let single_pipeline = context
            .device
            .new_compute_pipeline_state_with_function(&single_fn)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to create single pipeline: {}", e)))?;

        let batched_pipeline = context
            .device
            .new_compute_pipeline_state_with_function(&batched_fn)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to create batched pipeline: {}", e)))?;

        let residual_pipeline = context
            .device
            .new_compute_pipeline_state_with_function(&residual_fn)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to create residual pipeline: {}", e)))?;

        let inplace_pipeline = context
            .device
            .new_compute_pipeline_state_with_function(&inplace_fn)
            .map_err(|e| MetalError::compilation_failed(format!("Failed to create inplace pipeline: {}", e)))?;

        info!("RMS Norm pipelines created successfully");

        Ok(Self {
            single_pipeline,
            batched_pipeline,
            residual_pipeline,
            inplace_pipeline,
            context,
        })
    }

    /// Execute RMS normalization on a single vector
    ///
    /// # Arguments
    /// * `input` - Input vector [hidden_size]
    /// * `weight` - Weight vector [hidden_size]
    /// * `epsilon` - Small constant for numerical stability (typically 1e-6)
    pub fn forward_single(
        &self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
    ) -> Result<Vec<f32>, FerrumError> {
        let hidden_size = input.len();

        if weight.len() != hidden_size {
            return Err(MetalError::invalid_argument(format!(
                "Weight length ({}) must equal input length ({})",
                weight.len(),
                hidden_size
            )));
        }

        debug!("RMS Norm single: hidden_size={}", hidden_size);

        // Create buffers
        let input_buffer = self.create_buffer_with_data(bytemuck::cast_slice(input), "RMSNorm Input")?;
        let weight_buffer = self.create_buffer_with_data(bytemuck::cast_slice(weight), "RMSNorm Weight")?;

        let output_size = hidden_size * std::mem::size_of::<f32>();
        let output_buffer = self.context.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        output_buffer.set_label("RMSNorm Output");

        let hidden_size_u32 = hidden_size as u32;
        let hidden_size_buffer = self.create_buffer_with_data(
            bytemuck::bytes_of(&hidden_size_u32),
            "Hidden Size",
        )?;
        let epsilon_buffer = self.create_buffer_with_data(bytemuck::bytes_of(&epsilon), "Epsilon")?;

        // Execute
        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.single_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&hidden_size_buffer), 0);
        encoder.set_buffer(4, Some(&epsilon_buffer), 0);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(1, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const f32;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, hidden_size) };

        Ok(output_slice.to_vec())
    }

    /// Execute batched RMS normalization
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size, hidden_size]
    /// * `weight` - Weight vector [hidden_size]
    /// * `batch_size` - Number of vectors
    /// * `hidden_size` - Size of each vector
    /// * `epsilon` - Small constant for numerical stability
    pub fn forward_batched(
        &self,
        input: &[f32],
        weight: &[f32],
        batch_size: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, FerrumError> {
        if input.len() != batch_size * hidden_size {
            return Err(MetalError::invalid_argument(format!(
                "Input length ({}) must equal batch_size * hidden_size ({})",
                input.len(),
                batch_size * hidden_size
            )));
        }

        if weight.len() != hidden_size {
            return Err(MetalError::invalid_argument(format!(
                "Weight length ({}) must equal hidden_size ({})",
                weight.len(),
                hidden_size
            )));
        }

        debug!(
            "RMS Norm batched: batch_size={}, hidden_size={}",
            batch_size, hidden_size
        );

        // Create buffers
        let input_buffer = self.create_buffer_with_data(bytemuck::cast_slice(input), "RMSNorm Input")?;
        let weight_buffer = self.create_buffer_with_data(bytemuck::cast_slice(weight), "RMSNorm Weight")?;

        let output_size = batch_size * hidden_size * std::mem::size_of::<f32>();
        let output_buffer = self.context.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        output_buffer.set_label("RMSNorm Output");

        let batch_size_u32 = batch_size as u32;
        let hidden_size_u32 = hidden_size as u32;

        let batch_size_buffer = self.create_buffer_with_data(
            bytemuck::bytes_of(&batch_size_u32),
            "Batch Size",
        )?;
        let hidden_size_buffer = self.create_buffer_with_data(
            bytemuck::bytes_of(&hidden_size_u32),
            "Hidden Size",
        )?;
        let epsilon_buffer = self.create_buffer_with_data(bytemuck::bytes_of(&epsilon), "Epsilon")?;

        // Execute
        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.batched_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&batch_size_buffer), 0);
        encoder.set_buffer(4, Some(&hidden_size_buffer), 0);
        encoder.set_buffer(5, Some(&epsilon_buffer), 0);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(1, batch_size as u64, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const f32;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, batch_size * hidden_size) };

        Ok(output_slice.to_vec())
    }

    /// Execute fused RMS normalization with residual add
    ///
    /// Computes: output = RMSNorm(input + residual) * weight
    pub fn forward_with_residual(
        &self,
        input: &[f32],
        residual: &[f32],
        weight: &[f32],
        batch_size: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, FerrumError> {
        if input.len() != batch_size * hidden_size || residual.len() != batch_size * hidden_size {
            return Err(MetalError::invalid_argument(
                "Input and residual must have same length as batch_size * hidden_size".to_string(),
            ));
        }

        debug!(
            "RMS Norm with residual: batch_size={}, hidden_size={}",
            batch_size, hidden_size
        );

        // Create buffers
        let input_buffer = self.create_buffer_with_data(bytemuck::cast_slice(input), "Input")?;
        let residual_buffer = self.create_buffer_with_data(bytemuck::cast_slice(residual), "Residual")?;
        let weight_buffer = self.create_buffer_with_data(bytemuck::cast_slice(weight), "Weight")?;

        let output_size = batch_size * hidden_size * std::mem::size_of::<f32>();
        let output_buffer = self.context.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let batch_size_u32 = batch_size as u32;
        let hidden_size_u32 = hidden_size as u32;

        let batch_size_buffer = self.create_buffer_with_data(bytemuck::bytes_of(&batch_size_u32), "Batch")?;
        let hidden_size_buffer = self.create_buffer_with_data(bytemuck::bytes_of(&hidden_size_u32), "Hidden")?;
        let epsilon_buffer = self.create_buffer_with_data(bytemuck::bytes_of(&epsilon), "Eps")?;

        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.residual_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&residual_buffer), 0);
        encoder.set_buffer(2, Some(&weight_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_buffer(4, Some(&batch_size_buffer), 0);
        encoder.set_buffer(5, Some(&hidden_size_buffer), 0);
        encoder.set_buffer(6, Some(&epsilon_buffer), 0);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(1, batch_size as u64, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, batch_size * hidden_size) };

        Ok(output_slice.to_vec())
    }

    fn create_buffer_with_data(&self, data: &[u8], label: &str) -> Result<Buffer, FerrumError> {
        let buffer = self.context.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        buffer.set_label(label);
        Ok(buffer)
    }
}

/// High-level RMS Norm operations interface
pub struct RmsNormOps {
    pipeline: RmsNormPipeline,
}

impl RmsNormOps {
    /// Create RMS Norm operations with Metal acceleration
    pub fn new(context: Arc<MetalContext>) -> Result<Self, FerrumError> {
        let pipeline = RmsNormPipeline::new(context)?;
        Ok(Self { pipeline })
    }

    /// Apply RMS normalization
    pub fn forward(
        &self,
        input: &[f32],
        weight: &[f32],
        batch_size: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, FerrumError> {
        if batch_size == 1 {
            self.pipeline.forward_single(input, weight, epsilon)
        } else {
            self.pipeline.forward_batched(input, weight, batch_size, hidden_size, epsilon)
        }
    }

    /// Apply fused RMS normalization with residual connection
    pub fn forward_with_residual(
        &self,
        input: &[f32],
        residual: &[f32],
        weight: &[f32],
        batch_size: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, FerrumError> {
        self.pipeline.forward_with_residual(input, residual, weight, batch_size, hidden_size, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    async fn test_q4_0_matvec_pipeline() {
        // Create test data
        let nrows = 8;
        let ncols = 64; // 2 blocks per row
        let weights: Vec<f32> = (0..nrows * ncols).map(|i| (i as f32 - 256.0) * 0.001).collect();
        let input: Vec<f32> = (0..ncols).map(|i| i as f32 * 0.01).collect();
        
        // Create Metal context and pipeline
        let mut context = MetalContext::new().expect("Failed to create Metal context");
        context.load_shader_library().expect("Failed to load shader library");
        let context = Arc::new(context);
        
        let ops = Q4_0MatrixOps::new(context).expect("Failed to create Q4_0 ops");
        
        // Execute quantized matvec
        let result = ops.quantized_matvec(&weights, &input, nrows, ncols)
            .expect("Failed to execute quantized matvec");
        
        assert_eq!(result.len(), nrows);
        
        // Verify results by comparing with CPU implementation
        // (This would be a more thorough test in practice)
        for (i, &output_val) in result.iter().enumerate() {
            assert!(output_val.is_finite(), "Output {} is not finite: {}", i, output_val);
        }
    }

    #[tokio::test]
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    async fn test_rms_norm_single() {
        let hidden_size = 128;
        let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let epsilon = 1e-6;

        // Create Metal context and pipeline
        let mut context = MetalContext::new().expect("Failed to create Metal context");
        context.load_shader_library().expect("Failed to load shader library");
        let context = Arc::new(context);

        let ops = RmsNormOps::new(context).expect("Failed to create RMS Norm ops");

        // Execute RMS Norm
        let result = ops
            .forward(&input, &weight, 1, hidden_size, epsilon)
            .expect("Failed to execute RMS Norm");

        assert_eq!(result.len(), hidden_size);

        // Verify results
        // Compute expected RMS value
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_size as f32 + epsilon).sqrt();

        for (i, (&output_val, &input_val)) in result.iter().zip(input.iter()).enumerate() {
            let expected = input_val / rms * weight[i];
            let diff = (output_val - expected).abs();
            assert!(
                diff < 1e-4,
                "Output {} mismatch: got {}, expected {}, diff {}",
                i,
                output_val,
                expected,
                diff
            );
        }
    }

    #[tokio::test]
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    async fn test_rms_norm_batched() {
        let batch_size = 4;
        let hidden_size = 64;
        let input: Vec<f32> = (0..batch_size * hidden_size)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let weight: Vec<f32> = (0..hidden_size).map(|_| 1.0).collect();
        let epsilon = 1e-6;

        let mut context = MetalContext::new().expect("Failed to create Metal context");
        context.load_shader_library().expect("Failed to load shader library");
        let context = Arc::new(context);

        let ops = RmsNormOps::new(context).expect("Failed to create RMS Norm ops");

        let result = ops
            .forward(&input, &weight, batch_size, hidden_size, epsilon)
            .expect("Failed to execute batched RMS Norm");

        assert_eq!(result.len(), batch_size * hidden_size);

        // Verify each batch element
        for b in 0..batch_size {
            let start = b * hidden_size;
            let end = start + hidden_size;
            let batch_input = &input[start..end];
            let batch_output = &result[start..end];

            let sum_sq: f32 = batch_input.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_size as f32 + epsilon).sqrt();

            for i in 0..hidden_size {
                let expected = batch_input[i] / rms;
                let diff = (batch_output[i] - expected).abs();
                assert!(
                    diff < 1e-4,
                    "Batch {} output {} mismatch: got {}, expected {}",
                    b,
                    i,
                    batch_output[i],
                    expected
                );
            }
        }
    }
}
