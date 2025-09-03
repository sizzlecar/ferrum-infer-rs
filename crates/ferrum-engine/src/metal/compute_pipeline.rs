//! Metal compute pipeline management for quantized operations
//! 
//! This module provides high-level interfaces for executing quantized 
//! matrix operations on Apple GPU using Metal compute shaders.

use crate::metal::{MetalContext, MetalError};
use crate::metal::quantization::{MatvecParams, pack_matrix_q4_0_for_gpu, QK4_0};
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
    pub fn new(context: Arc<MetalContext>) -> Result<Self, MetalError> {
        let library = context.library()
            .ok_or_else(|| MetalError::Generic("Metal library not loaded".to_string()))?;
        
        let function = library.get_function("q4_0_matvec_main", None)
            .map_err(|e| MetalError::CompilationFailed(format!("Failed to find q4_0_matvec_main function: {}", e)))?;
        
        let pipeline_descriptor = ComputePipelineDescriptor::new();
        pipeline_descriptor.set_compute_function(Some(&function));
        pipeline_descriptor.set_label("Q4_0 MatVec Pipeline");
        
        let pipeline_state = context.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::CompilationFailed(format!("Failed to create pipeline state: {}", e)))?;
        
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
    ) -> Result<Vec<f32>, MetalError> {
        // Validate input dimensions
        if ncols % QK4_0 != 0 {
            return Err(MetalError::InvalidArgument(
                format!("ncols ({}) must be divisible by QK4_0 ({})", ncols, QK4_0)
            ));
        }
        
        if input.len() != ncols {
            return Err(MetalError::InvalidArgument(
                format!("Input vector length ({}) must equal ncols ({})", input.len(), ncols)
            ));
        }
        
        let blocks_per_row = ncols / QK4_0;
        let expected_weight_size = nrows * blocks_per_row * 5; // 5 u32 per block
        
        if weights.len() != expected_weight_size {
            return Err(MetalError::InvalidArgument(
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
    fn create_buffer_with_data(&self, data: &[u8], label: &str) -> Result<Buffer, MetalError> {
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
    pub fn new(context: Arc<MetalContext>) -> Result<Self, MetalError> {
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
    ) -> Result<Vec<f32>, MetalError> {
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
    ) -> Result<Vec<f32>, MetalError> {
        self.pipeline.execute_matvec(quantized_weights, input, nrows, ncols)
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
}
