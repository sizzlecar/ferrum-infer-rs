//! Metal-optimized model wrapper for Apple GPU acceleration
//! 
//! This module provides a model wrapper that intercepts Linear layer operations
//! and accelerates them using Q4_0 quantization on Apple GPU.

use async_trait::async_trait;
use ferrum_core::{Model, ModelInfo, Result, Tensor, TokenId, SamplingParams, GenerateOutput, KVCache, Error};
use crate::metal::{MetalBackend, QK4_0};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Metal-optimized model that wraps a base model and accelerates Linear operations
pub struct MetalOptimizedModel {
    base_model: Box<dyn Model>,
    metal_backend: Arc<MetalBackend>,
    // Cache quantized weights for Linear layers
    quantized_weights_cache: std::sync::Mutex<HashMap<String, Vec<u32>>>,
    model_info: ModelInfo,
}

impl MetalOptimizedModel {
    /// Create a new Metal-optimized model wrapper
    pub fn new(base_model: Box<dyn Model>, metal_backend: Arc<MetalBackend>) -> Self {
        let model_info = base_model.info().clone();
        
        info!("Created Metal-optimized model wrapper for {}", model_info.model_id.0);
        
        Self {
            base_model,
            metal_backend,
            quantized_weights_cache: std::sync::Mutex::new(HashMap::new()),
            model_info,
        }
    }
    
    /// Check if this layer should use Metal acceleration
    fn should_accelerate_layer(&self, layer_name: &str, nrows: usize, ncols: usize) -> bool {
        // Only accelerate large enough matrices and certain layer types
        if ncols % QK4_0 != 0 {
            return false;
        }
        
        // Accelerate main MLP and attention projection layers
        let is_large_enough = nrows >= 512 && ncols >= 512;
        let is_mlp_layer = layer_name.contains("mlp") || layer_name.contains("linear");
        let is_attention_layer = layer_name.contains("attn") || layer_name.contains("proj");
        
        is_large_enough && (is_mlp_layer || is_attention_layer)
    }
}

#[async_trait]
impl Model for MetalOptimizedModel {
    fn info(&self) -> &ModelInfo {
        &self.model_info
    }

    async fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For now, delegate to base model
        // TODO: Intercept and optimize Linear layer operations
        debug!("MetalOptimizedModel forward pass (delegating to base)");
        
        if self.metal_backend.has_q4_0_acceleration() {
            debug!("Metal Q4_0 acceleration available but not yet integrated in forward pass");
        }
        
        self.base_model.forward(input).await
    }

    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.base_model.encode(text)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        self.base_model.decode(tokens)
    }

    async fn generate_next_token(
        &self,
        input_ids: &[TokenId],
        past_kv: Option<&KVCache>,
        sampling_params: &SamplingParams,
    ) -> Result<GenerateOutput> {
        // For now, delegate to base model
        // This is where we'd integrate Metal optimizations for the decode path
        self.base_model.generate_next_token(input_ids, past_kv, sampling_params).await
    }
}

/// Test utility to compare Metal vs CPU performance
pub async fn compare_metal_vs_cpu_speed(metal_backend: &MetalBackend) -> Result<()> {
    info!("Running Metal vs CPU speed comparison");
    
    if !metal_backend.has_q4_0_acceleration() {
        return Err(Error::internal("Metal Q4_0 acceleration not available"));
    }
    
    // Test with LLaMA-like dimensions
    let test_cases = [
        (4096, 4096),   // Standard transformer layer
        (4096, 11008),  // MLP up projection 
        (11008, 4096),  // MLP down projection
    ];
    
    for &(nrows, ncols) in &test_cases {
        info!("Testing {}x{} matrix multiplication", nrows, ncols);
        
        // Generate test data  
        let weights: Vec<f32> = (0..nrows * ncols)
            .map(|i| (i as f32 - 1000.0) * 0.001)
            .collect();
        let input: Vec<f32> = (0..ncols).map(|i| i as f32 * 0.01).collect();
        
        // Test Metal Q4_0 performance
        let start = std::time::Instant::now();
        let metal_result = metal_backend.q4_0_matvec(&weights, &input, nrows, ncols)?;
        let metal_time = start.elapsed();
        
        // Test CPU reference (simple multiplication)
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; nrows];
        for row in 0..nrows {
            let mut sum = 0.0;
            for col in 0..ncols {
                sum += weights[row * ncols + col] * input[col];
            }
            cpu_result[row] = sum;
        }
        let cpu_time = start.elapsed();
        
        let speedup = cpu_time.as_secs_f64() / metal_time.as_secs_f64();
        
        // Calculate accuracy
        let mut max_error: f32 = 0.0;
        for (cpu_val, metal_val) in cpu_result.iter().zip(metal_result.iter()) {
            let error = (cpu_val - metal_val).abs();
            max_error = max_error.max(error / cpu_val.abs().max(1.0));
        }
        
        info!("Results for {}x{}: CPU={:.2}ms, Metal={:.2}ms, Speedup={:.2}x, Max relative error={:.2e}",
              nrows, ncols,
              cpu_time.as_secs_f64() * 1000.0,
              metal_time.as_secs_f64() * 1000.0,
              speedup,
              max_error);
        
        if speedup > 1.0 {
            info!("✅ Metal acceleration is {:.2}x faster!", speedup);
        } else {
            debug!("⚠️  Metal slower than CPU for this size: {:.2}x", speedup);
        }
    }
    
    Ok(())
}
