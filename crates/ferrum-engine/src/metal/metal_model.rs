//! Metal-optimized model wrapper for Apple GPU acceleration
//!
//! This module provides utilities for Metal-accelerated model operations.

use crate::metal::{MetalBackend, QK4_0};
use ferrum_types::{FerrumError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Cache for quantized weights
pub struct QuantizedWeightsCache {
    /// Cached quantized weights by layer name
    weights: std::sync::Mutex<HashMap<String, Vec<u32>>>,
}

impl QuantizedWeightsCache {
    pub fn new() -> Self {
        Self {
            weights: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Get cached quantized weights for a layer
    pub fn get(&self, layer_name: &str) -> Option<Vec<u32>> {
        self.weights.lock().unwrap().get(layer_name).cloned()
    }

    /// Store quantized weights for a layer
    pub fn insert(&self, layer_name: &str, weights: Vec<u32>) {
        self.weights
            .lock()
            .unwrap()
            .insert(layer_name.to_string(), weights);
    }

    /// Check if weights are cached for a layer
    pub fn contains(&self, layer_name: &str) -> bool {
        self.weights.lock().unwrap().contains_key(layer_name)
    }

    /// Clear all cached weights
    pub fn clear(&self) {
        self.weights.lock().unwrap().clear();
    }
}

impl Default for QuantizedWeightsCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a layer should use Metal acceleration based on dimensions
pub fn should_accelerate_layer(layer_name: &str, nrows: usize, ncols: usize) -> bool {
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

/// Test utility to compare Metal vs CPU performance
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub async fn compare_metal_vs_cpu_speed(metal_backend: &MetalBackend) -> Result<()> {
    info!("Running Metal vs CPU speed comparison");

    if !metal_backend.has_q4_0_acceleration() {
        return Err(FerrumError::internal(
            "Metal Q4_0 acceleration not available",
        ));
    }

    // Test with LLaMA-like dimensions
    let test_cases = [
        (4096, 4096),  // Standard transformer layer
        (4096, 11008), // MLP up projection
        (11008, 4096), // MLP down projection
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

        info!(
            "Results for {}x{}: CPU={:.2}ms, Metal={:.2}ms, Speedup={:.2}x, Max relative error={:.2e}",
            nrows,
            ncols,
            cpu_time.as_secs_f64() * 1000.0,
            metal_time.as_secs_f64() * 1000.0,
            speedup,
            max_error
        );

        if speedup > 1.0 {
            info!("✅ Metal acceleration is {:.2}x faster!", speedup);
        } else {
            debug!("⚠️  Metal slower than CPU for this size: {:.2}x", speedup);
        }
    }

    Ok(())
}

/// Stub for non-Metal platforms
#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
pub async fn compare_metal_vs_cpu_speed(_metal_backend: &MetalBackend) -> Result<()> {
    Err(FerrumError::unsupported(
        "Metal comparison not available on this platform",
    ))
}
