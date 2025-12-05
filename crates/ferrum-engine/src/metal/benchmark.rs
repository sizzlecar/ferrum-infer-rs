//! Benchmarking suite for Metal Q4_0 optimizations
//!
//! Compares performance between CPU, standard Candle, and Metal Q4_0 implementations.

use crate::metal::MetalBackend;
use ferrum_types::{Device, Result};
use std::time::Instant;
use tracing::{debug, info};

/// Benchmark configuration for different matrix sizes
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Matrix dimensions to test: (rows, cols)
    pub matrix_sizes: Vec<(usize, usize)>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Whether to validate correctness against CPU reference
    pub validate_correctness: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            // Common transformer model sizes
            matrix_sizes: vec![
                (4096, 4096),  // Standard LLaMA
                (4096, 11008), // LLaMA MLP up_proj/gate_proj
                (11008, 4096), // LLaMA MLP down_proj
                (8192, 8192),  // Larger models
                (8192, 22016), // Larger MLP
            ],
            iterations: 10,
            validate_correctness: true,
        }
    }
}

/// Benchmark results for a single test case
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub matrix_size: (usize, usize),
    pub cpu_time_ms: f64,
    pub metal_time_ms: Option<f64>,
    pub speedup: Option<f64>,
    pub correctness_error: Option<f64>, // L2 error vs CPU reference
}

/// Run comprehensive benchmarks comparing CPU and Metal Q4_0 implementations
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub async fn run_q4_0_benchmarks(config: BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
    info!("Running Q4_0 matrix multiplication benchmarks");

    let mut results = Vec::new();

    // Create Metal backend
    let mut metal_backend = MetalBackend::new(Device::Metal)?;
    metal_backend.initialize().await?;

    let has_metal = metal_backend.has_q4_0_acceleration();
    if !has_metal {
        info!("Metal acceleration not available, benchmarking CPU only");
    }

    for (nrows, ncols) in config.matrix_sizes {
        info!("Benchmarking {}x{} matrix...", nrows, ncols);

        // Generate random test data
        let weights: Vec<f32> = (0..nrows * ncols)
            .map(|i| ((i as f32 * 0.001).sin() * 2.0))
            .collect();
        let input: Vec<f32> = (0..ncols).map(|i| (i as f32 * 0.01).cos()).collect();

        // CPU reference implementation
        let cpu_start = Instant::now();
        let mut cpu_output = vec![0.0f32; nrows];
        for _ in 0..config.iterations {
            for i in 0..nrows {
                let mut sum = 0.0f32;
                for j in 0..ncols {
                    sum += weights[i * ncols + j] * input[j];
                }
                cpu_output[i] = sum;
            }
        }
        let cpu_time_ms = cpu_start.elapsed().as_secs_f64() * 1000.0 / config.iterations as f64;

        // Metal Q4_0 implementation (if available)
        let (metal_time_ms, speedup, correctness_error) = if has_metal {
            let metal_start = Instant::now();
            let mut metal_output = vec![0.0f32; nrows];
            for _ in 0..config.iterations {
                metal_output = metal_backend.q4_0_matvec(&weights, &input, nrows, ncols)?;
            }
            let metal_time = metal_start.elapsed().as_secs_f64() * 1000.0 / config.iterations as f64;

            // Calculate correctness error
            let error = if config.validate_correctness {
                let mut l2_error = 0.0f64;
                for i in 0..nrows {
                    let diff = (cpu_output[i] - metal_output[i]) as f64;
                    l2_error += diff * diff;
                }
                Some(l2_error.sqrt() / nrows as f64)
            } else {
                None
            };

            let speedup = cpu_time_ms / metal_time;
            (Some(metal_time), Some(speedup), error)
        } else {
            (None, None, None)
        };

        let result = BenchmarkResult {
            matrix_size: (nrows, ncols),
            cpu_time_ms,
            metal_time_ms,
            speedup,
            correctness_error,
        };

        debug!("{:?}", result);
        results.push(result);
    }

    Ok(results)
}

/// Stub for non-Metal platforms
#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
pub async fn run_q4_0_benchmarks(_config: BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
    Err(ferrum_types::FerrumError::unsupported(
        "Metal benchmarks not available on this platform",
    ))
}

/// Print benchmark results in a formatted table
pub fn print_benchmark_results(results: &[BenchmarkResult]) {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    Q4_0 Matrix Multiplication Benchmarks              ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Matrix Size      │   CPU (ms)  │  Metal (ms) │  Speedup  │  L2 Error  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    for result in results {
        let metal_str = result
            .metal_time_ms
            .map(|t| format!("{:>10.2}", t))
            .unwrap_or_else(|| "    N/A".to_string());
        let speedup_str = result
            .speedup
            .map(|s| format!("{:>8.2}x", s))
            .unwrap_or_else(|| "   N/A".to_string());
        let error_str = result
            .correctness_error
            .map(|e| format!("{:>10.2e}", e))
            .unwrap_or_else(|| "    N/A".to_string());

        println!(
            "║ {:>5}x{:<6}    │ {:>10.2} │ {:>10} │ {:>9} │ {:>10} ║",
            result.matrix_size.0, result.matrix_size.1, result.cpu_time_ms, metal_str, speedup_str, error_str
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝\n");
}
