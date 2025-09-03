//! Benchmarking suite for Metal Q4_0 optimizations
//! 
//! Compares performance between CPU, standard Candle, and Metal Q4_0 implementations.

use crate::metal::MetalBackend;
use crate::candle_backend::CandleBackend;
use ferrum_core::{Device, Result, Backend};
use std::time::Instant;
use tracing::{info, debug};

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
                (4096, 4096),   // Standard LLaMA
                (4096, 11008),  // LLaMA MLP up_proj/gate_proj  
                (11008, 4096),  // LLaMA MLP down_proj
                (8192, 8192),   // Larger models
                (8192, 22016),  // Larger MLP
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
pub async fn run_q4_0_benchmarks(config: BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
    info!("Starting Q4_0 Metal optimization benchmarks");
    info!("Testing {} matrix sizes with {} iterations each", 
          config.matrix_sizes.len(), config.iterations);
    
    // Initialize backends
    let cpu_backend = CandleBackend::new(Device::CPU)?;
    let mut metal_backend = MetalBackend::new(Device::Metal)?;
    metal_backend.initialize().await?;
    
    let mut results = Vec::new();
    
    for &(nrows, ncols) in &config.matrix_sizes {
        info!("Benchmarking {}×{} matrix", nrows, ncols);
        
        let result = benchmark_single_size(
            &cpu_backend,
            &metal_backend,
            nrows,
            ncols,
            config.iterations,
            config.validate_correctness,
        ).await?;
        
        info!("Results for {}×{}: CPU={:.2}ms, Metal={:.2}ms, Speedup={:.2}x",
              nrows, ncols, 
              result.cpu_time_ms,
              result.metal_time_ms.unwrap_or(0.0),
              result.speedup.unwrap_or(0.0));
        
        results.push(result);
    }
    
    // Print summary
    print_benchmark_summary(&results);
    
    Ok(results)
}

/// Benchmark a single matrix size
async fn benchmark_single_size(
    _cpu_backend: &CandleBackend,
    metal_backend: &MetalBackend,
    nrows: usize,
    ncols: usize,
    iterations: usize,
    validate: bool,
) -> Result<BenchmarkResult> {
    // Generate test data
    let weights = generate_test_weights(nrows, ncols);
    let input = generate_test_input(ncols);
    
    // CPU benchmark
    let cpu_time = benchmark_cpu_matvec(&weights, &input, nrows, ncols, iterations)?;
    
    // Metal benchmark (if available)
    let (metal_time, correctness_error) = if metal_backend.has_q4_0_acceleration() {
        let metal_time = benchmark_metal_q4_0(&metal_backend, &weights, &input, nrows, ncols, iterations)?;
        
        let error = if validate {
            validate_correctness(&weights, &input, nrows, ncols, &metal_backend)?
        } else {
            None
        };
        
        (Some(metal_time), error)
    } else {
        debug!("Metal Q4_0 acceleration not available for {}×{}", nrows, ncols);
        (None, None)
    };
    
    let speedup = metal_time.map(|mt| cpu_time / mt);
    
    Ok(BenchmarkResult {
        matrix_size: (nrows, ncols),
        cpu_time_ms: cpu_time,
        metal_time_ms: metal_time,
        speedup,
        correctness_error,
    })
}

/// Generate test weight matrix with realistic values
fn generate_test_weights(nrows: usize, ncols: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    
    (0..nrows * ncols)
        .map(|i| {
            let row = i / ncols;
            let col = i % ncols;
            // Generate values that will quantize well to Q4_0
            0.1 * ((row as f32 * PI / nrows as f32).sin() + (col as f32 * PI / ncols as f32).cos())
        })
        .collect()
}

/// Generate test input vector
fn generate_test_input(ncols: usize) -> Vec<f32> {
    (0..ncols).map(|i| (i as f32 + 1.0) / ncols as f32).collect()
}

/// Benchmark CPU matrix-vector multiplication
fn benchmark_cpu_matvec(
    weights: &[f32],
    input: &[f32],
    nrows: usize,
    ncols: usize,
    iterations: usize,
) -> Result<f64> {
    // Warmup
    let _ = cpu_matvec_reference(weights, input, nrows, ncols);
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cpu_matvec_reference(weights, input, nrows, ncols);
    }
    let elapsed = start.elapsed();
    
    Ok(elapsed.as_secs_f64() * 1000.0 / iterations as f64)
}

/// Benchmark Metal Q4_0 matrix-vector multiplication  
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn benchmark_metal_q4_0(
    backend: &MetalBackend,
    weights: &[f32],
    input: &[f32],
    nrows: usize,
    ncols: usize,
    iterations: usize,
) -> Result<f64> {
    // Warmup
    let _ = backend.q4_0_matvec(weights, input, nrows, ncols)?;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.q4_0_matvec(weights, input, nrows, ncols)?;
    }
    let elapsed = start.elapsed();
    
    Ok(elapsed.as_secs_f64() * 1000.0 / iterations as f64)
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn benchmark_metal_q4_0(
    _backend: &MetalBackend,
    _weights: &[f32],
    _input: &[f32],
    _nrows: usize,
    _ncols: usize,
    _iterations: usize,
) -> Result<f64> {
    Err(ferrum_core::Error::internal("Metal not available on this platform"))
}

/// Reference CPU implementation for correctness validation
fn cpu_matvec_reference(
    weights: &[f32],
    input: &[f32], 
    nrows: usize,
    ncols: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; nrows];
    
    for row in 0..nrows {
        let mut sum = 0.0;
        for col in 0..ncols {
            sum += weights[row * ncols + col] * input[col];
        }
        output[row] = sum;
    }
    
    output
}

/// Validate correctness of Metal implementation against CPU reference
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn validate_correctness(
    weights: &[f32],
    input: &[f32],
    nrows: usize,
    ncols: usize,
    backend: &MetalBackend,
) -> Result<Option<f64>> {
    let cpu_result = cpu_matvec_reference(weights, input, nrows, ncols);
    let metal_result = backend.q4_0_matvec(weights, input, nrows, ncols)?;
    
    // Calculate L2 relative error
    let mut error_sum = 0.0;
    let mut norm_sum = 0.0;
    
    for (cpu_val, metal_val) in cpu_result.iter().zip(metal_result.iter()) {
        let diff = cpu_val - metal_val;
        error_sum += diff * diff;
        norm_sum += cpu_val * cpu_val;
    }
    
    let relative_error = if norm_sum > 0.0 {
        (error_sum / norm_sum).sqrt()
    } else {
        error_sum.sqrt()
    };
    
    Ok(Some(relative_error as f64))
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn validate_correctness(
    _weights: &[f32],
    _input: &[f32],
    _nrows: usize,
    _ncols: usize,
    _backend: &MetalBackend,
) -> Result<Option<f64>> {
    Ok(None)
}

/// Print formatted benchmark summary
fn print_benchmark_summary(results: &[BenchmarkResult]) {
    println!("\n=== Apple GPU Q4_0 Optimization Benchmark Results ===");
    println!("{:>12} {:>12} {:>12} {:>10} {:>12}", "Matrix Size", "CPU (ms)", "Metal (ms)", "Speedup", "Error");
    println!("{:-<70}", "");
    
    let mut total_speedup = 0.0;
    let mut valid_tests = 0;
    
    for result in results {
        let (rows, cols) = result.matrix_size;
        let size_str = format!("{}×{}", rows, cols);
        
        let metal_str = result.metal_time_ms
            .map(|t| format!("{:.2}", t))
            .unwrap_or_else(|| "N/A".to_string());
        
        let speedup_str = result.speedup
            .map(|s| {
                total_speedup += s;
                valid_tests += 1;
                format!("{:.2}x", s)
            })
            .unwrap_or_else(|| "N/A".to_string());
        
        let error_str = result.correctness_error
            .map(|e| format!("{:.2e}", e))
            .unwrap_or_else(|| "N/A".to_string());
        
        println!("{:>12} {:>12.2} {:>12} {:>10} {:>12}",
                 size_str, result.cpu_time_ms, metal_str, speedup_str, error_str);
    }
    
    if valid_tests > 0 {
        let avg_speedup = total_speedup / valid_tests as f64;
        println!("{:-<70}", "");
        println!("{:>12} {:>12} {:>12} {:>10.2}x {:>12}", "", "", "Average", avg_speedup, "");
    }
    println!();
}

/// Run a quick smoke test for Q4_0 operations  
pub async fn smoke_test_q4_0() -> Result<()> {
    info!("Running Q4_0 Metal optimization smoke test");
    
    let config = BenchmarkConfig {
        matrix_sizes: vec![(256, 256)], // Small test
        iterations: 3,
        validate_correctness: true,
    };
    
    let results = run_q4_0_benchmarks(config).await?;
    
    if let Some(result) = results.first() {
        if let Some(error) = result.correctness_error {
            if error > 0.1 {  // 10% error threshold
                return Err(ferrum_core::Error::internal(
                    format!("Q4_0 implementation has high error: {:.2e}", error)
                ));
            }
        }
        
        info!("Q4_0 smoke test passed with error: {:.2e}", 
              result.correctness_error.unwrap_or(0.0));
    }
    
    Ok(())
}
