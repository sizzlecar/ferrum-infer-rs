//! Apple GPU benchmark command for Q4_0 optimizations

use clap::Args;
use ferrum_core::{Device, Error, Result, Backend};
use ferrum_engine::{MetalBackend};
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use ferrum_engine::metal::{run_q4_0_benchmarks, smoke_test_q4_0, BenchmarkConfig};
use tracing::{info, error};

#[derive(Args)]
pub struct AppleGpuBenchmarkArgs {
    /// Run full benchmark suite
    #[arg(long)]
    full: bool,
    
    /// Number of iterations per test
    #[arg(long, default_value = "10")]
    iterations: usize,
    
    /// Skip correctness validation (faster)
    #[arg(long)]
    skip_validation: bool,
    
    /// Only test specific matrix size (format: ROWSxCOLS, e.g., 4096x4096)
    #[arg(long)]
    matrix_size: Option<String>,
}

impl AppleGpuBenchmarkArgs {
    pub async fn run(&self) -> Result<()> {
        // Check if we're on Apple platform
        if !cfg!(any(target_os = "macos", target_os = "ios")) {
            return Err(Error::internal("Apple GPU benchmarks only available on macOS/iOS"));
        }
        
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            // Check if Metal backend is available
            let mut backend = MetalBackend::new(Device::Metal)?;
            backend.initialize().await?;
        
        if !backend.has_metal_acceleration() {
            return Err(Error::internal("Metal acceleration not available"));
        }
        
        if !backend.has_q4_0_acceleration() {
            return Err(Error::internal("Q4_0 acceleration not available"));
        }
        
        info!("Apple GPU (Metal) detected and Q4_0 acceleration available");
        
            if self.full {
                self.run_full_benchmark().await
            } else {
                self.run_smoke_test().await
            }
        }
        
        #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
        {
            Err(Error::internal("Metal not available on this platform"))
        }
    }
    
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    async fn run_smoke_test(&self) -> Result<()> {
        info!("Running Q4_0 Apple GPU smoke test...");
        
        match smoke_test_q4_0().await {
            Ok(_) => {
                info!("✅ Q4_0 Apple GPU smoke test passed");
                Ok(())
            }
            Err(e) => {
                error!("❌ Q4_0 Apple GPU smoke test failed: {}", e);
                Err(e)
            }
        }
    }
    
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    async fn run_full_benchmark(&self) -> Result<()> {
        info!("Running full Q4_0 Apple GPU benchmark suite...");
        
        let mut config = BenchmarkConfig::default();
        config.iterations = self.iterations;
        config.validate_correctness = !self.skip_validation;
        
        // Override matrix sizes if specified
        if let Some(ref size_str) = self.matrix_size {
            let (rows, cols) = parse_matrix_size(size_str)?;
            config.matrix_sizes = vec![(rows, cols)];
            info!("Testing custom matrix size: {}×{}", rows, cols);
        }
        
        match run_q4_0_benchmarks(config).await {
            Ok(results) => {
                info!("✅ Benchmark suite completed with {} test cases", results.len());
                
                // Check for any concerning results
                for result in &results {
                    if let Some(error) = result.correctness_error {
                        if error > 0.05 {  // 5% error threshold
                            error!("⚠️  High quantization error detected for {}×{}: {:.2e}", 
                                   result.matrix_size.0, result.matrix_size.1, error);
                        }
                    }
                    
                    if let Some(speedup) = result.speedup {
                        if speedup < 1.0 {
                            error!("⚠️  Metal slower than CPU for {}×{}: {:.2}x", 
                                   result.matrix_size.0, result.matrix_size.1, speedup);
                        }
                    }
                }
                
                Ok(())
            }
            Err(e) => {
                error!("❌ Benchmark suite failed: {}", e);
                Err(e)
            }
        }
    }
}

/// Parse matrix size string (e.g., "4096x4096") into (rows, cols)
fn parse_matrix_size(size_str: &str) -> Result<(usize, usize)> {
    let parts: Vec<&str> = size_str.split('x').collect();
    if parts.len() != 2 {
        return Err(Error::internal(format!("Invalid matrix size format: {}. Use ROWSxCOLS", size_str)));
    }
    
    let rows = parts[0].parse::<usize>()
        .map_err(|_| Error::internal(format!("Invalid row count: {}", parts[0])))?;
    let cols = parts[1].parse::<usize>()
        .map_err(|_| Error::internal(format!("Invalid column count: {}", parts[1])))?;
    
    if cols % 32 != 0 {
        return Err(Error::internal(format!("Column count must be divisible by 32 for Q4_0, got {}", cols)));
    }
    
    Ok((rows, cols))
}
