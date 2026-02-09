//! Simplified Metal backend for MVP that extends Candle with Apple GPU acceleration
//!
//! This backend provides a pragmatic hybrid approach:
//! - Delegates most operations to CandleBackend for compatibility
//! - Accelerates specific operations with Metal when available
//! - Provides seamless fallback when Metal is not available

use ferrum_types::{Device, FerrumError, Result};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::metal::{MetalContext, Q4_0MatrixOps};

/// Metal-accelerated backend that wraps standard compute operations
///
/// This backend provides optional Metal GPU acceleration while
/// maintaining compatibility with the standard compute path.
pub struct MetalBackend {
    metal_context: Option<Arc<MetalContext>>,
    q4_0_ops: Option<Q4_0MatrixOps>,
    device: Device,
    initialized: bool,
}

impl MetalBackend {
    /// Create a new Metal backend
    pub fn new(device: Device) -> Result<Self> {
        debug!("Creating Metal backend for device: {:?}", device);

        Ok(Self {
            metal_context: None,
            q4_0_ops: None,
            device,
            initialized: false,
        })
    }

    /// Initialize the Metal context and operations
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        debug!("Initializing Metal backend");

        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            if matches!(self.device, Device::Metal) {
                match MetalContext::new() {
                    Ok(mut context) => {
                        // Try to load shader library
                        if let Err(e) = context.load_shader_library() {
                            warn!("Failed to load Metal shaders: {}. Falling back to CPU.", e);
                        } else {
                            debug!("Metal context initialized successfully");
                            let context_arc = Arc::new(context);

                            // Initialize Q4_0 operations
                            match Q4_0MatrixOps::new(context_arc.clone()) {
                                Ok(ops) => {
                                    info!("Q4_0 matrix operations initialized");
                                    self.q4_0_ops = Some(ops);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to initialize Q4_0 operations: {}. Using fallback.",
                                        e
                                    );
                                }
                            }

                            self.metal_context = Some(context_arc);
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to create Metal context: {}. Falling back to CPU.",
                            e
                        );
                    }
                }
            }
        }

        self.initialized = true;

        if self.has_metal_acceleration() {
            info!("Metal backend initialized with GPU acceleration");
        } else {
            info!("Metal backend initialized with CPU fallback");
        }

        Ok(())
    }

    /// Check if Metal acceleration is available
    pub fn has_metal_acceleration(&self) -> bool {
        self.metal_context.is_some()
    }

    /// Check if Q4_0 quantized operations are available
    pub fn has_q4_0_acceleration(&self) -> bool {
        self.q4_0_ops.is_some()
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the Metal context (if available)
    pub fn metal_context(&self) -> Option<&Arc<MetalContext>> {
        self.metal_context.as_ref()
    }

    /// Get backend name
    pub fn name(&self) -> &str {
        if self.has_metal_acceleration() {
            "Metal"
        } else {
            "Metal (CPU fallback)"
        }
    }

    /// Check if backend supports specific device
    pub fn supports_device(&self, device: &Device) -> bool {
        match device {
            Device::CPU => true,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => true,
            Device::CUDA(_) => false,
            Device::ROCm(_) => false,
        }
    }

    /// Execute optimized Q4_0 matrix-vector multiplication
    ///
    /// This method provides a high-performance path for Linear layers
    /// using Q4_0 quantization on Apple GPU
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub fn q4_0_matvec(
        &self,
        weights: &[f32],
        input: &[f32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Vec<f32>> {
        if let Some(ref ops) = self.q4_0_ops {
            ops.quantized_matvec(weights, input, nrows, ncols)
                .map_err(|e| FerrumError::internal(format!("Q4_0 matvec failed: {}", e)))
        } else {
            Err(FerrumError::internal("Q4_0 operations not available"))
        }
    }

    /// Execute pre-quantized Q4_0 matrix-vector multiplication
    ///
    /// Use this for cached/pre-processed weights to avoid repeated quantization
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub fn q4_0_matvec_quantized(
        &self,
        quantized_weights: &[u32],
        input: &[f32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Vec<f32>> {
        if let Some(ref ops) = self.q4_0_ops {
            ops.execute_quantized_matvec(quantized_weights, input, nrows, ncols)
                .map_err(|e| FerrumError::internal(format!("Q4_0 quantized matvec failed: {}", e)))
        } else {
            Err(FerrumError::internal("Q4_0 operations not available"))
        }
    }

    /// Stub for non-Metal platforms
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    pub fn q4_0_matvec(
        &self,
        _weights: &[f32],
        _input: &[f32],
        _nrows: usize,
        _ncols: usize,
    ) -> Result<Vec<f32>> {
        Err(FerrumError::internal(
            "Q4_0 operations not available on this platform",
        ))
    }

    /// Stub for non-Metal platforms
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    pub fn q4_0_matvec_quantized(
        &self,
        _quantized_weights: &[u32],
        _input: &[f32],
        _nrows: usize,
        _ncols: usize,
    ) -> Result<Vec<f32>> {
        Err(FerrumError::internal(
            "Q4_0 operations not available on this platform",
        ))
    }
}
