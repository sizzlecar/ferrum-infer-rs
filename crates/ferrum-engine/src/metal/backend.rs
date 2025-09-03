//! Simplified Metal backend for MVP that extends Candle with Apple GPU acceleration
//!
//! This backend provides a pragmatic hybrid approach:
//! - Delegates most operations to CandleBackend for compatibility
//! - Accelerates specific operations with Metal when available
//! - Provides seamless fallback when Metal is not available

use async_trait::async_trait;
use ferrum_core::{
    Backend, BackendCapabilities, DataType, Device, Error, Model, Result, Tensor,
};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::candle_backend::CandleBackend;
use crate::metal::{MetalContext, Q4_0MatrixOps};

/// Metal-accelerated backend that wraps CandleBackend
pub struct MetalBackend {
    candle_backend: CandleBackend,
    metal_context: Option<Arc<MetalContext>>,
    q4_0_ops: Option<Q4_0MatrixOps>,
    device: Device,
}

impl MetalBackend {
    /// Create a new Metal backend
    pub fn new(device: Device) -> Result<Self> {
        debug!("Creating Metal backend for device: {:?}", device);
        
        // Create the underlying Candle backend
        let candle_backend = CandleBackend::new(device.clone())?;
        
        Ok(Self {
            candle_backend,
            metal_context: None,
            q4_0_ops: None,
            device,
        })
    }

    /// Initialize Metal context if on Apple platform
    async fn initialize_metal_context(&mut self) -> Result<()> {
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
                                    warn!("Failed to initialize Q4_0 operations: {}. Using fallback.", e);
                                }
                            }
                            
                            self.metal_context = Some(context_arc);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to create Metal context: {}. Falling back to CPU.", e);
                    }
                }
            }
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
                .map_err(|e| Error::internal(format!("Q4_0 matvec failed: {}", e)))
        } else {
            Err(Error::internal("Q4_0 operations not available"))
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
                .map_err(|e| Error::internal(format!("Q4_0 quantized matvec failed: {}", e)))
        } else {
            Err(Error::internal("Q4_0 operations not available"))
        }
    }
}

#[async_trait]
impl Backend for MetalBackend {
    async fn initialize(&mut self) -> Result<()> {
        debug!("Initializing Metal backend");
        
        // Initialize the underlying Candle backend
        self.candle_backend.initialize().await?;
        
        // Initialize Metal context for acceleration
        self.initialize_metal_context().await?;
        
        if self.has_metal_acceleration() {
            debug!("Metal backend initialized with GPU acceleration");
        } else {
            debug!("Metal backend initialized with CPU fallback");
        }
        
        Ok(())
    }

    fn create_tensor(&self, data: Vec<f32>, shape: Vec<usize>, device: &Device) -> Result<Tensor> {
        // Delegate to Candle backend
        self.candle_backend.create_tensor(data, shape, device)
    }

    async fn load_weights(
        &self,
        path: &str,
        dtype: DataType,
        device: &Device,
    ) -> Result<Box<dyn Model>> {
        debug!("Loading model with Metal backend: {}", path);
        
        // Load base model with Candle backend
        let base_model = self.candle_backend.load_weights(path, dtype, device).await?;
        
        // Wrap with Metal acceleration if available
        if self.has_metal_acceleration() && self.has_q4_0_acceleration() {
            debug!("Wrapping model with Metal acceleration");
            let metal_model = crate::metal::MetalOptimizedModel::new(
                base_model,
                Arc::new(MetalBackend::new(device.clone())?) // Create a reference for the model
            );
            Ok(Box::new(metal_model))
        } else {
            debug!("Metal acceleration not available, using standard model");
            Ok(base_model)
        }
    }

    fn name(&self) -> &str {
        if self.has_metal_acceleration() {
            "Metal+Candle"
        } else {
            "Candle"
        }
    }

    fn supports_device(&self, device: &Device) -> bool {
        match device {
            Device::CPU => true,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => true,
            Device::CUDA(_) => self.candle_backend.supports_device(device),
            Device::ROCm(_) => self.candle_backend.supports_device(device),
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        let mut capabilities = self.candle_backend.capabilities();
        
        // Enhanced capabilities when Metal acceleration is available
        if self.has_metal_acceleration() {
            capabilities.supports_flash_attention = true;
            capabilities.max_batch_size = 64; // Increased with Metal optimization
            capabilities.max_sequence_length = 4096; // Increased with Metal optimization
        }
        
        capabilities
    }
}