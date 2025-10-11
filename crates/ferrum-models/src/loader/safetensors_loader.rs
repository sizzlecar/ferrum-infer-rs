//! SafeTensors weight loader with Candle integration

use candle_core::{DType, Device as CandleDevice};
use candle_nn::VarBuilder;
use ferrum_types::{FerrumError, Result};
use std::path::Path;
use tracing::{info, debug};

/// SafeTensors weight loader
pub struct SafeTensorsLoader {
    model_dir: std::path::PathBuf,
}

impl SafeTensorsLoader {
    /// Create new loader
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }
    
    /// Load weights to VarBuilder for model construction
    pub fn load_varbuilder(
        &self,
        device: &CandleDevice,
        dtype: DType,
    ) -> Result<VarBuilder> {
        info!("📦 Loading model weights from: {:?}", self.model_dir);
        
        // Check for single file
        let single_file = self.model_dir.join("model.safetensors");
        if single_file.exists() {
            info!("  📄 Loading single SafeTensors file...");
            
            // Load SafeTensors file
            let tensors = vec![single_file.clone()];
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&tensors, dtype, device)
                    .map_err(|e| FerrumError::model(format!("Failed to load SafeTensors: {}", e)))?
            };
            
            info!("  ✅ SafeTensors weights loaded successfully");
            return Ok(vb);
        }
        
        // Check for sharded model
        let index_file = self.model_dir.join("model.safetensors.index.json");
        if index_file.exists() {
            info!("  📚 Loading sharded SafeTensors model...");
            
            // Use Candle's built-in sharded loading with mmap
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[self.model_dir.clone()],
                    dtype,
                    device,
                )
                .map_err(|e| FerrumError::model(format!("Failed to load sharded weights: {}", e)))?
            };
            
            info!("  ✅ Sharded weights loaded successfully");
            return Ok(vb);
        }
        
        Err(FerrumError::model(
            "No SafeTensors files found in model directory",
        ))
    }
}

