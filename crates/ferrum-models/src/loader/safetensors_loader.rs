//! SafeTensors weight loader with Candle integration

use candle_core::{DType, Device as CandleDevice};
use candle_nn::VarBuilder;
use ferrum_types::{FerrumError, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// SafeTensors weight loader
pub struct SafeTensorsLoader {
    model_dir: PathBuf,
}

impl SafeTensorsLoader {
    /// Create new loader
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }

    /// Load weights to VarBuilder for model construction
    pub fn load_varbuilder(&self, device: &CandleDevice, dtype: DType) -> Result<VarBuilder> {
        info!("📦 Loading model weights from: {:?}", self.model_dir);

        // Check for single file first
        let single_file = self.model_dir.join("model.safetensors");
        if single_file.exists() {
            info!("  📄 Loading single SafeTensors file...");

            let tensors = vec![single_file.clone()];
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&tensors, dtype, device)
                    .map_err(|e| FerrumError::model(format!("Failed to load SafeTensors: {}", e)))?
            };

            info!("  ✅ SafeTensors weights loaded successfully");
            return Ok(vb);
        }

        // Check for sharded model (model.safetensors.index.json)
        let index_file = self.model_dir.join("model.safetensors.index.json");
        if index_file.exists() {
            info!("  📚 Loading sharded SafeTensors model...");

            // Parse index file to get shard filenames
            let shard_files = self.parse_sharded_index(&index_file)?;
            info!("  📦 Found {} shards", shard_files.len());

            // Build full paths for all shards
            let shard_paths: Vec<PathBuf> = shard_files
                .iter()
                .map(|f| self.model_dir.join(f))
                .collect();

            // Verify all shards exist
            for path in &shard_paths {
                if !path.exists() {
                    return Err(FerrumError::model(format!(
                        "Missing shard file: {}",
                        path.display()
                    )));
                }
            }

            // Load all shards with mmap
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, device).map_err(|e| {
                    FerrumError::model(format!("Failed to load sharded weights: {}", e))
                })?
            };

            info!("  ✅ Sharded weights loaded successfully ({} shards)", shard_files.len());
            return Ok(vb);
        }

        // Check for pytorch model
        let pytorch_file = self.model_dir.join("pytorch_model.bin");
        if pytorch_file.exists() {
            return Err(FerrumError::model(
                "PyTorch .bin format is not supported. Please use SafeTensors format.",
            ));
        }

        Err(FerrumError::model(format!(
            "No SafeTensors files found in model directory: {}",
            self.model_dir.display()
        )))
    }

    /// Parse model.safetensors.index.json to get unique shard filenames
    fn parse_sharded_index(&self, index_file: &Path) -> Result<Vec<String>> {
        let content = std::fs::read_to_string(index_file)
            .map_err(|e| FerrumError::io_str(format!("Failed to read index file: {}", e)))?;

        let index: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| FerrumError::model(format!("Failed to parse index JSON: {}", e)))?;

        // Extract unique shard filenames from weight_map
        let weight_map = index
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or_else(|| FerrumError::model("Invalid index: missing 'weight_map'"))?;

        let shards: HashSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();

        // Sort shard names for consistent ordering
        let mut shard_list: Vec<String> = shards.into_iter().collect();
        shard_list.sort();

        if shard_list.is_empty() {
            return Err(FerrumError::model("No shards found in index file"));
        }

        debug!("Shards to load: {:?}", shard_list);
        Ok(shard_list)
    }
}
