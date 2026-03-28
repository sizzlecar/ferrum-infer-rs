//! SafeTensors weight loader with Candle integration

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ferrum_types::{FerrumError, Result};
use std::collections::{HashMap, HashSet};
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
    pub fn load_varbuilder(&self, device: &CandleDevice, dtype: DType) -> Result<VarBuilder<'_>> {
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
            let shard_paths: Vec<PathBuf> =
                shard_files.iter().map(|f| self.model_dir.join(f)).collect();

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

            info!(
                "  ✅ Sharded weights loaded successfully ({} shards)",
                shard_files.len()
            );
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

    /// Load GPTQ quantized model: dequantize INT4→FP16 on CPU, return VarBuilder.
    ///
    /// GPTQ safetensors use .qweight/.scales/.qzeros instead of .weight.
    /// This method dequantizes them to FP16 so candle can load the model normally.
    pub fn load_varbuilder_gptq(
        &self,
        qconfig: &super::QuantizeConfig,
        device: &CandleDevice,
        dtype: DType,
    ) -> Result<VarBuilder<'static>> {
        info!("Loading GPTQ model from: {:?}", self.model_dir);

        // 1. Load GPTQ packed weights
        let gptq_weights = super::load_gptq_weights(&self.model_dir, qconfig)?;

        // 2. Load non-quantized tensors from safetensors
        let st_files = self.find_all_safetensor_files()?;
        let mut tensor_map: HashMap<String, Tensor> = HashMap::new();

        for path in &st_files {
            let data = std::fs::read(path)
                .map_err(|e| FerrumError::model(format!("read {}: {e}", path.display())))?;
            let st = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| FerrumError::model(format!("parse {}: {e}", path.display())))?;

            for (name, _) in st.tensors() {
                // Skip quantization-specific tensors
                if name.ends_with(".qweight")
                    || name.ends_with(".scales")
                    || name.ends_with(".qzeros")
                    || name.ends_with(".g_idx")
                {
                    continue;
                }

                let view = st
                    .tensor(&name)
                    .map_err(|e| FerrumError::model(format!("tensor {name}: {e}")))?;

                let candle_dtype = match view.dtype() {
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::BF16 => DType::BF16,
                    safetensors::Dtype::F32 => DType::F32,
                    safetensors::Dtype::F64 => DType::F64,
                    other => {
                        debug!("Skipping tensor {name} with dtype {:?}", other);
                        continue;
                    }
                };

                let tensor = Tensor::from_raw_buffer(
                    view.data(),
                    candle_dtype,
                    view.shape(),
                    &CandleDevice::Cpu,
                )
                .map_err(|e| FerrumError::model(format!("tensor {name}: {e}")))?
                .to_device(device)
                .map_err(|e| FerrumError::model(format!("tensor {name} to device: {e}")))?
                .to_dtype(dtype)
                .map_err(|e| FerrumError::model(format!("tensor {name} to dtype: {e}")))?;

                tensor_map.insert(name.to_string(), tensor);
            }
        }
        info!("Loaded {} non-quantized tensors", tensor_map.len());

        // 3. Dequantize quantized weights → FP16, add as .weight
        for (prefix, gw) in &gptq_weights {
            let dequant = gw.dequantize_cpu();
            let f32_data: Vec<f32> = dequant.iter().map(|x| x.to_f32()).collect();
            // GPTQ stores [K, N] (in_features, out_features) but candle Linear
            // expects [N, K] (out_features, in_features). Transpose after reshape.
            let tensor = Tensor::new(&f32_data[..], &CandleDevice::Cpu)
                .map_err(|e| FerrumError::model(format!("dequant {prefix}: {e}")))?
                .reshape(&[gw.k, gw.n])
                .map_err(|e| FerrumError::model(format!("dequant {prefix} reshape: {e}")))?
                .t()
                .map_err(|e| FerrumError::model(format!("dequant {prefix} transpose: {e}")))?
                .contiguous()
                .map_err(|e| FerrumError::model(format!("dequant {prefix} contiguous: {e}")))?
                .to_device(device)
                .map_err(|e| FerrumError::model(format!("dequant {prefix} to device: {e}")))?
                .to_dtype(dtype)
                .map_err(|e| FerrumError::model(format!("dequant {prefix} to dtype: {e}")))?;

            let weight_name = format!("{prefix}.weight");
            debug!("Dequantized: {weight_name} [{}, {}]", gw.n, gw.k);
            tensor_map.insert(weight_name, tensor);
        }
        info!(
            "Total tensors: {} ({} dequantized from INT4)",
            tensor_map.len(),
            gptq_weights.len()
        );

        Ok(VarBuilder::from_tensors(tensor_map, dtype, device))
    }

    /// Find all safetensor files (single or sharded).
    fn find_all_safetensor_files(&self) -> Result<Vec<PathBuf>> {
        let single = self.model_dir.join("model.safetensors");
        if single.exists() {
            return Ok(vec![single]);
        }
        let index_file = self.model_dir.join("model.safetensors.index.json");
        if index_file.exists() {
            let shard_files = self.parse_sharded_index(&index_file)?;
            return Ok(shard_files.iter().map(|f| self.model_dir.join(f)).collect());
        }
        Ok(vec![])
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
