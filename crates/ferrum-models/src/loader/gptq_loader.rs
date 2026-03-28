//! GPTQ quantized model loader.
//!
//! Loads GPTQ INT4 weights from HuggingFace safetensors and provides:
//! - Dequantized FP16 tensors for candle prefill (via VarBuilder)
//! - Packed INT4 weights for CUDA runner decode (via GpuQuantWeight)

use ferrum_types::{FerrumError, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// GPTQ quantization config (from quantize_config.json).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizeConfig {
    pub bits: usize,
    pub group_size: i64,
    #[serde(default)]
    pub sym: bool,
    #[serde(default)]
    pub desc_act: bool,
    #[serde(default)]
    pub quant_method: String,
}

impl QuantizeConfig {
    /// Try to load quantize_config.json from model directory.
    /// Returns None if file doesn't exist (non-quantized model).
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("quantize_config.json");
        if !path.exists() {
            // Also check config.json for embedded quantization_config
            let config_path = model_dir.join("config.json");
            if config_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(qc) = config.get("quantization_config") {
                            if let Ok(qconfig) =
                                serde_json::from_value::<QuantizeConfig>(qc.clone())
                            {
                                tracing::info!("GPTQ config found in config.json: {:?}", qconfig);
                                return Ok(Some(qconfig));
                            }
                        }
                    }
                }
            }
            return Ok(None);
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| FerrumError::model(format!("read quantize_config.json: {e}")))?;
        let config: QuantizeConfig = serde_json::from_str(&content)
            .map_err(|e| FerrumError::model(format!("parse quantize_config.json: {e}")))?;
        tracing::info!("GPTQ config: {:?}", config);
        Ok(Some(config))
    }

    pub fn effective_group_size(&self, k: usize) -> usize {
        if self.group_size <= 0 {
            k // per-channel
        } else {
            self.group_size as usize
        }
    }
}

/// Packed GPTQ weights for one linear layer (CPU-side, before GPU upload).
#[derive(Debug)]
pub struct GptqLayerWeights {
    /// Packed INT4 weights: [K/8, N] as int32
    pub qweight: Vec<i32>,
    /// Per-group scales: [K/group_size, N] as f16 bytes
    pub scales: Vec<half::f16>,
    /// Per-group zero-points: [K/group_size, N/8] as int32 (None for symmetric)
    pub qzeros: Option<Vec<i32>>,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
    pub symmetric: bool,
}

impl GptqLayerWeights {
    /// Dequantize to FP16 on CPU. Returns [K, N] in row-major order.
    pub fn dequantize_cpu(&self) -> Vec<half::f16> {
        let mut output = vec![half::f16::ZERO; self.k * self.n];
        let packed_rows = self.k / 8;

        for packed_row in 0..packed_rows {
            for col in 0..self.n {
                let packed = self.qweight[packed_row * self.n + col];
                let base_k = packed_row * 8;
                let group = base_k / self.group_size;
                let scale = self.scales[group * self.n + col].to_f32();

                let zero = if self.symmetric {
                    8
                } else if let Some(ref qz) = self.qzeros {
                    let zp_packed = qz[group * (self.n / 8) + col / 8];
                    let zp_shift = (col % 8) * 4;
                    (zp_packed >> zp_shift) & 0xF
                } else {
                    8
                };

                for i in 0..8 {
                    let val = (packed >> (i * 4)) & 0xF;
                    let dequantized = (val - zero) as f32 * scale;
                    output[(base_k + i as usize) * self.n + col] = half::f16::from_f32(dequantized);
                }
            }
        }
        output
    }
}

/// Load GPTQ packed weights from safetensors files.
///
/// Returns a map of layer_prefix → GptqLayerWeights.
/// Layer prefixes are like "model.layers.0.self_attn.q_proj".
pub fn load_gptq_weights(
    model_dir: &Path,
    qconfig: &QuantizeConfig,
) -> Result<HashMap<String, GptqLayerWeights>> {
    use safetensors::SafeTensors;

    let safetensor_files = find_safetensor_files(model_dir)?;
    if safetensor_files.is_empty() {
        return Err(FerrumError::model("No safetensor files found"));
    }

    let mut result = HashMap::new();

    // Collect all qweight tensor names to find layer prefixes
    for path in &safetensor_files {
        let data = std::fs::read(path)
            .map_err(|e| FerrumError::model(format!("read {}: {e}", path.display())))?;
        let st = SafeTensors::deserialize(&data)
            .map_err(|e| FerrumError::model(format!("parse {}: {e}", path.display())))?;

        for (name, _) in st.tensors() {
            if !name.ends_with(".qweight") {
                continue;
            }
            let prefix = name.strip_suffix(".qweight").unwrap().to_string();

            // Load qweight
            let qw_tensor = st
                .tensor(&format!("{prefix}.qweight"))
                .map_err(|e| FerrumError::model(format!("{prefix}.qweight: {e}")))?;
            let qweight: Vec<i32> = bytemuck::cast_slice(qw_tensor.data()).to_vec();
            let qw_shape = qw_tensor.shape();
            let packed_k = qw_shape[0]; // K/8
            let n = qw_shape[1];
            let k = packed_k * 8;

            // Load scales
            let sc_tensor = st
                .tensor(&format!("{prefix}.scales"))
                .map_err(|e| FerrumError::model(format!("{prefix}.scales: {e}")))?;
            let scales: Vec<half::f16> = bytemuck::cast_slice(sc_tensor.data()).to_vec();

            // Load qzeros (optional for symmetric)
            let qzeros = if !qconfig.sym {
                let qz_tensor = st
                    .tensor(&format!("{prefix}.qzeros"))
                    .map_err(|e| FerrumError::model(format!("{prefix}.qzeros: {e}")))?;
                Some(bytemuck::cast_slice(qz_tensor.data()).to_vec())
            } else {
                None
            };

            let gs = qconfig.effective_group_size(k);

            tracing::debug!(
                "GPTQ layer: {prefix} K={k} N={n} group_size={gs} sym={}",
                qconfig.sym
            );

            result.insert(
                prefix,
                GptqLayerWeights {
                    qweight,
                    scales,
                    qzeros,
                    k,
                    n,
                    group_size: gs,
                    symmetric: qconfig.sym,
                },
            );
        }
    }

    tracing::info!("Loaded {} GPTQ quantized layers", result.len());
    Ok(result)
}

fn find_safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    // Check single file
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        files.push(single);
        return Ok(files);
    }

    // Check sharded index
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| FerrumError::model(format!("read index: {e}")))?;
        let index: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| FerrumError::model(format!("parse index: {e}")))?;
        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            let mut seen = std::collections::HashSet::new();
            for filename in weight_map.values().filter_map(|v| v.as_str()) {
                if seen.insert(filename.to_string()) {
                    let path = model_dir.join(filename);
                    if path.exists() {
                        files.push(path);
                    }
                }
            }
        }
    }

    Ok(files)
}
