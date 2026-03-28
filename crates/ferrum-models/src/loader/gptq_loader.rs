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

    tracing::info!("Loaded {} GPTQ quantized layers (raw)", result.len());

    // Fuse separate q/k/v → qkv_proj and gate/up → gate_up_proj.
    // GPTQ stores separate projections, but the CUDA runner expects fused weights.
    fuse_qkv_and_gate_up(&mut result);

    tracing::info!(
        "After fusion: {} GPTQ layers (includes fused qkv_proj, gate_up_proj)",
        result.len()
    );
    Ok(result)
}

/// Fuse separate q/k/v projections into qkv_proj, and gate/up into gate_up_proj.
/// GPTQ packing is [K/8, N] — fusing along N dimension means concatenating columns.
fn fuse_qkv_and_gate_up(weights: &mut HashMap<String, GptqLayerWeights>) {
    let prefixes: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".self_attn.q_proj"))
        .map(|k| k.strip_suffix(".self_attn.q_proj").unwrap().to_string())
        .collect();

    for layer_prefix in &prefixes {
        // Fuse q + k + v → qkv_proj
        let q_key = format!("{layer_prefix}.self_attn.q_proj");
        let k_key = format!("{layer_prefix}.self_attn.k_proj");
        let v_key = format!("{layer_prefix}.self_attn.v_proj");
        if let (Some(q), Some(k), Some(v)) = (
            weights.get(&q_key),
            weights.get(&k_key),
            weights.get(&v_key),
        ) {
            if q.k == k.k && q.k == v.k {
                let fused = fuse_columns(&[q, k, v]);
                let fused_key = format!("{layer_prefix}.self_attn.qkv_proj");
                tracing::info!(
                    "Fused {q_key}+{k_key}+{v_key} → {fused_key} K={} N={}",
                    fused.k,
                    fused.n
                );
                weights.insert(fused_key, fused);
            }
        }

        // Fuse gate + up → gate_up_proj
        let gate_key = format!("{layer_prefix}.mlp.gate_proj");
        let up_key = format!("{layer_prefix}.mlp.up_proj");
        if let (Some(gate), Some(up)) = (weights.get(&gate_key), weights.get(&up_key)) {
            if gate.k == up.k {
                let fused = fuse_columns(&[gate, up]);
                let fused_key = format!("{layer_prefix}.mlp.gate_up_proj");
                tracing::info!(
                    "Fused {gate_key}+{up_key} → {fused_key} K={} N={}",
                    fused.k,
                    fused.n
                );
                weights.insert(fused_key, fused);
            }
        }
    }
}

/// Fuse multiple GPTQ weights along the N (output) dimension.
/// All weights must have the same K. Result has N = sum(w.n for w in weights).
///
/// qweight layout: [K/8, N] — concatenate columns.
/// scales layout: [K/gs, N] — concatenate columns.
/// qzeros layout: [K/gs, N/8] — concatenate columns (trickier due to packing).
fn fuse_columns(parts: &[&GptqLayerWeights]) -> GptqLayerWeights {
    let k = parts[0].k;
    let gs = parts[0].group_size;
    let sym = parts[0].symmetric;
    let total_n: usize = parts.iter().map(|p| p.n).sum();
    let packed_k = k / 8;
    let num_groups = k / gs;

    // Fuse qweight [K/8, N] — row by row, concatenate columns
    let mut qweight = vec![0i32; packed_k * total_n];
    let mut col_offset = 0;
    for part in parts {
        for row in 0..packed_k {
            for col in 0..part.n {
                qweight[row * total_n + col_offset + col] = part.qweight[row * part.n + col];
            }
        }
        col_offset += part.n;
    }

    // Fuse scales [K/gs, N]
    let mut scales = vec![half::f16::ZERO; num_groups * total_n];
    col_offset = 0;
    for part in parts {
        for row in 0..num_groups {
            for col in 0..part.n {
                scales[row * total_n + col_offset + col] = part.scales[row * part.n + col];
            }
        }
        col_offset += part.n;
    }

    // Fuse qzeros [K/gs, N/8] — need to unpack, concatenate, repack
    let qzeros = if !sym {
        let mut all_zeros = vec![0u8; num_groups * total_n];
        let mut col_off = 0usize;
        for part in parts {
            if let Some(ref qz) = part.qzeros {
                let part_n8 = part.n / 8;
                for row in 0..num_groups {
                    for col in 0..part.n {
                        let packed = qz[row * part_n8 + col / 8];
                        let val = ((packed >> ((col % 8) * 4)) & 0xF) as u8;
                        all_zeros[row * total_n + col_off + col] = val;
                    }
                }
            }
            col_off += part.n;
        }
        // Repack
        let total_n8 = total_n / 8;
        let mut packed_zeros = vec![0i32; num_groups * total_n8];
        for row in 0..num_groups {
            for col in 0..total_n {
                let val = all_zeros[row * total_n + col] as i32;
                packed_zeros[row * total_n8 + col / 8] |= val << ((col % 8) * 4);
            }
        }
        Some(packed_zeros)
    } else {
        None
    };

    GptqLayerWeights {
        qweight,
        scales,
        qzeros,
        k,
        n: total_n,
        group_size: gs,
        symmetric: sym,
    }
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
