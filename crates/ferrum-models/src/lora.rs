//! Startup LoRA adapter loader and validator.
//!
//! G4 supports PEFT-style startup adapters with:
//! - adapter_config.json
//! - adapter_model.safetensors
//!
//! The production server registry uses the validated metadata here. Actual
//! hot-loading, GGUF LoRA, and multi-adapter composition are intentionally out
//! of scope.

use ferrum_kernels::backend::Backend;
use ferrum_quantization::{DenseLinear, Linear};
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoraAdapterConfig {
    pub r: usize,
    pub lora_alpha: usize,
    pub target_modules: Vec<String>,
    pub base_model_name_or_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraTensorPair {
    pub target_module: String,
    pub a_tensor: String,
    pub b_tensor: String,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
}

#[derive(Debug, Clone)]
pub struct StartupLoraAdapter {
    pub name: String,
    pub public_model_id: String,
    pub path: PathBuf,
    pub config: LoraAdapterConfig,
    pub tensors: Vec<LoraTensorPair>,
}

#[derive(Debug, Clone)]
pub struct StartupLoraSpec {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveLoraAdapter {
    pub name: String,
    pub path: PathBuf,
}

pub struct RuntimeLoraLinear<B: Backend> {
    pub layer_index: Option<usize>,
    pub target_module: String,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub scaling: f32,
    a: DenseLinear<B>,
    b: DenseLinear<B>,
}

pub struct RuntimeLoraAdapter<B: Backend> {
    pub name: String,
    pub path: PathBuf,
    pub config: LoraAdapterConfig,
    pub linears: Vec<RuntimeLoraLinear<B>>,
}

impl<B: Backend> RuntimeLoraAdapter<B> {
    pub fn apply_projection(
        &self,
        ctx: &mut B::Context,
        layer_index: usize,
        target_module: &str,
        input: &B::Buffer,
        out: &mut B::Buffer,
        m: usize,
    ) -> Result<usize> {
        let mut applied = 0usize;
        for linear in self.linears.iter().filter(|linear| {
            linear.target_module == target_module
                && linear
                    .layer_index
                    .map(|idx| idx == layer_index)
                    .unwrap_or(true)
        }) {
            linear.apply(ctx, input, out, m)?;
            applied += 1;
        }
        Ok(applied)
    }

    pub fn supports_projection(
        &self,
        layer_index: usize,
        target_module: &str,
        in_features: usize,
        out_features: usize,
    ) -> bool {
        self.linears.iter().any(|linear| {
            linear.target_module == target_module
                && linear.in_features == in_features
                && linear.out_features == out_features
                && linear
                    .layer_index
                    .map(|idx| idx == layer_index)
                    .unwrap_or(true)
        })
    }
}

impl<B: Backend> RuntimeLoraLinear<B> {
    fn apply(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        out: &mut B::Buffer,
        m: usize,
    ) -> Result<()> {
        let mut low_rank = B::alloc(m * self.rank);
        let mut delta = B::alloc(m * self.out_features);
        self.a.forward(ctx, input, &mut low_rank, m);
        self.b.forward(ctx, &low_rank, &mut delta, m);
        B::scaled_add_inplace(ctx, out, &delta, self.scaling, m * self.out_features);
        Ok(())
    }
}

const SUPPORTED_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
    "linear",
];

pub fn default_lora_model_id(base_model_id: &str, name: &str) -> String {
    format!("{base_model_id}:{name}")
}

pub fn render_lora_model_id(template: &str, base_model_id: &str, name: &str) -> String {
    template
        .replace("<base>", base_model_id)
        .replace("<name>", name)
}

pub fn load_startup_lora_adapter(
    name: impl Into<String>,
    path: impl AsRef<Path>,
    public_model_id: impl Into<String>,
) -> Result<StartupLoraAdapter> {
    let name = name.into();
    validate_lora_name(&name)?;
    let path = path.as_ref().to_path_buf();
    if !path.is_dir() {
        return Err(FerrumError::config(format!(
            "LoRA adapter path does not exist or is not a directory: {}",
            path.display()
        )));
    }

    let config_path = path.join("adapter_config.json");
    let weights_path = path.join("adapter_model.safetensors");
    if !config_path.is_file() {
        return Err(FerrumError::config(format!(
            "LoRA adapter missing adapter_config.json: {}",
            path.display()
        )));
    }
    if !weights_path.is_file() {
        return Err(FerrumError::config(format!(
            "LoRA adapter missing adapter_model.safetensors: {}",
            path.display()
        )));
    }

    let config: LoraAdapterConfig = serde_json::from_slice(
        &std::fs::read(&config_path)
            .map_err(|e| FerrumError::config(format!("read adapter_config.json: {e}")))?,
    )
    .map_err(|e| FerrumError::config(format!("invalid adapter_config.json: {e}")))?;
    validate_lora_config(&config)?;

    let weights = std::fs::read(&weights_path)
        .map_err(|e| FerrumError::config(format!("read adapter_model.safetensors: {e}")))?;
    let tensors = SafeTensors::deserialize(&weights)
        .map_err(|e| FerrumError::config(format!("invalid adapter_model.safetensors: {e}")))?;
    let pairs = collect_lora_tensor_pairs(&config, &tensors)?;

    Ok(StartupLoraAdapter {
        name,
        public_model_id: public_model_id.into(),
        path,
        config,
        tensors: pairs,
    })
}

pub fn load_startup_lora_adapters(
    base_model_id: &str,
    template: Option<&str>,
    specs: &[StartupLoraSpec],
) -> Result<Vec<StartupLoraAdapter>> {
    let mut seen_names = HashSet::new();
    let mut seen_ids = HashSet::new();
    let template = template.unwrap_or("<base>:<name>");
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        if !seen_names.insert(spec.name.clone()) {
            return Err(FerrumError::config(format!(
                "duplicate LoRA adapter name: {}",
                spec.name
            )));
        }
        let public_model_id = render_lora_model_id(template, base_model_id, &spec.name);
        if !seen_ids.insert(public_model_id.clone()) {
            return Err(FerrumError::config(format!(
                "duplicate LoRA public model id: {public_model_id}"
            )));
        }
        out.push(load_startup_lora_adapter(
            spec.name.clone(),
            &spec.path,
            public_model_id,
        )?);
    }
    Ok(out)
}

pub fn load_runtime_lora_adapter<B: Backend>(
    adapter: &ActiveLoraAdapter,
) -> Result<RuntimeLoraAdapter<B>> {
    let startup = load_startup_lora_adapter(
        adapter.name.clone(),
        &adapter.path,
        default_lora_model_id("base", &adapter.name),
    )?;
    let weights_path = adapter.path.join("adapter_model.safetensors");
    let weights = std::fs::read(&weights_path)
        .map_err(|e| FerrumError::config(format!("read adapter_model.safetensors: {e}")))?;
    let tensors = SafeTensors::deserialize(&weights)
        .map_err(|e| FerrumError::config(format!("invalid adapter_model.safetensors: {e}")))?;

    let mut linears = Vec::with_capacity(startup.tensors.len());
    for pair in &startup.tensors {
        let a = tensors
            .tensor(&pair.a_tensor)
            .map_err(|e| FerrumError::config(format!("read LoRA tensor {}: {e}", pair.a_tensor)))?;
        let b = tensors
            .tensor(&pair.b_tensor)
            .map_err(|e| FerrumError::config(format!("read LoRA tensor {}: {e}", pair.b_tensor)))?;
        let a_data = tensor_data_to_f32(a.dtype(), a.data())?;
        let b_data = tensor_data_to_f32(b.dtype(), b.data())?;
        linears.push(RuntimeLoraLinear {
            layer_index: parse_lora_layer_index(&pair.a_tensor),
            target_module: pair.target_module.clone(),
            in_features: pair.in_features,
            out_features: pair.out_features,
            rank: pair.rank,
            scaling: startup.config.lora_alpha as f32 / startup.config.r as f32,
            a: DenseLinear::<B>::from_rows(&a_data, pair.rank, pair.in_features),
            b: DenseLinear::<B>::from_rows(&b_data, pair.out_features, pair.rank),
        });
    }

    Ok(RuntimeLoraAdapter {
        name: adapter.name.clone(),
        path: adapter.path.clone(),
        config: startup.config,
        linears,
    })
}

fn validate_lora_name(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Err(FerrumError::config(format!(
            "invalid LoRA adapter name {name:?}; use [A-Za-z0-9_.-]"
        )));
    }
    Ok(())
}

fn validate_lora_config(config: &LoraAdapterConfig) -> Result<()> {
    if config.r == 0 {
        return Err(FerrumError::config("LoRA adapter rank r must be > 0"));
    }
    if config.target_modules.is_empty() {
        return Err(FerrumError::config(
            "LoRA adapter target_modules must not be empty",
        ));
    }
    for target in &config.target_modules {
        if !SUPPORTED_TARGET_MODULES.contains(&target.as_str()) {
            return Err(FerrumError::config(format!(
                "unsupported LoRA target module: {target}"
            )));
        }
    }
    Ok(())
}

fn collect_lora_tensor_pairs(
    config: &LoraAdapterConfig,
    tensors: &SafeTensors<'_>,
) -> Result<Vec<LoraTensorPair>> {
    let names: Vec<String> = tensors.names().iter().map(|s| (*s).to_string()).collect();
    let name_set: HashSet<&str> = names.iter().map(String::as_str).collect();
    let mut pairs = Vec::new();

    for target in &config.target_modules {
        for a_name in names
            .iter()
            .filter(|name| is_lora_a_for_target(name, target))
        {
            let b_name = a_name.replace(".lora_A.weight", ".lora_B.weight");
            if !name_set.contains(b_name.as_str()) {
                return Err(FerrumError::config(format!(
                    "LoRA tensor pair missing B tensor for {a_name}"
                )));
            }
            let a = tensors
                .tensor(a_name)
                .map_err(|e| FerrumError::config(format!("read LoRA tensor {a_name}: {e}")))?;
            let b = tensors
                .tensor(&b_name)
                .map_err(|e| FerrumError::config(format!("read LoRA tensor {b_name}: {e}")))?;
            validate_lora_dtype(a_name, a.dtype())?;
            validate_lora_dtype(&b_name, b.dtype())?;
            let a_shape = a.shape();
            let b_shape = b.shape();
            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(FerrumError::config(format!(
                    "LoRA tensors must be 2-D: {a_name} shape={a_shape:?}, {b_name} shape={b_shape:?}"
                )));
            }
            let rank = a_shape[0];
            if rank != config.r || b_shape[1] != config.r {
                return Err(FerrumError::config(format!(
                    "LoRA rank mismatch for {target}: config r={}, A shape={a_shape:?}, B shape={b_shape:?}",
                    config.r
                )));
            }
            pairs.push(LoraTensorPair {
                target_module: target.clone(),
                a_tensor: a_name.clone(),
                b_tensor: b_name,
                in_features: a_shape[1],
                out_features: b_shape[0],
                rank,
            });
        }
    }

    if pairs.is_empty() {
        return Err(FerrumError::config(
            "LoRA adapter did not contain any supported target module tensor pairs",
        ));
    }

    let mut unique = HashMap::<String, LoraTensorPair>::new();
    for pair in pairs {
        unique.entry(pair.a_tensor.clone()).or_insert(pair);
    }
    Ok(unique.into_values().collect())
}

fn is_lora_a_for_target(name: &str, target: &str) -> bool {
    let suffix = format!(".{target}.lora_A.weight");
    name.ends_with(&suffix) || name == format!("{target}.lora_A.weight")
}

fn validate_lora_dtype(name: &str, dtype: Dtype) -> Result<()> {
    match dtype {
        Dtype::F16 | Dtype::BF16 | Dtype::F32 => Ok(()),
        other => Err(FerrumError::config(format!(
            "unsupported LoRA tensor dtype for {name}: {other:?}"
        ))),
    }
}

fn tensor_data_to_f32(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(FerrumError::config(
                    "F32 LoRA tensor byte length is not divisible by 4",
                ));
            }
            Ok(data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect())
        }
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(FerrumError::config(
                    "F16 LoRA tensor byte length is not divisible by 2",
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(FerrumError::config(
                    "BF16 LoRA tensor byte length is not divisible by 2",
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|bytes| bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
                .collect())
        }
        other => Err(FerrumError::config(format!(
            "unsupported LoRA tensor dtype for runtime load: {other:?}"
        ))),
    }
}

fn parse_lora_layer_index(name: &str) -> Option<usize> {
    let marker = ".layers.";
    let start = name.find(marker)? + marker.len();
    let digits: String = name[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}
