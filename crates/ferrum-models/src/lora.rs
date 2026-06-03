//! Startup LoRA adapter loader and validator.
//!
//! G4 supports PEFT-style startup adapters with:
//! - adapter_config.json
//! - adapter_model.safetensors
//!
//! The production server registry uses the validated metadata here. Actual
//! hot-loading, GGUF LoRA, and multi-adapter composition are intentionally out
//! of scope.

use ferrum_types::{FerrumError, Result};
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
        for a_name in names.iter().filter(|name| is_lora_a_for_target(name, target)) {
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
