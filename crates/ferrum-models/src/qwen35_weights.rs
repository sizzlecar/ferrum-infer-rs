//! Qwen3.5 / Qwen3.6 safetensors inventory and manifest validation.
//!
//! This module only inspects safetensors metadata. It does not materialize
//! tensor data, so it can run before the W3 executor allocates model weights.

use std::collections::BTreeSet;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

use ferrum_kernels::backend::Backend;
use ferrum_quantization::{Linear, WeightLoader};
use ferrum_types::{FerrumError, Result as FerrumResult};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::qwen35_config::{
    Qwen35LayerType, Qwen35MlpKind, Qwen35TextConfig, Qwen35WeightManifest, Qwen35WeightSpec,
};

const PREFIX_CANDIDATES: &[&str] = &["model.language_model", "model"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35WeightInventory {
    names: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35WeightValidation {
    pub prefix: String,
    pub missing_required: Vec<String>,
    pub present_required: Vec<String>,
    pub present_optional: Vec<String>,
    pub missing_optional: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ResolvedWeightSpec {
    pub role: String,
    pub name: String,
    pub required: bool,
    pub present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ResolvedLayerWeights {
    pub layer_index: usize,
    pub attention: Qwen35LayerType,
    pub mlp: Qwen35MlpKind,
    pub tensors: Vec<Qwen35ResolvedWeightSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ResolvedWeightPlan {
    pub prefix: String,
    pub global_tensors: Vec<Qwen35ResolvedWeightSpec>,
    pub layers: Vec<Qwen35ResolvedLayerWeights>,
}

pub struct Qwen35WeightPlanLoader<'a, B: Backend> {
    plan: &'a Qwen35ResolvedWeightPlan,
    inner: &'a dyn WeightLoader<B>,
}

impl<'a, B: Backend> Qwen35WeightPlanLoader<'a, B> {
    pub fn new(plan: &'a Qwen35ResolvedWeightPlan, inner: &'a dyn WeightLoader<B>) -> Self {
        Self { plan, inner }
    }

    pub fn plan(&self) -> &'a Qwen35ResolvedWeightPlan {
        self.plan
    }

    pub fn has_global_tensor(&self, role: &str) -> bool {
        self.plan
            .global_tensor(role)
            .is_some_and(|tensor| tensor.present && self.inner.has_tensor(&tensor.name))
    }

    pub fn has_layer_tensor(&self, layer_index: usize, role: &str) -> bool {
        self.plan
            .layer_tensor(layer_index, role)
            .is_some_and(|tensor| self.inner.has_tensor(&tensor.name))
    }

    pub fn load_global_tensor(&self, role: &str) -> FerrumResult<B::Buffer> {
        let tensor = self.required_global_tensor(role)?;
        self.inner.load_tensor(&tensor.name)
    }

    pub fn load_layer_tensor(&self, layer_index: usize, role: &str) -> FerrumResult<B::Buffer> {
        let tensor = self.required_layer_tensor(layer_index, role)?;
        self.inner.load_tensor(&tensor.name)
    }

    pub fn load_global_linear(&self, role: &str) -> FerrumResult<Box<dyn Linear<B>>> {
        let tensor = self.required_global_tensor(role)?;
        self.inner.load_linear(&linear_module_name(&tensor.name))
    }

    pub fn load_layer_linear(
        &self,
        layer_index: usize,
        role: &str,
    ) -> FerrumResult<Box<dyn Linear<B>>> {
        let tensor = self.required_layer_tensor(layer_index, role)?;
        self.inner.load_linear(&linear_module_name(&tensor.name))
    }

    pub fn load_layer_dense_gate_up_linear(
        &self,
        layer_index: usize,
    ) -> FerrumResult<Box<dyn Linear<B>>> {
        let gate = self.required_layer_tensor(layer_index, "mlp_gate")?;
        let up = self.required_layer_tensor(layer_index, "mlp_up")?;
        let gate_module = linear_module_name(&gate.name);
        let up_module = linear_module_name(&up.name);
        let prefix = gate_module.strip_suffix("gate_proj").ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 dense MLP gate tensor for layer {layer_index} does not end with \
                 gate_proj: {gate_module}"
            ))
        })?;
        let expected_up = format!("{prefix}up_proj");
        if up_module != expected_up {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense MLP gate/up tensors for layer {layer_index} do not share a \
                 fusion prefix: gate={gate_module} up={up_module}"
            )));
        }

        self.inner.load_linear(&format!("{prefix}gate_up_proj"))
    }

    pub fn load_layer_linear_attention_qkvz(
        &self,
        layer_index: usize,
    ) -> FerrumResult<Box<dyn Linear<B>>> {
        self.load_fused_layer_linear(
            layer_index,
            "linear_attn_qkv",
            "linear_attn_z",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_qkvz",
        )
    }

    pub fn load_layer_linear_attention_ba(
        &self,
        layer_index: usize,
    ) -> FerrumResult<Box<dyn Linear<B>>> {
        self.load_fused_layer_linear(
            layer_index,
            "linear_attn_b",
            "linear_attn_a",
            "in_proj_b",
            "in_proj_a",
            "in_proj_ba",
        )
    }

    pub fn load_layer_stacked_gptq_experts(
        &self,
        layer_index: usize,
        num_experts: usize,
        proj_names: &[&str],
    ) -> FerrumResult<(
        std::sync::Arc<dyn ferrum_kernels::MarlinExpertStack<B>>,
        usize,
        usize,
    )> {
        let expert_prefix = format!(
            "{}.layers.{layer_index}.mlp.experts.{{e}}.",
            self.plan.prefix
        );
        self.inner
            .load_stacked_gptq_experts(&expert_prefix, num_experts, proj_names)
    }

    fn required_global_tensor(&self, role: &str) -> FerrumResult<&Qwen35ResolvedWeightSpec> {
        let tensor = self.plan.global_tensor(role).ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 resolved weight plan has no global tensor role {role:?}"
            ))
        })?;
        if !tensor.present {
            return Err(FerrumError::model(format!(
                "Qwen3.5 global tensor role {role:?} is absent: {}",
                tensor.name
            )));
        }
        Ok(tensor)
    }

    fn required_layer_tensor(
        &self,
        layer_index: usize,
        role: &str,
    ) -> FerrumResult<&Qwen35ResolvedWeightSpec> {
        self.plan.layer_tensor(layer_index, role).ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 resolved weight plan has no present layer tensor role {role:?} at layer {layer_index}"
            ))
        })
    }

    fn load_fused_layer_linear(
        &self,
        layer_index: usize,
        first_role: &str,
        second_role: &str,
        first_suffix: &str,
        second_suffix: &str,
        fused_suffix: &str,
    ) -> FerrumResult<Box<dyn Linear<B>>> {
        let first = self.required_layer_tensor(layer_index, first_role)?;
        let second = self.required_layer_tensor(layer_index, second_role)?;
        let first_module = linear_module_name(&first.name);
        let second_module = linear_module_name(&second.name);
        let prefix = first_module.strip_suffix(first_suffix).ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 layer {layer_index} tensor for role {first_role:?} does not end with \
                 {first_suffix}: {first_module}"
            ))
        })?;
        let expected_second = format!("{prefix}{second_suffix}");
        if second_module != expected_second {
            return Err(FerrumError::model(format!(
                "Qwen3.5 layer {layer_index} fused projection roles {first_role:?}/{second_role:?} \
                 do not share a fusion prefix: first={first_module} second={second_module}"
            )));
        }

        self.inner.load_linear(&format!("{prefix}{fused_suffix}"))
    }
}

impl Qwen35WeightInventory {
    pub fn from_names(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            names: names.into_iter().map(Into::into).collect(),
        }
    }

    pub fn from_safetensors_dir(model_dir: &Path) -> Result<Self, String> {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            return Self::from_safetensors_files([single]);
        }

        let index = model_dir.join("model.safetensors.index.json");
        if index.exists() {
            return Self::from_safetensors_index(model_dir, &index);
        }

        let mut files = fs::read_dir(model_dir)
            .map_err(|err| format!("read_dir {model_dir:?}: {err}"))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        files.sort();
        if files.is_empty() {
            Err(format!("no safetensors files found in {model_dir:?}"))
        } else {
            Self::from_safetensors_files(files)
        }
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn validate_manifest(&self, manifest: &Qwen35WeightManifest) -> Qwen35WeightValidation {
        let mut missing_required = Vec::new();
        let mut present_required = Vec::new();
        let mut present_optional = Vec::new();
        let mut missing_optional = Vec::new();

        for tensor in manifest.global_tensors.iter().chain(
            manifest
                .layers
                .iter()
                .flat_map(|layer| layer.tensors.iter()),
        ) {
            let present = self.contains_weight_spec(tensor);
            match (tensor.required, present) {
                (true, true) => present_required.push(tensor.name.clone()),
                (true, false) => missing_required.push(tensor.name.clone()),
                (false, true) => present_optional.push(tensor.name.clone()),
                (false, false) => missing_optional.push(tensor.name.clone()),
            }
        }

        Qwen35WeightValidation {
            prefix: manifest.prefix.clone(),
            missing_required,
            present_required,
            present_optional,
            missing_optional,
        }
    }

    pub fn resolve_manifest(
        &self,
        manifest: &Qwen35WeightManifest,
    ) -> Result<Qwen35ResolvedWeightPlan, String> {
        let validation = self.validate_manifest(manifest);
        if !validation.missing_required.is_empty() {
            return Err(format!(
                "missing {} required Qwen3.5/Qwen3.6 tensors for prefix {}: {}",
                validation.missing_required.len(),
                validation.prefix,
                validation
                    .missing_required
                    .iter()
                    .take(12)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        Ok(Qwen35ResolvedWeightPlan {
            prefix: manifest.prefix.clone(),
            global_tensors: self.resolve_weight_specs(&manifest.global_tensors),
            layers: manifest
                .layers
                .iter()
                .map(|layer| Qwen35ResolvedLayerWeights {
                    layer_index: layer.layer_index,
                    attention: layer.attention,
                    mlp: layer.mlp,
                    tensors: self.resolve_weight_specs(&layer.tensors),
                })
                .collect(),
        })
    }

    pub fn detect_prefix_and_validate(
        &self,
        config: &Qwen35TextConfig,
    ) -> Result<Qwen35WeightValidation, String> {
        let mut best: Option<Qwen35WeightValidation> = None;
        for prefix in PREFIX_CANDIDATES {
            let manifest = config.weight_manifest(*prefix)?;
            let validation = self.validate_manifest(&manifest);
            if validation.missing_required.is_empty() {
                return Ok(validation);
            }
            if best.as_ref().is_none_or(|current| {
                validation.missing_required.len() < current.missing_required.len()
            }) {
                best = Some(validation);
            }
        }
        let best = best.expect("PREFIX_CANDIDATES is non-empty");
        Err(format!(
            "missing {} required Qwen3.5/Qwen3.6 tensors for prefix {}: {}",
            best.missing_required.len(),
            best.prefix,
            best.missing_required
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    pub fn detect_prefix_and_resolve(
        &self,
        config: &Qwen35TextConfig,
    ) -> Result<Qwen35ResolvedWeightPlan, String> {
        let mut best: Option<Qwen35WeightValidation> = None;
        for prefix in PREFIX_CANDIDATES {
            let manifest = config.weight_manifest(*prefix)?;
            let validation = self.validate_manifest(&manifest);
            if validation.missing_required.is_empty() {
                return self.resolve_manifest(&manifest);
            }
            if best.as_ref().is_none_or(|current| {
                validation.missing_required.len() < current.missing_required.len()
            }) {
                best = Some(validation);
            }
        }
        let best = best.expect("PREFIX_CANDIDATES is non-empty");
        Err(format!(
            "missing {} required Qwen3.5/Qwen3.6 tensors for prefix {}: {}",
            best.missing_required.len(),
            best.prefix,
            best.missing_required
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    fn from_safetensors_files(files: impl IntoIterator<Item = PathBuf>) -> Result<Self, String> {
        let mut names = BTreeSet::new();
        for path in files {
            let file = File::open(&path).map_err(|err| format!("open {path:?}: {err}"))?;
            let mmap = unsafe { Mmap::map(&file).map_err(|err| format!("mmap {path:?}: {err}"))? };
            let safetensors =
                SafeTensors::deserialize(&mmap).map_err(|err| format!("parse {path:?}: {err}"))?;
            names.extend(safetensors.names().into_iter().map(|name| name.to_string()));
        }
        Ok(Self { names })
    }

    fn from_safetensors_index(model_dir: &Path, index: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(index).map_err(|err| format!("read {index:?}: {err}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).map_err(|err| format!("parse {index:?}: {err}"))?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{index:?} missing weight_map"))?;
        let shard_files = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .collect::<BTreeSet<_>>();
        for shard in shard_files {
            let path = model_dir.join(shard);
            if !path.exists() {
                return Err(format!("missing safetensors shard {path:?}"));
            }
        }
        Ok(Self {
            names: weight_map.keys().cloned().collect(),
        })
    }

    fn contains_weight_spec(&self, tensor: &Qwen35WeightSpec) -> bool {
        !self.matching_names(tensor).is_empty()
    }

    fn matching_names(&self, tensor: &Qwen35WeightSpec) -> Vec<String> {
        if !tensor.name.contains('*') {
            if tensor.name.ends_with(".lm_head.weight") && self.contains("lm_head.weight") {
                return vec!["lm_head.weight".to_string()];
            }
            return self
                .contains(&tensor.name)
                .then(|| tensor.name.clone())
                .into_iter()
                .collect();
        }
        let mut pieces = tensor.name.splitn(2, '*');
        let prefix = pieces.next().unwrap_or("");
        let suffix = pieces.next().unwrap_or("");
        self.names
            .iter()
            .filter(|name| name.starts_with(prefix) && name.ends_with(suffix))
            .cloned()
            .collect()
    }

    fn resolve_weight_specs(&self, specs: &[Qwen35WeightSpec]) -> Vec<Qwen35ResolvedWeightSpec> {
        specs
            .iter()
            .flat_map(|spec| {
                let matches = self.matching_names(spec);
                if matches.is_empty() {
                    vec![Qwen35ResolvedWeightSpec {
                        role: spec.role.clone(),
                        name: spec.name.clone(),
                        required: spec.required,
                        present: false,
                    }]
                } else {
                    matches
                        .into_iter()
                        .map(|name| Qwen35ResolvedWeightSpec {
                            role: spec.role.clone(),
                            name,
                            required: spec.required,
                            present: true,
                        })
                        .collect()
                }
            })
            .collect()
    }
}

impl Qwen35WeightValidation {
    pub fn is_pass(&self) -> bool {
        self.missing_required.is_empty()
    }
}

impl Qwen35ResolvedWeightPlan {
    pub fn validation(&self) -> Qwen35WeightValidation {
        let mut missing_required = Vec::new();
        let mut present_required = Vec::new();
        let mut present_optional = Vec::new();
        let mut missing_optional = Vec::new();

        for tensor in self
            .global_tensors
            .iter()
            .chain(self.layers.iter().flat_map(|layer| layer.tensors.iter()))
        {
            match (tensor.required, tensor.present) {
                (true, true) => present_required.push(tensor.name.clone()),
                (true, false) => missing_required.push(tensor.name.clone()),
                (false, true) => present_optional.push(tensor.name.clone()),
                (false, false) => missing_optional.push(tensor.name.clone()),
            }
        }

        Qwen35WeightValidation {
            prefix: self.prefix.clone(),
            missing_required,
            present_required,
            present_optional,
            missing_optional,
        }
    }

    pub fn layer_tensor(
        &self,
        layer_index: usize,
        role: &str,
    ) -> Option<&Qwen35ResolvedWeightSpec> {
        self.layers
            .iter()
            .find(|layer| layer.layer_index == layer_index)
            .and_then(|layer| {
                layer
                    .tensors
                    .iter()
                    .find(|tensor| tensor.role == role && tensor.present)
            })
    }

    pub fn global_tensor(&self, role: &str) -> Option<&Qwen35ResolvedWeightSpec> {
        self.global_tensors
            .iter()
            .find(|tensor| tensor.role == role)
    }
}

fn linear_module_name(tensor_name: &str) -> String {
    tensor_name
        .strip_suffix(".weight")
        .unwrap_or(tensor_name)
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::{FerrumError, Result as FerrumResult};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use tempfile::TempDir;

    use super::*;

    fn dense_config() -> Qwen35TextConfig {
        Qwen35TextConfig::from_hf_config_str(
            r#"{
              "model_type": "qwen3_5",
              "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 32,
                "tie_word_embeddings": true
              }
            }"#,
        )
        .unwrap()
    }

    fn moe_config() -> Qwen35TextConfig {
        Qwen35TextConfig::from_hf_config_str(
            r#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8,
                "tie_word_embeddings": false
              }
            }"#,
        )
        .unwrap()
    }

    fn write_safetensors(dir: &Path, names: &[String]) {
        let tensors = names
            .iter()
            .map(|name| {
                let bytes: &'static [u8] =
                    Box::leak(0.0f32.to_le_bytes().to_vec().into_boxed_slice());
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, vec![1], bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            tensors,
            &None::<HashMap<String, String>>,
            &dir.join("model.safetensors"),
        )
        .unwrap();
    }

    struct RecordingLoader {
        tensors: HashMap<String, Vec<f32>>,
        linear_names: Mutex<Vec<String>>,
    }

    impl RecordingLoader {
        fn from_names(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
            Self {
                tensors: names
                    .into_iter()
                    .map(|name| (name.into(), vec![1.0]))
                    .collect(),
                linear_names: Mutex::new(Vec::new()),
            }
        }

        fn linear_names(&self) -> Vec<String> {
            self.linear_names.lock().unwrap().clone()
        }
    }

    impl WeightLoader<CpuBackend> for RecordingLoader {
        fn load_tensor(&self, name: &str) -> FerrumResult<Vec<f32>> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| FerrumError::model(format!("missing tensor {name}")))
        }

        fn load_linear(
            &self,
            name: &str,
        ) -> FerrumResult<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            self.linear_names.lock().unwrap().push(name.to_string());
            Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(
                &[0.0, 0.0],
                1,
                2,
            )))
        }

        fn has_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    #[test]
    fn validates_single_file_safetensors_against_manifest() {
        let config = dense_config();
        let manifest = config.weight_manifest("model").unwrap();
        let names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let tmp = TempDir::new().unwrap();
        write_safetensors(tmp.path(), &names);

        let inventory = Qwen35WeightInventory::from_safetensors_dir(tmp.path()).unwrap();
        let validation = inventory.detect_prefix_and_validate(&config).unwrap();
        assert_eq!(validation.prefix, "model");
        assert!(validation.is_pass());
        assert!(validation
            .missing_optional
            .contains(&"model.lm_head.weight".to_string()));
    }

    #[test]
    fn reports_missing_required_tensor() {
        let config = dense_config();
        let manifest = config.weight_manifest("model").unwrap();
        let mut names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        names.retain(|name| name != "model.layers.3.self_attn.q_proj.weight");
        let inventory = Qwen35WeightInventory::from_names(names);
        let err = inventory
            .detect_prefix_and_validate(&config)
            .expect_err("missing full-attention q_proj should fail");
        assert!(err.contains("self_attn.q_proj.weight"), "{err}");
    }

    #[test]
    fn resolves_manifest_to_executor_tensor_plan() {
        let config = dense_config();
        let manifest = config.weight_manifest("model").unwrap();
        let names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let inventory = Qwen35WeightInventory::from_names(names);

        let plan = inventory.detect_prefix_and_resolve(&config).unwrap();
        let validation = plan.validation();

        assert_eq!(plan.prefix, "model");
        assert!(validation.is_pass());
        assert!(plan
            .global_tensors
            .iter()
            .any(|tensor| tensor.role == "embed_tokens"
                && tensor.name == "model.embed_tokens.weight"
                && tensor.present));
        assert!(plan
            .global_tensors
            .iter()
            .any(|tensor| tensor.role == "lm_head"
                && tensor.name == "model.lm_head.weight"
                && !tensor.present
                && !tensor.required));
        assert_eq!(
            plan.layer_tensor(0, "linear_attn_qkv")
                .map(|tensor| tensor.name.as_str()),
            Some("model.layers.0.linear_attn.in_proj_qkv.weight")
        );
    }

    #[test]
    fn resolves_optional_wildcard_expert_aliases_when_present() {
        let config = moe_config();
        let manifest = config.weight_manifest("model").unwrap();
        let mut names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        names.push("model.layers.0.mlp.experts.0.gate_proj.weight".to_string());
        let inventory = Qwen35WeightInventory::from_names(names);

        let plan = inventory.detect_prefix_and_resolve(&config).unwrap();

        assert_eq!(
            plan.layer_tensor(0, "moe_per_expert_gate_proj")
                .map(|tensor| tensor.name.as_str()),
            Some("model.layers.0.mlp.experts.0.gate_proj.weight")
        );
        assert!(plan
            .validation()
            .present_optional
            .contains(&"model.layers.0.mlp.experts.0.gate_proj.weight".to_string()));
    }

    #[test]
    fn resolves_top_level_lm_head_alias_for_language_model_prefix() {
        let config = moe_config();
        let manifest = config.weight_manifest("model.language_model").unwrap();
        let mut names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        names.retain(|name| name != "model.language_model.lm_head.weight");
        names.push("lm_head.weight".to_string());
        let inventory = Qwen35WeightInventory::from_names(names);

        let plan = inventory.detect_prefix_and_resolve(&config).unwrap();

        assert_eq!(plan.prefix, "model.language_model");
        assert_eq!(
            plan.global_tensor("lm_head")
                .map(|tensor| (tensor.name.as_str(), tensor.present)),
            Some(("lm_head.weight", true))
        );
        assert!(plan.validation().is_pass());
    }

    #[test]
    fn planned_loader_loads_by_role_and_strips_linear_weight_suffix() {
        let config = dense_config();
        let manifest = config.weight_manifest("model").unwrap();
        let names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let inventory = Qwen35WeightInventory::from_names(names.clone());
        let plan = inventory.detect_prefix_and_resolve(&config).unwrap();
        let loader = RecordingLoader::from_names(names);
        let planned = Qwen35WeightPlanLoader::<CpuBackend>::new(&plan, &loader);

        assert_eq!(planned.plan().prefix, "model");
        assert!(planned.has_global_tensor("embed_tokens"));
        assert_eq!(
            planned.load_global_tensor("embed_tokens").unwrap(),
            vec![1.0]
        );
        assert!(planned.has_layer_tensor(0, "linear_attn_qkv"));
        let linear = planned.load_layer_linear(0, "linear_attn_qkv").unwrap();

        assert_eq!(linear.in_features(), 2);
        assert_eq!(
            loader.linear_names(),
            vec!["model.layers.0.linear_attn.in_proj_qkv".to_string()]
        );
        let err = planned
            .load_global_tensor("lm_head")
            .expect_err("tied lm_head is optional and absent");
        assert!(err.to_string().contains("lm_head"), "{err}");
    }

    #[test]
    fn planned_loader_loads_packed_linear_attention_projection_names() {
        let config = dense_config();
        let manifest = config.weight_manifest("model").unwrap();
        let names = manifest
            .global_tensors
            .iter()
            .chain(
                manifest
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|tensor| tensor.required)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let inventory = Qwen35WeightInventory::from_names(names.clone());
        let plan = inventory.detect_prefix_and_resolve(&config).unwrap();
        let loader = RecordingLoader::from_names(names);
        let planned = Qwen35WeightPlanLoader::<CpuBackend>::new(&plan, &loader);

        let qkvz = planned.load_layer_linear_attention_qkvz(0).unwrap();
        let ba = planned.load_layer_linear_attention_ba(0).unwrap();

        assert_eq!(qkvz.in_features(), 2);
        assert_eq!(ba.in_features(), 2);
        assert_eq!(
            loader.linear_names(),
            vec![
                "model.layers.0.linear_attn.in_proj_qkvz".to_string(),
                "model.layers.0.linear_attn.in_proj_ba".to_string(),
            ]
        );
    }

    #[test]
    fn reads_sharded_index_weight_map_without_loading_tensor_data() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), []).unwrap();
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "shard-00001.safetensors",
                    "model.norm.weight": "shard-00001.safetensors"
                }
            })
            .to_string(),
        )
        .unwrap();

        let inventory = Qwen35WeightInventory::from_safetensors_dir(tmp.path()).unwrap();
        assert!(inventory.contains("model.embed_tokens.weight"));
        assert!(inventory.contains("model.norm.weight"));
    }
}
