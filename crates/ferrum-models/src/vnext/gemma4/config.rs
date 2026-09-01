use std::collections::BTreeSet;

use ferrum_interfaces::vnext::CanonicalRational;
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::EXTERNAL_METADATA_ID;

const HF_ARCHITECTURE: &str = "Gemma4UnifiedForConditionalGeneration";
const HF_MODEL_TYPE: &str = "gemma4_unified";
const HF_TEXT_MODEL_TYPE: &str = "gemma4_unified_text";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Gemma4LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Gemma4CompressedTensorsRecipe {
    pub format: String,
    pub quant_method: String,
    pub quantization_status: String,
    pub version: String,
    pub group_size: u64,
    pub num_bits: u64,
    pub symmetric: bool,
    pub dynamic: bool,
    pub strategy: String,
    pub weight_type: String,
    pub targets: Vec<String>,
    pub ignored_modules: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Gemma4SemanticConfig {
    pub hidden_size: u64,
    pub layer_count: u64,
    pub attention_head_count: u64,
    pub local_kv_head_count: u64,
    pub global_kv_head_count: u64,
    pub local_head_dim: u64,
    pub global_head_dim: u64,
    pub intermediate_size: u64,
    pub vocabulary_size: u64,
    pub maximum_sequence_tokens: u64,
    pub sliding_window_tokens: u64,
    pub rms_norm_epsilon: CanonicalRational,
    pub local_rope_theta: CanonicalRational,
    pub global_rope_theta: CanonicalRational,
    pub global_partial_rotary_factor: CanonicalRational,
    pub final_logit_softcap: CanonicalRational,
    pub layer_types: Vec<Gemma4LayerType>,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub quantization: Gemma4CompressedTensorsRecipe,
}

impl Gemma4SemanticConfig {
    pub(super) fn parse(raw: &[u8]) -> Result<Self, String> {
        let value: Value = serde_json::from_slice(raw)
            .map_err(|error| format!("parse Gemma 4 config.json: {error}"))?;
        let root = value
            .as_object()
            .ok_or_else(|| "Gemma 4 config.json root must be an object".to_owned())?;
        validate_outer_identity(root)?;

        let text = required_object(root, "text_config")?;
        validate_outer_text_consistency(root, text)?;
        validate_supported_text_semantics(text)?;

        let rope = required_object(text, "rope_parameters")?;
        let local_rope = required_object(rope, "sliding_attention")?;
        let global_rope = required_object(rope, "full_attention")?;
        require_string_eq(local_rope, "rope_type", "default")?;
        require_string_eq(global_rope, "rope_type", "proportional")?;

        let layer_types = required_array(text, "layer_types")?
            .iter()
            .enumerate()
            .map(|(index, value)| match value.as_str() {
                Some("sliding_attention") => Ok(Gemma4LayerType::SlidingAttention),
                Some("full_attention") => Ok(Gemma4LayerType::FullAttention),
                other => Err(format!(
                    "text_config.layer_types[{index}] must be sliding_attention or full_attention, got {other:?}"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let config = Self {
            hidden_size: required_positive_u64(text, "hidden_size")?,
            layer_count: required_positive_u64(text, "num_hidden_layers")?,
            attention_head_count: required_positive_u64(text, "num_attention_heads")?,
            local_kv_head_count: required_positive_u64(text, "num_key_value_heads")?,
            global_kv_head_count: required_positive_u64(text, "num_global_key_value_heads")?,
            local_head_dim: required_positive_u64(text, "head_dim")?,
            global_head_dim: required_positive_u64(text, "global_head_dim")?,
            intermediate_size: required_positive_u64(text, "intermediate_size")?,
            vocabulary_size: required_positive_u64(text, "vocab_size")?,
            maximum_sequence_tokens: required_positive_u64(text, "max_position_embeddings")?,
            sliding_window_tokens: required_positive_u64(text, "sliding_window")?,
            rms_norm_epsilon: required_positive_rational(text, "rms_norm_eps")?,
            local_rope_theta: required_positive_rational(local_rope, "rope_theta")?,
            global_rope_theta: required_positive_rational(global_rope, "rope_theta")?,
            global_partial_rotary_factor: required_positive_rational(
                global_rope,
                "partial_rotary_factor",
            )?,
            final_logit_softcap: required_positive_rational(text, "final_logit_softcapping")?,
            layer_types,
            tie_word_embeddings: required_bool(text, "tie_word_embeddings")?,
            attention_k_eq_v: required_bool(text, "attention_k_eq_v")?,
            quantization: parse_quantization(root)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate_semantic_source(raw: &[u8]) -> Result<(), String> {
        Self::parse(raw).map(|_| ())
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.layer_count),
            ("num_attention_heads", self.attention_head_count),
            ("num_key_value_heads", self.local_kv_head_count),
            ("num_global_key_value_heads", self.global_kv_head_count),
            ("head_dim", self.local_head_dim),
            ("global_head_dim", self.global_head_dim),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocabulary_size),
            ("max_position_embeddings", self.maximum_sequence_tokens),
            ("sliding_window", self.sliding_window_tokens),
        ] {
            if value == 0 {
                return Err(format!("{field} must be positive"));
            }
        }
        if self.layer_types.len() != usize::try_from(self.layer_count).unwrap_or(usize::MAX) {
            return Err(format!(
                "layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.layer_count
            ));
        }
        if !self
            .layer_types
            .iter()
            .any(|kind| *kind == Gemma4LayerType::SlidingAttention)
            || !self
                .layer_types
                .iter()
                .any(|kind| *kind == Gemma4LayerType::FullAttention)
        {
            return Err(
                "Gemma 4 Unified text requires both sliding and full attention layers".to_owned(),
            );
        }
        for (field, heads) in [
            ("num_key_value_heads", self.local_kv_head_count),
            ("num_global_key_value_heads", self.global_kv_head_count),
        ] {
            if heads > self.attention_head_count || !self.attention_head_count.is_multiple_of(heads)
            {
                return Err(format!("num_attention_heads must be divisible by {field}"));
            }
        }
        if self.sliding_window_tokens > self.maximum_sequence_tokens {
            return Err("sliding_window exceeds max_position_embeddings".to_owned());
        }
        if !self.tie_word_embeddings {
            return Err("Gemma 4 Unified text contract requires tied word embeddings".to_owned());
        }
        if !self.attention_k_eq_v {
            return Err("Gemma 4 Unified full attention requires attention_k_eq_v=true".to_owned());
        }
        if self.rms_norm_epsilon.numerator() <= 0
            || self.local_rope_theta.numerator() <= 0
            || self.global_rope_theta.numerator() <= 0
            || self.global_partial_rotary_factor.numerator() <= 0
            || self.final_logit_softcap.numerator() <= 0
        {
            return Err("Gemma 4 rational semantic values must be positive".to_owned());
        }
        if self.global_partial_rotary_factor.numerator() as u64
            > self.global_partial_rotary_factor.denominator()
        {
            return Err("full-attention partial_rotary_factor must be <= 1".to_owned());
        }
        let rope_numerator = self
            .global_head_dim
            .checked_mul(self.global_partial_rotary_factor.numerator() as u64)
            .ok_or_else(|| "full-attention rotary width overflows".to_owned())?;
        if !rope_numerator.is_multiple_of(self.global_partial_rotary_factor.denominator()) {
            return Err("full-attention partial rotary width must be integral".to_owned());
        }
        let rope_dim = rope_numerator / self.global_partial_rotary_factor.denominator();
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
            return Err("full-attention partial rotary width must be positive and even".to_owned());
        }
        self.quantization.validate()?;
        Ok(())
    }

    pub(super) fn query_features(&self, layer_type: Gemma4LayerType) -> Result<u64, String> {
        self.attention_head_count
            .checked_mul(self.head_dim(layer_type))
            .ok_or_else(|| "query projection width overflows u64".to_owned())
    }

    pub(super) fn kv_features(&self, layer_type: Gemma4LayerType) -> Result<u64, String> {
        self.kv_head_count(layer_type)
            .checked_mul(self.head_dim(layer_type))
            .ok_or_else(|| "key/value projection width overflows u64".to_owned())
    }

    pub(super) const fn head_dim(&self, layer_type: Gemma4LayerType) -> u64 {
        match layer_type {
            Gemma4LayerType::SlidingAttention => self.local_head_dim,
            Gemma4LayerType::FullAttention => self.global_head_dim,
        }
    }

    pub(super) const fn kv_head_count(&self, layer_type: Gemma4LayerType) -> u64 {
        match layer_type {
            Gemma4LayerType::SlidingAttention => self.local_kv_head_count,
            Gemma4LayerType::FullAttention => self.global_kv_head_count,
        }
    }

    pub(super) fn rope_dim(&self, layer_type: Gemma4LayerType) -> u64 {
        match layer_type {
            Gemma4LayerType::SlidingAttention => self.local_head_dim,
            Gemma4LayerType::FullAttention => {
                self.global_head_dim * self.global_partial_rotary_factor.numerator() as u64
                    / self.global_partial_rotary_factor.denominator()
            }
        }
    }

    pub(super) const fn rope_frequency_denominator(&self, layer_type: Gemma4LayerType) -> u64 {
        self.head_dim(layer_type)
    }

    pub(super) const fn rope_theta(&self, layer_type: Gemma4LayerType) -> CanonicalRational {
        match layer_type {
            Gemma4LayerType::SlidingAttention => self.local_rope_theta,
            Gemma4LayerType::FullAttention => self.global_rope_theta,
        }
    }

    pub(super) const fn sliding_window(&self, layer_type: Gemma4LayerType) -> Option<u64> {
        match layer_type {
            Gemma4LayerType::SlidingAttention => Some(self.sliding_window_tokens),
            Gemma4LayerType::FullAttention => None,
        }
    }

    pub(super) fn embedding_scale(&self) -> Result<CanonicalRational, String> {
        let rounded = bf16::from_f32((self.hidden_size as f32).sqrt()).to_f32();
        CanonicalRational::from_decimal_str(&rounded.to_string()).map_err(|error| error.to_string())
    }

    pub(super) const fn external_metadata_id(&self) -> &'static str {
        EXTERNAL_METADATA_ID
    }
}

impl Gemma4CompressedTensorsRecipe {
    fn validate(&self) -> Result<(), String> {
        if self.format != "pack-quantized"
            || self.quant_method != "compressed-tensors"
            || self.quantization_status != "compressed"
            || self.version.trim().is_empty()
            || self.group_size != 32
            || self.num_bits != 4
            || !self.symmetric
            || self.dynamic
            || self.strategy != "group"
            || self.weight_type != "int"
            || self.targets != ["Linear"]
        {
            return Err(
                "Gemma 4 requires compressed-tensors pack-quantized symmetric static INT4 Linear weights with group_size=32"
                    .to_owned(),
            );
        }
        for required in [
            "lm_head",
            "model.embed_vision.patch_dense",
            "model.embed_vision.multimodal_embedder.embedding_projection",
            "model.embed_audio.embedding_projection",
        ] {
            if !self.ignored_modules.contains(required) {
                return Err(format!(
                    "compressed-tensors ignore list is missing required dense exclusion {required:?}"
                ));
            }
        }
        Ok(())
    }
}

fn validate_outer_identity(root: &Map<String, Value>) -> Result<(), String> {
    let architectures = required_array(root, "architectures")?;
    if architectures.len() != 1 || architectures[0].as_str() != Some(HF_ARCHITECTURE) {
        return Err(format!(
            "architectures must be exactly [\"{HF_ARCHITECTURE}\"]"
        ));
    }
    require_string_eq(root, "model_type", HF_MODEL_TYPE)?;
    require_string_eq(root, "dtype", "bfloat16")?;
    require_string_eq(root, "hidden_act", "gelu_pytorch_tanh")?;
    if !required_bool(root, "tie_word_embeddings")? {
        return Err("outer tie_word_embeddings must be true".to_owned());
    }
    for field in [
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ] {
        require_null(root, field)?;
    }
    Ok(())
}

fn validate_outer_text_consistency(
    root: &Map<String, Value>,
    text: &Map<String, Value>,
) -> Result<(), String> {
    for field in ["hidden_size", "intermediate_size"] {
        if root.get(field) != text.get(field) {
            return Err(format!("outer {field} differs from text_config.{field}"));
        }
    }
    let outer_activation = required_string(root, "hidden_act")?;
    let text_activation = required_string(text, "hidden_activation")?;
    if outer_activation != text_activation {
        return Err("outer hidden_act differs from text_config.hidden_activation".to_owned());
    }
    if root.get("tie_word_embeddings") != text.get("tie_word_embeddings") {
        return Err("outer tie_word_embeddings differs from text_config".to_owned());
    }
    Ok(())
}

fn validate_supported_text_semantics(text: &Map<String, Value>) -> Result<(), String> {
    require_string_eq(text, "model_type", HF_TEXT_MODEL_TYPE)?;
    require_string_eq(text, "dtype", "bfloat16")?;
    require_string_eq(text, "hidden_activation", "gelu_pytorch_tanh")?;
    require_string_eq(text, "use_bidirectional_attention", "vision")?;
    for field in ["attention_bias", "enable_moe_block", "use_double_wide_mlp"] {
        if required_bool(text, field)? {
            return Err(format!("unsupported text_config.{field}=true"));
        }
    }
    if !required_bool(text, "use_cache")? {
        return Err("text_config.use_cache must be true".to_owned());
    }
    for field in [
        "hidden_size_per_layer_input",
        "vocab_size_per_layer_input",
        "num_kv_shared_layers",
    ] {
        if required_u64(text, field)? != 0 {
            return Err(format!("unsupported non-zero text_config.{field}"));
        }
    }
    for field in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
        require_null(text, field)?;
    }
    let dropout = required_rational(text, "attention_dropout")?;
    if dropout.numerator() != 0 {
        return Err("text_config.attention_dropout must be zero".to_owned());
    }
    Ok(())
}

fn parse_quantization(root: &Map<String, Value>) -> Result<Gemma4CompressedTensorsRecipe, String> {
    let quantization = required_object(root, "quantization_config")?;
    require_string_eq(quantization, "format", "pack-quantized")?;
    require_string_eq(quantization, "quant_method", "compressed-tensors")?;
    require_string_eq(quantization, "quantization_status", "compressed")?;
    require_null(quantization, "kv_cache_scheme")?;
    require_empty_object(quantization, "sparsity_config")?;
    require_empty_object(quantization, "transform_config")?;

    let groups = required_object(quantization, "config_groups")?;
    if groups.len() != 1 {
        return Err("quantization_config.config_groups must contain exactly group_0".to_owned());
    }
    let group = required_object(groups, "group_0")?;
    require_string_eq(group, "format", "pack-quantized")?;
    require_null(group, "input_activations")?;
    require_null(group, "output_activations")?;
    let targets = required_array(group, "targets")?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("quantization target {index} must be a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = required_object(group, "weights")?;
    require_null(weights, "actorder")?;
    require_null(weights, "block_structure")?;
    require_null(weights, "scale_dtype")?;
    require_null(weights, "zp_dtype")?;
    require_empty_object(weights, "observer_kwargs")?;
    require_string_eq(weights, "observer", "memoryless_minmax")?;

    let ignored_modules = required_array(quantization, "ignore")?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("quantization ignore entry {index} must be a string"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let recipe = Gemma4CompressedTensorsRecipe {
        format: required_string(quantization, "format")?.to_owned(),
        quant_method: required_string(quantization, "quant_method")?.to_owned(),
        quantization_status: required_string(quantization, "quantization_status")?.to_owned(),
        version: required_string(quantization, "version")?.to_owned(),
        group_size: required_positive_u64(weights, "group_size")?,
        num_bits: required_positive_u64(weights, "num_bits")?,
        symmetric: required_bool(weights, "symmetric")?,
        dynamic: required_bool(weights, "dynamic")?,
        strategy: required_string(weights, "strategy")?.to_owned(),
        weight_type: required_string(weights, "type")?.to_owned(),
        targets,
        ignored_modules,
    };
    recipe.validate()?;
    Ok(recipe)
}

fn required_object<'a>(
    root: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Map<String, Value>, String> {
    root.get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{field} must be an object"))
}

fn required_array<'a>(root: &'a Map<String, Value>, field: &str) -> Result<&'a [Value], String> {
    root.get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{field} must be an array"))
}

fn required_string<'a>(root: &'a Map<String, Value>, field: &str) -> Result<&'a str, String> {
    root.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))
}

fn require_string_eq(root: &Map<String, Value>, field: &str, expected: &str) -> Result<(), String> {
    let actual = required_string(root, field)?;
    if actual != expected {
        return Err(format!("{field} must be {expected:?}, got {actual:?}"));
    }
    Ok(())
}

fn required_u64(root: &Map<String, Value>, field: &str) -> Result<u64, String> {
    root.get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be a non-negative integer"))
}

fn required_positive_u64(root: &Map<String, Value>, field: &str) -> Result<u64, String> {
    let value = required_u64(root, field)?;
    if value == 0 {
        return Err(format!("{field} must be positive"));
    }
    Ok(value)
}

fn required_bool(root: &Map<String, Value>, field: &str) -> Result<bool, String> {
    root.get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{field} must be a boolean"))
}

fn required_rational(root: &Map<String, Value>, field: &str) -> Result<CanonicalRational, String> {
    let number = root
        .get(field)
        .and_then(Value::as_number)
        .ok_or_else(|| format!("{field} must be numeric"))?;
    CanonicalRational::from_decimal_str(&number.to_string()).map_err(|error| error.to_string())
}

fn required_positive_rational(
    root: &Map<String, Value>,
    field: &str,
) -> Result<CanonicalRational, String> {
    let value = required_rational(root, field)?;
    if value.numerator() <= 0 {
        return Err(format!("{field} must be positive"));
    }
    Ok(value)
}

fn require_null(root: &Map<String, Value>, field: &str) -> Result<(), String> {
    match root.get(field) {
        Some(Value::Null) => Ok(()),
        other => Err(format!("{field} must be null, got {other:?}")),
    }
}

fn require_empty_object(root: &Map<String, Value>, field: &str) -> Result<(), String> {
    let object = required_object(root, field)?;
    if !object.is_empty() {
        return Err(format!("{field} must be an empty object"));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn tiny_semantic_config() -> Gemma4SemanticConfig {
    Gemma4SemanticConfig::parse(&serde_json::to_vec(&tests::fixture_value()).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn fixture_value() -> Value {
        serde_json::from_str(
            r#"{
              "architectures":["Gemma4UnifiedForConditionalGeneration"],
              "dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh",
              "hidden_size":64,"intermediate_size":128,"model_type":"gemma4_unified",
              "moe_intermediate_size":null,"num_experts":null,"num_experts_per_tok":null,
              "tie_word_embeddings":true,
              "quantization_config":{
                "config_groups":{"group_0":{
                  "format":"pack-quantized","input_activations":null,
                  "output_activations":null,"targets":["Linear"],
                  "weights":{"actorder":null,"block_structure":null,"dynamic":false,
                    "group_size":32,"num_bits":4,"observer":"memoryless_minmax",
                    "observer_kwargs":{},"scale_dtype":null,"strategy":"group",
                    "symmetric":true,"type":"int","zp_dtype":null}}},
                "format":"pack-quantized",
                "ignore":["lm_head","model.embed_vision.patch_dense",
                  "model.embed_vision.multimodal_embedder.embedding_projection",
                  "model.embed_audio.embedding_projection"],
                "kv_cache_scheme":null,"quant_method":"compressed-tensors",
                "quantization_status":"compressed","sparsity_config":{},
                "transform_config":{},"version":"0.17.1.test"},
              "text_config":{
                "attention_bias":false,"attention_dropout":0.0,"attention_k_eq_v":true,
                "dtype":"bfloat16","enable_moe_block":false,"final_logit_softcapping":30.0,
                "global_head_dim":32,"head_dim":16,"hidden_activation":"gelu_pytorch_tanh",
                "hidden_size":64,"hidden_size_per_layer_input":0,"intermediate_size":128,
                "layer_types":["sliding_attention","full_attention"],
                "max_position_embeddings":128,"model_type":"gemma4_unified_text",
                "moe_intermediate_size":null,"num_attention_heads":4,"num_experts":null,
                "num_global_key_value_heads":2,"num_hidden_layers":2,"num_key_value_heads":4,
                "num_kv_shared_layers":0,"rms_norm_eps":0.000001,
                "rope_parameters":{
                  "full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,
                    "rope_type":"proportional"},
                  "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},
                "sliding_window":32,"tie_word_embeddings":true,"top_k_experts":null,
                "use_bidirectional_attention":"vision","use_cache":true,
                "use_double_wide_mlp":false,"vocab_size":256,"vocab_size_per_layer_input":0}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_typed_hybrid_attention_and_proportional_rope() {
        let config = tiny_semantic_config();
        assert_eq!(config.layer_types.len(), 2);
        assert_eq!(
            config
                .query_features(Gemma4LayerType::SlidingAttention)
                .unwrap(),
            64
        );
        assert_eq!(
            config
                .kv_features(Gemma4LayerType::SlidingAttention)
                .unwrap(),
            64
        );
        assert_eq!(
            config
                .query_features(Gemma4LayerType::FullAttention)
                .unwrap(),
            128
        );
        assert_eq!(
            config.kv_features(Gemma4LayerType::FullAttention).unwrap(),
            64
        );
        assert_eq!(config.rope_dim(Gemma4LayerType::FullAttention), 8);
        assert_eq!(
            config.rope_frequency_denominator(Gemma4LayerType::FullAttention),
            32
        );
        assert_eq!(
            config.embedding_scale(),
            CanonicalRational::new(8, 1).map_err(|e| e.to_string())
        );
    }

    #[test]
    fn wrong_compressed_tensors_recipe_fails_closed_before_allocation() {
        let mut value = fixture_value();
        value["quantization_config"]["config_groups"]["group_0"]["weights"]["symmetric"] =
            Value::Bool(false);
        let error = Gemma4SemanticConfig::parse(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.contains("symmetric static INT4"), "{error}");

        let mut value = fixture_value();
        value["text_config"]["hidden_size_per_layer_input"] = Value::from(32);
        let error = Gemma4SemanticConfig::parse(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.contains("hidden_size_per_layer_input"), "{error}");
    }
}
