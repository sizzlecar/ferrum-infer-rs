use std::collections::BTreeSet;

use ferrum_interfaces::vnext::CanonicalRational;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::EXTERNAL_METADATA_ID;

const HF_ARCHITECTURE: &str = "GptOssForCausalLM";
const HF_MODEL_TYPE: &str = "gpt_oss";
const MXFP4_QUANT_METHOD: &str = "mxfp4";
const MXFP4_BITS_PER_WEIGHT: u8 = 4;
const MXFP4_GROUP_SIZE: u32 = 32;
const MXFP4_PACKED_BYTES_PER_GROUP: u32 = 16;
const MXFP4_SCALE_EXPONENT_BIAS: i16 = 127;
const GPT_OSS_VOCABULARY_SIZE: u64 = 201_088;
const GPT_OSS_MAXIMUM_SEQUENCE_TOKENS: u64 = 131_072;
const GPT_OSS_EXPERT_COUNT: u64 = 32;
const GPT_OSS_EXPERTS_PER_TOKEN: u64 = 4;
const GPT_OSS_SLIDING_WINDOW: u64 = 128;
const MXFP4_DENSE_EXCLUSIONS: [&str; 4] = [
    "model.layers.*.self_attn",
    "model.layers.*.mlp.router",
    "model.embed_tokens",
    "lm_head",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GptOssLayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GptOssRopeType {
    Yarn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GptOssRopeScaling {
    pub rope_type: GptOssRopeType,
    pub factor: CanonicalRational,
    pub original_max_position_embeddings: u64,
    pub beta_fast: CanonicalRational,
    pub beta_slow: CanonicalRational,
    pub truncate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GptOssMxfp4Config {
    pub bits_per_weight: u8,
    pub group_size: u32,
    pub packed_bytes_per_group: u32,
    pub scale_exponent_bias: i16,
    pub modules_to_not_convert: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GptOssSemanticConfig {
    pub hidden_size: u64,
    pub layer_count: u64,
    pub attention_head_count: u64,
    pub kv_head_count: u64,
    pub head_dim: u64,
    pub vocabulary_size: u64,
    pub maximum_sequence_tokens: u64,
    pub intermediate_size: u64,
    pub expert_count: u64,
    pub experts_per_token: u64,
    pub layer_types: Vec<GptOssLayerType>,
    pub sliding_window: u64,
    pub rms_norm_epsilon: CanonicalRational,
    pub rope_theta: CanonicalRational,
    pub rope_scaling: GptOssRopeScaling,
    pub swiglu_limit: CanonicalRational,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
}

impl GptOssSemanticConfig {
    pub(super) fn parse(raw: &[u8]) -> Result<(Self, GptOssMxfp4Config), String> {
        let value: Value = serde_json::from_slice(raw)
            .map_err(|error| format!("parse GPT-OSS config.json: {error}"))?;
        let root = value
            .as_object()
            .ok_or_else(|| "GPT-OSS config.json root must be an object".to_owned())?;

        validate_identity(root)?;
        validate_fixed_runtime_semantics(root)?;

        let layer_types = parse_layer_types(root)?;
        let rope_scaling = parse_rope_scaling(root)?;
        let experts_per_token = required_positive_u64(root, "num_experts_per_tok")?;
        let alias_experts_per_token = required_positive_u64(root, "experts_per_token")?;
        if alias_experts_per_token != experts_per_token {
            return Err(
                "experts_per_token and num_experts_per_tok must describe the same top-K".to_owned(),
            );
        }

        let semantic = Self {
            hidden_size: required_positive_u64(root, "hidden_size")?,
            layer_count: required_positive_u64(root, "num_hidden_layers")?,
            attention_head_count: required_positive_u64(root, "num_attention_heads")?,
            kv_head_count: required_positive_u64(root, "num_key_value_heads")?,
            head_dim: required_positive_u64(root, "head_dim")?,
            vocabulary_size: required_positive_u64(root, "vocab_size")?,
            maximum_sequence_tokens: required_positive_u64(root, "max_position_embeddings")?,
            intermediate_size: required_positive_u64(root, "intermediate_size")?,
            expert_count: required_positive_u64(root, "num_local_experts")?,
            experts_per_token,
            layer_types,
            sliding_window: required_positive_u64(root, "sliding_window")?,
            rms_norm_epsilon: required_rational(root, "rms_norm_eps")?,
            rope_theta: required_rational(root, "rope_theta")?,
            rope_scaling,
            swiglu_limit: required_rational(root, "swiglu_limit")?,
            tie_word_embeddings: required_bool(root, "tie_word_embeddings")?,
            attention_bias: required_bool(root, "attention_bias")?,
        };
        semantic.validate()?;

        let mxfp4 = parse_mxfp4(root)?;
        mxfp4.validate()?;
        Ok((semantic, mxfp4))
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.layer_count),
            ("num_attention_heads", self.attention_head_count),
            ("num_key_value_heads", self.kv_head_count),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocabulary_size),
            ("max_position_embeddings", self.maximum_sequence_tokens),
            ("intermediate_size", self.intermediate_size),
            ("num_local_experts", self.expert_count),
            ("num_experts_per_tok", self.experts_per_token),
            ("sliding_window", self.sliding_window),
        ] {
            if value == 0 {
                return Err(format!("{field} must be a positive integer"));
            }
        }

        if self.kv_head_count > self.attention_head_count
            || !self.attention_head_count.is_multiple_of(self.kv_head_count)
        {
            return Err("num_attention_heads must be divisible by num_key_value_heads".to_owned());
        }
        if !matches!(self.head_dim, 32 | 64) {
            return Err("head_dim must be 32 or 64 for the supported GPT-OSS providers".to_owned());
        }
        self.query_features()?;
        self.kv_features()?;

        if self.experts_per_token > self.expert_count {
            return Err("num_experts_per_tok must not exceed num_local_experts".to_owned());
        }
        if self.expert_count != GPT_OSS_EXPERT_COUNT
            || self.experts_per_token != GPT_OSS_EXPERTS_PER_TOKEN
        {
            return Err("GPT-OSS requires 32 local experts with top-4 routing".to_owned());
        }
        if !self.hidden_size.is_multiple_of(u64::from(MXFP4_GROUP_SIZE))
            || !self
                .intermediate_size
                .is_multiple_of(u64::from(MXFP4_GROUP_SIZE))
        {
            return Err(
                "hidden_size and intermediate_size must be divisible by the MXFP4 group size 32"
                    .to_owned(),
            );
        }

        let expected_layers = usize::try_from(self.layer_count)
            .map_err(|_| "num_hidden_layers exceeds host address space".to_owned())?;
        if self.layer_types.len() != expected_layers {
            return Err(format!(
                "layer_types length {} differs from num_hidden_layers {}",
                self.layer_types.len(),
                self.layer_count
            ));
        }
        if !self.layer_count.is_multiple_of(2) {
            return Err(
                "num_hidden_layers must form complete sliding/full attention pairs".to_owned(),
            );
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if index.is_multiple_of(2) {
                GptOssLayerType::SlidingAttention
            } else {
                GptOssLayerType::FullAttention
            };
            if *layer_type != expected {
                return Err(format!(
                    "layer_types[{index}] must be {expected:?}, got {layer_type:?}"
                ));
            }
        }

        if self.vocabulary_size != GPT_OSS_VOCABULARY_SIZE {
            return Err(format!(
                "vocab_size must be {GPT_OSS_VOCABULARY_SIZE} for the locked GPT-OSS tokenizer"
            ));
        }
        if self.maximum_sequence_tokens != GPT_OSS_MAXIMUM_SEQUENCE_TOKENS {
            return Err(format!(
                "max_position_embeddings must be {GPT_OSS_MAXIMUM_SEQUENCE_TOKENS}"
            ));
        }
        if self.sliding_window != GPT_OSS_SLIDING_WINDOW {
            return Err(format!("sliding_window must be {GPT_OSS_SLIDING_WINDOW}"));
        }
        if !self.attention_bias {
            return Err("attention_bias must be true for GPT-OSS Q/K/V/O projections".to_owned());
        }
        require_exact_rational(&self.rms_norm_epsilon, 1, 100_000, "rms_norm_eps")?;
        require_exact_rational(&self.rope_theta, 150_000, 1, "rope_theta")?;
        require_exact_rational(&self.swiglu_limit, 7, 1, "swiglu_limit")?;
        self.rope_scaling.validate(self.maximum_sequence_tokens)?;
        Ok(())
    }

    pub(super) fn query_features(&self) -> Result<u64, String> {
        self.attention_head_count
            .checked_mul(self.head_dim)
            .ok_or_else(|| "query projection width overflows u64".to_owned())
    }

    pub(super) fn kv_features(&self) -> Result<u64, String> {
        self.kv_head_count
            .checked_mul(self.head_dim)
            .ok_or_else(|| "key/value projection width overflows u64".to_owned())
    }

    pub(super) const fn attention_uses_sinks(&self) -> bool {
        true
    }

    pub(super) const fn router_has_bias(&self) -> bool {
        true
    }

    pub(super) const fn experts_have_bias(&self) -> bool {
        true
    }

    pub(super) fn external_metadata_id(&self) -> &'static str {
        EXTERNAL_METADATA_ID
    }
}

impl GptOssRopeScaling {
    fn validate(&self, maximum_sequence_tokens: u64) -> Result<(), String> {
        if self.rope_type != GptOssRopeType::Yarn {
            return Err("rope_scaling.rope_type must be \"yarn\"".to_owned());
        }
        require_exact_rational(&self.factor, 32, 1, "rope_scaling.factor")?;
        require_exact_rational(&self.beta_fast, 32, 1, "rope_scaling.beta_fast")?;
        require_exact_rational(&self.beta_slow, 1, 1, "rope_scaling.beta_slow")?;
        if self.original_max_position_embeddings != 4096 {
            return Err("rope_scaling.original_max_position_embeddings must be 4096".to_owned());
        }
        if self.truncate {
            return Err("rope_scaling.truncate must be false".to_owned());
        }
        if self.original_max_position_embeddings >= maximum_sequence_tokens {
            return Err(
                "rope_scaling original context must be smaller than max_position_embeddings"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

impl GptOssMxfp4Config {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.bits_per_weight != MXFP4_BITS_PER_WEIGHT
            || self.group_size != MXFP4_GROUP_SIZE
            || self.packed_bytes_per_group != MXFP4_PACKED_BYTES_PER_GROUP
            || self.scale_exponent_bias != MXFP4_SCALE_EXPONENT_BIAS
        {
            return Err(
                "native MXFP4 requires E2M1 4-bit values, 32-value/16-byte groups, and E8M0 exponent bias 127"
                    .to_owned(),
            );
        }
        let expected = expected_mxfp4_exclusions();
        if self.modules_to_not_convert != expected {
            return Err(format!(
                "quantization_config.modules_to_not_convert must equal {expected:?}"
            ));
        }
        Ok(())
    }

    pub(super) const fn quant_method(&self) -> &'static str {
        MXFP4_QUANT_METHOD
    }
}

fn validate_identity(root: &Map<String, Value>) -> Result<(), String> {
    let architectures = root
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| "architectures must be an array".to_owned())?;
    if architectures.as_slice() != [Value::String(HF_ARCHITECTURE.to_owned())] {
        return Err(format!(
            "architectures must contain only {HF_ARCHITECTURE:?}"
        ));
    }
    if required_string(root, "model_type")? != HF_MODEL_TYPE {
        return Err(format!("model_type must be {HF_MODEL_TYPE:?}"));
    }
    Ok(())
}

fn validate_fixed_runtime_semantics(root: &Map<String, Value>) -> Result<(), String> {
    if required_string(root, "hidden_act")? != "silu" {
        return Err("hidden_act must be \"silu\" for the GPT-OSS clamped SwiGLU".to_owned());
    }
    let attention_dropout = required_rational(root, "attention_dropout")?;
    require_exact_rational(&attention_dropout, 0, 1, "attention_dropout")?;
    if !required_bool(root, "use_cache")? {
        return Err("use_cache must be true".to_owned());
    }
    if required_bool(root, "output_router_logits")? {
        return Err("output_router_logits must be false for inference".to_owned());
    }
    if let Some(torch_dtype) = root.get("torch_dtype").filter(|value| !value.is_null()) {
        if torch_dtype.as_str() != Some("bfloat16") {
            return Err("torch_dtype, when present, must be \"bfloat16\"".to_owned());
        }
    }
    Ok(())
}

fn parse_layer_types(root: &Map<String, Value>) -> Result<Vec<GptOssLayerType>, String> {
    let values = root
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| "layer_types must be an array".to_owned())?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value.as_str() {
            Some("sliding_attention") => Ok(GptOssLayerType::SlidingAttention),
            Some("full_attention") => Ok(GptOssLayerType::FullAttention),
            _ => Err(format!(
                "layer_types[{index}] must be \"sliding_attention\" or \"full_attention\""
            )),
        })
        .collect()
}

fn parse_rope_scaling(root: &Map<String, Value>) -> Result<GptOssRopeScaling, String> {
    let rope = root
        .get("rope_scaling")
        .and_then(Value::as_object)
        .ok_or_else(|| "rope_scaling must be an object".to_owned())?;
    reject_unknown_fields(
        rope,
        "rope_scaling",
        &[
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "truncate",
        ],
    )?;
    let rope_type = match required_string(rope, "rope_type")? {
        "yarn" => GptOssRopeType::Yarn,
        other => {
            return Err(format!(
                "rope_scaling.rope_type must be \"yarn\", got {other:?}"
            ))
        }
    };
    let parsed = GptOssRopeScaling {
        rope_type,
        factor: required_rational(rope, "factor")?,
        original_max_position_embeddings: required_positive_u64(
            rope,
            "original_max_position_embeddings",
        )?,
        beta_fast: required_rational(rope, "beta_fast")?,
        beta_slow: required_rational(rope, "beta_slow")?,
        truncate: required_bool(rope, "truncate")?,
    };
    let initial_context_length = required_positive_u64(root, "initial_context_length")?;
    if initial_context_length != parsed.original_max_position_embeddings {
        return Err(
            "initial_context_length must equal rope_scaling.original_max_position_embeddings"
                .to_owned(),
        );
    }
    Ok(parsed)
}

fn parse_mxfp4(root: &Map<String, Value>) -> Result<GptOssMxfp4Config, String> {
    let quantization = root
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| "quantization_config must be an object".to_owned())?;
    reject_unknown_fields(
        quantization,
        "quantization_config",
        &["quant_method", "modules_to_not_convert"],
    )?;
    if required_string(quantization, "quant_method")? != MXFP4_QUANT_METHOD {
        return Err("quantization_config.quant_method must be \"mxfp4\"".to_owned());
    }
    let modules = quantization
        .get("modules_to_not_convert")
        .and_then(Value::as_array)
        .ok_or_else(|| "quantization_config.modules_to_not_convert must be an array".to_owned())?;
    let mut modules_to_not_convert = BTreeSet::new();
    for (index, module) in modules.iter().enumerate() {
        let module = module
            .as_str()
            .filter(|module| !module.is_empty())
            .ok_or_else(|| {
                format!(
                "quantization_config.modules_to_not_convert[{index}] must be a non-empty string"
            )
            })?;
        if !modules_to_not_convert.insert(module.to_owned()) {
            return Err(format!(
                "quantization_config.modules_to_not_convert contains duplicate module {module:?}"
            ));
        }
    }
    Ok(GptOssMxfp4Config {
        bits_per_weight: MXFP4_BITS_PER_WEIGHT,
        group_size: MXFP4_GROUP_SIZE,
        packed_bytes_per_group: MXFP4_PACKED_BYTES_PER_GROUP,
        scale_exponent_bias: MXFP4_SCALE_EXPONENT_BIAS,
        modules_to_not_convert,
    })
}

fn expected_mxfp4_exclusions() -> BTreeSet<String> {
    MXFP4_DENSE_EXCLUSIONS
        .into_iter()
        .map(str::to_owned)
        .collect()
}

fn reject_unknown_fields(
    root: &Map<String, Value>,
    context: &str,
    allowed: &[&str],
) -> Result<(), String> {
    if let Some(field) = root.keys().find(|field| {
        !allowed
            .iter()
            .any(|allowed_field| field.as_str() == *allowed_field)
    }) {
        return Err(format!("unsupported {context} field {field:?}"));
    }
    Ok(())
}

fn required_positive_u64(root: &Map<String, Value>, field: &str) -> Result<u64, String> {
    root.get(field)
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{field} must be a positive integer"))
}

fn required_bool(root: &Map<String, Value>, field: &str) -> Result<bool, String> {
    root.get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{field} must be a boolean"))
}

fn required_string<'a>(root: &'a Map<String, Value>, field: &str) -> Result<&'a str, String> {
    root.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))
}

fn required_rational(root: &Map<String, Value>, field: &str) -> Result<CanonicalRational, String> {
    let number = root
        .get(field)
        .and_then(Value::as_number)
        .ok_or_else(|| format!("{field} must be a JSON number"))?;
    CanonicalRational::from_decimal_str(&number.to_string())
        .map_err(|error| format!("{field}: {error}"))
}

fn require_exact_rational(
    value: &CanonicalRational,
    numerator: i64,
    denominator: u64,
    field: &str,
) -> Result<(), String> {
    if value.numerator() != numerator || value.denominator() != denominator {
        return Err(format!(
            "{field} must equal {numerator}/{denominator}, got {}/{}",
            value.numerator(),
            value.denominator()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_types(layer_count: usize) -> Vec<Value> {
        (0..layer_count)
            .map(|index| {
                Value::String(
                    if index.is_multiple_of(2) {
                        "sliding_attention"
                    } else {
                        "full_attention"
                    }
                    .to_owned(),
                )
            })
            .collect()
    }

    fn reference_config(
        hidden_size: u64,
        layer_count: usize,
        attention_heads: u64,
        kv_heads: u64,
        head_dim: u64,
        intermediate_size: u64,
        tie_word_embeddings: bool,
    ) -> Value {
        serde_json::json!({
            "architectures": ["GptOssForCausalLM"],
            "attention_bias": true,
            "attention_dropout": 0.0,
            "experts_per_token": 4,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "hidden_size": hidden_size,
            "initial_context_length": 4096,
            "intermediate_size": intermediate_size,
            "layer_types": layer_types(layer_count),
            "max_position_embeddings": 131072,
            "model_type": "gpt_oss",
            "num_attention_heads": attention_heads,
            "num_experts_per_tok": 4,
            "num_hidden_layers": layer_count,
            "num_key_value_heads": kv_heads,
            "num_local_experts": 32,
            "output_router_logits": false,
            "quantization_config": {
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.layers.*.mlp.router",
                    "model.embed_tokens",
                    "lm_head"
                ],
                "quant_method": "mxfp4"
            },
            "rms_norm_eps": 0.00001,
            "rope_scaling": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "truncate": false
            },
            "rope_theta": 150000,
            "sliding_window": 128,
            "swiglu_limit": 7.0,
            "tie_word_embeddings": tie_word_embeddings,
            "use_cache": true,
            "vocab_size": 201088
        })
    }

    fn official_20b_config() -> Value {
        reference_config(2880, 24, 64, 8, 64, 2880, false)
    }

    fn tiny_canary_config() -> Value {
        let mut config = reference_config(32, 2, 2, 1, 32, 64, true);
        config["torch_dtype"] = Value::String("bfloat16".to_owned());
        config
    }

    #[test]
    fn official_20b_semantics_and_native_mxfp4_recipe_are_exact() {
        let (semantic, mxfp4) =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&official_20b_config()).unwrap())
                .unwrap();

        assert_eq!(semantic.hidden_size, 2880);
        assert_eq!(semantic.layer_count, 24);
        assert_eq!(semantic.query_features().unwrap(), 4096);
        assert_eq!(semantic.kv_features().unwrap(), 512);
        assert_eq!(semantic.intermediate_size, 2880);
        assert_eq!(semantic.layer_types.len(), 24);
        assert!(semantic.attention_bias);
        assert!(semantic.attention_uses_sinks());
        assert!(semantic.router_has_bias());
        assert!(semantic.experts_have_bias());
        assert!(!semantic.tie_word_embeddings);
        assert_eq!(semantic.external_metadata_id(), EXTERNAL_METADATA_ID);

        assert_eq!(mxfp4.quant_method(), "mxfp4");
        assert_eq!(mxfp4.bits_per_weight, 4);
        assert_eq!(mxfp4.group_size, 32);
        assert_eq!(mxfp4.packed_bytes_per_group, 16);
        assert_eq!(mxfp4.scale_exponent_bias, 127);
        assert_eq!(mxfp4.modules_to_not_convert, expected_mxfp4_exclusions());
    }

    #[test]
    fn tiny_same_architecture_canary_preserves_semantics_at_small_dimensions() {
        let (semantic, mxfp4) =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&tiny_canary_config()).unwrap())
                .unwrap();

        assert_eq!(semantic.hidden_size, 32);
        assert_eq!(semantic.layer_count, 2);
        assert_eq!(semantic.query_features().unwrap(), 64);
        assert_eq!(semantic.kv_features().unwrap(), 32);
        assert_eq!(semantic.intermediate_size, 64);
        assert_eq!(
            semantic.layer_types,
            [
                GptOssLayerType::SlidingAttention,
                GptOssLayerType::FullAttention
            ]
        );
        assert!(semantic.tie_word_embeddings);
        assert_eq!(mxfp4.group_size, 32);
    }

    #[test]
    fn wrong_mxfp4_recipe_or_dense_exclusions_fail_closed() {
        let mut wrong_method = official_20b_config();
        wrong_method["quantization_config"]["quant_method"] = Value::String("fp4".to_owned());
        let error =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&wrong_method).unwrap()).unwrap_err();
        assert!(error.contains("quant_method"), "{error}");

        let mut missing_router = official_20b_config();
        missing_router["quantization_config"]["modules_to_not_convert"] =
            serde_json::json!(["model.layers.*.self_attn", "model.embed_tokens", "lm_head"]);
        let error =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&missing_router).unwrap()).unwrap_err();
        assert!(error.contains("modules_to_not_convert"), "{error}");
    }

    #[test]
    fn layer_schedule_drift_fails_closed() {
        let mut wrong_order = official_20b_config();
        wrong_order["layer_types"][0] = Value::String("full_attention".to_owned());
        let error =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&wrong_order).unwrap()).unwrap_err();
        assert!(error.contains("layer_types[0]"), "{error}");

        let mut wrong_length = tiny_canary_config();
        wrong_length["layer_types"] = serde_json::json!(["sliding_attention"]);
        let error =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&wrong_length).unwrap()).unwrap_err();
        assert!(error.contains("layer_types length"), "{error}");
    }

    #[test]
    fn yarn_contract_drift_fails_closed() {
        for (field, replacement) in [
            ("factor", Value::from(16.0)),
            ("beta_fast", Value::from(16.0)),
            ("beta_slow", Value::from(2.0)),
            ("truncate", Value::Bool(true)),
        ] {
            let mut config = official_20b_config();
            config["rope_scaling"][field] = replacement;
            let error =
                GptOssSemanticConfig::parse(&serde_json::to_vec(&config).unwrap()).unwrap_err();
            assert!(error.contains(field), "{field}: {error}");
        }

        let mut wrong_type = official_20b_config();
        wrong_type["rope_scaling"]["rope_type"] = Value::String("dynamic".to_owned());
        let error =
            GptOssSemanticConfig::parse(&serde_json::to_vec(&wrong_type).unwrap()).unwrap_err();
        assert!(error.contains("rope_type"), "{error}");
    }

    #[test]
    fn attention_bias_and_clamped_swiglu_are_not_optional() {
        for (field, replacement) in [
            ("attention_bias", Value::Bool(false)),
            ("swiglu_limit", Value::from(8.0)),
        ] {
            let mut config = official_20b_config();
            config[field] = replacement;
            let error =
                GptOssSemanticConfig::parse(&serde_json::to_vec(&config).unwrap()).unwrap_err();
            assert!(error.contains(field), "{field}: {error}");
        }
    }
}
