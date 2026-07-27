use ferrum_interfaces::vnext::CanonicalRational;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::EXTERNAL_METADATA_ID;

const HF_ARCHITECTURE: &str = "Qwen3MoeForCausalLM";
const HF_MODEL_TYPE: &str = "qwen3_moe";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Qwen3MoeGptqConfig {
    pub bits: u8,
    pub group_size: u32,
    pub desc_act: bool,
    pub sym: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Qwen3MoeSemanticConfig {
    pub hidden_size: u64,
    pub layer_count: u64,
    pub attention_head_count: u64,
    pub kv_head_count: u64,
    pub head_dim: u64,
    pub vocabulary_size: u64,
    pub maximum_sequence_tokens: u64,
    pub expert_count: u64,
    pub experts_per_token: u64,
    pub expert_intermediate_size: u64,
    pub normalize_topk: bool,
    pub rms_norm_epsilon: CanonicalRational,
    pub rope_theta: CanonicalRational,
    pub tie_word_embeddings: bool,
    pub quantization: Qwen3MoeGptqConfig,
}

impl Qwen3MoeSemanticConfig {
    pub(super) fn parse(raw: &[u8]) -> Result<Self, String> {
        let value: Value = serde_json::from_slice(raw)
            .map_err(|error| format!("parse semantic config.json: {error}"))?;
        let root = value
            .as_object()
            .ok_or_else(|| "semantic config.json root must be an object".to_owned())?;

        validate_identity(root)?;
        validate_supported_semantics(root)?;

        let config = Self {
            hidden_size: required_positive_u64(root, "hidden_size")?,
            layer_count: required_positive_u64(root, "num_hidden_layers")?,
            attention_head_count: required_positive_u64(root, "num_attention_heads")?,
            kv_head_count: required_positive_u64(root, "num_key_value_heads")?,
            head_dim: required_positive_u64(root, "head_dim")?,
            vocabulary_size: required_positive_u64(root, "vocab_size")?,
            maximum_sequence_tokens: required_positive_u64(root, "max_position_embeddings")?,
            expert_count: required_positive_u64(root, "num_experts")?,
            experts_per_token: required_positive_u64(root, "num_experts_per_tok")?,
            expert_intermediate_size: required_positive_u64(root, "moe_intermediate_size")?,
            normalize_topk: required_bool(root, "norm_topk_prob")?,
            rms_norm_epsilon: required_positive_rational(root, "rms_norm_eps", true)?,
            rope_theta: required_positive_rational(root, "rope_theta", false)?,
            tie_word_embeddings: required_bool(root, "tie_word_embeddings")?,
            quantization: parse_quantization(root)?,
        };
        config.validate()?;
        Ok(config)
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

    pub(super) fn validate(&self) -> Result<(), String> {
        if self.kv_head_count > self.attention_head_count
            || !self.attention_head_count.is_multiple_of(self.kv_head_count)
        {
            return Err("num_attention_heads must be divisible by num_key_value_heads".to_owned());
        }
        if self.experts_per_token > self.expert_count {
            return Err("num_experts_per_tok must not exceed num_experts".to_owned());
        }
        let query_features = self.query_features()?;
        let kv_features = self.kv_features()?;
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("query projection width", query_features),
            ("key/value projection width", kv_features),
            ("vocab_size", self.vocabulary_size),
            ("moe_intermediate_size", self.expert_intermediate_size),
        ] {
            if !value.is_multiple_of(16) {
                return Err(format!("{field} must be divisible by 16 for Marlin"));
            }
        }
        let group_size = u64::from(self.quantization.group_size);
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.expert_intermediate_size),
        ] {
            if !value.is_multiple_of(group_size) {
                return Err(format!(
                    "{field} must be divisible by GPTQ group_size {group_size}"
                ));
            }
        }
        for (field, output_features, input_features) in [
            ("q_proj", query_features, self.hidden_size),
            ("k_proj", kv_features, self.hidden_size),
            ("v_proj", kv_features, self.hidden_size),
            ("o_proj", self.hidden_size, query_features),
            (
                "expert gate/up",
                self.expert_intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| "expert gate/up width overflows u64".to_owned())?,
                self.hidden_size,
            ),
            (
                "expert down",
                self.hidden_size,
                self.expert_intermediate_size,
            ),
        ] {
            validate_marlin_thread_tile(field, output_features, input_features)?;
        }
        if self.rms_norm_epsilon.numerator() <= 0
            || self.rms_norm_epsilon.numerator() as u64 > self.rms_norm_epsilon.denominator()
            || self.rope_theta.numerator() <= 0
        {
            return Err("RMS epsilon and RoPE theta must be canonical positive values".to_owned());
        }
        self.quantization.validate()
    }

    pub(super) fn external_metadata_id(&self) -> &'static str {
        EXTERNAL_METADATA_ID
    }
}

impl Qwen3MoeGptqConfig {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.bits != 4
            || self.group_size == 0
            || !self.group_size.is_power_of_two()
            || self.desc_act
            || !self.sym
        {
            return Err(
                "Qwen3 MoE vNext requires symmetric GPTQ INT4, power-of-two group_size, and desc_act=false"
                    .to_owned(),
            );
        }
        Ok(())
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

fn validate_supported_semantics(root: &Map<String, Value>) -> Result<(), String> {
    if required_string(root, "hidden_act")? != "silu" {
        return Err("hidden_act must be \"silu\"".to_owned());
    }
    if required_string(root, "torch_dtype")? != "float16" {
        return Err("torch_dtype must be \"float16\" for the current vNext providers".to_owned());
    }
    if required_bool(root, "attention_bias")? {
        return Err("attention_bias=true is not supported".to_owned());
    }
    if root
        .get("use_sliding_window")
        .is_some_and(|value| value.as_bool() != Some(false))
    {
        return Err("use_sliding_window must be false".to_owned());
    }
    for field in ["rope_scaling", "sliding_window"] {
        if root.get(field).is_some_and(|value| !value.is_null()) {
            return Err(format!("{field} is not supported and must be null"));
        }
    }
    if required_positive_u64(root, "decoder_sparse_step")? != 1 {
        return Err("decoder_sparse_step must be 1 for all-layer MoE".to_owned());
    }
    let mlp_only_layers = root
        .get("mlp_only_layers")
        .and_then(Value::as_array)
        .ok_or_else(|| "mlp_only_layers must be an array".to_owned())?;
    if !mlp_only_layers.is_empty() {
        return Err("mlp_only_layers must be empty for all-layer MoE".to_owned());
    }
    Ok(())
}

fn validate_marlin_thread_tile(
    field: &str,
    output_features: u64,
    input_features: u64,
) -> Result<(), String> {
    let supported = (output_features.is_multiple_of(64) && input_features.is_multiple_of(128))
        || (output_features.is_multiple_of(128) && input_features.is_multiple_of(64));
    if output_features == 0 || input_features == 0 || !supported {
        return Err(format!(
            "{field} shape N={output_features}, K={input_features} does not satisfy a Marlin 64x128 or 128x64 thread tile"
        ));
    }
    Ok(())
}

fn parse_quantization(root: &Map<String, Value>) -> Result<Qwen3MoeGptqConfig, String> {
    let quantization = root
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| "quantization_config must be an object".to_owned())?;
    if required_string(quantization, "quant_method")? != "gptq"
        || required_string(quantization, "checkpoint_format")? != "gptq"
    {
        return Err("quantization_config must describe a GPTQ checkpoint".to_owned());
    }
    if quantization
        .get("static_groups")
        .is_some_and(|value| value.as_bool() != Some(false))
    {
        return Err("quantization_config.static_groups must be false".to_owned());
    }
    let bits = required_positive_u64(quantization, "bits")?;
    let group_size = required_positive_u64(quantization, "group_size")?;
    let parsed = Qwen3MoeGptqConfig {
        bits: u8::try_from(bits).map_err(|_| "GPTQ bits exceeds u8".to_owned())?,
        group_size: u32::try_from(group_size)
            .map_err(|_| "GPTQ group_size exceeds u32".to_owned())?,
        desc_act: required_bool(quantization, "desc_act")?,
        sym: required_bool(quantization, "sym")?,
    };
    parsed.validate()?;
    Ok(parsed)
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

fn required_positive_rational(
    root: &Map<String, Value>,
    field: &str,
    at_most_one: bool,
) -> Result<CanonicalRational, String> {
    let number = root
        .get(field)
        .and_then(Value::as_number)
        .ok_or_else(|| format!("{field} must be a JSON number"))?;
    let value = CanonicalRational::from_decimal_str(&number.to_string())
        .map_err(|error| format!("{field}: {error}"))?;
    if value.numerator() <= 0 || (at_most_one && value.numerator() as u64 > value.denominator()) {
        return Err(format!(
            "{field} must be positive{}",
            if at_most_one {
                " and no greater than one"
            } else {
                ""
            }
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_config() -> Value {
        serde_json::json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "norm_topk_prob": true,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0,
            "rope_scaling": null,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "attention_bias": false,
            "use_sliding_window": false,
            "sliding_window": null,
            "torch_dtype": "float16",
            "quantization_config": {
                "bits": 4,
                "checkpoint_format": "gptq",
                "desc_act": false,
                "group_size": 128,
                "quant_method": "gptq",
                "static_groups": false,
                "sym": true
            }
        })
    }

    #[test]
    fn production_m3_semantics_are_exact_and_typed() {
        let raw = serde_json::to_vec(&reference_config()).unwrap();
        let parsed = Qwen3MoeSemanticConfig::parse(&raw).unwrap();

        assert_eq!(parsed.hidden_size, 2048);
        assert_eq!(parsed.layer_count, 48);
        assert_eq!(parsed.query_features().unwrap(), 4096);
        assert_eq!(parsed.kv_features().unwrap(), 512);
        assert_eq!(parsed.expert_count, 128);
        assert_eq!(parsed.experts_per_token, 8);
        assert_eq!(parsed.expert_intermediate_size, 768);
        assert_eq!(parsed.external_metadata_id(), EXTERNAL_METADATA_ID);
        assert_eq!(
            parsed.rms_norm_epsilon,
            CanonicalRational::new(1, 1_000_000).unwrap()
        );
    }

    #[test]
    fn unsupported_semantics_fail_before_weight_discovery() {
        for (field, replacement) in [
            ("hidden_act", Value::String("gelu".to_owned())),
            ("rope_scaling", serde_json::json!({"type": "dynamic"})),
            ("attention_bias", Value::Bool(true)),
            ("decoder_sparse_step", Value::from(2)),
            ("mlp_only_layers", serde_json::json!([3])),
        ] {
            let mut value = reference_config();
            value
                .as_object_mut()
                .unwrap()
                .insert(field.to_owned(), replacement);
            let error =
                Qwen3MoeSemanticConfig::parse(&serde_json::to_vec(&value).unwrap()).unwrap_err();
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn model_identity_cannot_alias_qwen35_or_dense_qwen3() {
        for architecture in ["Qwen3ForCausalLM", "Qwen3_5MoeForConditionalGeneration"] {
            let mut value = reference_config();
            value["architectures"] = serde_json::json!([architecture]);
            let error =
                Qwen3MoeSemanticConfig::parse(&serde_json::to_vec(&value).unwrap()).unwrap_err();
            assert!(error.contains("architectures"), "{error}");
        }
    }
}
