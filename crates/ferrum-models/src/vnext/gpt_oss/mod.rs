//! Typed GPT-OSS routed-MoE package for the production vNext path.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    ExternalModelMetadataId, ModelFamilyId, ModelFamilyProvider, ModelFamilyRegistration,
    ModelProgram, ModelSemanticMetadata, TypedFamilyRegistration, VNextError,
    WeightComponentSource, WeightSchema,
};
use ferrum_quantization::{Mxfp4SafetensorsSource, SafetensorsArchive};
use ferrum_types::{DataType, ModelOutputProtocol};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    hf_metadata::parse_hf_model_semantic_metadata_with_external_template,
    CausalLanguageModelDescriptor, PreparedProductionModel, ProductionModelSourceBundle,
    ProductionWeightArtifact,
};

mod config;
mod program;
mod weights;

use config::GptOssSemanticConfig;
use weights::GptOssWeightManifest;

pub const FAMILY_ID: &str = "family.gpt_oss.routed_moe";
pub const EXTERNAL_METADATA_ID: &str = "hf.architecture.GptOssForCausalLM";

const HARMONY_BOS_TOKEN_ID: u32 = 199_998;
const HARMONY_ENDOFTEXT_TOKEN_ID: u32 = 199_999;
const HARMONY_RETURN_TOKEN_ID: u32 = 200_002;
const HARMONY_CALL_TOKEN_ID: u32 = 200_012;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GptOssFamilyConfig {
    semantic: GptOssSemanticConfig,
    metadata: ModelSemanticMetadata,
    weights: GptOssWeightManifest,
}

pub struct GptOssFamilyProvider {
    family_id: ModelFamilyId,
}

impl GptOssFamilyProvider {
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            family_id: ModelFamilyId::new(FAMILY_ID)?,
        })
    }

    fn validate_typed_config(&self, config: &GptOssFamilyConfig) -> Result<(), VNextError> {
        config
            .semantic
            .validate()
            .map_err(|reason| invalid_config("semantic", reason))?;
        config.weights.validate(&config.semantic)?;
        let expected_eos = BTreeSet::from([
            HARMONY_ENDOFTEXT_TOKEN_ID,
            HARMONY_RETURN_TOKEN_ID,
            HARMONY_CALL_TOKEN_ID,
        ]);
        if config.metadata.template.template.is_empty()
            || config.metadata.template.source_file != "chat_template.jinja"
            || config.metadata.special_tokens.bos_token_id != Some(HARMONY_BOS_TOKEN_ID)
            || config.metadata.special_tokens.pad_token_id != Some(HARMONY_ENDOFTEXT_TOKEN_ID)
            || config.metadata.special_tokens.eos_token_ids != expected_eos
        {
            return Err(invalid_config(
                "metadata",
                "GPT-OSS requires chat_template.jinja and the locked Harmony BOS/PAD/CALL/RETURN/ENDOFTEXT token contract",
            ));
        }
        Ok(())
    }
}

impl ModelFamilyProvider for GptOssFamilyProvider {
    type Config = GptOssFamilyConfig;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([ExternalModelMetadataId::new(EXTERNAL_METADATA_ID)
            .expect("GPT-OSS external metadata id is static and valid")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.validate_typed_config(config)?;
        let typed = serde_json::to_value(config).map_err(|error| VNextError::Serialization {
            context: "serialize GPT-OSS family config",
            message: error.to_string(),
        })?;
        if raw != &typed {
            return Err(invalid_config(
                "config",
                "GPT-OSS family input is not the exact typed configuration",
            ));
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        ExternalModelMetadataId::new(config.semantic.external_metadata_id())
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: GptOssFamilyConfig = serde_json::from_value(raw.clone())
            .map_err(|error| invalid_config("config", error.to_string()))?;
        self.validate_typed_config(&config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        config.weights.weight_schema(&config.semantic)
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        program::build_semantic_program(&self.family_id, &config.semantic, &config.weights)
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(config.metadata.clone())
    }
}

pub(super) fn validate_semantic_config(
    expected_metadata_id: &ExternalModelMetadataId,
    raw: &[u8],
) -> ferrum_types::Result<()> {
    if expected_metadata_id.as_str() != EXTERNAL_METADATA_ID {
        return Err(ferrum_types::FerrumError::internal(format!(
            "GPT-OSS semantic validator received unowned metadata identity {expected_metadata_id}"
        )));
    }
    GptOssSemanticConfig::parse(raw)
        .map(|_| ())
        .map_err(ferrum_types::FerrumError::model)
}

pub fn prepare_from_model_dir(model_dir: &Path) -> ferrum_types::Result<PreparedProductionModel> {
    let sources = Arc::new(super::open_registered_colocated_safetensors(model_dir)?);
    prepare_from_sources(sources)
}

pub(super) fn prepare_from_sources(
    sources: Arc<ProductionModelSourceBundle>,
) -> ferrum_types::Result<PreparedProductionModel> {
    let tokenizer_config = sources.tokenizer_config_json().ok_or_else(|| {
        ferrum_types::FerrumError::model("GPT-OSS source missing tokenizer_config.json")
    })?;
    let chat_template = sources.chat_template_jinja().ok_or_else(|| {
        ferrum_types::FerrumError::model("GPT-OSS source missing chat_template.jinja")
    })?;
    let generation_config = sources.generation_config_json().ok_or_else(|| {
        ferrum_types::FerrumError::model("GPT-OSS source missing generation_config.json")
    })?;
    let model_config: Value = serde_json::from_slice(sources.config_json())
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let metadata = parse_hf_model_semantic_metadata_with_external_template(
        &model_config,
        tokenizer_config,
        chat_template,
        generation_config,
    )
    .map_err(ferrum_types::FerrumError::model)?;
    let (semantic, quantization) = GptOssSemanticConfig::parse(sources.config_json())
        .map_err(ferrum_types::FerrumError::model)?;
    let weight_root = match sources.weights() {
        ProductionWeightArtifact::SafetensorsDirectory(weight_root) => weight_root,
        ProductionWeightArtifact::GgufFile(_) => {
            return Err(ferrum_types::FerrumError::model(
                "GPT-OSS accepts only the locked native MXFP4 safetensors source",
            ))
        }
    };
    let archive = SafetensorsArchive::open(weight_root)?;
    let weights = GptOssWeightManifest::load(&archive, &semantic, &quantization)
        .map_err(ferrum_types::FerrumError::model)?;
    finish_preparation(
        sources,
        Mxfp4SafetensorsSource::new(archive),
        GptOssFamilyConfig {
            semantic,
            metadata,
            weights,
        },
    )
}

fn finish_preparation<W>(
    sources: Arc<ProductionModelSourceBundle>,
    weights: W,
    config: GptOssFamilyConfig,
) -> ferrum_types::Result<PreparedProductionModel>
where
    W: WeightComponentSource + 'static,
{
    let descriptor = production_descriptor(&config)?;
    let raw = serde_json::to_value(config)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let provider = GptOssFamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let family = TypedFamilyRegistration::new(provider)
        .prepare(&raw)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(PreparedProductionModel::new(
        family, weights, descriptor, sources,
    ))
}

fn production_descriptor(
    config: &GptOssFamilyConfig,
) -> ferrum_types::Result<CausalLanguageModelDescriptor> {
    let schema = config
        .weights
        .weight_schema(&config.semantic)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let parameter_count = schema.tensors.iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(
                tensor
                    .logical_elements()
                    .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?,
            )
            .ok_or_else(|| ferrum_types::FerrumError::model("parameter count overflows u64"))
    })?;
    Ok(CausalLanguageModelDescriptor::new(
        "gpt_oss",
        parameter_count,
        usize::try_from(config.semantic.hidden_size)
            .map_err(|_| ferrum_types::FerrumError::model("hidden_size exceeds usize"))?,
        usize::try_from(config.semantic.layer_count)
            .map_err(|_| ferrum_types::FerrumError::model("layer_count exceeds usize"))?,
        usize::try_from(config.semantic.attention_head_count)
            .map_err(|_| ferrum_types::FerrumError::model("attention_head_count exceeds usize"))?,
        usize::try_from(config.semantic.kv_head_count)
            .map_err(|_| ferrum_types::FerrumError::model("kv_head_count exceeds usize"))?,
        usize::try_from(config.semantic.head_dim)
            .map_err(|_| ferrum_types::FerrumError::model("head_dim exceeds usize"))?,
        usize::try_from(config.semantic.vocabulary_size)
            .map_err(|_| ferrum_types::FerrumError::model("vocabulary_size exceeds usize"))?,
        usize::try_from(config.semantic.maximum_sequence_tokens).map_err(|_| {
            ferrum_types::FerrumError::model("maximum_sequence_tokens exceeds usize")
        })?,
        DataType::FP16,
    )?
    .with_output_protocol(ModelOutputProtocol::HarmonyGptOss))
}

pub(super) fn family_registration() -> ferrum_types::Result<Box<dyn ModelFamilyRegistration>> {
    let provider = GptOssFamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(Box::new(TypedFamilyRegistration::new(provider)))
}

fn invalid_config(field: impl Into<String>, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: FAMILY_ID.to_owned(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use ferrum_interfaces::vnext::{
        SpecialTokenCollision, SpecialTokenCollisionPolicy, SpecialTokenMetadata, SpecialTokenRole,
        TemplateMetadata, MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES,
    };

    use super::*;

    fn tiny_config() -> GptOssFamilyConfig {
        let raw = serde_json::to_vec(&serde_json::json!({
            "architectures": ["GptOssForCausalLM"], "attention_bias": true,
            "attention_dropout": 0.0, "experts_per_token": 4, "head_dim": 32,
            "hidden_act": "silu", "hidden_size": 32, "initial_context_length": 4096,
            "intermediate_size": 64, "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 131072, "model_type": "gpt_oss",
            "num_attention_heads": 2, "num_experts_per_tok": 4, "num_hidden_layers": 2,
            "num_key_value_heads": 1, "num_local_experts": 32, "output_router_logits": false,
            "quantization_config": {"modules_to_not_convert": [
                "model.layers.*.self_attn", "model.layers.*.mlp.router",
                "model.embed_tokens", "lm_head"
            ], "quant_method": "mxfp4"},
            "rms_norm_eps": 0.00001,
            "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
                "original_max_position_embeddings": 4096, "rope_type": "yarn", "truncate": false},
            "rope_theta": 150000, "sliding_window": 128, "swiglu_limit": 7.0,
            "tie_word_embeddings": true, "use_cache": true, "vocab_size": 201088
        }))
        .unwrap();
        let (semantic, quantization) = GptOssSemanticConfig::parse(&raw).unwrap();
        let collision =
            SpecialTokenCollision::new(SpecialTokenRole::Eos, SpecialTokenRole::Pad).unwrap();
        GptOssFamilyConfig {
            weights: weights::expected_manifest(&semantic, &quantization).unwrap(),
            semantic,
            metadata: ModelSemanticMetadata {
                template: TemplateMetadata {
                    template: "{{ messages }}".to_owned(),
                    source_file: "chat_template.jinja".to_owned(),
                    sha256: "0".repeat(64),
                },
                special_tokens: SpecialTokenMetadata {
                    bos_token_id: Some(HARMONY_BOS_TOKEN_ID),
                    eos_token_ids: BTreeSet::from([
                        HARMONY_ENDOFTEXT_TOKEN_ID,
                        HARMONY_RETURN_TOKEN_ID,
                        HARMONY_CALL_TOKEN_ID,
                    ]),
                    pad_token_id: Some(HARMONY_ENDOFTEXT_TOKEN_ID),
                    collision_policy: SpecialTokenCollisionPolicy::new(BTreeSet::from([collision])),
                },
            },
        }
    }

    #[test]
    fn production_family_is_canonical_and_selects_harmony() {
        let config = tiny_config();
        let descriptor = production_descriptor(&config).unwrap();
        assert_eq!(
            descriptor.output_protocol(),
            ModelOutputProtocol::HarmonyGptOss
        );
        let raw = serde_json::to_value(config).unwrap();
        let family = TypedFamilyRegistration::new(GptOssFamilyProvider::new().unwrap())
            .prepare(&raw)
            .unwrap();
        assert!(serde_json::to_vec(&family).unwrap().len() <= MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES);
        assert_eq!(family.program().states().len(), 2);
        assert_eq!(family.external_metadata_id().as_str(), EXTERNAL_METADATA_ID);
    }

    #[test]
    fn harmony_terminal_set_drift_is_rejected() {
        let mut config = tiny_config();
        config
            .metadata
            .special_tokens
            .eos_token_ids
            .remove(&HARMONY_CALL_TOKEN_ID);
        let error = GptOssFamilyProvider::new()
            .unwrap()
            .validate_typed_config(&config)
            .unwrap_err();
        assert!(error.to_string().contains("Harmony"));
    }
}
