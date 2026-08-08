//! Typed Qwen3 routed-MoE package for the production vNext path.
//!
//! Semantic metadata, checkpoint headers, the logical program, and the GPTQ
//! physical layout are validated before a backend receives any weight payload.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    ExternalModelMetadataId, ModelFamilyId, ModelFamilyProvider, ModelFamilyRegistration,
    ModelProgram, ModelSemanticMetadata, PreparedModelFamily, TypedFamilyRegistration, VNextError,
    WeightComponentSource, WeightSchema,
};
use ferrum_quantization::{
    GgufWeightComponentSource, GptqMarlinSafetensorsSource, SafetensorsArchive,
};
use ferrum_types::DataType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    hf_metadata::parse_hf_model_semantic_metadata, CausalLanguageModelDescriptor,
    PreparedProductionModel, ProductionModelSourceBundle, ProductionWeightArtifact,
};

mod config;
mod program;
mod weights;

use config::Qwen3MoeSemanticConfig;
use weights::Qwen3MoeWeightManifest;

pub const FAMILY_ID: &str = "family.qwen3.routed_moe";
pub const EXTERNAL_METADATA_ID: &str = "hf.architecture.Qwen3MoeForCausalLM";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3MoeFamilyConfig {
    semantic: Qwen3MoeSemanticConfig,
    metadata: ModelSemanticMetadata,
    weights: Qwen3MoeWeightManifest,
}

pub struct Qwen3MoeFamilyProvider {
    family_id: ModelFamilyId,
}

impl Qwen3MoeFamilyProvider {
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            family_id: ModelFamilyId::new(FAMILY_ID)?,
        })
    }

    fn validate_typed_config(&self, config: &Qwen3MoeFamilyConfig) -> Result<(), VNextError> {
        config
            .semantic
            .validate()
            .map_err(|reason| invalid_config("semantic", reason))?;
        config.weights.validate(&config.semantic)?;
        if config.metadata.template.template.is_empty()
            || config.metadata.template.source_file != "tokenizer_config.json"
            || config.metadata.special_tokens.eos_token_ids.is_empty()
        {
            return Err(invalid_config(
                "metadata",
                "chat template source and EOS token set must be explicit",
            ));
        }
        Ok(())
    }
}

impl ModelFamilyProvider for Qwen3MoeFamilyProvider {
    type Config = Qwen3MoeFamilyConfig;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([ExternalModelMetadataId::new(EXTERNAL_METADATA_ID)
            .expect("Qwen3 MoE external metadata id is static and valid")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.validate_typed_config(config)?;
        let typed = serde_json::to_value(config).map_err(|error| VNextError::Serialization {
            context: "serialize Qwen3 MoE family config",
            message: error.to_string(),
        })?;
        if raw != &typed {
            return Err(invalid_config(
                "config",
                "Qwen3 MoE family input is not the exact typed configuration",
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
        let config: Qwen3MoeFamilyConfig = serde_json::from_value(raw.clone())
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
            "Qwen3 MoE semantic validator received unowned metadata identity {expected_metadata_id}"
        )));
    }
    Qwen3MoeSemanticConfig::validate_semantic_source(raw)
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
        ferrum_types::FerrumError::model("tokenizer source missing tokenizer_config.json")
    })?;
    let model_config: Value = serde_json::from_slice(sources.config_json())
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let metadata = parse_hf_model_semantic_metadata(&model_config, tokenizer_config)
        .map_err(ferrum_types::FerrumError::model)?;
    match sources.weights() {
        ProductionWeightArtifact::SafetensorsDirectory(weight_root) => {
            let weight_config = sources.weight_config_json().ok_or_else(|| {
                ferrum_types::FerrumError::model(
                    "Qwen3 MoE GPTQ weight source is missing physical config.json",
                )
            })?;
            let (semantic, quantization) =
                Qwen3MoeSemanticConfig::parse_sources(sources.config_json(), weight_config)
                    .map_err(ferrum_types::FerrumError::model)?;
            let archive = SafetensorsArchive::open(weight_root)?;
            let weights = Qwen3MoeWeightManifest::load(&archive, &semantic, &quantization)
                .map_err(ferrum_types::FerrumError::model)?;
            let config = Qwen3MoeFamilyConfig {
                semantic,
                metadata,
                weights,
            };
            finish_preparation(sources, GptqMarlinSafetensorsSource::new(archive), config)
        }
        ProductionWeightArtifact::GgufFile(path) => {
            let semantic = Qwen3MoeSemanticConfig::parse_semantic_source(sources.config_json())
                .map_err(ferrum_types::FerrumError::model)?;
            let source = GgufWeightComponentSource::open(path)?;
            let weights = Qwen3MoeWeightManifest::load_gguf(&source, &semantic)
                .map_err(ferrum_types::FerrumError::model)?;
            let config = Qwen3MoeFamilyConfig {
                semantic,
                metadata,
                weights,
            };
            finish_preparation(sources, source, config)
        }
    }
}

fn finish_preparation<W>(
    sources: Arc<ProductionModelSourceBundle>,
    weights: W,
    config: Qwen3MoeFamilyConfig,
) -> ferrum_types::Result<PreparedProductionModel>
where
    W: WeightComponentSource + 'static,
{
    let descriptor = production_descriptor(&config)?;
    let raw = serde_json::to_value(config)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let provider = Qwen3MoeFamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let family = TypedFamilyRegistration::new(provider)
        .prepare(&raw)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(PreparedProductionModel::new(
        family, weights, descriptor, sources,
    ))
}

fn production_descriptor(
    config: &Qwen3MoeFamilyConfig,
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
    CausalLanguageModelDescriptor::new(
        "qwen3_moe",
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
    )
}

pub(super) fn family_registration() -> ferrum_types::Result<Box<dyn ModelFamilyRegistration>> {
    let provider = Qwen3MoeFamilyProvider::new()
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
    use std::collections::BTreeSet;

    use ferrum_interfaces::vnext::{
        CanonicalRational, SpecialTokenCollisionPolicy, SpecialTokenMetadata, TemplateMetadata,
        MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES,
    };

    use super::*;

    fn production_config() -> Qwen3MoeFamilyConfig {
        let semantic = Qwen3MoeSemanticConfig {
            hidden_size: 2048,
            layer_count: 48,
            attention_head_count: 32,
            kv_head_count: 4,
            head_dim: 128,
            vocabulary_size: 151_936,
            maximum_sequence_tokens: 40_960,
            expert_count: 128,
            experts_per_token: 8,
            expert_intermediate_size: 768,
            normalize_topk: true,
            rms_norm_epsilon: CanonicalRational::new(1, 1_000_000).unwrap(),
            rope_theta: CanonicalRational::new(1_000_000, 1).unwrap(),
            tie_word_embeddings: false,
        };
        let quantization = config::Qwen3MoeGptqConfig {
            bits: 4,
            group_size: 128,
            desc_act: false,
            sym: true,
        };
        Qwen3MoeFamilyConfig {
            weights: weights::expected_manifest(&semantic, &quantization).unwrap(),
            semantic,
            metadata: ModelSemanticMetadata {
                template: TemplateMetadata {
                    template: "{{ messages }}".to_owned(),
                    source_file: "tokenizer_config.json".to_owned(),
                    sha256: "0".repeat(64),
                },
                special_tokens: SpecialTokenMetadata {
                    bos_token_id: None,
                    eos_token_ids: BTreeSet::from([151_645]),
                    pad_token_id: None,
                    collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
                },
            },
        }
    }

    #[test]
    fn production_family_package_is_canonical_and_below_wire_limit() {
        let config = production_config();
        let raw = serde_json::to_value(config).unwrap();
        let family = TypedFamilyRegistration::new(Qwen3MoeFamilyProvider::new().unwrap())
            .prepare(&raw)
            .unwrap();
        let bytes = serde_json::to_vec(&family).unwrap();

        eprintln!("QWEN3_MOE_PREPARED_FAMILY_WIRE_BYTES={}", bytes.len());
        assert!(bytes.len() <= MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES);
        assert_eq!(family.program().states().len(), 48);
        assert_eq!(family.external_metadata_id().as_str(), EXTERNAL_METADATA_ID);
    }
}
