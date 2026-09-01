//! Typed Gemma 4 Unified text-only package for the production vNext path.
//!
//! The outer checkpoint remains multimodal, but every non-text tensor is
//! explicitly inventoried and excluded before the immutable text program is
//! built. No Gemma 3 or legacy executor behavior is reused.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    ExternalModelMetadataId, ModelFamilyId, ModelFamilyProvider, ModelFamilyRegistration,
    ModelProgram, ModelSemanticMetadata, PreparedModelFamily, TypedFamilyRegistration, VNextError,
    WeightComponentSource, WeightSchema,
};
use ferrum_quantization::{CompressedTensorsMarlinSafetensorsSource, SafetensorsArchive};
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

use config::Gemma4SemanticConfig;
use weights::Gemma4WeightManifest;

pub const FAMILY_ID: &str = "family.gemma4_unified.text";
pub const EXTERNAL_METADATA_ID: &str = "hf.architecture.Gemma4UnifiedForConditionalGeneration";
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gemma4FamilyConfig {
    semantic: Gemma4SemanticConfig,
    metadata: ModelSemanticMetadata,
    weights: Gemma4WeightManifest,
}

pub struct Gemma4FamilyProvider {
    family_id: ModelFamilyId,
}

impl Gemma4FamilyProvider {
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            family_id: ModelFamilyId::new(FAMILY_ID)?,
        })
    }

    fn validate_typed_config(&self, config: &Gemma4FamilyConfig) -> Result<(), VNextError> {
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

impl ModelFamilyProvider for Gemma4FamilyProvider {
    type Config = Gemma4FamilyConfig;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([ExternalModelMetadataId::new(EXTERNAL_METADATA_ID)
            .expect("Gemma 4 external metadata id is static and valid")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.validate_typed_config(config)?;
        let typed = serde_json::to_value(config).map_err(|error| VNextError::Serialization {
            context: "serialize Gemma 4 family config",
            message: error.to_string(),
        })?;
        if raw != &typed {
            return Err(invalid_config(
                "config",
                "Gemma 4 family input is not the exact typed configuration",
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
        let config: Gemma4FamilyConfig = serde_json::from_value(raw.clone())
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
            "Gemma 4 semantic validator received unowned metadata identity {expected_metadata_id}"
        )));
    }
    Gemma4SemanticConfig::validate_semantic_source(raw).map_err(ferrum_types::FerrumError::model)
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
    let semantic = Gemma4SemanticConfig::parse(sources.config_json())
        .map_err(ferrum_types::FerrumError::model)?;
    match sources.weights() {
        ProductionWeightArtifact::SafetensorsDirectory(weight_root) => {
            let archive = SafetensorsArchive::open(weight_root)?;
            let weights = Gemma4WeightManifest::load(&archive, &semantic)
                .map_err(ferrum_types::FerrumError::model)?;
            let config = Gemma4FamilyConfig {
                semantic,
                metadata,
                weights,
            };
            finish_preparation(
                sources,
                CompressedTensorsMarlinSafetensorsSource::new(archive),
                config,
            )
        }
        ProductionWeightArtifact::GgufFile(_) => Err(ferrum_types::FerrumError::model(
            "Gemma 4 Unified text requires the typed compressed-tensors safetensors source; GGUF is not a supported fallback",
        )),
    }
}

fn finish_preparation<W>(
    sources: Arc<ProductionModelSourceBundle>,
    weights: W,
    config: Gemma4FamilyConfig,
) -> ferrum_types::Result<PreparedProductionModel>
where
    W: WeightComponentSource + 'static,
{
    let descriptor = production_descriptor(&config)?;
    let raw = serde_json::to_value(config)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let provider = Gemma4FamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let family = TypedFamilyRegistration::new(provider)
        .prepare(&raw)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(PreparedProductionModel::new(
        family, weights, descriptor, sources,
    ))
}

fn production_descriptor(
    config: &Gemma4FamilyConfig,
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
        "gemma4_unified_text",
        parameter_count,
        usize::try_from(config.semantic.hidden_size)
            .map_err(|_| ferrum_types::FerrumError::model("hidden_size exceeds usize"))?,
        usize::try_from(config.semantic.layer_count)
            .map_err(|_| ferrum_types::FerrumError::model("layer_count exceeds usize"))?,
        usize::try_from(config.semantic.attention_head_count)
            .map_err(|_| ferrum_types::FerrumError::model("attention_head_count exceeds usize"))?,
        usize::try_from(config.semantic.local_kv_head_count)
            .map_err(|_| ferrum_types::FerrumError::model("local_kv_head_count exceeds usize"))?,
        usize::try_from(config.semantic.local_head_dim)
            .map_err(|_| ferrum_types::FerrumError::model("local_head_dim exceeds usize"))?,
        usize::try_from(config.semantic.vocabulary_size)
            .map_err(|_| ferrum_types::FerrumError::model("vocabulary_size exceeds usize"))?,
        usize::try_from(config.semantic.maximum_sequence_tokens).map_err(|_| {
            ferrum_types::FerrumError::model("maximum_sequence_tokens exceeds usize")
        })?,
        DataType::FP16,
    )
}

pub(super) fn family_registration() -> ferrum_types::Result<Box<dyn ModelFamilyRegistration>> {
    let provider = Gemma4FamilyProvider::new()
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
