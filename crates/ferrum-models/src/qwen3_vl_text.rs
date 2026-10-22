//! Qwen3-VL text-only conditioning component, ending before the final RMSNorm.
//!
//! Embedding lookup, tokenization and position generation are caller adapters.
//! This family consumes their actual BF16 tensors through ordinary managed
//! execution. It contains no vision branch, logits head, paged cache or tokens
//! masquerading as tensor work. Numerical profiles distinguish our FP32 online
//! attention from Transformers' eager BF16-probability implementation.

mod config;
mod program;
mod weights;
pub use config::{
    Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextKeyRun, Qwen3VLTextLayout,
};
pub use weights::Qwen3VLTextExpectedWeight;

use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct Qwen3VLTextEncoderFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl Qwen3VLTextEncoderFamily {
    pub const FAMILY_ID: &'static str = "family.qwen3_vl.text_encoder";
    pub const REFERENCE_PROFILE_ID: &'static str = "qwen3vl.text.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen3vl.text.cuda.bf16_f32.v1";
    pub const EMBEDDINGS_ID: &'static str = "value.qwen3_vl.text.embeddings";
    pub const COSINE_ID: &'static str = "value.qwen3_vl.text.cosine";
    pub const SINE_ID: &'static str = "value.qwen3_vl.text.sine";
    pub const OUTPUT_ID: &'static str = "value.qwen3_vl.text.pre_final_norm";
    pub const EMBEDDING_EXTERNAL_NAME: &'static str = "model.language_model.embed_tokens.weight";
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen3_vl.text_encoder")?,
        })
    }
}
impl ModelFamilyProvider for Qwen3VLTextEncoderFamily {
    type Config = Qwen3VLTextConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        config.validate()?;
        if serde_json::to_value(config).map_err(|e| invalid("config", e.to_string()))? != *raw {
            return Err(invalid(
                "config",
                "requires the exact canonical checked Qwen3-VL text/layout configuration",
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
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| invalid("config", e.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(weights::schema(config, &self.id)?))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let graph = program::build(config, &self.id)?;
        let boundaries = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .cloned()
            .map(|id| (id, ElementType::Bf16))
            .collect::<BTreeMap<_, _>>();
        let operations = [
            DENSE_LINEAR_BF16_OPERATION_ID,
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            HALF_SPLIT_ROPE_BF16_OPERATION_ID,
            STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            SILU_MUL_BF16_OPERATION_ID,
        ]
        .into_iter()
        .map(|id| {
            Ok(NumericalOperationContract {
                operation_id: OperationId::new(id)?,
                version: ContractVersion::new(1, 0),
                multiplication_type: if id == PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID {
                    None
                } else {
                    Some(ElementType::F32)
                },
                accumulation_type: Some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
        let profile = |name: &str| -> Result<NumericalExecutionProfile, VNextError> {
            Ok(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(|e| invalid("profile", e))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: boundaries.clone(),
                states: vec![],
                kv_storage: vec![],
                operations: operations.clone(),
            })
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![
                profile(Self::REFERENCE_PROFILE_ID)?,
                profile(Self::CUDA_PROFILE_ID)?,
            ],
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID)
                .map_err(|e| invalid("profile", e))?],
        )
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid(
                "profile",
                "text component profile differs from declared BF16 boundaries and FP32 arithmetic",
            ));
        }
        program::build(config, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: Qwen3VLTextEncoderFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}
#[cfg(test)]
mod tests;
