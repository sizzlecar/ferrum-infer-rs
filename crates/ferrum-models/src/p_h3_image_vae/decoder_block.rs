//! One real PH3 ViT decoder block under an explicit Ferrum F32 preset.
//!
//! Source: ComfyUI `56f179`, complete `comfy/ldm/minimax/vae.py`,
//! TransformerBlock/Attention/FeedForward. This profile uses F32 full attention;
//! upstream's optional INT8 attention quantizes Q/K/V and, for F32 inputs,
//! rounds its output through BF16. Neither that route nor the default FP16 VAE
//! is represented by this independent numerical profile.
//!
//! The caller owns embedded patch/register/CLS rows and their exact positions.
//! This component performs no latent denormalization, token construction,
//! image time coordinate synthesis, posterior sampling or unpatchification.
use super::PH3ImageVaeDecoderConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod checkpoint;
mod config;
mod program;
mod weights;
pub use checkpoint::PH3ImageVaeDecoderBlockCheckpoint;
pub use config::{PH3ImageVaeDecoderBlockConfig, PH3ImageVaeDecoderBlockExecutionPreset};

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageVaeDecoderBlockLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
#[derive(Clone, Debug)]
pub struct PH3ImageVaeDecoderBlockFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageVaeDecoderBlockLimits,
}
impl PH3ImageVaeDecoderBlockFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.vae.decoder_block";
    pub const CUDA_F32_PROFILE_ID: &'static str =
        "p_h3_image.vae.decoder_block.cuda.f32_a8.full_attention.fma_residual.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.vae.decoder_block.input";
    pub const COSINE_ID: &'static str = "value.p_h3_image.vae.decoder_block.cosine";
    pub const SINE_ID: &'static str = "value.p_h3_image.vae.decoder_block.sine";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.vae.decoder_block.output";
    pub fn new(limits: PH3ImageVaeDecoderBlockLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("limits", "capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new(
                "metadata.p_h3_image.vae.decoder_block.comfy_int8",
            )?,
            limits,
        })
    }
    fn check(&self, c: &PH3ImageVaeDecoderBlockConfig) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_vec(c)
            .map_err(|e| invalid("config", e.to_string()))?
            .len()
            > self.limits.max_metadata_bytes
        {
            return Err(invalid("config", "metadata exceeds declared capacity"));
        }
        weights::schema(c, self.limits.max_stored_bytes, &self.id)?;
        // Compare the actual emitted graph with the limit. No duplicated node
        // count can under-admit a later change to the vertical block.
        if program::build(c, &self.id)?
            .blocks()
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.nodes.len()))
            .is_none_or(|n| n > self.limits.max_plan_nodes)
        {
            return Err(invalid("config", "block plan exceeds declared capacity"));
        }
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageVaeDecoderBlockFamily {
    type Config = PH3ImageVaeDecoderBlockConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| invalid("config", e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.check(c)?;
        if serde_json::to_value(c).map_err(|e| invalid("config", e.to_string()))? != *raw {
            return Err(invalid(
                "config",
                "requires the canonical checked block and explicit preset",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.check(c)?;
        Ok(ModelWeights::Stored(weights::schema(
            c,
            self.limits.max_stored_bytes,
            &self.id,
        )?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.check(c)?;
        let graph = program::build(c, &self.id)?;
        let boundaries: BTreeMap<ProgramValueId, ElementType> = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .map(|id| (id.clone(), ElementType::F32))
            .collect();
        let used: BTreeSet<_> = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.clone())
            .collect();
        let operations = used
            .into_iter()
            .map(|operation_id| {
                let staged = (operation_id.as_str() == COMFY_CONVROT_W8A8_F32_OPERATION_ID)
                    .then(StagedNumericalArithmetic::comfy_convrot_w8a8_f32);
                let arithmetic = !matches!(
                    operation_id.as_str(),
                    GROUPED_COLUMN_SLICE_F32_OPERATION_ID | NAN_TO_NUM_F32_OPERATION_ID
                ) && staged.is_none();
                NumericalOperationContract {
                    multiplication_type: (arithmetic
                        && operation_id.as_str() != CHANNEL_BIAS_ADD_F32_OPERATION_ID)
                        .then_some(ElementType::F32),
                    accumulation_type: arithmetic.then_some(ElementType::F32),
                    staged_arithmetic: staged,
                    composite_arithmetic: None,
                    operation_id,
                    version: ContractVersion::new(1, 0),
                }
            })
            .collect();
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::CUDA_F32_PROFILE_ID)
                .map_err(|e| invalid("profile", e))?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
            boundaries,
            states: vec![],
            kv_storage: vec![],
            operations,
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![profile],
            vec![NumericalProfileId::new(Self::CUDA_F32_PROFILE_ID)
                .map_err(|e| invalid("profile", e))?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid(
                "profile",
                "block profile boundaries/arithmetic differ",
            ));
        }
        program::build(c, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: PH3ImageVaeDecoderBlockFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}
#[cfg(test)]
pub(super) mod tests;
