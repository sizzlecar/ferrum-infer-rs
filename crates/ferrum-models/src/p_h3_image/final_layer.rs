//! Stored curve final heads. Only target rows are normalized; F32 modulation
//! never crosses the ordinary DiT block's BF16 scale/shift boundaries.
//! Returned image/audio values are positive BF16 velocities for the native
//! dataward solver, not the negative Comfy outer model-wrapper predictions.
use super::{
    PH3ImageCurveLerpArithmetic, PH3ImageCurveTimeEmbedding, PH3ImageRequestLayout,
    PH3ImageTransformerConfig,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
mod config;
mod program;
mod weights;
pub use checkpoint::PH3ImageFinalLayerCheckpoint;
use config::tensor;
pub use config::{PH3ImageFinalLayerConfig, PH3ImageFinalLayerRequest, PH3ImageFinalLayerTimeRun};
#[derive(Clone, Copy, Debug)]
pub struct PH3ImageFinalLayerLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
#[derive(Clone, Debug)]
pub struct PH3ImageFinalLayerFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageFinalLayerLimits,
}
impl PH3ImageFinalLayerFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.final_layer";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "p_h3_image.final_layer.reference.bf16_norm.f32_modulation.positive_bf16_velocity.v1";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.final_layer.cuda.bf16_norm.f32_modulation.positive_bf16_velocity.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.final_layer.input";
    pub const BASIS_ID: &'static str = "value.p_h3_image.final_layer.basis";
    pub const IMAGE_VELOCITY_ID: &'static str = "value.p_h3_image.final_layer.image_velocity";
    pub const AUDIO_VELOCITY_ID: &'static str = "value.p_h3_image.final_layer.audio_velocity";
    pub fn new(limits: PH3ImageFinalLayerLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.p_h3_image.final_layer.comfy_curve")?,
            limits,
        })
    }
    fn check(&self, c: &PH3ImageFinalLayerConfig) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_vec(c)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > self.limits.max_metadata_bytes
        {
            return Err(invalid("configuration exceeds metadata capacity"));
        }
        weights::schema(c, self.limits.max_stored_bytes, &self.id)?;
        if program::build(c, &self.id)?
            .blocks()
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.nodes.len()))
            .is_none_or(|n| n > self.limits.max_plan_nodes)
        {
            return Err(invalid("final plan exceeds node capacity"));
        }
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageFinalLayerFamily {
    type Config = PH3ImageFinalLayerConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.check(c)?;
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("requires canonical final-layer configuration"));
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
        let specs = program::tensor_specs(c, &graph)?;
        let boundaries: BTreeMap<ProgramValueId, ElementType> = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .map(|v| (v.clone(), specs[v].element_type))
            .collect();
        let used: BTreeSet<_> = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.clone())
            .collect();
        let operations: Vec<_> = used
            .into_iter()
            .map(|operation_id| {
                let arithmetic = matches!(
                    operation_id.as_str(),
                    AFFINE_LINEAR_F32_OPERATION_ID
                        | ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID
                        | ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID
                );
                NumericalOperationContract {
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: arithmetic.then_some(ElementType::F32),
                    accumulation_type: arithmetic.then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect();
        let profile = |name| -> Result<NumericalExecutionProfile, VNextError> {
            Ok(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::IMAGE_VELOCITY_ID)?,
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
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("final numerical profile differs"));
        }
        program::build(c, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 final layer: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
