//! One complete curve-form MiniMax-H3 DiT block, including its stored F32
//! AdaLN projection. This is a real TensorComponent, not an image pipeline.
use super::{
    PH3ImageCurveTimeEmbedding, PH3ImageModality, PH3ImageRequestLayout, PH3ImageTransformerConfig,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
mod config;
mod program;
mod projection_storage;
mod weights;
pub use checkpoint::PH3ImageCurveBlockCheckpoint;
pub use config::{PH3ImageCurveBlockConfig, PH3ImageCurveBlockRequest, PH3ImageCurveBlockRowRun};
pub use projection_storage::{PH3ImageBlockProjectionStorage, PH3ImageProjectionStorage};

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageCurveBlockLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
#[derive(Clone, Debug)]
pub struct PH3ImageCurveBlockFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageCurveBlockLimits,
}
impl PH3ImageCurveBlockFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.curve_block";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.curve_block.cuda.precise_w4_swiglu_f32.native_attention.v1";
    pub const CUDA_MIXED_PROFILE_ID: &'static str =
        "p_h3_image.curve_block.cuda.precise_mixed_convrot_swiglu_f32.native_attention.v1";
    pub fn cuda_profile_id(config: &PH3ImageCurveBlockConfig) -> &'static str {
        if config.projection_storage().all_w4() {
            Self::CUDA_PROFILE_ID
        } else {
            Self::CUDA_MIXED_PROFILE_ID
        }
    }
    pub const INPUT_ID: &'static str = "value.p_h3_image.curve_block.input";
    pub const BASIS_ID: &'static str = "value.p_h3_image.curve_block.basis";
    pub const COSINE_ID: &'static str = "value.p_h3_image.curve_block.cosine";
    pub const SINE_ID: &'static str = "value.p_h3_image.curve_block.sine";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.curve_block.output";
    pub fn new(limits: PH3ImageCurveBlockLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("block capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.p_h3_image.curve_block.comfy_w4")?,
            limits,
        })
    }
    fn check(&self, c: &PH3ImageCurveBlockConfig) -> Result<(), VNextError> {
        c.validate()?;
        if c.metadata_bytes()? > self.limits.max_metadata_bytes
            || c.estimated_plan_nodes()? > self.limits.max_plan_nodes
        {
            return Err(invalid("block exceeds declared metadata/plan capacity"));
        }
        weights::schema(c, self.limits.max_stored_bytes)?;
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageCurveBlockFamily {
    type Config = PH3ImageCurveBlockConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("curve block config must be canonical"));
        }
        self.check(c)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.check(c)?;
        Ok(ModelWeights::Stored(weights::schema(
            c,
            self.limits.max_stored_bytes,
        )?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.check(c)?;
        let body = program::build(self, c)?;
        let used_operations: BTreeSet<_> = body
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .map(|node| node.operation_id.as_str())
            .collect();
        let mut profiles = Vec::new();
        for name in [Self::cuda_profile_id(c)] {
            let mut operations = Vec::new();
            for &operation in &used_operations {
                let staged = match operation {
                    COMFY_CONVROT_W4A8_BF16_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_w4a8_bf16())
                    }
                    COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w4a8_bf16())
                    }
                    COMFY_CONVROT_W8A8_BF16_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_w8a8_bf16())
                    }
                    COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w8a8_bf16())
                    }
                    _ => None,
                };
                let arithmetic = !matches!(
                    operation,
                    CAST_F32_TO_BF16_OPERATION_ID
                        | RESHAPE_COPY_BF16_OPERATION_ID
                        | ROW_GATHER_BF16_OPERATION_ID
                        | BATCHED_ROW_CONCAT_BF16_OPERATION_ID
                ) && staged.is_none();
                operations.push(NumericalOperationContract {
                    operation_id: OperationId::new(operation)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: arithmetic.then_some(ElementType::F32),
                    accumulation_type: arithmetic.then_some(ElementType::F32),
                    staged_arithmetic: staged,
                    composite_arithmetic: None,
                });
            }
            profiles.push(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: program::boundaries(&body)?,
                states: vec![],
                kv_storage: vec![],
                operations,
            });
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![NumericalProfileId::new(Self::cuda_profile_id(c)).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        self.check(c)?;
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown curve block numerical profile"));
        }
        program::build(self, c)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 curve block: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
