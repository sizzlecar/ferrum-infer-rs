//! Stored BF16 conditioning and W8 token refinement from exact ComfyUI
//! `56f179f6c50dad0c4c860ae4da427415c8e86415`, minimax/model.py:250-280.
//! This private component is not complete image-model support. Its precise
//! numerical profile does not certify a Comfy/Torch binary or serving SLO.
use super::PH3ImageTransformerConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
mod config;
mod program;
mod weights;
pub(super) use checkpoint::PH3ImageTokenRefinerCheckpoint;
pub(super) use config::PH3ImageTokenRefinerConfig;

#[derive(Clone, Copy, Debug)]
pub(super) struct PH3ImageTokenRefinerLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
#[derive(Clone, Debug)]
pub(super) struct PH3ImageTokenRefinerFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageTokenRefinerLimits,
}
impl PH3ImageTokenRefinerFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.token_refiner";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.token_refiner.cuda.precise_w8_swiglu_f32.native_attention.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.token_refiner.input";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.token_refiner.output";
    pub fn new(limits: PH3ImageTokenRefinerLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.p_h3_image.token_refiner.comfy_w8")?,
            limits,
        })
    }
    fn check(&self, c: &PH3ImageTokenRefinerConfig) -> Result<(), VNextError> {
        c.validate()?;
        if c.metadata_bytes()? > self.limits.max_metadata_bytes
            || c.plan_nodes()? > self.limits.max_plan_nodes
        {
            return Err(invalid(
                "configuration exceeds declared metadata/plan capacity",
            ));
        }
        weights::schema(c, self.limits.max_stored_bytes)?;
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageTokenRefinerFamily {
    type Config = PH3ImageTokenRefinerConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *r {
            return Err(invalid("configuration must be canonical"));
        }
        self.check(c)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(r, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(r.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(r, &c)?;
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
        let operations = body
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.as_str())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|operation| {
                let staged = match operation {
                    COMFY_CONVROT_W8A8_BF16_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_w8a8_bf16())
                    }
                    COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w8a8_bf16())
                    }
                    _ => None,
                };
                let arithmetic = staged.is_none()
                    && !matches!(
                        operation,
                        RESHAPE_COPY_BF16_OPERATION_ID | ROW_GATHER_BF16_OPERATION_ID
                    );
                Ok(NumericalOperationContract {
                    operation_id: OperationId::new(operation)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: arithmetic.then_some(ElementType::F32),
                    accumulation_type: arithmetic.then_some(ElementType::F32),
                    staged_arithmetic: staged,
                    composite_arithmetic: None,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let boundaries = body
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .cloned()
            .map(|v| (v, ElementType::Bf16))
            .collect();
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: NumericalProfileId::new(Self::CUDA_PROFILE_ID).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations,
            }],
            vec![NumericalProfileId::new(Self::CUDA_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        self.check(c)?;
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown numerical profile"));
        }
        program::build(self, c)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 token refiner: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
