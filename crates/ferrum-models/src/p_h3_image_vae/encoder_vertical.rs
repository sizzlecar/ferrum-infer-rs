//! A real single-frame F32 encoder vertical: stem, selected width-changing
//! ResnetBlock3D and that level's stride-2 downsample. This is a qualification
//! component extracted from the declared full CNN architecture, not a complete
//! encoder prefix: intervening earlier-level blocks are intentionally absent.
//! Literal T=1 causal semantics select the last original temporal weight tap.
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
mod config;
mod program;
mod weights;
pub(crate) use checkpoint::PH3ImageVaeEncoderVerticalCheckpoint;
pub(crate) use config::{PH3ImageVaeEncoderArchitecture, PH3ImageVaeEncoderVerticalConfig};
#[derive(Clone, Copy, Debug)]
pub(crate) struct PH3ImageVaeEncoderVerticalLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct PH3ImageVaeEncoderVerticalFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageVaeEncoderVerticalLimits,
}
impl PH3ImageVaeEncoderVerticalFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.vae.encoder_vertical";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "p_h3_image.vae.encoder_vertical.reference.f32.last_tap.reflect.gn_silu.v1";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.vae.encoder_vertical.cuda.f32_fma.last_tap.reflect.gn_silu.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.vae.encoder_vertical.input";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.vae.encoder_vertical.output";
    pub fn new(limits: PH3ImageVaeEncoderVerticalLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new(
                "metadata.p_h3_image.vae.encoder_vertical.literal_t1",
            )?,
            limits,
        })
    }
    fn check(&self, c: &PH3ImageVaeEncoderVerticalConfig) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_vec(c)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > self.limits.max_metadata_bytes
        {
            return Err(invalid("config exceeds metadata capacity"));
        }
        weights::schema(c, self.limits.max_stored_bytes, &self.id)?;
        if program::build(c, &self.id)?
            .blocks()
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.nodes.len()))
            .is_none_or(|n| n > self.limits.max_plan_nodes)
        {
            return Err(invalid("vertical exceeds plan-node capacity"));
        }
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageVaeEncoderVerticalFamily {
    type Config = PH3ImageVaeEncoderVerticalConfig;
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
            return Err(invalid("canonical config required"));
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
            .map(|v| (v.clone(), ElementType::F32))
            .collect();
        let operations = F32CnnOp::ALL
            .into_iter()
            .map(|kind| NumericalOperationContract {
                operation_id: OperationId::new(kind.operation_id()).unwrap(),
                version: ContractVersion::new(1, 0),
                multiplication_type: (kind != F32CnnOp::ResidualAdd).then_some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            })
            .collect::<Vec<_>>();
        let profile = |name| -> Result<NumericalExecutionProfile, VNextError> {
            Ok(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(invalid)?,
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
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("vertical profile differs"));
        }
        program::build(c, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 VAE encoder vertical: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
