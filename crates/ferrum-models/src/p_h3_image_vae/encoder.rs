//! Complete single-frame F32 CNN encoder through quant_conv. Pixel domain,
//! tiling/stitching, posterior noise and latent normalization are explicit
//! caller-owned boundaries. Every declared level/block is executed on device.
use super::encoder_vertical::PH3ImageVaeEncoderArchitecture;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
pub(super) mod checkpoint_source;
mod config;
mod program;
mod weights;
pub(crate) use checkpoint::PH3ImageVaeEncoderCheckpoint;
pub(crate) use config::PH3ImageVaeEncoderConfig;
#[derive(Clone, Copy, Debug)]
pub(crate) struct PH3ImageVaeEncoderLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct PH3ImageVaeEncoderFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageVaeEncoderLimits,
}
impl PH3ImageVaeEncoderFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.vae.encoder";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "p_h3_image.vae.encoder.reference.full_cnn.f32.last_tap.reflect.gn_silu.v1";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.vae.encoder.cuda.full_cnn.f32_fma.last_tap.reflect.gn_silu.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.vae.encoder.input";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.vae.encoder.moments";
    pub fn new(limits: PH3ImageVaeEncoderLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new(
                "metadata.p_h3_image.vae.encoder.full_literal_t1",
            )?,
            limits,
        })
    }
    fn check(&self, c: &PH3ImageVaeEncoderConfig) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_vec(c)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > self.limits.max_metadata_bytes
        {
            return Err(invalid("config exceeds metadata capacity"));
        }
        if c.plan_nodes()? > self.limits.max_plan_nodes {
            return Err(invalid(
                "full CNN exceeds plan-node capacity before expansion",
            ));
        }
        weights::schema(c, self.limits.max_stored_bytes, &self.id)?;
        program::build(c, &self.id)?;
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageVaeEncoderFamily {
    type Config = PH3ImageVaeEncoderConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(r.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(r, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.check(c)?;
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *r {
            return Err(invalid("canonical full CNN config required"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(r, c)?;
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
            .map(|kind| {
                Ok(NumericalOperationContract {
                    operation_id: OperationId::new(kind.operation_id())?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (kind != F32CnnOp::ResidualAdd)
                        .then_some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
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
            return Err(invalid("full CNN profile differs"));
        }
        program::build(c, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 VAE full encoder: {}", reason.into()),
    }
}
#[cfg(any(test, feature = "test-support"))]
pub(in crate::p_h3_image_vae) mod test_fixture;
#[cfg(test)]
pub(super) mod tests;
