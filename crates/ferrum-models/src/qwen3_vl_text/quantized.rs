//! Canonical Comfy decoder stack with genuine F32 activations.
//!
//! Embeddings and the already interleaved mRoPE cosine/sine are borrowed caller
//! inputs. Seven independent projections per decoder use their own checkpoint
//! metadata to select W4A8 or W8A8. Norm vectors retain their original BF16
//! checkpoint identity and are exactly promoted during ordinary weight loading.
//! This component does not round the decoder activations through BF16, merge
//! gate/up, or replace online A8 execution with dequantized dense multiplication.
use super::{Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as DenseFamily};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod checkpoint;
mod config;
mod program;
mod visual;
mod weights;
pub use checkpoint::Qwen3VLTextComfyCheckpoint;
pub use config::{Qwen3VLTextComfyConfig, Qwen3VLTextComfyProjectionQuantization};
pub use visual::Qwen3VLTextComfyInputMode;

#[derive(Clone, Copy, Debug)]
pub struct Qwen3VLTextComfyLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}

#[derive(Clone, Debug)]
pub struct Qwen3VLTextComfyFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: Qwen3VLTextComfyLimits,
}
impl Qwen3VLTextComfyFamily {
    pub const FAMILY_ID: &'static str = "family.qwen3_vl.text_encoder.comfy_f32";
    /// Actual canonical A8 CUDA execution. Scalar CPU oracles are separate;
    /// there is currently no executable Reference provider for raw W4/W8.
    pub const CUDA_PROFILE_ID: &'static str =
        "qwen3vl.text.comfy.cuda.precise_f32_a8.online_attention.v1";
    /// Adds exact F32 main-embedding replacement and three selected-row F32
    /// residual additions. This is a component contract, not a qualification
    /// against a particular upstream CUDA binary.
    pub const CUDA_VISION_PROFILE_ID: &'static str =
        "qwen3vl.text.comfy.cuda.visual_f32_a8.online_attention.v1";
    pub const EMBEDDINGS_ID: &'static str = DenseFamily::EMBEDDINGS_ID;
    pub const VISUAL_EMBEDDINGS_ID: &'static str = "value.qwen3_vl.text.comfy.visual_embeddings";
    pub const DEEPSTACK_IDS: [&'static str; 3] = DenseFamily::DEEPSTACK_IDS;
    pub const COSINE_ID: &'static str = DenseFamily::COSINE_ID;
    pub const SINE_ID: &'static str = DenseFamily::SINE_ID;
    pub const OUTPUT_ID: &'static str = DenseFamily::OUTPUT_ID;
    pub fn cuda_profile_id(c: &Qwen3VLTextComfyConfig) -> &'static str {
        if c.visual_inputs().is_some() {
            Self::CUDA_VISION_PROFILE_ID
        } else {
            Self::CUDA_PROFILE_ID
        }
    }
    pub fn hidden_state_output_id(index: u64) -> Result<ProgramValueId, VNextError> {
        DenseFamily::hidden_state_output_id(index)
    }
    pub fn new(limits: Qwen3VLTextComfyLimits) -> Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_stored_bytes == 0
            || limits.max_plan_nodes == 0
        {
            return Err(invalid("limits", "capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen3_vl.text_encoder.comfy_f32")?,
            limits,
        })
    }
    fn check(&self, c: &Qwen3VLTextComfyConfig) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_vec(c)
            .map_err(|e| invalid("config", e.to_string()))?
            .len()
            > self.limits.max_metadata_bytes
        {
            return Err(invalid("config", "metadata exceeds declared capacity"));
        }
        if c.estimated_plan_nodes()? > self.limits.max_plan_nodes {
            return Err(invalid("config", "decoder plan exceeds declared capacity"));
        }
        weights::schema(c, self.limits.max_stored_bytes, &self.id)?;
        Ok(())
    }
}
impl ModelFamilyProvider for Qwen3VLTextComfyFamily {
    type Config = Qwen3VLTextComfyConfig;
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
                "requires the canonical checked configuration",
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
                let staged = match operation_id.as_str() {
                    COMFY_CONVROT_W4A8_F32_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_w4a8_f32())
                    }
                    COMFY_CONVROT_W8A8_F32_OPERATION_ID => {
                        Some(StagedNumericalArithmetic::comfy_convrot_w8a8_f32())
                    }
                    _ => None,
                };
                NumericalOperationContract {
                    multiplication_type: (staged.is_none()
                        && !matches!(
                            operation_id.as_str(),
                            PLAIN_RESIDUAL_ADD_F32_OPERATION_ID
                                | ROW_SCATTER_REPLACE_F32_OPERATION_ID
                                | ROW_SCATTER_ADD_F32_OPERATION_ID
                        ))
                    .then_some(ElementType::F32),
                    accumulation_type: (staged.is_none()
                        && operation_id.as_str() != ROW_SCATTER_REPLACE_F32_OPERATION_ID)
                        .then_some(ElementType::F32),
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    staged_arithmetic: staged,
                    composite_arithmetic: None,
                }
            })
            .collect::<Vec<_>>();
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
            vec![profile(Self::cuda_profile_id(c))?],
            vec![NumericalProfileId::new(Self::cuda_profile_id(c))
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
                "profile boundaries or arithmetic differ",
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
        family_id: Qwen3VLTextComfyFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
