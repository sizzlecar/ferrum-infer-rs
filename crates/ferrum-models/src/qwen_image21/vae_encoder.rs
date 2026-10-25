//! Complete single-frame Qwen-Image 2.1 VAE encoder and posterior mode.
//!
//! Source: vllm-omni c0551d22674d0db81305c800b66ad83035fdc980,
//! `autoencoder_kl_qwenimage21.py`: AvgDown3D, ResidualDownBlock,
//! Encoder3d and `_encode`. This path has no feature cache, so temporal
//! convolutions are not executed. Temporal zero padding in the averaging
//! shortcuts remains observable even for one frame.
//!
//! The caller supplies already normalized RGBA BF16 NCHW pixels. The graph
//! emits the first half of quant_conv's channels (posterior mean), without
//! sampling or clipping. The caller separately applies the checkpoint's
//! BF16 subtraction/division by latent mean/std before passing these latents
//! to the denoiser. No video, tiling or Torch bitwise equivalence is implied.

mod config;
mod graph;
mod weights;
pub use config::QwenImage21VaeEncoderConfig;
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub use weights::QwenImage21VaeEncoderWeight;

#[derive(Debug, Clone)]
pub struct QwenImage21VaeEncoderFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl QwenImage21VaeEncoderFamily {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.vae.encoder";
    pub const REFERENCE_PROFILE_ID: &'static str = "qwen_image21.vae.encoder.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.vae.encoder.cuda.bf16_f32.v1";
    pub const METAL_PROFILE_ID: &'static str = "qwen_image21.vae.encoder.metal.bf16_f32.v1";
    pub const INPUT_ID: &'static str = "value.qwen_image21.vae.encoder.normalized_rgba";
    pub const OUTPUT_ID: &'static str = "value.qwen_image21.vae.encoder.posterior_mean";
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.vae.encoder")?,
        })
    }
}
impl ModelFamilyProvider for QwenImage21VaeEncoderFamily {
    type Config = QwenImage21VaeEncoderConfig;
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
                "requires canonical single-frame encoder configuration",
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
        Ok(ModelWeights::Stored(weights::schema(
            &config.expected_weights()?,
            &self.id,
        )?))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let graph = graph::build(config)?;
        let operations = graph
            .nodes
            .iter()
            .map(|n| n.operation_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|operation_id| {
                let transport = operation_id.as_str() == SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID;
                let add = operation_id.as_str() == PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID;
                NumericalOperationContract {
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (!transport && !add).then_some(ElementType::F32),
                    accumulation_type: (!transport).then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect::<Vec<_>>();
        let boundaries = graph
            .specs
            .into_keys()
            .map(|id| (id, ElementType::Bf16))
            .collect::<BTreeMap<_, _>>();
        let profiles = [
            Self::REFERENCE_PROFILE_ID,
            Self::CUDA_PROFILE_ID,
            Self::METAL_PROFILE_ID,
        ]
        .into_iter()
        .map(|id| {
            Ok(NumericalExecutionProfile {
                id: NumericalProfileId::new(id).map_err(|e| invalid("profile", e))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: boundaries.clone(),
                operations: operations.clone(),
                kv_storage: vec![],
                states: vec![],
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
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
            return Err(invalid("profile", "undeclared encoder profile"));
        }
        let graph = graph::build(config)?;
        let references = graph
            .weights
            .iter()
            .map(|w| {
                Ok(WeightReference {
                    weight_id: w.weight_id()?,
                    value_id: w.value_id()?,
                    tensor: spec(w.dimensions()),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        ModelProgram::new(
            self.id.clone(),
            vec![ProgramValueId::new(Self::INPUT_ID)?],
            vec![ProgramBlock {
                id: "block.qwen_image21.vae.encoder".into(),
                nodes: graph.nodes,
            }],
            vec![],
            references,
            vec![ProgramValueId::new(Self::OUTPUT_ID)?],
        )?
        .with_input_tensor_specs(config.input_tensor_specs()?)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn spec(dimensions: &[u64]) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: dimensions.to_vec(),
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn tensor(dimensions: &[u64]) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(
        dimensions.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21VaeEncoderFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}
#[cfg(test)]
mod tests;
