//! Single-frame VAE decoder prefix: post_quant_conv -> decoder.conv_in ->
//! decoder.mid_block.resnets.0. Input is already denormalized BF16 NCHW.
//!
//! Fixed upstream: vllm-omni c0551d22674d0db81305c800b66ad83035fdc980,
//! autoencoder_kl_qwenimage21.py:166-283,392-463,514-550,898-905,1383-1390.
//! There is no attention, upsampling, final clamp, pixel conversion, or image
//! generation claim. Explicit padding and the independent normalization storage
//! boundaries remain declared; backend profiles do not promise Torch parity.

mod config;
mod graph;
mod weights;
pub use config::QwenImage21VaeStemConfig;
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub use weights::QwenImage21VaeStemWeight;

#[derive(Debug, Clone)]
pub struct QwenImage21VaeStemFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl QwenImage21VaeStemFamily {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.vae.decoder_stem";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "qwen_image21.vae.decoder_stem.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.vae.decoder_stem.cuda.bf16_f32.v1";
    pub const INPUT_ID: &'static str = "value.qwen_image21.vae.decoder_stem.denormalized_latent";
    pub const OUTPUT_IDS: [&'static str; 9] = [
        "value.qwen_image21.vae.decoder_stem.post_quant",
        "value.qwen_image21.vae.decoder_stem.conv_in",
        "value.qwen_image21.vae.decoder_stem.norm1",
        "value.qwen_image21.vae.decoder_stem.silu1",
        "value.qwen_image21.vae.decoder_stem.conv1",
        "value.qwen_image21.vae.decoder_stem.norm2",
        "value.qwen_image21.vae.decoder_stem.silu2",
        "value.qwen_image21.vae.decoder_stem.conv2",
        "value.qwen_image21.vae.decoder_stem.residual",
    ];
    pub const WEIGHT_NAMES: [&'static str; 10] = [
        "post_quant_conv.weight",
        "post_quant_conv.bias",
        "decoder.conv_in.weight",
        "decoder.conv_in.bias",
        "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.mid_block.resnets.0.conv2.bias",
    ];
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.vae.decoder_stem")?,
        })
    }
}
impl ModelFamilyProvider for QwenImage21VaeStemFamily {
    type Config = QwenImage21VaeStemConfig;
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
                "requires exact canonical decoder-prefix geometry",
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
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| invalid("config", e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(weights::schema(config, &self.id)?))
    }
    fn numerical_profiles(&self, _: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let operations = [
            CONV2D_BF16_OPERATION_ID,
            CHANNEL_L2_NORM_BF16_OPERATION_ID,
            SILU_NCHW_BF16_OPERATION_ID,
            PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
        ]
        .into_iter()
        .map(|id| {
            Ok(NumericalOperationContract {
                operation_id: OperationId::new(id)?,
                version: ContractVersion::new(1, 0),
                multiplication_type: if id == PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID {
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
        let mut profiles = Vec::new();
        for id in [Self::REFERENCE_PROFILE_ID, Self::CUDA_PROFILE_ID] {
            profiles.push(NumericalExecutionProfile {
                id: NumericalProfileId::new(id).map_err(|e| invalid("profile", e))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_IDS[8])?,
                boundaries: Self::OUTPUT_IDS
                    .into_iter()
                    .map(|id| Ok((ProgramValueId::new(id)?, ElementType::Bf16)))
                    .collect::<Result<BTreeMap<_, _>, VNextError>>()?,
                states: vec![],
                kv_storage: vec![],
                operations: operations.clone(),
            });
        }
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
            return Err(invalid(
                "profile",
                "undeclared decoder-prefix numerical profile",
            ));
        }
        graph::build(config, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21VaeStemFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}
#[cfg(test)]
mod tests;
