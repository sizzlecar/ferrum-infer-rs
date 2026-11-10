//! HF image presentation with explicitly native F32 visual arithmetic.
//! BF16 checkpoint storage is promoted through the existing managed source.
//! This is not a claim of equality to an upstream Torch/Torchvision binary.
use super::{comfy::*, Qwen3VLVisionArchitecture, Qwen3VLVisionGrid};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod pixels;
mod position;
mod processor;
mod weights;
pub use pixels::{Qwen3VLVisionHfImagePlan, QWEN3_VL_HF_IMAGE_PREPROCESS_PROFILE};
pub use position::QWEN3_VL_HF_POSITION_PROFILE;
pub use processor::{Qwen3VLVisionHfProcessor, Qwen3VLVisionHfProcessorLimits};
pub use weights::Qwen3VLVisionHfWeightAdapter;
pub type Qwen3VLVisionHfLimits = Qwen3VLVisionComfyLimits;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Qwen3VLVisionHfConfig {
    base: Qwen3VLVisionComfyConfig,
    processor_sha256: String,
}
impl<'de> Deserialize<'de> for Qwen3VLVisionHfConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            base: Qwen3VLVisionComfyConfig,
            processor_sha256: String,
        }
        let wire = Wire::deserialize(d)?;
        let result = Self {
            base: wire.base,
            processor_sha256: wire.processor_sha256,
        };
        result.check().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}
impl Qwen3VLVisionHfConfig {
    pub fn new(
        architecture: Qwen3VLVisionArchitecture,
        grid: Qwen3VLVisionGrid,
        processor: &Qwen3VLVisionHfProcessor,
    ) -> Result<Self> {
        processor.check_architecture(&architecture)?;
        let c = Self {
            base: Qwen3VLVisionComfyConfig::new(architecture, grid)?,
            processor_sha256: processor.source_sha256().to_owned(),
        };
        c.check()?;
        Ok(c)
    }
    fn check(&self) -> Result<()> {
        ensure!(
            self.architecture().deepstack_layers().is_empty() && self.grid().temporal() == 1,
            "HF single-image main-only vision requires no DeepStack and temporal grid one"
        );
        ensure!(
            self.processor_sha256.len() == 64
                && self
                    .processor_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid immutable processor identity"
        );
        Ok(())
    }
    pub fn architecture(&self) -> &Qwen3VLVisionArchitecture {
        self.base.architecture()
    }
    pub fn grid(&self) -> &Qwen3VLVisionGrid {
        self.base.grid()
    }
    pub fn rows(&self) -> std::result::Result<u64, VNextError> {
        self.base.rows()
    }
    pub fn merged_rows(&self) -> std::result::Result<u64, VNextError> {
        self.base.merged_rows()
    }
    pub fn processor_sha256(&self) -> &str {
        &self.processor_sha256
    }
    pub fn compile_options(&self) -> std::result::Result<ProgramPlanCompileOptions, VNextError> {
        self.base.compile_options()
    }
}

/// Reuses the already executable F32 patch/block/merger topology. The distinct
/// family and profiles bind HF producer contracts without changing Comfy math.
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionHfFamily {
    id: ModelFamilyId,
    inner: Qwen3VLVisionComfyFamily,
}
impl Qwen3VLVisionHfFamily {
    pub const FAMILY_ID: &'static str = "family.qwen3_5.vision_hf_input_f32";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "qwen3_5.vision_hf_input.reference.bf16_storage_f32_compute.native_aa_positions.v1";
    pub const CUDA_PROFILE_ID: &'static str =
        "qwen3_5.vision_hf_input.cuda.bf16_storage_f32_compute.native_aa_positions.v1";
    pub const CUDA_FUSED_F32_PROFILE_ID: &'static str =
        "qwen3_5.vision_hf_input.cuda.bf16_storage_fused_f32_attention.native_aa_positions.v1";
    // Internal ports deliberately retain the reused executable graph IDs.
    pub const PATCHES_ID: &'static str = Qwen3VLVisionComfyFamily::PATCHES_ID;
    pub const POSITIONS_ID: &'static str = Qwen3VLVisionComfyFamily::POSITIONS_ID;
    pub const COSINE_ID: &'static str = Qwen3VLVisionComfyFamily::COSINE_ID;
    pub const SINE_ID: &'static str = Qwen3VLVisionComfyFamily::SINE_ID;
    pub const OUTPUT_ID: &'static str = Qwen3VLVisionComfyFamily::OUTPUT_ID;
    pub fn new(limits: Qwen3VLVisionHfLimits) -> std::result::Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            inner: Qwen3VLVisionComfyFamily::new(limits)?,
        })
    }
    fn check(&self, c: &Qwen3VLVisionHfConfig) -> std::result::Result<(), VNextError> {
        c.check().map_err(invalid)?;
        self.inner.weights(&c.base)?;
        Ok(())
    }
}
impl ModelFamilyProvider for Qwen3VLVisionHfFamily {
    type Config = Qwen3VLVisionHfConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([
            ExternalModelMetadataId::new("metadata.qwen3_5.vision_hf_input_f32")
                .expect("constant metadata"),
        ])
    }
    fn parse_config(
        &self,
        raw: &serde_json::Value,
    ) -> std::result::Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> std::result::Result<(), VNextError> {
        self.check(c)?;
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical HF visual config"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> std::result::Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        ExternalModelMetadataId::new("metadata.qwen3_5.vision_hf_input_f32")
    }
    fn weights(&self, c: &Self::Config) -> std::result::Result<ModelWeights, VNextError> {
        self.check(c)?;
        self.inner.weights(&c.base)
    }
    fn numerical_profiles(
        &self,
        c: &Self::Config,
    ) -> std::result::Result<FamilyNumericalProfiles, VNextError> {
        self.check(c)?;
        let base = self.inner.numerical_profiles(&c.base)?;
        let profiles = [
            (
                Self::REFERENCE_PROFILE_ID,
                Qwen3VLVisionComfyFamily::REFERENCE_PROFILE_ID,
            ),
            (
                Self::CUDA_PROFILE_ID,
                Qwen3VLVisionComfyFamily::CUDA_PROFILE_ID,
            ),
            (
                Self::CUDA_FUSED_F32_PROFILE_ID,
                Qwen3VLVisionComfyFamily::CUDA_FUSED_F32_PROFILE_ID,
            ),
        ]
        .into_iter()
        .map(|(new, old)| {
            let mut p = base
                .resolve(&NumericalProfileId::new(old).map_err(invalid)?)?
                .clone();
            p.id = NumericalProfileId::new(new).map_err(invalid)?;
            p.family_id = self.id.clone();
            Ok(p)
        })
        .collect::<std::result::Result<Vec<_>, VNextError>>()?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> std::result::Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("HF visual numerical profile differs"));
        }
        let old = match p.id.as_str() {
            Self::CUDA_PROFILE_ID => Qwen3VLVisionComfyFamily::CUDA_PROFILE_ID,
            Self::CUDA_FUSED_F32_PROFILE_ID => Qwen3VLVisionComfyFamily::CUDA_FUSED_F32_PROFILE_ID,
            _ => Qwen3VLVisionComfyFamily::REFERENCE_PROFILE_ID,
        };
        let profiles = self.inner.numerical_profiles(&c.base)?;
        let g = self.inner.semantic_program(
            &c.base,
            profiles.resolve(&NumericalProfileId::new(old).map_err(invalid)?)?,
        )?;
        ModelProgram::new(
            self.id.clone(),
            g.inputs().to_vec(),
            g.blocks().to_vec(),
            vec![],
            g.weights().to_vec(),
            g.outputs().to_vec(),
        )?
        .with_input_tensor_specs(g.input_tensor_specs().clone())
    }
    fn semantic_metadata(
        &self,
        _: &Self::Config,
    ) -> std::result::Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: Qwen3VLVisionHfFamily::FAMILY_ID.into(),
        field: "vision_hf_input_f32".into(),
        reason: e.to_string(),
    }
}

#[cfg(test)]
mod tests;
