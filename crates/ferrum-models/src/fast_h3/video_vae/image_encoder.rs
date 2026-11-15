//! Literal T=1 keyframe encoder. Reuses the complete causal CNN implementation;
//! only the declared checkpoint dialect differs from its Comfy importer.
//! Pixel normalization, posterior sampling and spatial tiling are caller-owned.
use super::FastH3VideoVaeConfig;
use crate::p_h3_image_vae::encoder::{
    CausalVaeEncoderSourceLayout, PH3ImageVaeEncoderCheckpoint, PH3ImageVaeEncoderConfig,
    PH3ImageVaeEncoderFamily, PH3ImageVaeEncoderLimits,
};
use anyhow::{ensure, Result};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct FastH3ImageEncoderLimits {
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_plan_nodes: usize,
    pub maximum_input_bytes: u64,
    pub maximum_output_bytes: u64,
}

/// Original F32 causal weights are retained; KT>1 kernels upload only the last
/// temporal plane because all earlier T=1 input planes are causally zero.
pub struct PreparedFastH3ImageEncoder {
    config: PH3ImageVaeEncoderConfig,
    source: Arc<SafetensorsArchive>,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
}

impl PreparedFastH3ImageEncoder {
    pub const INPUT_ID: &'static str = PH3ImageVaeEncoderFamily::INPUT_ID;
    pub const OUTPUT_ID: &'static str = PH3ImageVaeEncoderFamily::OUTPUT_ID;

    /// Input is normalized F32 NCHW; output is F32 N(2C)HW posterior moments.
    /// This entrypoint never accepts multi-frame inputs or a substituted VAE class.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        architecture: &FastH3VideoVaeConfig,
        source: Arc<SafetensorsArchive>,
        batch: u64,
        height: u64,
        width: u64,
        limits: FastH3ImageEncoderLimits,
        profile: FastH3ImageEncoderProfile,
        host: &HostMemoryAccount,
    ) -> Result<Self> {
        ensure!(
            limits.maximum_input_bytes > 0 && limits.maximum_output_bytes > 0,
            "image encoder activation capacities must be positive"
        );
        let config = PH3ImageVaeEncoderConfig::new(
            architecture.causal_encoder_architecture()?,
            batch,
            height,
            width,
        )?
        .with_source_layout(CausalVaeEncoderSourceLayout::Diffusers);
        let resolve = |s: &ProgramTensorSpec| {
            ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
        };
        let input = resolve(&config.input_tensor_specs()?[&ProgramValueId::new(Self::INPUT_ID)?])?;
        let output = resolve(&config.output_tensor_spec()?)?;
        ensure!(
            input.minimum_storage_bytes()? <= limits.maximum_input_bytes
                && output.minimum_storage_bytes()? <= limits.maximum_output_bytes,
            "image encoder input or posterior exceeds declared capacity"
        );
        let internal_limits = PH3ImageVaeEncoderLimits {
            max_metadata_bytes: limits.maximum_metadata_bytes,
            max_stored_bytes: limits.maximum_stored_bytes,
            max_plan_nodes: limits.maximum_plan_nodes,
        };
        // One complete original-source validation, before any static upload.
        PH3ImageVaeEncoderCheckpoint::new(&config, &source, host.clone(), internal_limits)?;
        let profile = match profile {
            FastH3ImageEncoderProfile::Reference => PH3ImageVaeEncoderFamily::REFERENCE_PROFILE_ID,
            FastH3ImageEncoderProfile::Cuda => PH3ImageVaeEncoderFamily::CUDA_PROFILE_ID,
        };
        let profile = NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?;
        let family = TypedFamilyRegistration::new(PH3ImageVaeEncoderFamily::new(internal_limits)?)
            .prepare_with_profile(&serde_json::to_value(&config)?, &profile)?;
        let options = config.compile_options()?;
        Ok(Self {
            config,
            source,
            family,
            options,
            input,
            output,
        })
    }
    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    pub fn input_spec(&self) -> &ResolvedTensorSpec {
        &self.input
    }
    pub fn output_spec(&self) -> &ResolvedTensorSpec {
        &self.output
    }
    pub fn weight_source(&self, host: HostMemoryAccount) -> FastH3ImageEncoderSource<'_> {
        FastH3ImageEncoderSource {
            checkpoint: PH3ImageVaeEncoderCheckpoint::view_validated(
                &self.config,
                &self.source,
                host,
            ),
        }
    }
}

/// Explicit arithmetic policy, independent of model names and checkpoint dialect.
#[derive(Clone, Copy, Debug)]
pub enum FastH3ImageEncoderProfile {
    Reference,
    Cuda,
}

pub struct FastH3ImageEncoderSource<'a> {
    checkpoint: PH3ImageVaeEncoderCheckpoint<'a>,
}
impl WeightComponentSource for FastH3ImageEncoderSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.checkpoint.component(c)
    }
}
