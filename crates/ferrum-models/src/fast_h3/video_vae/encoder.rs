//! Complete causal NCTHW encoder, retaining every original F32 temporal tap.
//! Clip padding, spatial tiles, posterior sampling and normalization belong to
//! the caller. This component emits original posterior moments through quant_conv.
use super::FastH3VideoVaeConfig;
use crate::p_h3_image_vae::PH3ImageVaeEncoderArchitecture;
use anyhow::{ensure, Result};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
mod family;
mod program;
#[cfg(test)]
pub(super) mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3VideoEncoderGeometry {
    pub batch: u64,
    pub frames: u64,
    pub height: u64,
    pub width: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct FastH3VideoEncoderLimits {
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_plan_nodes: usize,
    /// Bound each declared activation before graph expansion/static upload.
    /// Actual overlapping activations and workspace still require runtime admission.
    pub maximum_activation_bytes: u64,
}
#[derive(Clone, Copy, Debug)]
pub enum FastH3VideoEncoderProfile {
    Reference,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    architecture: PH3ImageVaeEncoderArchitecture,
    geometry: FastH3VideoEncoderGeometry,
}
impl Config {
    fn validate(&self, limits: FastH3VideoEncoderLimits) -> Result<(), VNextError> {
        self.architecture.validate()?;
        if limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
            || limits.maximum_plan_nodes == 0
            || limits.maximum_activation_bytes == 0
        {
            return Err(invalid("positive component capacities required"));
        }
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("configuration exceeds metadata capacity"));
        }
        let g = self.geometry;
        if g.batch == 0 || g.frames == 0 || g.height < 2 || g.width < 2 {
            return Err(invalid(
                "positive batch/frames and reflect-valid spatial input required",
            ));
        }
        let a = &self.architecture;
        let blocks = usize::try_from(a.layers_per_block()).map_err(invalid)?;
        let mut nodes = 4usize;
        for (level, &channels) in a.channels().iter().enumerate() {
            nodes = nodes
                .checked_add(
                    blocks
                        .checked_mul(5)
                        .ok_or_else(|| invalid("node count overflows"))?,
                )
                .and_then(|v| {
                    v.checked_add(usize::from(
                        level > 0 && a.channels()[level - 1] != channels,
                    ))
                })
                .and_then(|v| {
                    v.checked_add(usize::from(
                        a.spatial_downsample_factors()[level] > 1
                            || a.temporal_downsample_factors()[level] > 1,
                    ))
                })
                .filter(|n| *n <= limits.maximum_plan_nodes)
                .ok_or_else(|| invalid("complete encoder exceeds plan-node capacity"))?;
        }
        Ok(())
    }
}

pub struct PreparedFastH3VideoEncoder {
    source: Arc<SafetensorsArchive>,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
}
impl PreparedFastH3VideoEncoder {
    pub const INPUT_ID: &'static str = "value.fast_h3.video_encoder.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.video_encoder.moments";

    pub fn prepare(
        architecture: &FastH3VideoVaeConfig,
        source: Arc<SafetensorsArchive>,
        geometry: FastH3VideoEncoderGeometry,
        limits: FastH3VideoEncoderLimits,
        profile: FastH3VideoEncoderProfile,
    ) -> Result<Self> {
        ensure!(
            geometry.frames <= architecture.clip_length() as u64,
            "video encoder component consumes at most one original temporal clip"
        );
        let config = Config {
            architecture: architecture.causal_encoder_architecture()?,
            geometry,
        };
        let graph = program::build(&config, limits)?;
        for c in &graph.schema.components {
            let t = source.tensor(&c.external_names[0])?;
            ensure!(
                t.element_type() == Some(ElementType::F32) && t.shape() == c.dimensions,
                "video encoder requires original F32 shape for {}",
                c.external_names[0]
            );
            ensure!(
                t.bytes()
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "video encoder source {} is nonfinite",
                c.external_names[0]
            );
        }
        let profile = match profile {
            FastH3VideoEncoderProfile::Reference => family::Family::REFERENCE_PROFILE,
            FastH3VideoEncoderProfile::Cuda => family::Family::CUDA_PROFILE,
        };
        let family = TypedFamilyRegistration::new(family::Family::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(&config)?,
                &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
            )?;
        ensure!(
            *family.program() == graph.program,
            "prepared causal encoder graph changed"
        );
        let input = resolved(&graph.specs[&id(Self::INPUT_ID)?])?;
        let output = resolved(&graph.specs[&id(Self::OUTPUT_ID)?])?;
        Ok(Self {
            source,
            family,
            options: ProgramPlanCompileOptions::new(graph.specs)?,
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
    /// Original archive ownership covers full OICTHW tensors; no temporal slice
    /// or derived Host weight allocation is introduced for video execution.
    pub fn weight_source(&self) -> &dyn WeightComponentSource {
        self.source.as_ref()
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(name: &str) -> Result<T, VNextError> {
    T::try_from(name.to_owned())
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn resolved(t: &ProgramTensorSpec) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(t.dimensions.clone(), t.element_type, t.layout.clone())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 causal video encoder: {e}"),
    }
}
