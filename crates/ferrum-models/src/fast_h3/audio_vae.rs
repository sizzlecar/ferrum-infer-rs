//! Original F32 audio VAE components. Stereo channels are independent mono
//! batch items, with shared checked source loading and managed execution.
use anyhow::{ensure, Context, Result};
use ferrum_quantization::{SafetensorsArchive, TensorArchive};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
mod config;
mod encoder;
mod family;
mod program;
mod source;
mod transport;
pub use transport::{FastH3AudioLatentLayout, FastH3AudioLatentUnits, FastH3AudioWaveform};
#[cfg(test)]
mod tests;
pub use config::{FastH3AudioComponentLimits, FastH3AudioDecoderLimits, FastH3AudioVaeConfig};
pub use encoder::{PreparedFastH3AudioEncoderProjection, PreparedFastH3AudioEncoderTrunk};
pub use source::PreparedFastH3AudioDecoder;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DecoderConfig {
    architecture: FastH3AudioVaeConfig,
    requests: u64,
    latent_samples: u64,
    weights: Vec<AudioWeight>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AudioWeight {
    name: String,
    component: WeightComponentSpec,
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 audio VAE: {e}"),
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> Result<T, VNextError> {
    T::try_from(s.into())
}
fn tensor(shape: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: shape,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn weight_id(name: &str) -> Result<WeightId, VNextError> {
    id(format!("weight.fast_h3.audio.{name}"))
}
fn weight_value(name: &str) -> Result<ProgramValueId, VNextError> {
    id(format!("value.fast_h3.audio.weight.{name}"))
}
impl DecoderConfig {
    fn batch(&self) -> Result<u64, VNextError> {
        self.requests
            .checked_mul(2)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("stereo batch overflow"))
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        weight_schema(
            &self.weights,
            "weight-layout.fast_h3.audio_decoder.original_f32.v1",
        )
    }
}
fn weight_schema(weights: &[AudioWeight], layout: &str) -> Result<WeightSchema, VNextError> {
    Ok(WeightSchema {
        format_id: id("weight-format.safetensors.dense")?,
        version: ContractVersion::new(1, 0),
        layout_id: id(layout)?,
        components: weights.iter().map(|w| w.component.clone()).collect(),
        tensors: weights
            .iter()
            .map(|w| {
                Ok(WeightTensorSpec {
                    id: weight_id(&w.name)?,
                    dimensions: w.component.dimensions.clone(),
                    logical_element_type: ElementType::F32,
                    physical_layout: PhysicalWeightLayout::Dense {
                        component_id: w.component.id.clone(),
                    },
                    required: true,
                })
            })
            .collect::<Result<_, VNextError>>()?,
    })
}
