//! FastH3 Ref2VA checkpoint, PDD sampling, and packed temporal geometry.
//!
//! These are checked model contracts, not a registered executable pipeline.
//! They follow FastVideo `866a3cdb91a85aef8561847365fc8564add9e4c8`;
//! P-H3 image weights and their eight-step schedule are not interchangeable.

mod attention_core;
mod attention_input;
mod attention_output;
mod attention_projection;
mod checkpoint;
mod denoising;
mod dit_backbone;
mod dit_block;
mod ffn;
mod initial_media;
mod layout;
mod managed_sampling;
mod modulation;
mod nvfp4_scalars;
mod package;
mod pdd_projection;
mod positions;
mod reference_image;
mod reference_video;
pub use reference_image::{
    FastH3ReferenceImagePlan, PreparedFastH3ReferenceImage, FAST_H3_MAX_REFERENCE_IMAGES,
    FAST_H3_REFERENCE_IMAGE_PROFILE, FAST_H3_REFERENCE_IMAGE_SHORT_EDGE,
};
pub use reference_video::{
    FastH3DecodedVideoGeometry, FastH3ReferenceVideoCanvas, FastH3ReferenceVideoLimits,
    FastH3ReferenceVideoPlan,
};
mod sampling;
mod text_conditioning;
mod time_conditioning;
pub mod vsa;

pub use attention_core::{
    FastH3AttentionCoreConfig, FastH3AttentionCoreFamily, FastH3AttentionCoreLimits,
    FastH3AttentionCoreSource,
};
pub use attention_input::{
    FastH3AttentionInputConfig, FastH3AttentionInputFamily, FastH3AttentionInputLimits,
    FastH3AttentionInputSource,
};
pub use attention_output::{
    FastH3AttentionMerge, FastH3AttentionOutputConfig, FastH3AttentionOutputFamily,
    FastH3AttentionOutputLimits, FastH3AttentionOutputSource,
};
pub use attention_projection::{
    FastH3AttentionProjection, FastH3AttentionProjectionConfig, FastH3AttentionProjectionFamily,
    FastH3AttentionProjectionGroup, FastH3AttentionProjectionLimits,
    FastH3AttentionProjectionSource, FastH3UnitAlphaDerivation,
};
pub use checkpoint::{FastH3Checkpoint, FastH3MetadataLimits, FastH3TransformerConfig};
pub use denoising::{
    FastH3DenoisingBatch, FastH3DenoisingBatchPlan, FastH3DenoisingConfig, FastH3DenoisingFamily,
    FastH3DenoisingLimits, FastH3DenoisingRequest, FastH3DenoisingSource,
    OwnedFastH3DenoisingSource,
};
pub use dit_backbone::{
    FastH3DitBackboneConfig, FastH3DitBackboneFamily, FastH3DitBackboneLimits,
    FastH3DitBackboneSource,
};
pub use dit_block::{
    FastH3DitBlockConfig, FastH3DitBlockFamily, FastH3DitBlockLimits, FastH3DitBlockSource,
};
pub use ffn::{
    FastH3FfnConfig, FastH3FfnFamily, FastH3FfnLimits, FastH3FfnSource, FastH3ScalarDerivation,
};
pub use initial_media::{
    prepare_fast_h3_keyframe_media, prepare_fast_h3_reference_media, FastH3ReferenceLatents,
    FAST_H3_INITIAL_MEDIA_PROFILE,
};
pub use layout::{
    FastH3PackedLayout, FastH3Reference, FastH3Segment, FastH3SegmentKind, FastH3VideoGeometry,
};
pub use managed_sampling::{
    FastH3ManagedContinuation, FastH3PredictionView, FastH3SamplingHostRequirements,
    FastH3VelocityInput, PreparedFastH3Advance,
};
pub use modulation::{
    FastH3ConditioningRole, FastH3ModulationConfig, FastH3ModulationFamily, FastH3ModulationLimits,
    FastH3ModulationSource, FastH3ModulationStage, FastH3TextTagRun, FastH3TextTokenTag,
};
pub use package::{
    FastH3Nvfp4Profile, FastH3Nvfp4WeightOperands, FastH3TransformerPackage, FastH3Weight,
    FastH3WeightLimits, FastH3WeightStorage,
};
pub use pdd_projection::{
    FastH3OutputModality, FastH3PddProjectionConfig, FastH3PddProjectionFamily,
    FastH3PddProjectionLimits, FastH3PddProjectionSource,
};
pub use positions::{FastH3PositionLimits, FastH3PositionPlan};
pub use sampling::{FastH3ModalityStep, FastH3PddPlan, FastH3PddSchedule, FastH3PddStep};
pub use text_conditioning::{
    FastH3ImagePresentation, FastH3MediaPresentation, FastH3PresentationLimits,
    FastH3PresentationReference, FastH3TextHostFootprint, FastH3VisionFeatureView,
};
pub use time_conditioning::{
    FastH3ConditioningClock, FastH3ConditioningSchedule, FastH3ConditioningTable,
    FastH3TimeConditioningConfig, FastH3TimeConditioningFamily, FastH3TimeConditioningLimits,
    FastH3TimeConditioningSource,
};

pub const FAST_H3_INFERENCE_SCHEMA: &str = "fasth3-inference-contract-v1";
pub const FAST_H3_GRID_MAX_T: f64 = 0.999;

/// Original CUDA scalar div_(255) multiplies by the rounded F32 reciprocal.
/// Both visual-reference encoders share these separate F32 rounding boundaries.
fn normalize_reference_rgb(pixel: u8, channel: usize) -> f32 {
    let unit = f32::from(pixel) * (1f64 / 255.) as f32;
    let centered = unit - [0.485f32, 0.456, 0.406][channel];
    centered / [0.229f32, 0.224, 0.225][channel]
}

#[cfg(test)]
mod tests;

mod qk_norm_rope;
pub use qk_norm_rope::{
    FastH3QkNormRopeConfig, FastH3QkNormRopeFamily, FastH3QkNormRopeLimits, FastH3QkNormRopeSource,
};

mod output_conditioning;
pub use output_conditioning::{
    FastH3OutputConditioningLimits, FastH3OutputConditioningSource, FastH3OutputConditioningTable,
    FastH3OutputNormConfig, FastH3OutputNormFamily, FastH3OutputTimeConfig, FastH3OutputTimeFamily,
};

mod final_output;
pub use final_output::{
    FastH3FinalOutputConfig, FastH3FinalOutputFamily, FastH3FinalOutputLimits,
    FastH3FinalOutputSource, FastH3PredictionInput, OwnedFastH3FinalOutputSource,
};

mod input_projection;
pub use input_projection::{
    FastH3InputLimits, FastH3InputPackingConfig, FastH3InputPackingFamily,
    FastH3InputProjectionConfig, FastH3InputProjectionFamily, FastH3InputProjectionPhase,
    FastH3InputProjectionSource,
};

mod text_refiner;
pub use text_refiner::{
    FastH3TextRefinerConfig, FastH3TextRefinerFamily, FastH3TextRefinerLimits,
    FastH3TextRefinerSource,
};

pub mod audio_vae;
pub mod video_vae;
