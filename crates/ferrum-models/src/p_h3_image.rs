//! P-H3-Image metadata, request-local sampling and executable text connector.
//!
//! The stored F32 projection and managed adapters implement the connector;
//! the remaining checked contracts do not expose a complete image pipeline.
//! MiniMax-H3 uses a data-ward velocity and `t = 1 - sigma`.

mod block;
mod config;
mod curve_stack;
mod curve_time;
mod dit_forward;
mod final_layer;
mod image_input;
mod layout;
mod pipeline_adapter;
mod prompt_expansion;
mod rotary;
mod sampling_schedule;
#[cfg(any(test, feature = "test-support"))]
pub(crate) mod test_support;
mod text_conditioning;
mod text_connector;
mod text_connector_managed;
mod text_connector_projection;
mod token_refiner;
mod weights;

pub use block::{
    PH3ImageBlockProjectionStorage, PH3ImageCurveBlockCheckpoint, PH3ImageCurveBlockConfig,
    PH3ImageCurveBlockFamily, PH3ImageCurveBlockLimits, PH3ImageCurveBlockRequest,
    PH3ImageCurveBlockRowRun, PH3ImageProjectionStorage,
};
pub use config::{PH3ImageTextConnectorConfig, PH3ImageTransformerConfig};
pub use curve_time::{PH3ImageCurveLerpArithmetic, PH3ImageCurveTimeEmbedding};
pub use final_layer::{
    PH3ImageFinalLayerCheckpoint, PH3ImageFinalLayerConfig, PH3ImageFinalLayerFamily,
    PH3ImageFinalLayerLimits, PH3ImageFinalLayerRequest, PH3ImageFinalLayerTimeRun,
};
pub use image_input::{
    PH3ImageEditPixelHostRequirements, PH3ImageEditPixelPlan, PH3ImageEditPixelPolicy,
    PH3ImageEditPixels, PH3_IMAGE_EDIT_PIXEL_PROFILE,
};
pub use layout::{
    PH3ImageModality, PH3ImagePackedBatchLayout, PH3ImageReferenceImage, PH3ImageRequestLayout,
    PH3ImageRowKind,
};
pub use pipeline_adapter::{
    PH3ImageConditioningKey, PH3ImageDenoiserAdapter, PH3ImageDenoiserAttentionPolicy,
    PH3ImageDenoiserFamilyConfig, PH3ImageDenoiserFamilyProvider, PH3ImageDenoiserLimits,
    PH3ImageDenoiserPhase, PH3ImageDenoiserReference, PH3ImageDenoiserRequest,
    PH3ImageDenoiserSample, PH3ImageRefinedTextCache, PH3ImageRefinementCacheHostRequirement,
    PH3ImageRefinementDestination, PH3ImageRefinementHostRequirements, PH3ImageRefinementSession,
    PH3ImageTextRefinementError, PreparedPH3ImageDenoiser,
};
pub use prompt_expansion::{
    ph3_image_edit_prompt_form, ph3_image_expansion_aspect_ratio, PH3ImageEditPromptForm,
    PH3ImagePromptExpansionError, PH3ImagePromptExpansionGeneration,
    PH3ImagePromptExpansionImageRun, PH3ImagePromptExpansionImageTokens,
    PH3ImagePromptExpansionInput, PH3ImagePromptExpansionLimits, PH3ImagePromptExpansionProtocol,
    PH3ImagePromptExpansionRequest, PH3ImagePromptExpansionRotary, PH3ImagePromptExpansionTask,
    PH3_IMAGE_EDIT_EXPANSION_MAX_NEW_TOKENS, PH3_IMAGE_EXPANSION_ROTARY_PROFILE,
    PH3_IMAGE_PROMPT_EXPANSION_MAX_IMAGES, PH3_IMAGE_PROMPT_EXPANSION_PROTOCOL,
    PH3_IMAGE_T2I_EXPANSION_MAX_NEW_TOKENS,
};
pub use rotary::{PH3ImageCheckpointRotary, PH3ImageRotaryEmbedding};
pub use sampling_schedule::{
    PH3ImageManagedSampling, PH3ImageRequestSchedule, PH3ImageSamplingLimits,
    PH3ImageSamplingSchedule, PH3ImageSamplingStep, PH3ImageSamplingVelocity,
    PH3ImageVelocityBatchLayout, PH3ImageVelocityPrecision, PH3ImageVelocityRequest,
    PreparedPH3ImageSamplingAdvance,
};
pub use text_conditioning::{
    PH3ImageEditPrompt, PH3ImageTextConditioning, PH3ImageTextConditioningCheckpoint,
    PH3ImageTextConditioningConfig, PH3ImageTextConditioningError, PH3ImageTextConditioningFamily,
    PH3ImageTextConditioningHostPhase, PH3ImageTextConditioningHostPhaseBudget,
    PH3ImageTextConditioningHostRequirements, PH3ImageTextConditioningLimits,
    PH3ImageTextConditioningOutput, PH3ImageTextConditioningRun, PH3ImageTextPromptBatch,
    PH3ImageTextTokenizer, PH3ImageTextTokenizerLimits, PH3ImageTextWeightAdapter,
    PH3ImageTextWeightSource, PH3_IMAGE_EDIT_PRESENTATION_PROFILE, PH3_IMAGE_EDIT_ROTARY_PROFILE,
    PH3_IMAGE_TEXT_HIDDEN_STATES, PH3_IMAGE_TEXT_PRESENTATION_PROFILE,
    PH3_IMAGE_TEXT_ROTARY_PROFILE,
};
pub use text_connector::{
    PH3ImageHiddenState, PH3ImageTextConnectorAffineArithmetic, PH3ImageTextConnectorReference,
    PH3ImageTextConnectorShape, PH3ImageTextConnectorWeights,
};
pub use text_connector_managed::{
    PH3ImageBf16HiddenState, PH3ImageConditioningPrecision, PH3ImageF32HiddenState,
    PH3ImageTextConnectorCheckpoint, PH3ImageTextConnectorManagedAdapter,
};
pub use text_connector_projection::{
    PH3ImageTextConnectorCheckpointLayout, PH3ImageTextConnectorProjectionConfig,
    PH3ImageTextConnectorProjectionFamily, PH3ImageTextConnectorProjectionLimits,
};
pub use weights::{
    PH3ImageTransformerCheckpointLayout, PH3ImageTransformerWeightSpec, PH3ImageTransformerWeights,
    PH3ImageWeightLimits,
};

pub const PH3_IMAGE_TRANSFORMER_CLASS: &str = "MiniMaxH3Transformer3DModel";
pub const PH3_IMAGE_TEXT_CONNECTOR_CLASS: &str = "PH3ImageTextConnector";

#[cfg(test)]
mod test_fixture;
