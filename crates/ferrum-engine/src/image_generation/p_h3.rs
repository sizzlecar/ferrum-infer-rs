//! Native P-H3 component composition. Source validation is device-free;
//! execution uses the existing image execution owner and service contracts.
mod assets;
mod conditioning;
mod conditioning_diagnostics;
mod continuous;
#[cfg(feature = "cuda")]
pub mod cuda;
mod directory;
mod dit_input_diagnostics;
mod edit_noise;
mod editing;
pub(crate) mod expansion;
mod final_latent_diagnostics;
mod generation;
mod host_budget;
mod metadata;
mod noise;
mod pipeline;
mod reference;
mod request;
mod source;
mod stages;
#[cfg(test)]
mod tests;
mod text;
mod trace;
mod vision;
pub use assets::{PH3ImageAssetLimits, PH3ImageAssets, PH3ImageMetadata, PH3ImageSources};
pub use conditioning_diagnostics::{
    PH3ImageMainTextObservation, PH3ImageMainTextObserver, PH3ImageMainTextTap,
    PH3ImageMainVisionFeature, PH3ImageMainVisionObservation, PH3ImageMainVisionObserver,
    PH3ImageMainVisionPort, PH3ImageMainVisionRgba,
};
#[cfg(test)]
pub(in crate::image_generation) use continuous::clean_decoder_admission;
pub use directory::PH3ImageArchivePaths;
pub use dit_input_diagnostics::{PH3ImageDitInputObservation, PH3ImageDitInputObserver};
pub use edit_noise::PH3ImageEditNoisePolicy;
pub use expansion::{
    PH3ImagePromptExpansionDiagnosticPolicy, PH3ImagePromptExpansionMtp,
    PH3ImagePromptExpansionOptions, PH3ImagePromptExpansionSourceBacking,
    PH3ImagePromptExpansionVisionReuse, PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES,
};
pub use final_latent_diagnostics::{PH3ImageFinalLatentObservation, PH3ImageFinalLatentObserver};
pub use noise::PH3ImageInitialNoisePolicy;
pub use pipeline::{PH3ImagePipeline, PH3ImagePipelineOptions};
pub use source::{PH3ImageSourceAdapter, PH3ImageStageSource};
pub use vision::reuse::PH3ImageMainVisionReuse;
