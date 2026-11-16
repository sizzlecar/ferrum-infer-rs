//! FastH3 stages on caller-owned Core runtimes, Host accounts and serial actors.
//! The existing iterative worker owns admission and membership. These stages
//! neither create a device runtime nor introduce another scheduler or cache.

pub mod assets;
#[cfg(feature = "cuda-fast-h3")]
pub mod composition;
pub mod conditioning;
#[cfg(feature = "cuda-fast-h3")]
pub mod cuda;
pub mod decode;
pub mod denoising;
#[cfg(feature = "cuda-fast-h3")]
mod worker;

/// Supported product reference kinds and the released Ref2VA cardinalities.
pub const REFERENCE_LIMITS: super::VideoReferenceLimits = super::VideoReferenceLimits {
    minimum_visual: 1,
    maximum_images: ferrum_models::fast_h3::FAST_H3_MAX_REFERENCE_IMAGES,
    maximum_videos: 3,
    maximum_audio: 3,
    maximum_total: 12,
};
