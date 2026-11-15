//! The original MiniMax-H3 video VAE, including its full-depth packed decoder.
//!
//! Import and temporal contracts are separate from an executable decoder. The
//! decoder attends across a complete temporal clip; this is not framewise image
//! decoding and does not reuse the P-H3 image encoder's last-temporal-tap view.

mod blocks;
mod clip;
mod config;
mod encoder;
mod image_encoder;
mod image_tiles;
mod keyframe_posterior;
mod latent_layout;
mod package;
mod spatial;
mod temporal;
mod temporal_encode;
mod temporal_stitch;
mod video_tiles;

pub use super::nvfp4_scalars::FastH3DecoderAlphaDerivation;
pub use blocks::{
    FastH3VideoDecoderBlocksConfig, FastH3VideoDecoderBlocksFamily, FastH3VideoDecoderBlocksLimits,
    FastH3VideoDecoderBlocksSource,
};
pub use clip::{
    FastH3VideoClipConfig, FastH3VideoClipFamily, FastH3VideoClipLimits, FastH3VideoClipSource,
    FastH3VideoRawClip,
};
pub use config::FastH3VideoVaeConfig;
pub use encoder::{
    FastH3VideoEncoderGeometry, FastH3VideoEncoderLimits, FastH3VideoEncoderProfile,
    PreparedFastH3VideoEncoder,
};
pub use image_encoder::{
    FastH3ImageEncoderLimits, FastH3ImageEncoderProfile, FastH3ImageEncoderSource,
    PreparedFastH3ImageEncoder,
};
pub use image_tiles::{
    FastH3ImageMomentError, FastH3ImageMomentRequest, FastH3ImageMoments, FastH3ImageSpatialPolicy,
    PreparedFastH3TiledImageEncoder, FAST_H3_IMAGE_TILE_PROFILE,
};
pub use keyframe_posterior::{
    sample_fast_h3_keyframe_posterior, sample_fast_h3_video_posterior,
    FAST_H3_KEYFRAME_POSTERIOR_PROFILE, FAST_H3_VIDEO_POSTERIOR_PROFILE,
};
pub use latent_layout::FastH3VideoLatentLayout;
pub use package::{FastH3VideoDecoderPackage, FastH3VideoDecoderWeightLimits};
pub use spatial::{
    FastH3VideoLatentUnits, FastH3VideoSpatialCanvas, FastH3VideoSpatialPlan,
    FastH3VideoSpatialPolicy,
};
pub use temporal::{FastH3VideoDecodeChunk, FastH3VideoDecodePlan};
pub use temporal_encode::{FastH3VideoEncodeChunk, FastH3VideoEncodePlan};
pub use temporal_stitch::{
    FastH3PreparedVideoChunk, FastH3VideoFrameChunk, FastH3VideoRgbChunk,
    FastH3VideoTemporalAssembler, FastH3VideoTemporalLimits,
};
pub use video_tiles::{
    FastH3VideoMomentError, FastH3VideoMomentRequest, FastH3VideoMoments,
    FastH3VideoTemporalPolicy, PreparedFastH3TiledVideoEncoder,
};

#[cfg(test)]
mod tests;
