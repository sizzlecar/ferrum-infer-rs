//! Managed, single-image input for Qwen-Image 2.1 editing.
//!
//! The first codec is static 8-bit RGB/RGBA PNG. Encoded input and every pixel
//! or model-input backing belong to the caller's single Host account. This
//! module creates no files, runtimes or network requests. PNG decoder-internal
//! storage has its own explicit codec limit; that limit is not a measurement
//! of total RSS or part of the managed backing-byte counter.

mod png;
mod preprocess;
mod resize;

pub use png::png_dimensions_hint;
pub use preprocess::{QwenImage21ConditionInputs, QwenImage21ImageProcessor};

use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryError, ManagedHostBytes};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageInputError {
    #[error("invalid input image: {0}")]
    Invalid(&'static str),
    #[error("unsupported input image: {0}")]
    Unsupported(&'static str),
    #[error("input image exceeds declared limit: {0}")]
    Limit(&'static str),
    #[error("invalid image processor configuration: {0}")]
    Configuration(String),
    #[error(transparent)]
    Host(#[from] HostMemoryError),
    #[error("PNG decode: {0}")]
    Png(#[from] ::png::DecodingError),
    #[error("image tensor: {0}")]
    Tensor(String),
}

pub type ImageInputResult<T> = Result<T, ImageInputError>;

/// Independent ingress, decompression and decoded-image bounds. These are
/// product admission limits, not claims about what every backend can execute.
#[derive(Debug, Clone, Copy)]
pub struct ImageInputLimits {
    maximum_encoded_bytes: usize,
    maximum_pixels: usize,
    maximum_width: u32,
    maximum_height: u32,
    decoder_workspace_bytes: usize,
}

impl Default for ImageInputLimits {
    /// Initial single-image ingress preset: 17 MiB encoded PNG, 24 million
    /// decoded pixels, at most 8192 pixels per axis, and a 64 MiB best-effort
    /// PNG codec workspace limit. These are caps, not reserved Host payload.
    fn default() -> Self {
        Self {
            maximum_encoded_bytes: Self::DEFAULT_MAXIMUM_ENCODED_BYTES,
            maximum_pixels: 24_000_000,
            maximum_width: 8192,
            maximum_height: 8192,
            decoder_workspace_bytes: 64 * 1024 * 1024,
        }
    }
}

impl ImageInputLimits {
    /// Includes PNG framing above the 16 MiB RGBA payload of a 2048² image.
    /// This is an ingress cap; every retained backing still needs Host admission.
    pub const DEFAULT_MAXIMUM_ENCODED_BYTES: usize = 17 * 1024 * 1024;

    pub fn new(
        maximum_encoded_bytes: usize,
        maximum_pixels: usize,
        maximum_width: u32,
        maximum_height: u32,
        decoder_workspace_bytes: usize,
    ) -> ImageInputResult<Self> {
        if maximum_encoded_bytes == 0
            || maximum_encoded_bytes > isize::MAX as usize
            || maximum_pixels == 0
            || maximum_pixels > isize::MAX as usize / 4
            || maximum_width == 0
            || maximum_height == 0
            || decoder_workspace_bytes == 0
            || decoder_workspace_bytes > isize::MAX as usize
        {
            return Err(ImageInputError::Limit("invalid or unaddressable limits"));
        }
        Ok(Self {
            maximum_encoded_bytes,
            maximum_pixels,
            maximum_width,
            maximum_height,
            decoder_workspace_bytes,
        })
    }

    pub fn maximum_encoded_bytes(self) -> usize {
        self.maximum_encoded_bytes
    }

    pub fn maximum_pixels(self) -> usize {
        self.maximum_pixels
    }

    pub fn maximum_width(self) -> u32 {
        self.maximum_width
    }

    pub fn maximum_height(self) -> u32 {
        self.maximum_height
    }

    pub fn decoder_workspace_bytes(self) -> usize {
        self.decoder_workspace_bytes
    }

    fn rgba_bytes(self, width: u32, height: u32) -> ImageInputResult<usize> {
        if width == 0 || height == 0 || width > self.maximum_width || height > self.maximum_height {
            return Err(ImageInputError::Limit("image dimensions"));
        }
        let pixels = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| ImageInputError::Limit("pixel count exceeds address space"))?;
        if pixels > self.maximum_pixels {
            return Err(ImageInputError::Limit("decoded pixels"));
        }
        pixels
            .checked_mul(4)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(ImageInputError::Limit("RGBA bytes exceed address space"))
    }
}

/// Immutable HWC RGBA8 owner. Original RGB is distinguished from original
/// RGBA because Pillow resampling premultiplies alpha only for the latter.
#[derive(Debug, Clone)]
pub struct ManagedRgba8Image {
    bytes: ManagedHostBytes,
    width: u32,
    height: u32,
    source_has_alpha: bool,
}

impl ManagedRgba8Image {
    pub fn decode_png(
        host: &HostMemoryAccount,
        encoded: &ManagedHostBytes,
        limits: ImageInputLimits,
    ) -> ImageInputResult<Self> {
        png::decode(host, encoded, limits)
    }

    /// Accept already decoded RGBA without copying. This is an explicit RGBA
    /// declaration; callers must not label arbitrary bytes as decoded pixels.
    pub fn from_rgba8(
        host: &HostMemoryAccount,
        bytes: ManagedHostBytes,
        width: u32,
        height: u32,
        limits: ImageInputLimits,
    ) -> ImageInputResult<Self> {
        check_input_owner(host, &bytes)?;
        if bytes.len() != limits.rgba_bytes(width, height)? {
            return Err(ImageInputError::Invalid(
                "RGBA byte extent differs from dimensions",
            ));
        }
        Ok(Self {
            bytes,
            width,
            height,
            source_has_alpha: true,
        })
    }

    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn source_has_alpha(&self) -> bool {
        self.source_has_alpha
    }

    pub fn validate_parent(&self, host: &HostMemoryAccount) -> ImageInputResult<()> {
        check_input_owner(host, &self.bytes)
    }
}

fn check_input_owner(host: &HostMemoryAccount, bytes: &ManagedHostBytes) -> ImageInputResult<()> {
    if !host.owns(bytes) || bytes.class() != HostMemoryClass::InputPreprocess {
        return Err(ImageInputError::Invalid(
            "foreign Host parent or non-InputPreprocess backing",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
