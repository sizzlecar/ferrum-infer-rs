use std::mem::size_of;

use thiserror::Error;

/// Axes of a dense channels-first spatial latent, `[N, C, H, W]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialLatentAxis {
    Batch,
    Channels,
    Height,
    Width,
}

/// Checked dimensions for a single-frame, unpatched spatial latent.
///
/// The two supported dense layouts are `[N, C, H, W]` and `[N, H * W, C]`.
/// A VAE's `[N, C, 1, H, W]` view has the same element order as the former.
/// Dimensions are latent dimensions, not output image pixels: model adapters
/// own spatial compression, alignment, channel counts, and patch constraints.
///
/// This is a CPU indexing contract, not evidence of tensor ownership, dtype,
/// device readiness, or permission to allocate. It grants no runtime authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingleFrameLatentShape {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    spatial_positions: usize,
    elements: usize,
}

impl SingleFrameLatentShape {
    pub fn new(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, LatentLayoutError> {
        for (axis, dimension) in [
            (SpatialLatentAxis::Batch, batch),
            (SpatialLatentAxis::Channels, channels),
            (SpatialLatentAxis::Height, height),
            (SpatialLatentAxis::Width, width),
        ] {
            if dimension == 0 {
                return Err(LatentLayoutError::ZeroDimension { axis });
            }
        }
        let spatial_positions = height
            .checked_mul(width)
            .ok_or(LatentLayoutError::ElementCountOverflow)?;
        let elements = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(spatial_positions))
            .ok_or(LatentLayoutError::ElementCountOverflow)?;
        // Every supported element occupies at least one byte. Wider element
        // types receive an additional byte-size check before any buffer write.
        checked_byte_len(elements, 1)?;
        Ok(Self {
            batch,
            channels,
            height,
            width,
            spatial_positions,
            elements,
        })
    }

    /// Accept a channels-first VAE view only when its frame axis is exactly one.
    /// Multi-frame tensors require a separate layout contract, even when their
    /// flattened element count happens to match a spatial tensor.
    pub fn from_ncfhw_dims(dims: [usize; 5]) -> Result<Self, LatentLayoutError> {
        let [batch, channels, frames, height, width] = dims;
        if frames != 1 {
            return Err(LatentLayoutError::UnsupportedFrameCount { frames });
        }
        Self::new(batch, channels, height, width)
    }

    pub const fn nchw_dims(self) -> [usize; 4] {
        [self.batch, self.channels, self.height, self.width]
    }

    pub const fn ncfhw_dims(self) -> [usize; 5] {
        [self.batch, self.channels, 1, self.height, self.width]
    }

    pub const fn sequence_dims(self) -> [usize; 3] {
        [self.batch, self.spatial_positions, self.channels]
    }

    pub const fn element_count(self) -> usize {
        self.elements
    }

    /// Storage occupied by one complete input or output buffer of `T`.
    /// Zero-sized elements are unsupported; this is not an allocation request.
    pub fn byte_len<T>(self) -> Result<usize, LatentLayoutError> {
        checked_byte_len(self.elements, size_of::<T>())
    }

    fn validate_buffers<T>(self, input: usize, output: usize) -> Result<(), LatentLayoutError> {
        self.byte_len::<T>()?;
        if input != self.elements {
            return Err(LatentLayoutError::InputLengthMismatch {
                expected: self.elements,
                actual: input,
            });
        }
        if output != self.elements {
            return Err(LatentLayoutError::OutputLengthMismatch {
                expected: self.elements,
                actual: output,
            });
        }
        Ok(())
    }
}

fn checked_byte_len(elements: usize, element_bytes: usize) -> Result<usize, LatentLayoutError> {
    if element_bytes == 0 {
        return Err(LatentLayoutError::ZeroSizedElement);
    }
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or(LatentLayoutError::ByteCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(LatentLayoutError::AddressSpaceExceeded { bytes });
    }
    Ok(bytes)
}

/// Copy dense `[N, C, H, W]` into dense `[N, H * W, C]`, without patch grouping.
///
/// Both slices must contain exactly `shape.element_count()` elements. All
/// validation precedes writing, so errors leave `output` unchanged. This
/// function allocates no storage, and Rust's slice borrows exclude overlap.
/// `T: Copy` treats elements as opaque values; no arithmetic, conversion, or
/// finite-value check is performed. It does not establish a numerical dtype
/// contract or BF16 inference parity. Zero-sized elements are rejected.
pub fn pack_spatial_into<T: Copy>(
    shape: SingleFrameLatentShape,
    input: &[T],
    output: &mut [T],
) -> Result<(), LatentLayoutError> {
    shape.validate_buffers::<T>(input.len(), output.len())?;
    let positions = shape.spatial_positions;
    let image_elements = shape.elements / shape.batch;
    for (source, target) in input
        .chunks_exact(image_elements)
        .zip(output.chunks_exact_mut(image_elements))
    {
        for position in 0..positions {
            for channel in 0..shape.channels {
                target[position * shape.channels + channel] =
                    source[channel * positions + position];
            }
        }
    }
    Ok(())
}

/// Copy dense `[N, H * W, C]` into dense `[N, C, H, W]`.
/// The result is also the element order of `[N, C, 1, H, W]`.
///
/// This has the same exact-length, no-allocation, opaque-element, and
/// failure-before-write contract as [`pack_spatial_into`].
pub fn unpack_spatial_into<T: Copy>(
    shape: SingleFrameLatentShape,
    input: &[T],
    output: &mut [T],
) -> Result<(), LatentLayoutError> {
    shape.validate_buffers::<T>(input.len(), output.len())?;
    let positions = shape.spatial_positions;
    let image_elements = shape.elements / shape.batch;
    for (source, target) in input
        .chunks_exact(image_elements)
        .zip(output.chunks_exact_mut(image_elements))
    {
        for channel in 0..shape.channels {
            for position in 0..positions {
                target[channel * positions + position] =
                    source[position * shape.channels + channel];
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LatentLayoutError {
    #[error("spatial latent axis {axis:?} must be non-zero")]
    ZeroDimension { axis: SpatialLatentAxis },
    #[error("spatial latent layout requires one frame, received {frames}")]
    UnsupportedFrameCount { frames: usize },
    #[error("spatial latent element count overflows usize")]
    ElementCountOverflow,
    #[error("zero-sized spatial latent elements are unsupported")]
    ZeroSizedElement,
    #[error("spatial latent byte count overflows usize")]
    ByteCountOverflow,
    #[error("spatial latent buffer of {bytes} bytes exceeds isize::MAX")]
    AddressSpaceExceeded { bytes: usize },
    #[error("spatial latent input requires {expected} elements, received {actual}")]
    InputLengthMismatch { expected: usize, actual: usize },
    #[error("spatial latent output requires {expected} elements, received {actual}")]
    OutputLengthMismatch { expected: usize, actual: usize },
}

#[cfg(test)]
mod tests;
