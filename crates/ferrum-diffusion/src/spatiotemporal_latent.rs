use std::mem::size_of;

use thiserror::Error;

/// Axes of a dense `[N, C, T, H, W]` latent. Time is never the batch axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatiotemporalAxis {
    Batch,
    Channels,
    Time,
    Height,
    Width,
}

/// Checked dense latent and patch geometry.
///
/// Packing maps `[N, C, T, H, W]` to `[N, t*h*w, C*pt*ph*pw]`, with
/// token order `(t,h,w)` and channel-major within-patch order `(C,pt,ph,pw)`.
/// Every patch is complete; this contract adds no padding and has no hidden
/// invalid region. Model adapters own compression, alignment and cropping.
/// Dimensions describe latent coordinates, not output pixels or timestamps.
///
/// This is an indexing contract. It neither owns storage nor grants memory,
/// device submission, dtype conversion or completion authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatiotemporalLatentShape {
    dense: [usize; 5],
    patch: [usize; 3],
    grid: [usize; 3],
    tokens: usize,
    token_width: usize,
    elements: usize,
}

impl SpatiotemporalLatentShape {
    pub fn new(dense: [usize; 5], patch: [usize; 3]) -> Result<Self, SpatiotemporalLayoutError> {
        use SpatiotemporalAxis::*;
        for (axis, extent) in [Batch, Channels, Time, Height, Width]
            .into_iter()
            .zip(dense)
        {
            if extent == 0 {
                return Err(SpatiotemporalLayoutError::ZeroDimension { axis });
            }
        }
        let mut grid = [0; 3];
        for (i, axis) in [Time, Height, Width].into_iter().enumerate() {
            if patch[i] == 0 || dense[i + 2] % patch[i] != 0 {
                return Err(SpatiotemporalLayoutError::IncompletePatch {
                    axis,
                    extent: dense[i + 2],
                    patch: patch[i],
                });
            }
            grid[i] = dense[i + 2] / patch[i];
        }
        let elements = product(dense)?;
        let tokens = product(grid)?;
        let token_width = product([dense[1], patch[0], patch[1], patch[2]])?;
        checked_bytes(elements, 1)?;
        Ok(Self {
            dense,
            patch,
            grid,
            tokens,
            token_width,
            elements,
        })
    }

    pub const fn ncthw_dims(self) -> [usize; 5] {
        self.dense
    }
    pub const fn patch_dims(self) -> [usize; 3] {
        self.patch
    }
    pub const fn patch_grid(self) -> [usize; 3] {
        self.grid
    }
    pub const fn sequence_dims(self) -> [usize; 3] {
        [self.dense[0], self.tokens, self.token_width]
    }
    pub const fn tokens_per_batch(self) -> usize {
        self.tokens
    }
    pub const fn token_width(self) -> usize {
        self.token_width
    }
    pub const fn element_count(self) -> usize {
        self.elements
    }
    pub fn byte_len<T>(self) -> Result<usize, SpatiotemporalLayoutError> {
        checked_bytes(self.elements, size_of::<T>())
    }

    /// Visits `(packed_index, dense_index)` in increasing packed order.
    /// Both indices are strictly below `element_count()`. This permits a
    /// caller to read/write its existing managed byte buffers and retain its
    /// model-specific rounding without allocating a temporary typed tensor.
    /// The visitor cannot change the declared geometry or omit an axis.
    pub fn for_each_packed_index(self, mut visit: impl FnMut(usize, usize)) {
        self.for_each_packed_span(|packed, dense| {
            for (p, d) in packed.zip(dense) {
                visit(p, d);
            }
        });
    }

    /// Visits equal-length contiguous spans in packed and dense storage.
    /// Packed spans partition `0..element_count()` in increasing order; dense
    /// spans visit every element exactly once. A span is one within-patch row,
    /// so it never crosses a channel, time, or spatial patch boundary.
    ///
    /// This grants only indices, like `for_each_packed_index`. Callers retain
    /// their storage owners and validate their element widths and byte bounds.
    pub fn for_each_packed_span(
        self,
        mut visit: impl FnMut(std::ops::Range<usize>, std::ops::Range<usize>),
    ) {
        let [batch, channels, time, height, width] = self.dense;
        let [pt, ph, pw] = self.patch;
        let [gt, gh, gw] = self.grid;
        let mut packed = 0;
        for n in 0..batch {
            for t in 0..gt {
                for h in 0..gh {
                    for w in 0..gw {
                        for c in 0..channels {
                            for dt in 0..pt {
                                for dh in 0..ph {
                                    let dense = (((n * channels + c) * time + t * pt + dt)
                                        * height
                                        + h * ph
                                        + dh)
                                        * width
                                        + w * pw;
                                    visit(packed..packed + pw, dense..dense + pw);
                                    packed += pw;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn validate<T>(self, input: usize, output: usize) -> Result<(), SpatiotemporalLayoutError> {
        self.byte_len::<T>()?;
        if input != self.elements || output != self.elements {
            return Err(SpatiotemporalLayoutError::BufferLength {
                expected: self.elements,
                input,
                output,
            });
        }
        Ok(())
    }
}

/// Pack opaque elements into caller-owned storage. Validation precedes every
/// write; errors preserve `output`. No allocation or numeric conversion occurs.
pub fn pack_spatiotemporal_into<T: Copy>(
    shape: SpatiotemporalLatentShape,
    input: &[T],
    output: &mut [T],
) -> Result<(), SpatiotemporalLayoutError> {
    shape.validate::<T>(input.len(), output.len())?;
    shape.for_each_packed_span(|packed, dense| output[packed].copy_from_slice(&input[dense]));
    Ok(())
}

/// Inverse of [`pack_spatiotemporal_into`], with the same no-allocation and
/// validation-before-write contract.
pub fn unpack_spatiotemporal_into<T: Copy>(
    shape: SpatiotemporalLatentShape,
    input: &[T],
    output: &mut [T],
) -> Result<(), SpatiotemporalLayoutError> {
    shape.validate::<T>(input.len(), output.len())?;
    shape.for_each_packed_span(|packed, dense| output[dense].copy_from_slice(&input[packed]));
    Ok(())
}

fn product<const N: usize>(extents: [usize; N]) -> Result<usize, SpatiotemporalLayoutError> {
    extents.into_iter().try_fold(1usize, |a, b| {
        a.checked_mul(b)
            .ok_or(SpatiotemporalLayoutError::ElementCountOverflow)
    })
}

fn checked_bytes(elements: usize, width: usize) -> Result<usize, SpatiotemporalLayoutError> {
    if width == 0 {
        return Err(SpatiotemporalLayoutError::ZeroSizedElement);
    }
    let bytes = elements
        .checked_mul(width)
        .ok_or(SpatiotemporalLayoutError::ByteCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(SpatiotemporalLayoutError::AddressSpaceExceeded { bytes });
    }
    Ok(bytes)
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SpatiotemporalLayoutError {
    #[error("latent axis {axis:?} must be non-zero")]
    ZeroDimension { axis: SpatiotemporalAxis },
    #[error("latent axis {axis:?} extent {extent} does not fit complete patches of {patch}")]
    IncompletePatch {
        axis: SpatiotemporalAxis,
        extent: usize,
        patch: usize,
    },
    #[error("latent element count overflows usize")]
    ElementCountOverflow,
    #[error("zero-sized latent elements are unsupported")]
    ZeroSizedElement,
    #[error("latent byte count overflows usize")]
    ByteCountOverflow,
    #[error("latent buffer of {bytes} bytes exceeds isize::MAX")]
    AddressSpaceExceeded { bytes: usize },
    #[error("latent needs {expected} elements; input has {input}, output has {output}")]
    BufferLength {
        expected: usize,
        input: usize,
        output: usize,
    },
}

#[cfg(test)]
mod tests;
