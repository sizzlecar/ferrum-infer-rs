use thiserror::Error;

use crate::{LatentLayoutError, SingleFrameLatentShape};

/// Normalize dense single-frame `[N, C, H, W]` latents in FP32.
///
/// Each element in channel `c` becomes `(input - channel_mean[c]) /
/// channel_std[c]`. Statistics have exactly `C` elements and broadcast across
/// batch and spatial positions. The same storage order also represents a VAE's
/// `[N, C, 1, H, W]` view. Callers must supply that channels-first order;
/// slice lengths cannot prove layout or detect a packed spatial sequence.
///
/// Input and output lengths must exactly match `shape`. Inputs and means must
/// be finite; standard deviations must be finite and strictly positive. The
/// FP32 storage size and every result are checked before any output write, so
/// errors leave the entire output unchanged, including its existing bit patterns.
/// Output's previous contents need not be finite. No scratch storage is allocated;
/// a second pass repeats the validated arithmetic into caller-owned storage.
///
/// Subtraction and division are separate FP32 operations, with no reciprocal
/// substitution. This is a local CPU mathematical contract, not BF16/PyTorch
/// rounding parity, model configuration validation, or runtime allocation authority.
pub fn normalize_latent_f32_into(
    shape: SingleFrameLatentShape,
    input: &[f32],
    channel_mean: &[f32],
    channel_std: &[f32],
    output: &mut [f32],
) -> Result<(), LatentNormalizationError> {
    transform_into(
        Transform::Normalize,
        shape,
        input,
        channel_mean,
        channel_std,
        output,
    )
}

/// Denormalize dense single-frame `[N, C, H, W]` latents in FP32.
///
/// Each element in channel `c` becomes `input * channel_std[c] +
/// channel_mean[c]`. Multiplication and addition are separate FP32 operations;
/// fused multiply-add is not used. The shape, statistics, finite-value,
/// caller-owned storage, and failure-before-write contracts are identical to
/// [`normalize_latent_f32_into`]. Finite underflow to zero is permitted in both
/// directions; these finite-precision transforms need not be exact inverses.
pub fn denormalize_latent_f32_into(
    shape: SingleFrameLatentShape,
    input: &[f32],
    channel_mean: &[f32],
    channel_std: &[f32],
    output: &mut [f32],
) -> Result<(), LatentNormalizationError> {
    transform_into(
        Transform::Denormalize,
        shape,
        input,
        channel_mean,
        channel_std,
        output,
    )
}

#[derive(Clone, Copy)]
enum Transform {
    Normalize,
    Denormalize,
}

impl Transform {
    fn apply(self, value: f32, mean: f32, std: f32) -> f32 {
        match self {
            Self::Normalize => {
                let centered = value - mean;
                centered / std
            }
            Self::Denormalize => {
                let scaled = value * std;
                scaled + mean
            }
        }
    }
}

fn transform_into(
    transform: Transform,
    shape: SingleFrameLatentShape,
    input: &[f32],
    channel_mean: &[f32],
    channel_std: &[f32],
    output: &mut [f32],
) -> Result<(), LatentNormalizationError> {
    // A valid element count is not enough for an f32 slice's byte/address-space
    // boundary. Check it even when the supplied slices are shorter than shape.
    shape.byte_len::<f32>()?;
    let elements = shape.element_count();
    if input.len() != elements {
        return Err(LatentLayoutError::InputLengthMismatch {
            expected: elements,
            actual: input.len(),
        }
        .into());
    }
    if output.len() != elements {
        return Err(LatentLayoutError::OutputLengthMismatch {
            expected: elements,
            actual: output.len(),
        }
        .into());
    }
    let [_, positions, channels] = shape.sequence_dims();
    if channel_mean.len() != channels {
        return Err(LatentNormalizationError::MeanLengthMismatch {
            expected: channels,
            actual: channel_mean.len(),
        });
    }
    if channel_std.len() != channels {
        return Err(LatentNormalizationError::StdLengthMismatch {
            expected: channels,
            actual: channel_std.len(),
        });
    }
    for (channel, (&mean, &std)) in channel_mean.iter().zip(channel_std).enumerate() {
        if !mean.is_finite() {
            return Err(LatentNormalizationError::NonFiniteMean { channel });
        }
        if !std.is_finite() || std <= 0.0 {
            return Err(LatentNormalizationError::InvalidStd { channel });
        }
    }
    for (index, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(LatentNormalizationError::NonFiniteInput { index });
        }
        let channel = (index / positions) % channels;
        let result = transform.apply(value, channel_mean[channel], channel_std[channel]);
        if !result.is_finite() {
            return Err(LatentNormalizationError::NonFiniteResult { index });
        }
    }
    // Shape validation guarantees nonzero positions/channels. Shared immutable
    // input/statistic borrows cannot overlap the exclusive output borrow.
    for (index, (&value, target)) in input.iter().zip(output).enumerate() {
        let channel = (index / positions) % channels;
        *target = transform.apply(value, channel_mean[channel], channel_std[channel]);
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LatentNormalizationError {
    #[error(transparent)]
    Layout(#[from] LatentLayoutError),
    #[error("latent channel means require {expected} elements, received {actual}")]
    MeanLengthMismatch { expected: usize, actual: usize },
    #[error("latent channel standard deviations require {expected} elements, received {actual}")]
    StdLengthMismatch { expected: usize, actual: usize },
    #[error("latent mean for channel {channel} must be finite")]
    NonFiniteMean { channel: usize },
    #[error("latent standard deviation for channel {channel} must be finite and positive")]
    InvalidStd { channel: usize },
    #[error("latent input element {index} must be finite")]
    NonFiniteInput { index: usize },
    #[error("latent normalization result element {index} is not finite")]
    NonFiniteResult { index: usize },
}

#[cfg(test)]
mod tests;
