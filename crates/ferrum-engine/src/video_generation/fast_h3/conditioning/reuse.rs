//! Reuse is local to one immutable checkpoint/processor composition. Only
//! completed Host results may be retained; this grants no device residency.
#[cfg(any(test, feature = "cuda-fast-h3"))]
use crate::video_generation::VideoGenerationRequest;
use serde::{Deserialize, Serialize};
#[cfg(any(test, feature = "cuda-fast-h3"))]
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FastH3ConditioningReuse {
    Disabled {},
    /// Keep at most the last completed image/prompt result. The limit covers
    /// the full physical tensor backings, including padding and sliced owners.
    LastResult {
        maximum_retained_bytes: usize,
    },
}
impl Default for FastH3ConditioningReuse {
    fn default() -> Self {
        Self::LastResult {
            maximum_retained_bytes: 64 << 20,
        }
    }
}
impl FastH3ConditioningReuse {
    pub fn validate(self) -> anyhow::Result<()> {
        if let Self::LastResult {
            maximum_retained_bytes,
        } = self
        {
            anyhow::ensure!(
                maximum_retained_bytes > 0 && maximum_retained_bytes <= isize::MAX as usize,
                "conditioning reuse requires a positive addressable backing limit"
            );
        }
        Ok(())
    }
    #[cfg(any(test, feature = "cuda-fast-h3"))]
    pub(crate) fn retains(self, bytes: usize) -> bool {
        matches!(self, Self::LastResult { maximum_retained_bytes } if bytes <= maximum_retained_bytes)
    }
    /// A newly cached result keeps its original request-scoped Output credit
    /// through decoding. Its complete backing can coexist with every decoder
    /// buffer; the conditioning floor alone does not cover that overlap.
    #[cfg(any(test, feature = "cuda-fast-h3"))]
    pub(crate) fn output_claim(self, decoder_bytes: usize, floor: usize) -> anyhow::Result<usize> {
        let retained = match self {
            Self::Disabled {} => 0,
            Self::LastResult {
                maximum_retained_bytes,
            } => maximum_retained_bytes,
        };
        decoder_bytes
            .checked_add(retained)
            .map(|bytes| bytes.max(floor))
            .ok_or_else(|| ferrum_runtime::HostMemoryError::ArithmeticOverflow.into())
    }
}

/// Exact encoded inputs, kind, order and raw prompt, with length framing.
/// Audio is trimmed to the generated duration, so that duration is part of its
/// identity. Image-only conditioning remains independent of output geometry.
/// Model/tokenizer/profile identity is the enclosing conditioner's lifetime.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg(any(test, feature = "cuda-fast-h3"))]
pub(crate) struct ConditioningKey([u8; 32]);
#[cfg(any(test, feature = "cuda-fast-h3"))]
impl ConditioningKey {
    pub(crate) fn new(request: &VideoGenerationRequest) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"FastH3 original media conditioning v1");
        hash.update((request.parameters.prompt.len() as u64).to_le_bytes());
        hash.update(request.parameters.prompt.as_bytes());
        hash.update((request.references.len() as u64).to_le_bytes());
        let mut temporal = false;
        for reference in &request.references {
            use crate::video_generation::VideoReferenceKind;
            let tag = match reference.kind() {
                VideoReferenceKind::Image => 0,
                VideoReferenceKind::Video => {
                    temporal = true;
                    1
                }
                VideoReferenceKind::Audio => {
                    temporal = true;
                    2
                }
            };
            hash.update([tag]);
            hash.update((reference.bytes().len() as u64).to_le_bytes());
            hash.update(reference.bytes().bytes());
        }
        if temporal {
            hash.update(request.parameters.num_frames.to_le_bytes());
            hash.update(request.parameters.frame_rate.numerator.to_le_bytes());
            hash.update(request.parameters.frame_rate.denominator.to_le_bytes());
        }
        Self(hash.finalize().into())
    }
}

#[cfg(test)]
mod tests;
