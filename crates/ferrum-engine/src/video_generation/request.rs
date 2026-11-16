use super::{VideoReference, VideoReferenceLimits};
use crate::iterative_service::{service_error, ServiceError, ServiceErrorKind, ServiceRequest};
use ferrum_output::mp4::VideoAudioSpec;
use ferrum_runtime::HostMemoryAccount;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoFrameRate {
    pub numerator: u32,
    pub denominator: u32,
}

/// Product metadata; no latent dimensions, token tags, native budgets or model
/// sampling tables are accepted from a transport as request metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationParameters {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub frame_rate: VideoFrameRate,
    pub seed: u64,
}
#[derive(Debug, Clone)]
pub struct VideoGenerationRequest {
    pub parameters: VideoGenerationParameters,
    /// Encoded media in the model template's original reference order. The
    /// loaded adapter validates format and declares the actual supported count.
    pub references: Vec<VideoReference>,
}
#[derive(Debug, Clone, Copy)]
pub struct VideoRequestLimits {
    pub maximum_prompt_bytes: usize,
    pub references: VideoReferenceLimits,
    pub maximum_encoded_reference_bytes: usize,
}
impl VideoRequestLimits {
    pub(super) fn validate(self) -> Result<(), ServiceError> {
        self.references.validate()?;
        if self.maximum_prompt_bytes == 0
            || (self.references.maximum_total > 0 && self.maximum_encoded_reference_bytes == 0)
            || self.maximum_prompt_bytes > isize::MAX as usize
        {
            return Err(invalid("invalid bounded video ingress limits"));
        }
        Ok(())
    }
}
pub(super) fn invalid(message: impl Into<String>) -> ServiceError {
    service_error(ServiceErrorKind::InvalidRequest, message)
}
impl VideoGenerationRequest {
    pub(super) fn validate(
        &self,
        limits: VideoRequestLimits,
        host: &HostMemoryAccount,
        output: VideoAudioSpec,
    ) -> Result<(), ServiceError> {
        let p = &self.parameters;
        if p.prompt.is_empty()
            || p.prompt.len() > limits.maximum_prompt_bytes
            || p.width == 0
            || p.height == 0
            || p.num_frames == 0
            || p.frame_rate.numerator == 0
            || p.frame_rate.denominator == 0
        {
            return Err(invalid(
                "video prompt, dimensions, rate or reference count exceeds declared ingress limits",
            ));
        }
        limits
            .references
            .validate_kinds(self.references.iter().map(VideoReference::kind))?;
        let mut encoded = 0usize;
        for reference in &self.references {
            let image = reference.bytes();
            if image.is_empty() || !host.owns(image) {
                return Err(invalid(
                    "video references must retain nonempty backings from the original Host account",
                ));
            }
            encoded = encoded
                .checked_add(image.len())
                .ok_or_else(|| invalid("encoded reference extent overflow"))?;
        }
        if encoded > limits.maximum_encoded_reference_bytes {
            return Err(invalid("video references exceed the encoded-byte limit"));
        }
        if p.width != output.width
            || p.height != output.height
            || u64::from(p.num_frames) != output.frames
            || u64::from(p.frame_rate.numerator) * u64::from(output.fps_denominator)
                != u64::from(output.fps_numerator) * u64::from(p.frame_rate.denominator)
        {
            return Err(invalid(
                "resolved output plan differs from requested video geometry",
            ));
        }
        Ok(())
    }
}
impl ServiceRequest for VideoGenerationRequest {
    fn pending_bytes(&self) -> usize {
        self.references
            .iter()
            .fold(self.admission_bytes(), |bytes, image| {
                bytes.saturating_add(image.bytes().backing_bytes())
            })
    }
    fn admission_bytes(&self) -> usize {
        // The encoded backings are already charged to this Host account.
        self.parameters.prompt.capacity().saturating_add(
            self.references
                .capacity()
                .saturating_mul(std::mem::size_of::<VideoReference>()),
        )
    }
    fn kind(&self) -> &'static str {
        "video"
    }
}
