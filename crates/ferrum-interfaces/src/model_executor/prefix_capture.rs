//! Optional interest in a future, exact completed prefix. This is not admission
//! authority and does not reserve device capacity.

use ferrum_types::{RequestId, TokenId};
use std::{any::Any, fmt::Debug, time::Instant};

#[derive(Debug, Clone, Copy)]
pub struct PrefixCapturePlan {
    pub boundary: usize,
    /// Constraint on executed span lengths, not absolute token positions.
    pub span: crate::vnext::CheckpointTokenSpanConstraint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixCaptureStatus {
    Pending,
    Ready,
    /// Cancellation, pressure, expiry, or an unsupported/overshot boundary.
    /// The untouched follower may proceed through ordinary cold admission.
    Unavailable,
}

/// Executor-owned interest in one source incarnation. Ready retains immutable
/// checkpoint ownership independently of cache-index membership. Implementations
/// must not retain the source merely to keep a cancelled computation alive.
pub trait PrefixCaptureLease: Debug + Send + Sync {
    fn boundary(&self) -> usize;
    fn status(&self) -> PrefixCaptureStatus;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug, Clone, Copy)]
pub struct PrefixCaptureRequest<'a> {
    pub source_request_id: &'a RequestId,
    pub source_tokens: &'a [TokenId],
    pub maximum_sequence_tokens: usize,
    pub boundary: usize,
    pub expires_at: Instant,
}

/// Token lengths are only a boundary-planning hint. The caller must establish
/// exact token equality; capture and restore independently validate full input
/// and native source/target identity before making state available.
#[derive(Debug, Clone, Copy)]
pub struct PrefixCaptureBoundary<'a> {
    pub processed_tokens: usize,
    pub source_prompt_tokens: usize,
    pub common_prefix_tokens: usize,
    pub follower_prompt_tokens: &'a [usize],
}
