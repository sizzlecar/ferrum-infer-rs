//! Typed video ingress and output handoff around the existing iterative worker.
//! Model adapters own conditioning, solver and decoder execution; this facade
//! owns neither a second model loop nor an HTTP job registry.

pub mod fast_h3;
mod output;
mod reference;
mod request;
mod service;
pub use output::{RawVideoOutput, VideoGenerationResult, VideoOutputPlan};
pub use reference::{VideoReference, VideoReferenceKind, VideoReferenceLimits};
pub use request::{
    VideoFrameRate, VideoGenerationParameters, VideoGenerationRequest, VideoRequestLimits,
};
pub use service::{
    VideoCancellationHandle, VideoGeneratorOwner, VideoJob, VideoJobError, VideoService,
    VideoServiceConfig, VideoServiceRun, VideoServiceShutdown,
};
