use super::request::invalid;
use crate::iterative_service::{
    service_error, OutputGate, OutputGateError, OutputLease, ServiceError, ServiceErrorKind,
};
use ferrum_output::{
    mp4::{Mp4Artifact, Mp4Encoder, Mp4EncoderConfig, Mp4ErrorKind, Mp4Spool, VideoAudioSpec},
    OutputBudget, OutputSession,
};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use ferrum_types::RequestCorrelation;
use std::task::Waker;

/// Fenced model/decoder output. Construction does not copy or reclassify bytes.
/// The encoder consumes these same owners; final external file readers do not
/// acquire fictitious Host ownership.
#[derive(Debug)]
pub struct RawVideoOutput {
    spec: VideoAudioSpec,
    rgb: ManagedHostBytes,
    stereo_f32le: ManagedHostBytes,
}
impl RawVideoOutput {
    pub fn new(
        spec: VideoAudioSpec,
        rgb: ManagedHostBytes,
        stereo_f32le: ManagedHostBytes,
        host: &HostMemoryAccount,
    ) -> Result<Self, ServiceError> {
        let output = Self {
            spec,
            rgb,
            stereo_f32le,
        };
        output.validate(host, spec)?;
        Ok(output)
    }
    pub fn spec(&self) -> VideoAudioSpec {
        self.spec
    }
    pub(super) fn validate(
        &self,
        host: &HostMemoryAccount,
        expected: VideoAudioSpec,
    ) -> Result<(), ServiceError> {
        if self.spec != expected
            || !host.owns(&self.rgb)
            || !host.owns(&self.stereo_f32le)
            || self.rgb.class() != HostMemoryClass::Output
            || self.stereo_f32le.class() != HostMemoryClass::Output
            || self.rgb.len() != expected.rgb_bytes().map_err(|e| invalid(e.to_string()))?
            || self.stereo_f32le.len()
                != expected.pcm_bytes().map_err(|e| invalid(e.to_string()))?
        {
            return Err(invalid("raw video output differs from its resolved spec or original Output-class Host owners"));
        }
        Ok(())
    }
    pub(super) fn into_parts(self) -> (ManagedHostBytes, ManagedHostBytes) {
        (self.rgb, self.stereo_f32le)
    }
}

/// Model-resolved audio/video lengths and one real output-session admission.
/// Audio sample count is supplied by the model's output plan, never inferred
/// by truncating video frames/fps. This is not a deserializable wire request.
pub struct VideoOutputPlan {
    spec: VideoAudioSpec,
    encoder: Mp4Encoder,
    spool: Mp4Spool,
    session: OutputSession,
}
impl VideoOutputPlan {
    pub fn new(
        spec: VideoAudioSpec,
        encoder: Mp4EncoderConfig,
        spool: Mp4Spool,
        budget: &OutputBudget,
    ) -> Result<Self, ServiceError> {
        spec.rgb_bytes()
            .and_then(|_| spec.pcm_bytes())
            .map_err(|e| invalid(e.to_string()))?;
        let encoder =
            Mp4Encoder::new(encoder, spool.clone()).map_err(|e| invalid(e.to_string()))?;
        let session = budget
            .try_open_session()
            .map_err(|e| service_error(ServiceErrorKind::QueueFull, e.to_string()))?;
        Ok(Self {
            spec,
            encoder,
            spool,
            session,
        })
    }
    pub fn spec(&self) -> VideoAudioSpec {
        self.spec
    }
}
impl OutputGate for VideoOutputPlan {
    fn try_acquire(&self) -> Result<Option<OutputLease>, OutputGateError> {
        self.encoder
            .try_admit(self.session.clone(), self.spec)
            .map(|permit| permit.map(OutputLease::new))
            .map_err(|e| {
                if e.kind == Mp4ErrorKind::Capacity {
                    OutputGateError::Capacity(e.to_string())
                } else {
                    OutputGateError::Failed(e.to_string())
                }
            })
    }
    fn subscribe_capacity(&self, waker: Waker) -> OutputLease {
        OutputLease::new((
            self.session.subscribe_capacity(waker.clone()),
            self.spool.subscribe_capacity(waker),
        ))
    }
}
#[derive(Debug)]
pub struct VideoGenerationResult {
    pub correlation: RequestCorrelation,
    pub seed: u64,
    pub artifact: Mp4Artifact,
}
